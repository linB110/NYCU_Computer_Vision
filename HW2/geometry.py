import cv2 
import numpy as np

# for BA optimization
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

class SfMGeometry:
    def __init__(self, K1, K2=None):
        # camera intrinsic matrix K 
        self.K1 = K1
        self.K2 = K2 if K2 is not None else K1
    
    def normalize_pts(self, pts):
        """normalize points and return transformation matrix T"""
        
        # decenterize and scale points to have mean distance sqrt(2)
        center = np.mean(pts, axis=0)
        scale = np.sqrt(2) / np.mean(np.linalg.norm(pts - center, axis=1))
        
        T = np.array([
            [scale,     0, -scale * center[0]],
            [    0, scale, -scale * center[1]],
            [    0,     0,                  1]
        ])
        pts_h = np.hstack((pts, np.ones((len(pts), 1))))
        
        return (T @ pts_h.T).T, T

    def run_8_point(self, pts1, pts2):
        """Normalized 8-point algorithm and estimate Fundamental Matrix F"""
        assert len(pts1) == len(pts2) == 8
        n1, T1 = self.normalize_pts(pts1)
        n2, T2 = self.normalize_pts(pts2)
        A = np.array([
            [n2[i,0]*n1[i,0], n2[i,0]*n1[i,1], n2[i,0],
            n2[i,1]*n1[i,0], n2[i,1]*n1[i,1], n2[i,1],
            n1[i,0],         n1[i,1],          1]
            for i in range(8)
        ])
        _, _, V = np.linalg.svd(A)
        F = V[-1].reshape(3, 3)
        # enforce rank-2 constraint
        U, S, V = np.linalg.svd(F)
        S[2] = 0
        F = U @ np.diag(S) @ V
        
        # denormalize
        F = T2.T @ F @ T1
        
        return F / F[2, 2]
    
    def sampson_distance(self, pts1, pts2, F):
        p1 = np.hstack((pts1, np.ones((len(pts1),1))))
        p2 = np.hstack((pts2, np.ones((len(pts2),1))))
        Fp1 = (F @ p1.T).T
        FTp2 = (F.T @ p2.T).T
        
        # ref : https://amroamroamro.github.io/mexopencv/matlab/cv.sampsonDistance.html
        denom = Fp1[:,0]**2 + Fp1[:,1]**2 + FTp2[:,0]**2 + FTp2[:,1]**2
        reproj_err = np.einsum('ij,ij->i', p2, (F @ p1.T).T)**2 / denom
        
        return reproj_err

    def estimate_F_RANSAC(self, pts1, pts2, thres=1.0, max_iter=5000, conf=0.5):
        best_F, best_mask, best_count = None, None, 0
        n = len(pts1)
        
        for _ in range(max_iter):
            idx = np.random.choice(n, 8, replace=False)
            try:
                F = self.run_8_point(pts1[idx], pts2[idx])
            except Exception:
                continue
            err = self.sampson_distance(pts1, pts2, F)
            mask = err < thres
            if mask.sum() > best_count:
                best_F, best_mask, best_count = F, mask, mask.sum()
                if best_count >= n * conf:
                    break  # early stop
                
        if best_F is None:
            raise RuntimeError("RANSAC failed: no valid F found")
        
        return best_F, best_mask
    
    def compute_E_from_F(self, F):
        """E = K2^T @ F @ K1 with rank-2 constraint"""
        E = self.K2.T @ F @ self.K1
        U, S, V = np.linalg.svd(E)
        m = (S[0] + S[1]) / 2
        E = U @ np.diag([m, m, 0]) @ V
        
        return E

    def four_solutions(self, E):
        U, S, V = np.linalg.svd(E)
        
        if np.linalg.det(U @ V) < 0:
            V = -V
            
        W = np.array([[0,-1,0],[1,0,0],[0,0,1]])
        t = U[:, 2:]
        R1 = U @ W @ V
        R2 = U @ W.T @ V
        
        return [
            np.hstack((R1,  t)), np.hstack((R1, -t)),
            np.hstack((R2,  t)), np.hstack((R2, -t))
        ]

    def cheirality_check(self, pts1, pts2, P2_candidates):
        """select the solution with most points in front of both cameras"""
        P1 = self.K1 @ np.eye(3, 4)
        best_P2, best_count = None, 0
        
        for P2_rel in P2_candidates:
            P2 = self.K2 @ P2_rel
            pts4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
            pts3d = (pts4d[:3] / pts4d[3]).T
            
            # check if points are in front of both cameras
            front1 = pts3d[:, 2] > 0
            R2, t2 = P2_rel[:, :3], P2_rel[:, 3:]
            pts_cam2 = (R2 @ pts3d.T + t2).T
            front2 = pts_cam2[:, 2] > 0
            count = (front1 & front2).sum()
            if count > best_count:
                best_count, best_P2 = count, P2_rel
                
        return best_P2
    
    def estimate_pose(self, pts1, pts2):
        # RANSAC + 8-point F estimation
        F, mask = self.estimate_F_RANSAC(pts1, pts2)
        inlier_pts1 = pts1[mask]
        inlier_pts2 = pts2[mask]
        
        # E from F + 4 solutions
        E = self.compute_E_from_F(F)
        P2s = self.four_solutions(E)
        
        # Cheirality check
        best_P2 = self.cheirality_check(inlier_pts1, inlier_pts2, P2s)
        R, t = best_P2[:, :3], best_P2[:, 3:]
        
        # reconstruct full mask array for main.py compatibility
        full_mask = mask.astype(np.uint8).reshape(-1, 1)
        
        return R, t.reshape(3,1), full_mask, F  
    
    def triangulate_points(self, pts1, pts2, R, t, mask):
        # triangulate 3D points
        
        # filter out outliers using mask
        pts1_inliers = pts1[mask.ravel() == 1]
        pts2_inliers = pts2[mask.ravel() == 1]
        
        # origin : P1 = [I | 0]
        cord1 = self.K1 @ np.hstack((np.eye(3), np.zeros((3, 1))))
        
        # second camera : P2 = [R | t]
        cord2 = self.K2 @ np.hstack((R, t))
        
        # triangulate points
        pts4d_hom = cv2.triangulatePoints(cord1, cord2, pts1_inliers.T, pts2_inliers.T)
        
        # transfer to  Cartesian coordinates
        points_3d = pts4d_hom[:3] / pts4d_hom[3]
        
        return points_3d.T
        
    def reprojection_error(self, points_3d, pts1, pts2, R, t):
        """Mean reprojection error in pixels across both views."""
        P1 = self.K1 @ np.hstack([np.eye(3), np.zeros((3, 1))])
        P2 = self.K2 @ np.hstack([R, t])

        def project(P, X):
            X_h = np.hstack([X, np.ones((len(X), 1))]).T  # 4xN
            x_h = P @ X_h                                 # 3xN
            x = (x_h[:2] / x_h[2]).T                      # Nx2
            
            return x

        proj1 = project(P1, points_3d)
        proj2 = project(P2, points_3d)
        err1 = np.linalg.norm(proj1 - pts1, axis=1)
        err2 = np.linalg.norm(proj2 - pts2, axis=1)
        return err1, err2
    
    def check_F(self, F, pts1, pts2):
        """Algebraic error x'ᵀ F x for inlier matches — should be near 0."""
        N = len(pts1)
        p1h = np.hstack([pts1, np.ones((N, 1))])
        p2h = np.hstack([pts2, np.ones((N, 1))])
        errors = np.abs(np.sum(p2h * (F @ p1h.T).T, axis=1))
        
        return errors
    
    # ------------------------------------------------------------------
    # Bundle adjustment
    # ------------------------------------------------------------------
    @staticmethod
    def _project(K, R, t, pts3d):
        """Project Nx3 points through K[R|t], return Nx2 pixel coords."""
        pts_c = (R @ pts3d.T + t).T        # Nx3 in camera frame
        pts_h = (K @ pts_c.T).T            # Nx3 homogeneous image coords
        return pts_h[:, :2] / pts_h[:, 2:3]

    def _ba_residuals(self, params, pts1, pts2, n):
        R, _ = cv2.Rodrigues(params[:3])
        t    = params[3:6].reshape(3, 1)
        pts3d = params[6:].reshape(n, 3)
        r1 = self._project(self.K1, np.eye(3), np.zeros((3, 1)), pts3d) - pts1
        r2 = self._project(self.K2, R, t, pts3d) - pts2
        # sequential layout: all cam1 residuals then all cam2 residuals,
        # so it matches the sparsity structure (rows 0..2N-1 = cam1, 2N..4N-1 = cam2)
        return np.concatenate([r1.ravel(), r2.ravel()])

    @staticmethod
    def _ba_sparsity(n):
        # rows = 4N residuals (2 per camera per point)
        # cols = 6 + 3N params (6 camera-2 pose + 3N point coords)
        rows, cols = 4 * n, 6 + 3 * n
        J = lil_matrix((rows, cols), dtype=np.int8)
        for i in range(n):
            # cam1 residuals (rows 2i, 2i+1) depend only on point i (cols 6+3i .. 6+3i+2)
            J[2 * i:2 * i + 2,     6 + 3 * i:6 + 3 * i + 3] = 1
            # cam2 residuals (rows 2N+2i, 2N+2i+1) depend on pose (cols 0..5) + point i
            J[2 * n + 2 * i:2 * n + 2 * i + 2, :6]                    = 1
            J[2 * n + 2 * i:2 * n + 2 * i + 2, 6 + 3 * i:6 + 3 * i + 3] = 1
        return J

    def bundle_adjustment(self, points_3d, pts1, pts2, R, t):
        """Refine R, t, and 3D point positions to minimize reprojection error."""
        n = len(points_3d)
        rvec, _ = cv2.Rodrigues(R)
        x0 = np.hstack([rvec.ravel(), t.ravel(), points_3d.ravel()])

        result = least_squares(
            self._ba_residuals, x0,
            jac_sparsity=self._ba_sparsity(n),
            args=(pts1, pts2, n),
            method='trf', loss='linear',  
            max_nfev=500, verbose=0,
        )

        R_opt, _ = cv2.Rodrigues(result.x[:3])
        t_opt    = result.x[3:6].reshape(3, 1)
        pts_opt  = result.x[6:].reshape(n, 3)
        return pts_opt, R_opt, t_opt
        
