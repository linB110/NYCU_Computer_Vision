import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix


class SfMGeometry:
    def __init__(self, K1, K2=None):
        self.K1 = K1
        self.K2 = K2 if K2 is not None else K1

    # ------------------------------------------------------------------
    # Step 2: F via normalized 8-point + RANSAC
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_points(pts):
        # Hartley normalization: centroid -> origin, mean dist -> sqrt(2)
        centroid = pts.mean(axis=0)
        shifted = pts - centroid
        mean_dist = np.sqrt((shifted ** 2).sum(axis=1)).mean()
        scale = np.sqrt(2) / mean_dist if mean_dist > 1e-12 else 1.0
        T = np.array([[scale, 0, -scale * centroid[0]],
                      [0, scale, -scale * centroid[1]],
                      [0, 0, 1]])
        pts_h = np.hstack([pts, np.ones((len(pts), 1))])
        pts_norm = (T @ pts_h.T).T
        return pts_norm, T

    def compute_F_normalized_8pt(self, pts1, pts2):
        p1n, T1 = self._normalize_points(pts1)
        p2n, T2 = self._normalize_points(pts2)
        x1, y1 = p1n[:, 0], p1n[:, 1]
        x2, y2 = p2n[:, 0], p2n[:, 1]
        A = np.column_stack([x2 * x1, x2 * y1, x2,
                             y2 * x1, y2 * y1, y2,
                             x1, y1, np.ones_like(x1)])
        _, _, Vt = np.linalg.svd(A)
        F = Vt[-1].reshape(3, 3)
        # rank-2 enforcement
        U, S, Vt2 = np.linalg.svd(F)
        S[2] = 0
        F = U @ np.diag(S) @ Vt2
        # denormalize
        F = T2.T @ F @ T1
        return F / F[2, 2] if abs(F[2, 2]) > 1e-12 else F

    @staticmethod
    def _sampson_distance(F, pts1, pts2):
        N = len(pts1)
        p1h = np.hstack([pts1, np.ones((N, 1))])
        p2h = np.hstack([pts2, np.ones((N, 1))])
        Fp1 = (F @ p1h.T).T
        Ftp2 = (F.T @ p2h.T).T
        num = np.sum(p2h * Fp1, axis=1) ** 2
        denom = Fp1[:, 0] ** 2 + Fp1[:, 1] ** 2 + Ftp2[:, 0] ** 2 + Ftp2[:, 1] ** 2
        return num / np.maximum(denom, 1e-12)

    def compute_F_ransac(self, pts1, pts2, threshold=1.0, iters=2000, seed=42):
        rng = np.random.default_rng(seed)
        N = len(pts1)
        best_inliers = np.zeros(N, dtype=bool)
        best_count = 0
        for _ in range(iters):
            idx = rng.choice(N, 8, replace=False)
            try:
                F_cand = self.compute_F_normalized_8pt(pts1[idx], pts2[idx])
            except np.linalg.LinAlgError:
                continue
            d = self._sampson_distance(F_cand, pts1, pts2)
            inliers = d < threshold ** 2
            count = int(inliers.sum())
            if count > best_count:
                best_count = count
                best_inliers = inliers
        # refit on the full inlier set
        F = self.compute_F_normalized_8pt(pts1[best_inliers], pts2[best_inliers])
        return F, best_inliers

    # ------------------------------------------------------------------
    # Step 4: 4 candidate (R, t) from E
    # ------------------------------------------------------------------
    @staticmethod
    def decompose_E(E):
        # enforce two equal singular values and a zero third
        U, S, Vt = np.linalg.svd(E)
        m = (S[0] + S[1]) / 2
        E = U @ np.diag([m, m, 0]) @ Vt
        U, _, Vt = np.linalg.svd(E)
        # ensure proper rotations (det = +1) by flipping sign columns
        if np.linalg.det(U) < 0:
            U[:, -1] *= -1
        if np.linalg.det(Vt) < 0:
            Vt[-1, :] *= -1
        W = np.array([[0, -1, 0],
                      [1,  0, 0],
                      [0,  0, 1]])
        R1 = U @ W @ Vt
        R2 = U @ W.T @ Vt
        t = U[:, 2:3]
        return [(R1, t), (R1, -t), (R2, t), (R2, -t)]

    # ------------------------------------------------------------------
    # Step 5: cheirality test to pick the right (R, t)
    # ------------------------------------------------------------------
    def _count_in_front(self, R, t, pts1, pts2):
        P1 = self.K1 @ np.hstack([np.eye(3), np.zeros((3, 1))])
        P2 = self.K2 @ np.hstack([R, t])
        pts4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
        pts3d = pts4d[:3] / pts4d[3]
        z1 = pts3d[2]
        z2 = (R @ pts3d + t)[2]
        return int(np.sum((z1 > 0) & (z2 > 0)))

    def estimate_pose(self, pts1, pts2):
        # Step 2: F via normalized 8-point + RANSAC
        F, inliers = self.compute_F_ransac(pts1, pts2, threshold=1.0)

        # Essential matrix from F and intrinsics
        E = self.K2.T @ F @ self.K1

        # Step 4 + 5: decompose E into 4 candidates, pick the one with the
        # most points triangulated in front of both cameras.
        in1 = pts1[inliers]
        in2 = pts2[inliers]
        candidates = self.decompose_E(E)
        best = None
        best_count = -1
        for R, t in candidates:
            count = self._count_in_front(R, t, in1, in2)
            if count > best_count:
                best_count = count
                best = (R, t)
        R, t = best

        mask = inliers.astype(np.uint8).reshape(-1, 1)
        return R, t, mask, F

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Step 6: triangulation 
    # ------------------------------------------------------------------
    def triangulate_points(self, pts1, pts2, R, t, mask):
        pts1_inliers = pts1[mask.ravel() == 1]
        pts2_inliers = pts2[mask.ravel() == 1]
        cord1 = self.K1 @ np.hstack((np.eye(3), np.zeros((3, 1))))
        cord2 = self.K2 @ np.hstack((R, t))
        pts4d_hom = cv2.triangulatePoints(cord1, cord2, pts1_inliers.T, pts2_inliers.T)
        points_3d = pts4d_hom[:3] / pts4d_hom[3]
        return points_3d.T
