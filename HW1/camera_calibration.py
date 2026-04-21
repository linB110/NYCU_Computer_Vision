import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import camera_calibration_show_extrinsics as show
import os
from scipy.optimize import least_squares

# ─────────────────────────────────────────
# 1. 資料收集
# ─────────────────────────────────────────
def collect_points(image_dir: str, corner_x: int, corner_y: int, output_dir: str = "output"):
    """
    collect points from chessboard images

    Returns:
        objpoints: List[np.ndarray]  - 3D world coordinates 
        imgpoints: List[np.ndarray]  - 2D image coordinates
        img_size:  (width, height)
    """
    os.makedirs(output_dir, exist_ok=True)

    objp = np.zeros((corner_x * corner_y, 3), np.float32)
    objp[:, :2] = np.mgrid[0:corner_x, 0:corner_y].T.reshape(-1, 2)

    objpoints, imgpoints = [], []
    images = glob.glob(os.path.join(image_dir, "*.jpg"))
    img_size = None

    print("Start finding chessboard corners...")
    for fname in images:
        img  = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        base = os.path.splitext(os.path.basename(fname))[0]

        plt.figure(); plt.imshow(gray, cmap="gray")
        plt.savefig(f"{output_dir}/{base}_gray.png"); plt.close()

        print(f"  Finding corners: {fname}")
        ret, corners = cv2.findChessboardCorners(gray, (corner_x, corner_y), None)

        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)
            img_size = (gray.shape[1], gray.shape[0])

            cv2.drawChessboardCorners(img, (corner_x, corner_y), corners, ret)
            plt.figure(); plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.savefig(f"{output_dir}/{base}_corners.png"); plt.close()

    return objpoints, imgpoints, img_size


# ─────────────────────────────────────────
# 2. Homography
# ─────────────────────────────────────────
def calculate_homography(objp_single, imgp_single):
    """DLT + normalize 3×3 Homography。"""

    def normalize_points(pts):
        pts   = np.array(pts, dtype=np.float32)
        mean  = np.mean(pts, axis=0, keepdims=True)
        diff  = pts - mean
        scale = float(np.sqrt(2) / (np.mean(np.linalg.norm(diff, axis=1)) + 1e-8))
        T = np.array([[scale, 0, -scale * float(mean[0, 0])],
                      [0, scale, -scale * float(mean[0, 1])],
                      [0,     0,  1]], dtype=np.float32)
        ph = np.hstack((pts, np.ones((len(pts), 1), dtype=np.float32)))
        return (T @ ph.T).T[:, :2], T

    pts_obj = np.array(objp_single, dtype=np.float32)[:, :2]
    pts_img = np.array(imgp_single, dtype=np.float32).reshape(-1, 2)
    N = len(pts_obj)

    obj_n, T_obj = normalize_points(pts_obj)
    img_n, T_img = normalize_points(pts_img)

    A = np.zeros((2 * N, 9), dtype=np.float32)
    for i in range(N):
        x, y = float(obj_n[i, 0]), float(obj_n[i, 1])
        u, v = float(img_n[i, 0]), float(img_n[i, 1])
        A[2*i]   = [-x, -y, -1,  0,  0,  0, x*u, y*u, u]
        A[2*i+1] = [ 0,  0,  0, -x, -y, -1, x*v, y*v, v]

    _, _, Vt = np.linalg.svd(A)
    H_n = Vt[-1].reshape(3, 3)
    
    return np.linalg.inv(T_img) @ H_n @ T_obj


def compute_homographies(objpoints, imgpoints):
    """ Compute Homography for each image. """
    
    return [calculate_homography(op, ip) for op, ip in zip(objpoints, imgpoints)]


# ─────────────────────────────────────────
# 3. intrinsics matrix by Zhang's method）
# ─────────────────────────────────────────
def compute_intrinsic_matrix(homographies):
    """from all Homography to solve K"""

    def get_v(H, i, j):
        return np.array([
            H[0,i]*H[0,j],
            H[0,i]*H[1,j] + H[1,i]*H[0,j],
            H[1,i]*H[1,j],
            H[2,i]*H[0,j] + H[0,i]*H[2,j],
            H[2,i]*H[1,j] + H[1,i]*H[2,j],
            H[2,i]*H[2,j]
        ])

    V = []
    for H in homographies:
        V.append(get_v(H, 0, 1))
        V.append(get_v(H, 0, 0) - get_v(H, 1, 1))
    V = np.array(V)

    _, _, Vt = np.linalg.svd(V)
    b = Vt[-1]
    B11, B12, B22, B13, B23, B33 = b

    if B11 * (B11 * B22 - B12**2) <= 0:
        b = -b; B11, B12, B22, B13, B23, B33 = b

    v0      = (B12*B13 - B11*B23) / (B11*B22 - B12**2)
    lam     = B33 - (B13**2 + v0*(B12*B13 - B11*B23)) / B11
    alpha   = np.sqrt(lam / B11)
    beta    = np.sqrt(lam * B11 / (B11*B22 - B12**2))
    gamma   = -B12 * alpha**2 * beta / lam
    u0      = gamma * v0 / beta - B13 * alpha**2 / lam

    return np.array([[alpha, gamma, u0],
                     [0,      beta, v0],
                     [0,         0,  1]])


# ─────────────────────────────────────────
# 4. extrinsic parameters optimization 
# ─────────────────────────────────────────
def _build_extrinsic_params(homographies, K):
    K_inv  = np.linalg.inv(K)
    params = []

    for H in homographies:
        h1, h2, h3 = H[:, 0], H[:, 1], H[:, 2]

        lam1 = 1.0 / np.linalg.norm(K_inv @ h1)
        lam2 = 1.0 / np.linalg.norm(K_inv @ h2)
        sc   = (lam1 + lam2) / 2.0

        r1 = sc * (K_inv @ h1)
        r2 = sc * (K_inv @ h2)
        t  = sc * (K_inv @ h3)

        if t[2] < 0:
            r1, r2, t = -r1, -r2, -t

        r3 = np.cross(r1, r2)
        R  = np.column_stack((r1, r2, r3))
        U, _, Vt = np.linalg.svd(R)
        R = U @ Vt
        if np.linalg.det(R) < 0:
            U, _, Vt = np.linalg.svd(R)
            Vt[2] *= -1
            R = U @ Vt

        rvec, _ = cv2.Rodrigues(R)
        params.extend(rvec.flatten())
        params.extend(t.flatten())

    return params


# ─────────────────────────────────────────
# 5. Bundle Adjustment
# ─────────────────────────────────────────
def _project_points(objpts, rvec, tvec, K, dist):
    R, _ = cv2.Rodrigues(rvec.reshape(3, 1))
    pts_cam = (R @ objpts.T).T + tvec.reshape(1, 3)
    pts_n   = pts_cam[:, :2] / (pts_cam[:, 2:] + 1e-9)

    k1, k2, p1, p2, k3 = dist
    r2  = np.sum(pts_n**2, axis=1)
    rad = 1 + k1*r2 + k2*r2**2 + k3*r2**3
    dx  = 2*p1*pts_n[:,0]*pts_n[:,1] + p2*(r2 + 2*pts_n[:,0]**2)
    dy  = p1*(r2 + 2*pts_n[:,1]**2)  + 2*p2*pts_n[:,0]*pts_n[:,1]
    pd  = np.column_stack((pts_n[:,0]*rad + dx, pts_n[:,1]*rad + dy))

    u = K[0,0]*pd[:,0] + K[0,2]
    v = K[1,1]*pd[:,1] + K[1,2]
    
    return np.column_stack((u, v))


def _residuals(params, objpoints_list, imgpoints_list):
    K    = np.array([[params[0], 0, params[2]],
                     [0, params[1], params[3]],
                     [0,         0,          1]])
    dist = params[4:9]
    errs = []
    idx  = 9
    for op, ip in zip(objpoints_list, imgpoints_list):
        rvec = params[idx:idx+3]
        tvec = params[idx+3:idx+6]
        proj = _project_points(op, rvec, tvec, K, dist)
        errs.extend((proj - ip.reshape(-1, 2)).ravel())
        idx += 6
        
    return np.array(errs)


def optimize_calibration(K_init, objpoints, imgpoints, homographies):
    """Levenberg-Marquardt bundle adjustment"""
    
    params = [K_init[0,0], K_init[1,1], K_init[0,2], K_init[1,2],
              0., 0., 0., 0., 0.]
    params += _build_extrinsic_params(homographies, K_init)

    result = least_squares(_residuals, params,
                           args=(objpoints, imgpoints), method="lm")

    rms = np.sqrt(np.mean(result.fun**2))
    print(f"RMS Reprojection Error: {rms:.4f} px")

    K_opt   = np.array([[result.x[0], 0, result.x[2]],
                        [0, result.x[1], result.x[3]],
                        [0,           0,            1]])
    dist_opt = result.x[4:9]
    
    return K_opt, dist_opt, result.x


def extract_extrinsics(params, num_images):
    extrinsics = []
    idx = 9
    for _ in range(num_images):
        rvec = params[idx:idx+3].reshape(3, 1)
        tvec = params[idx+3:idx+6].reshape(3, 1)
        R, _ = cv2.Rodrigues(rvec)
        extrinsics.append(np.hstack((R, tvec)))
        idx += 6
        
    return np.array(extrinsics)


# ─────────────────────────────────────────
# 6. visualization
# ─────────────────────────────────────────
def visualize_extrinsics(mtx, extrinsics, output_dir="output"):
    fig = plt.figure(figsize=(10, 10))
    ax  = fig.add_subplot(111, projection="3d")

    cam_w, cam_h   = 0.064/0.1, 0.032/0.1
    scale_focal    = 1600
    board_w, board_h, sq = 8, 6, 1

    min_v, max_v = show.draw_camera_boards(
        ax, mtx, cam_w, cam_h, scale_focal,
        extrinsics, board_w, board_h, sq, True
    )

    mid   = [(mn + mx) * 0.5 for mn, mx in zip(min_v, max_v)]
    rang  = np.array([mx - mn for mn, mx in zip(min_v, max_v)]).max() / 2.0
    ax.set_xlim(mid[0]-rang, mid[0]+rang)
    ax.set_ylim(mid[1]-rang, 0)
    ax.set_zlim(mid[2]-rang, mid[2]+rang)

    ax.set_xlabel("x"); ax.set_ylabel("z"); ax.set_zlabel("-y")
    ax.set_title("Extrinsic Parameters Visualization")
    plt.savefig(f"{output_dir}/extrinsics.png")
    plt.close()
    print(f"Saved → {output_dir}/extrinsics.png")


# ─────────────────────────────────────────
# 7. main : calibrate_camera()
# ─────────────────────────────────────────
def calibrate_camera(
    image_dir:  str = "data",
    corner_x:   int = 7,
    corner_y:   int = 7,
    output_dir: str = "output",
    visualize:  bool = True,
):
    """
    fun full camera calibration pipeline

    Returns:
        mtx        (3, 3)  - optimized intrinsic matrix
        dist       (5,)    - distortion coefficients [k1, k2, p1, p2, k3]
        extrinsics (N,3,4) - extrinsic matrices for each image [R|t]
    """
    # Step 1: collect corner points
    objpoints, imgpoints, _ = collect_points(image_dir, corner_x, corner_y, output_dir)

    # Step 2: compute Homography
    homographies = compute_homographies(objpoints, imgpoints)

    # Step 3: initial intrinsic parameters
    K_init = compute_intrinsic_matrix(homographies)
    print("Initial K:\n", K_init)

    # Step 4: Bundle Adjustment
    mtx, dist, params_opt = optimize_calibration(K_init, objpoints, imgpoints, homographies)
    print("Optimized K:\n", mtx)
    print("Distortion coefficients:", dist)

    # Step 5: extract extrinsic parameters
    extrinsics = extract_extrinsics(params_opt, len(imgpoints))

    # Step 6: visualization 
    if visualize:
        visualize_extrinsics(mtx, extrinsics, output_dir)

    return mtx, dist, extrinsics


if __name__ == "__main__":
    mtx, dist, extrinsics = calibrate_camera(
        image_dir  = "data",
        corner_x   = 7,
        corner_y   = 7,
        output_dir = "output",
        visualize  = True,
    )