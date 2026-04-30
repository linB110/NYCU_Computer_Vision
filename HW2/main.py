import cv2
import numpy as np
import scipy.io as sio
import os
from extract_keypoints import KeypointExtractor
from geometry import SfMGeometry
from visualization import (draw_sift_matches, draw_keypoints,
                           draw_epipolar_lines, plot_3d_points)

_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# select data input file
# given data : Mensona, Statue
# or your own data
MODE = "my_data"  # "statue" or "mesona" or ... (TODO)
BA = True # bundle adjustment flag

def main():
    
    if MODE == "statue":
        K1 = np.array([[5426.566895, 0.678017, 330.096680],
                    [0, 5423.133301, 648.950012],
                    [0, 0, 1]])
        K2 = np.array([[5426.566895, 0.678017, 387.430023],
                    [0, 5423.133301, 620.616699],
                    [0, 0, 1]])
        
        img1 = cv2.imread('data/Statue1.bmp')
        img2 = cv2.imread('data/Statue2.bmp')
        tex_name = 'data/Statue1.bmp'

    elif MODE == "mesona":
        K1 = np.array([[1.4219, 0.0005, 0.5092],
                    [0, 1.4219, 0.3802],
                    [0, 0, 0.0010]])
        
        K2 = K1.copy()
        img1 = cv2.imread('data/Mesona1.JPG')
        img2 = cv2.imread('data/Mesona2.JPG')
        tex_name = 'data/Mesona1.JPG'
    elif MODE=="my_data":
        K1 = np.array([[4.32963581e+03, 0 ,2.13196307e+03],
                        [0, 4.31994272e+03 ,2.90475106e+03],
                        [0,0 ,1]])
        K2 = K1.copy()
        img1 = cv2.imread('my_data/tissue1.png')
        img2 = cv2.imread('my_data/tissue2.png')
        tex_name = 'my_data/tissue1.png'
        
    else:
        #TODO
        pass

    if img1 is None or img2 is None:
        raise FileNotFoundError(
            f"Failed to read input images for MODE='{MODE}'. "
            "Please check the image paths and file integrity."
        )

    # 2. feature extraction and matching
    extractor = KeypointExtractor('SIFT')
    kp1, kp2, matches = None, None, None
    
    # handling different extractors
    if extractor.kpt_type == 'LOFTR':
        pts1, pts2 = extractor.match_dense(img1, img2)
        print(f"matched points (LoFTR): {len(pts1)}")
    elif extractor.kpt_type == 'XFEAT':
        pts1, pts2, kp_stats = extractor.detect_and_match(img1, img2)
        print(f"matched points (XFeat): {len(pts1)}")
    else:
        kp1, desc1 = extractor.extract_keypoints(img1)
        kp2, desc2 = extractor.extract_keypoints(img2)
        print(f"extracted keypoints: kp1={len(kp1)}, kp2={len(kp2)}")
        matches = extractor.match_keypoints(desc1, desc2)
        pts1, pts2 = extractor.get_aligned_points(kp1, kp2, matches)
        print(f"matched points: {len(matches)}")

    # 2.1 extract color information for visualization
    colors = []
    for p in pts1:
        x, y = int(p[0]), int(p[1])
        color = img1[y, x][::-1] / 255.0
        colors.append(color)
    colors = np.array(colors)

    # 2.2 draw 2D keypoint scatter plots
    draw_keypoints(img1, img2, pts1, pts2, OUTPUT_DIR)

    # 3. estimate camera pose
    geometry = SfMGeometry(K1, K2)
    R, t, mask, F = geometry.estimate_pose(pts1, pts2)
    print(f"after RANSAC points: {np.sum(mask)}")
    if matches is not None:
        draw_sift_matches(
            img1, img2, kp1, kp2, matches, OUTPUT_DIR,
            filename="sift_matches.png",
            correspondence_scores=mask.ravel().astype(np.float32)
        )
        inlier_matches = [
            match for match, keep in zip(matches, mask.ravel())
            if keep > 0
        ]
        draw_sift_matches(
            img1, img2, kp1, kp2, inlier_matches, OUTPUT_DIR,
            filename="sift_matches_filtered.png",
            correspondence_scores=np.ones(len(inlier_matches), dtype=np.float32)
        )

    # 3.1 draw epipolar lines
    #     Recover F from E:  F = K2^{-T} @ E @ K1^{-1}
    E, _ = cv2.findEssentialMat(pts1, pts2, K1, method=cv2.RANSAC,
                                prob=0.999, threshold=1.0)
    F = np.linalg.inv(K2).T @ E @ np.linalg.inv(K1)
    F = F / F[2, 2]
    pts1_inlier = pts1[mask.ravel() > 0]
    pts2_inlier = pts2[mask.ravel() > 0]
    draw_epipolar_lines(img1, img2, pts1_inlier, pts2_inlier, F, OUTPUT_DIR)

    # 4. triangulate points
    points_3d = geometry.triangulate_points(pts1, pts2, R, t, mask)
    print(f"triangulate points: {len(points_3d)}")
    colors = colors[mask.ravel() > 0] 
    
    # 4.1 bundle adjustment: refine R, t, and 3D points
    if BA == True:
        print("running bundle adjustment...")
        points_3d, R, t = geometry.bundle_adjustment(points_3d, pts1_inlier, pts2_inlier, R, t)
        print("bundle adjustment done")
    
    # 4.2 validation 
    f_err = geometry.check_F(F, pts1_inlier, pts2_inlier)
    print(f"[validate] algebraic F error  — mean: {f_err.mean():.4f}, max: {f_err.max():.4f}  (ideal: ~0)")

    err1, err2 = geometry.reprojection_error(points_3d, pts1_inlier, pts2_inlier, R, t)
    print(f"[validate] reprojection error cam1 — mean: {err1.mean():.2f}px, max: {err1.max():.2f}px")
    print(f"[validate] reprojection error cam2 — mean: {err2.mean():.2f}px, max: {err2.max():.2f}px")
    print(f"[validate] (good SfM: mean < 2px)")
    
    # 5. pointcloud filter
    # filter out points behind the camera
    valid_idx = points_3d[:, 2] > 0
    points_3d = points_3d[valid_idx]
    colors = colors[valid_idx]
    print(f"in front of camera points: {len(points_3d)}")

    # 5.1 save 3D scatter plot
    plot_3d_points(points_3d, OUTPUT_DIR)
    
    # 6. visualization (open3d point cloud viewer)
    try:
        from visualizer import Visualizer
        visualizer = Visualizer()
        inlier_points = visualizer.clean_pointclouds(points_3d, colors)
        visualizer.show_pointclouds(inlier_points)
    except ImportError:
        print("[Warning] open3d not installed, skipping 3D point cloud viewer.")
    
    sio.savemat('sfm_data.mat', {'P': points_3d,    
                                 'p_img2': pts2[mask.ravel() > 0],    
                                 'M': np.hstack([R, t]),
                                 'tex_name': tex_name,    'im_index': 1})
    
if __name__ == "__main__":
    main()
