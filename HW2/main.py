import argparse
import os
import cv2
import numpy as np
import scipy.io as sio
from extract_keypoints import KeypointExtractor
from geometry import SfMGeometry
from visualizer import Visualizer


def parse_args():
    parser = argparse.ArgumentParser(description='SfM pipeline (HW2)')
    parser.add_argument('--mode', choices=['statue', 'mesona'], default='statue',
                        help='dataset to run on (default: statue)')
    parser.add_argument('--output-dir', default='output',
                        help='where to save epipolar plots etc. (default: output)')
    parser.add_argument('--keypoint-type', choices=['ORB', 'SIFT', 'SURF', 'BRISK', 'XFEAT'], default='SIFT',
                        help='keypoint type (default: SIFT)')
    parser.add_argument('--no-ba', action='store_true',
                        help='skip bundle adjustment')
    return parser.parse_args()


def main():
    args = parse_args()
    MODE = args.mode
    OUTPUT_DIR = args.output_dir
    os.makedirs(OUTPUT_DIR, exist_ok=True)

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
        K1 = K1 / K1[2, 2]               # normalize so that K[2, 2] = 1

        K2 = K1.copy()
        img1 = cv2.imread('data/Mesona1.JPG')
        img2 = cv2.imread('data/Mesona2.JPG')
        tex_name = 'data/Mesona1.JPG'
    else:
        raise ValueError(f"unknown mode: {MODE}")

    # 2. feature extraction and matching
    extractor = KeypointExtractor(args.keypoint_type)
    pts1, pts2, kp_stats = extractor.detect_and_match(img1, img2)
    print(f"[{args.keypoint_type}] keypoints: {kp_stats['n_kp1']} / {kp_stats['n_kp2']}")
    print(f"[{args.keypoint_type}] matches:   {kp_stats['n_matches']}")
    print(f"[{args.keypoint_type}] time:      extract {kp_stats['t_extract_s']:.3f}s  "
          f"match {kp_stats['t_match_s']:.3f}s  "
          f"total {kp_stats['t_extract_s'] + kp_stats['t_match_s']:.3f}s")

    # 2.1 extract color information for visualization
    colors = []
    for p in pts1:
        x, y = int(p[0]), int(p[1])
        color = img1[y, x][::-1] / 255.0
        colors.append(color)
    colors = np.array(colors)
    
    # 3. estimate camera pose (F via normalized 8-pt + RANSAC, then E -> 4 candidates -> cheirality)
    geometry = SfMGeometry(K1, K2)
    R, t, mask, F = geometry.estimate_pose(pts1, pts2)
    print(f"after RANSAC points: {np.sum(mask)}")

    # 3.1 epipolar line visualization
    inl = mask.ravel() > 0
    Visualizer.draw_epipolar_lines(
        img1, img2, pts1[inl], pts2[inl], F,
        num_lines=30, save_path=os.path.join(OUTPUT_DIR, f'epipolar_{MODE}.png'))

    # 4. triangulate points
    points_3d = geometry.triangulate_points(pts1, pts2, R, t, mask)
    print(f"triangulate points: {len(points_3d)}")
    inlier_pts1 = pts1[inl]
    inlier_pts2 = pts2[inl]
    colors = colors[inl]

    # 4.1 bundle adjustment: refine R, t, and 3D points
    if not args.no_ba:
        print("running bundle adjustment...")
        points_3d, R, t = geometry.bundle_adjustment(points_3d, inlier_pts1, inlier_pts2, R, t)
        print("bundle adjustment done")
    else:
        print("bundle adjustment skipped")

    # 4.2 validation
    f_err = geometry.check_F(F, pts1[inl], pts2[inl])
    print(f"[validate] algebraic F error  — mean: {f_err.mean():.4f}, max: {f_err.max():.4f}  (ideal: ~0)")

    err1, err2 = geometry.reprojection_error(points_3d, inlier_pts1, inlier_pts2, R, t)
    print(f"[validate] reprojection error cam1 — mean: {err1.mean():.2f}px, max: {err1.max():.2f}px")
    print(f"[validate] reprojection error cam2 — mean: {err2.mean():.2f}px, max: {err2.max():.2f}px")
    print(f"[validate] (good SfM: mean < 2px)")

    # 5. pointcloud filter
    # filter out points behind the camera
    valid_idx = points_3d[:, 2] > 0
    points_3d = points_3d[valid_idx]
    inlier_pts1 = inlier_pts1[valid_idx]
    inlier_pts2 = inlier_pts2[valid_idx]
    colors = colors[valid_idx]
    print(f"in front of camera points: {len(points_3d)}")

    # 6. save point cloud (no display on headless server)
    points_3d, colors, clean_mask = Visualizer.clean_pointclouds(points_3d, colors)
    inlier_pts1 = inlier_pts1[clean_mask]   # keep in sync for MATLAB export
    ply_path = os.path.join(OUTPUT_DIR, f'pointcloud_{MODE}.ply')
    Visualizer.save_pointclouds(points_3d, colors, ply_path)
    Visualizer.plot_pointcloud(points_3d, colors,
                               filename=os.path.join(OUTPUT_DIR, f'pointcloud_{MODE}.png'),
                               title=f'SfM Point Cloud — {MODE}')

    # 7. dump for MATLAB step-7 obj_main. 
    M_tex = K1 @ np.hstack([np.eye(3), np.zeros((3, 1))])
    sio.savemat(f'{OUTPUT_DIR}/{MODE}_sfm_data.mat', {'P': points_3d,
                                 'p_img2': inlier_pts1,
                                 'M': M_tex,
                                 'tex_name': tex_name,
                                 'im_index': 1})
    
if __name__ == "__main__":
    main()