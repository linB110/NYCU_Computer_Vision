import cv2
import numpy as np
import scipy.io as sio
from extract_keypoints import KeypointExtractor
from geometry import SfMGeometry
from visualizer import Visualizer


def main():
    
    # 1. initialize and read images
    K1 = np.array([[5426.566895,  0.678017,    330.096680],
                    [0,     5423.133301, 648.950012],
                    [0,     0,    1]])
    
    K2 = np.array([[5426.566895,  0.678017,    387.430023],
                    [0,     5423.133301, 620.616699],
                    [0,     0,    1]])
    
    
    img1 = cv2.imread('data/Statue1.bmp')
    img2 = cv2.imread('data/Statue2.bmp')
    
    # 2. feature extraction and matching
    extractor = KeypointExtractor('SIFT')
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
    
    # 3. estimate camera pose
    geometry = SfMGeometry(K1, K2)
    R, t, mask = geometry.estimate_pose(pts1, pts2)
    print(f"after RANSAC points: {np.sum(mask)}")

    # 4. triangulate points
    points_3d = geometry.triangulate_points(pts1, pts2, R, t, mask)
    print(f"triangulate points: {len(points_3d)}")
    colors = colors[mask.ravel() > 0] 
    
    # 5. pointcloud filter
    # filter out points behind the camera
    valid_idx = points_3d[:, 2] > 0
    points_3d = points_3d[valid_idx]
    colors = colors[valid_idx]
    print(f"in front of camera points: {len(points_3d)}")
    
    # 6. visualizaation
    visualizer = Visualizer()
    inlier_points = visualizer.clean_pointclouds(points_3d, colors)
    visualizer.show_pointclouds(inlier_points)
    
    # 7. save for matlab
    sio.savemat('sfm_data.mat', {'P': points_3d,    
                                 'p_img2': pts2[mask.ravel() > 0],    
                                 'M': np.hstack([R, t]),
                                 'tex_name': 'data/Statue1.bmp',    'im_index': 1})
    
if __name__ == "__main__":
    main()
