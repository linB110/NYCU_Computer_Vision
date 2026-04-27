import cv2
import numpy as np
import scipy.io as sio
import os
from extract_keypoints import KeypointExtractor
from geometry import SfMGeometry
from visualizer import Visualizer

_DIR = os.path.dirname(os.path.abspath(__file__))

# select data input file
# given data : Mensona, Statue
# or your own data
MODE = "statue"  # "statue" or "mesona" or ... (TODO)

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
                    [0, 1.4219, 0],
                    [0, 0, 0.0010]])
        
        K2 = K1.copy()
        img1 = cv2.imread('data/Mesona1.JPG')
        img2 = cv2.imread('data/Mesona2.JPG')
        tex_name = 'data/Mesona1.JPG'
    else:
        #TODO
        pass

    # 2. feature extraction and matching
    extractor = KeypointExtractor('LOFTR')
    if extractor.kpt_type == 'LOFTR':
        pts1, pts2 = extractor.match_dense(img1, img2)
        print(f"matched points (LoFTR): {len(pts1)}")
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
    
    sio.savemat('sfm_data.mat', {'P': points_3d,    
                                 'p_img2': pts2[mask.ravel() > 0],    
                                 'M': np.hstack([R, t]),
                                 'tex_name': tex_name,    'im_index': 1})
    
if __name__ == "__main__":
    main()
