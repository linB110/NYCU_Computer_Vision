import cv2 
import numpy as np

class SfMGeometry:
    def __init__(self, K):
        # camera intrinsic matrix K 
        self.K = K
    
    def estimate_pose(self, pts1, pts2):
        # compute RANSAC mask
        E, mask = cv2.findEssentialMat(pts1, pts2, self.K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        
        
        # compute R, t from essential matrix
        _, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, self.K, mask=mask)
        
        return R, t, mask_pose
    
    def triangulate_points(self, pts1, pts2, R, t, mask):
        # triangulate 3D points
        
        # filter out outliers using mask
        pts1_inliers = pts1[mask.ravel() == 1]
        pts2_inliers = pts2[mask.ravel() == 1]
        
        # origin : P1 = [I | 0]
        cord1 = self.K @ np.hstack((np.eye(3), np.zeros((3, 1))))
        
        # second camera : P2 = [R | t]
        cord2 = self.K @ np.hstack((R, t))
        
        # triangulate points
        pts4d_hom = cv2.triangulatePoints(cord1, cord2, pts1_inliers.T, pts2_inliers.T)
        
        # transfer to  Cartesian coordinates
        points_3d = pts4d_hom[:3] / pts4d_hom[3]
        
        return points_3d.T
        
        
        