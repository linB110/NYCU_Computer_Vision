import cv2
import numpy as np

class KeypointExtractor:
    def __init__(self, kpt_type, nFeatures=3000):
        # ketpoint type: ORB, SIFT, SURF, BRISK
        self.kpt_type = kpt_type.upper()
        self.nFeatures = nFeatures
        
        self.detector = self._get_detector()
        
    def _get_detector(self):
        if self.kpt_type == 'ORB':
            return cv2.ORB_create(self.nFeatures)
        
        elif self.kpt_type == 'SIFT':
            return cv2.SIFT_create(self.nFeatures)
        
        elif self.kpt_type == 'SURF':
            return cv2.xfeatures2d.SURF_create(self.nFeatures)
        
        elif self.kpt_type == 'BRISK':
            return cv2.BRISK_create(self.nFeatures)
        else:
            raise ValueError(f"Unsupported method: {self.kpt_type}")

    def extract_keypoints(self, img):
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
        return self.detector.detectAndCompute(img, None)
    
    def match_keypoints(self, desc1, desc2):
        if desc1 is None or desc2 is None:
            return []
        # binary descriptors (ORB, BRISK) use Hamming distance
        if self.kpt_type in ['ORB', 'BRISK']:
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(desc1, desc2)
            return sorted(matches, key=lambda x: x.distance)

        # float descriptors (SIFT, SURF) use L2 distance
        elif self.kpt_type in ['SIFT', 'SURF']:
            index_params = dict(algorithm=1, trees=5) 
            search_params = dict(checks=50)
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            knn_matches = flann.knnMatch(desc1, desc2, k=2)

            matches = []
            for m, n in knn_matches:
                if m.distance < 0.75 * n.distance:
                    matches.append(m)
                    
            return matches

    @staticmethod
    def get_aligned_points(kp1, kp2, matches):
        """
        return (x, y) coordinates of matched keypoints 
        in the format of two numpy arrays: pts1, pts2
        """
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
        
        return pts1, pts2
