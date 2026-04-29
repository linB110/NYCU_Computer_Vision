import time
import cv2
import numpy as np


class KeypointExtractor:
    def __init__(self, kpt_type, nFeatures=3000):
        self.kpt_type = kpt_type.upper()
        self.nFeatures = nFeatures

        if self.kpt_type == 'XFEAT':
            import sys, pathlib
            _repo = pathlib.Path(__file__).parent / 'accelerated_features'
            if str(_repo) not in sys.path:
                sys.path.insert(0, str(_repo))
            from modules.xfeat import XFeat as _XFeat
            self._xfeat = _XFeat()
        else:
            self.detector = self._get_detector()

    def _get_detector(self):
        if self.kpt_type == 'ORB':
            return cv2.ORB_create(self.nFeatures)
        elif self.kpt_type == 'SIFT':
            return cv2.SIFT_create(self.nFeatures)
        elif self.kpt_type == 'SURF':
            return cv2.xfeatures2d.SURF_create(self.nFeatures)
        elif self.kpt_type == 'BRISK':
            return cv2.BRISK_create()
        else:
            raise ValueError(f"Unsupported method: {self.kpt_type}")

    def detect_and_match(self, img1, img2):
        """Detect keypoints in both images and match them.
        Returns (pts1, pts2, stats) where pts are Nx2 float32 matched pixel coords
        and stats contains counts and timing."""
        t_start = time.perf_counter()

        if self.kpt_type == 'XFEAT':
            # XFeat accepts BGR numpy (H,W,C) directly; parse_input handles conversion
            out1 = self._xfeat.detectAndCompute(img1, top_k=self.nFeatures)[0]
            out2 = self._xfeat.detectAndCompute(img2, top_k=self.nFeatures)[0]
            t_extract = time.perf_counter() - t_start

            t1 = time.perf_counter()
            idxs0, idxs1 = self._xfeat.match(out1['descriptors'], out2['descriptors'], min_cossim=0.82)
            t_match = time.perf_counter() - t1

            pts1 = out1['keypoints'][idxs0].cpu().numpy().astype(np.float32)
            pts2 = out2['keypoints'][idxs1].cpu().numpy().astype(np.float32)
            n_kp1 = len(out1['keypoints'])
            n_kp2 = len(out2['keypoints'])

        else:
            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) if len(img1.shape) == 3 else img1
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) if len(img2.shape) == 3 else img2
            kp1, desc1 = self.detector.detectAndCompute(gray1, None)
            kp2, desc2 = self.detector.detectAndCompute(gray2, None)
            t_extract = time.perf_counter() - t_start

            t1 = time.perf_counter()
            matches = self.match_keypoints(desc1, desc2)
            t_match = time.perf_counter() - t1

            pts1, pts2 = self.get_aligned_points(kp1, kp2, matches)
            n_kp1, n_kp2 = len(kp1), len(kp2)

        stats = {
            'n_kp1': n_kp1, 'n_kp2': n_kp2, 'n_matches': len(pts1),
            't_extract_s': t_extract, 't_match_s': t_match,
        }
        return pts1, pts2, stats

    def match_keypoints(self, desc1, desc2):
        if desc1 is None or desc2 is None:
            return []
        if self.kpt_type in ['ORB', 'BRISK']:
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            return sorted(bf.match(desc1, desc2), key=lambda x: x.distance)
        elif self.kpt_type in ['SIFT', 'SURF']:
            index_params = dict(algorithm=1, trees=5)
            search_params = dict(checks=50)
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            knn_matches = flann.knnMatch(desc1, desc2, k=2)
            return [m for m, n in knn_matches if m.distance < 0.75 * n.distance]

    @staticmethod
    def get_aligned_points(kp1, kp2, matches):
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
        return pts1, pts2
