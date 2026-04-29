import cv2
import numpy as np
import torch
import sys, pathlib
import kornia.feature as KF
from models.superpoint import SuperPoint

class KeypointExtractor:
    def __init__(self, kpt_type, nFeatures=3000):
        # kpt_type: ORB, SIFT, SURF, BRISK, AKAZE, KAZE, XFeat, DISK, GFTT, LOFTR, SUPERPOINT
        self.kpt_type = kpt_type.upper()
        self.nFeatures = nFeatures
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
        elif self.kpt_type == 'AKAZE':
            return cv2.AKAZE_create()
        elif self.kpt_type == 'KAZE':
            return cv2.KAZE_create()
        
        # XFeat
        elif self.kpt_type == 'XFEAT':
            # if your GPU is not supported
            torch.cuda.is_available = lambda: False

            repo_path = pathlib.Path(__file__).parent / "accelerated_features"
            sys.path.insert(0, str(repo_path))
            from modules.xfeat import XFeat

            self._xfeat = XFeat()
            
        # --- Kornia Based Models ---
        elif self.kpt_type == 'DISK':
            self.disk_model = KF.DISK.from_pretrained('depth').to(self.device).eval()
            return None
        elif self.kpt_type == 'GFTT':
            self.keynet_model = KF.GFTTAffNetHardNet(num_features=self.nFeatures).to(self.device).eval()
            return None
        elif self.kpt_type == 'LOFTR':
            self.loftr_model = KF.LoFTR('outdoor').to(self.device).eval()
            return None
        # super points
        elif self.kpt_type == 'SUPERPOINT':
            #  SuperPoint parameter
            config = {
                'nms_radius': 4,
                'keypoint_threshold': 0.005,
                'max_keypoints': self.nFeatures
            }
            # read models/weights/superpoint_v1.pth
            self.superpoint_model = SuperPoint(config).to(self.device).eval()
            return None
        else:
            raise ValueError(f"Unsupported method: {self.kpt_type}")

    def extract_keypoints(self, img):
        # deep learning Tensor transformation
        if self.kpt_type in ['DISK', 'GFTT', 'SUPERPOINT']:
            if len(img.shape) == 3:
                # DISK  RGB，other use Gray
                input_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if self.kpt_type == 'DISK' else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                input_img = img
                if self.kpt_type == 'DISK': # DISK 
                    input_img = cv2.cvtColor(input_img, cv2.COLOR_GRAY2RGB)

            tensor = torch.from_numpy(input_img).float() / 255.0
            if len(tensor.shape) == 2: tensor = tensor.unsqueeze(0) # (1, H, W)
            else: tensor = tensor.permute(2, 0, 1) # (3, H, W)
            tensor = tensor.unsqueeze(0).to(self.device) # (1, C, H, W)

            with torch.no_grad():
                if self.kpt_type == 'DISK':
                    features = self.disk_model(tensor, self.nFeatures, pad_if_not_divisible=True)
                    kp_array = features[0].keypoints.cpu().numpy()
                    desc = features[0].descriptors.cpu().numpy()
                
                elif self.kpt_type == 'GFTT':
                    lafs, _, desc_t = self.keynet_model(tensor)
                    kp_array = lafs[0, :, :, 2].cpu().numpy()
                    desc = desc_t[0].cpu().numpy()

                elif self.kpt_type == 'SUPERPOINT':
                    # SuperPoint input dict format
                    out = self.superpoint_model({'image': tensor})
                    
                    # extract keypoints and descriptors
                    # output: keypoints [1, N, 2], descriptors [1, 256, N]
                    kp_array = out['keypoints'][0].cpu().numpy()
                    
                    # OpenCV descripters (N, 256)，transpose to permute(1, 0)
                    desc = out['descriptors'][0].permute(1, 0).cpu().numpy()

            kps = [cv2.KeyPoint(float(k[0]), float(k[1]), 1) for k in kp_array]
            return kps, desc

        if self.kpt_type == 'LOFTR':
            return [], None

        # OpenCV traditional keypoint detection
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return self.detector.detectAndCompute(img, None)

    def match_keypoints(self, desc1, desc2):
        if desc1 is None or desc2 is None: return []

        # binary descripter Hamming
        if self.kpt_type in ['ORB', 'BRISK', 'AKAZE']:
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(desc1, desc2)
            return sorted(matches, key=lambda x: x.distance)

        # descripter (SIFT, DISK, SuperPoint ) use L2
        else:
            index_params = dict(algorithm=1, trees=5)
            search_params = dict(checks=50)
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            knn_matches = flann.knnMatch(desc1, desc2, k=2)
            # Ratio Test
            matches = [m for m, n in knn_matches if m.distance < 0.75 * n.distance]
            return matches

    def get_aligned_points(self, kp1, kp2, matches):
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
        return pts1, pts2

    def match_dense(self, img1, img2, conf_threshold=0.5):
        assert self.kpt_type == 'LOFTR', "match_dense 僅供 LOFTR 使用"
        
        def preprocess(img):
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            return (torch.from_numpy(img).float() / 255.0).unsqueeze(0).unsqueeze(0).to(self.device)

        t0, t1 = preprocess(img1), preprocess(img2)
        with torch.no_grad():
            out = self.loftr_model({'image0': t0, 'image1': t1})

        mask = out['confidence'].cpu().numpy() >= conf_threshold
        return out['keypoints0'].cpu().numpy()[mask], out['keypoints1'].cpu().numpy()[mask]
    
    
    # =========== XFeat ============
    def detect_and_match(self, img1, img2):
        import time
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
