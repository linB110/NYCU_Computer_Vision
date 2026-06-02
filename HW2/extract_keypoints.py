import cv2
import numpy as np
import torch
import sys, pathlib
import kornia.feature as KF
from models.superpoint import SuperPoint


class KeypointExtractor:
    def __init__(self, kpt_type, nFeatures=3000):
        self.kpt_type = kpt_type.upper()
        self.nFeatures = nFeatures
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.detector = self._get_detector()

    # ------------------------------------------------------------------ #
    #  Detector / Model initialization
    # ------------------------------------------------------------------ #
    def _get_detector(self):
        # ---- OpenCV Classical ----
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

        elif self.kpt_type == 'HARRIS':
            self._harris_params = dict(
                maxCorners=self.nFeatures,
                qualityLevel=0.01,
                minDistance=5,
                blockSize=3,
                useHarrisDetector=True,
                k=0.04
            )
            return None

        elif self.kpt_type == 'FAST':
            # FAST detector
            return cv2.FastFeatureDetector_create(threshold=20, nonmaxSuppression=True)

        elif self.kpt_type == 'AGAST':
            return cv2.AgastFeatureDetector_create(threshold=10, nonmaxSuppression=True)

        elif self.kpt_type == 'MSER':
            self._mser = cv2.MSER_create()
            self._orb_desc = cv2.ORB_create(self.nFeatures)
            return None

        # ---- XFeat ----
        elif self.kpt_type == 'XFEAT':
            torch.cuda.is_available = lambda: False
            repo_path = pathlib.Path(__file__).parent / "accelerated_features"
            sys.path.insert(0, str(repo_path))
            from modules.xfeat import XFeat
            self._xfeat = XFeat()
            return None

        # ---- Kornia ----
        elif self.kpt_type == 'DISK':
            self.disk_model = KF.DISK.from_pretrained('depth').to(self.device).eval()
            return None

        elif self.kpt_type == 'GFTT':
            self.keynet_model = KF.GFTTAffNetHardNet(num_features=self.nFeatures).to(self.device).eval()
            return None

        elif self.kpt_type == 'LOFTR':
            self.loftr_model = KF.LoFTR('outdoor').to(self.device).eval()
            return None

        elif self.kpt_type == 'KEYNET':
            self._keynet_model = KF.KeyNet(pretrained=True).to(self.device).eval()
            self._hardnet    = KF.HardNet8(pretrained=True).to(self.device).eval()
            self._affnet     = KF.LAFAffNetShapeEstimator(pretrained=True).to(self.device).eval()
            return None

        # ---- SuperPoint ----
        elif self.kpt_type == 'SUPERPOINT':
            config = {
                'nms_radius': 4,
                'keypoint_threshold': 0.005,
                'max_keypoints': self.nFeatures
            }
            self.superpoint_model = SuperPoint(config).to(self.device).eval()
            return None

        # ---- External repos ----
        elif self.kpt_type == 'R2D2':
            repo_path = pathlib.Path(__file__).parent / "r2d2"
            sys.path.insert(0, str(repo_path))
            from tools.dataloader import norm_RGB
            from nets.r2d2 import R2D2 as R2D2Net
            self._r2d2_model = R2D2Net()
            ckpt = torch.load(repo_path / "models/r2d2_WASF_N16.pt", map_location=self.device)
            self._r2d2_model.load_state_dict(ckpt['state_dict'])
            self._r2d2_model = self._r2d2_model.to(self.device).eval()
            self._r2d2_norm  = norm_RGB
            return None

        elif self.kpt_type == 'D2NET':
            repo_path = pathlib.Path(__file__).parent / "d2-net"
            sys.path.insert(0, str(repo_path))
            from lib.model_test import D2Net as D2NetModel
            self._d2net_model = D2NetModel(
                model_file=str(repo_path / "models/d2_tf.pth"),
                use_relu=True,
                use_cuda=(self.device.type == 'cuda')
            )
            return None

        elif self.kpt_type == 'ALIKE':
            repo_path = pathlib.Path(__file__).parent / "ALIKE"
            sys.path.insert(0, str(repo_path))
            from alike import ALike, configs
            self._alike_model = ALike(
                **configs['alike-l'],
                device=self.device,
                top_k=self.nFeatures,
                scores_th=0.2,
                n_limit=self.nFeatures
            )
            return None

        else:
            raise ValueError(f"Unsupported method: {self.kpt_type}")

    # ------------------------------------------------------------------ #
    #  Keypoint extraction
    # ------------------------------------------------------------------ #
    def extract_keypoints(self, img):
        # gray conversion
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()

        # ============================================================
        #  Classical OpenCV (no descriptor computation in extract_keypoints)
        # ============================================================
        if self.kpt_type == 'HARRIS':
            corners = cv2.goodFeaturesToTrack(gray, **self._harris_params)
            if corners is None:
                return [], None
            kps = [cv2.KeyPoint(float(c[0][0]), float(c[0][1]), 5) for c in corners]
            brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
            kps, desc = brief.compute(gray, kps)
            return kps, desc

        elif self.kpt_type == 'FAST':
            kps = self.detector.detect(gray, None)
            kps = sorted(kps, key=lambda k: -k.response)[:self.nFeatures]
            brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
            kps, desc = brief.compute(gray, kps)
            return kps, desc

        elif self.kpt_type == 'AGAST':
            kps = self.detector.detect(gray, None)
            kps = sorted(kps, key=lambda k: -k.response)[:self.nFeatures]
            brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
            kps, desc = brief.compute(gray, kps)
            return kps, desc

        elif self.kpt_type == 'MSER':
            msers, _ = self._mser.detectRegions(gray)
            kps_raw = []
            for region in msers:
                x, y, w, h = cv2.boundingRect(region)
                kps_raw.append(cv2.KeyPoint(x + w / 2, y + h / 2, float(max(w, h))))
            kps, desc = self._orb_desc.compute(gray, kps_raw)
            return kps, desc

        # ============================================================
        #  Classical OpenCV (with built-in detectAndCompute)
        # ============================================================
        elif self.kpt_type in ['ORB', 'SIFT', 'SURF', 'BRISK', 'AKAZE', 'KAZE']:
            return self.detector.detectAndCompute(gray, None)

        # ============================================================
        #  Deep Learning — Kornia / custom tensor path
        # ============================================================
        elif self.kpt_type == 'DISK':
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if len(img.shape) == 3 \
                  else cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            tensor = torch.from_numpy(rgb).float().permute(2, 0, 1).unsqueeze(0).to(self.device) / 255.0
            with torch.no_grad():
                features = self.disk_model(tensor, self.nFeatures, pad_if_not_divisible=True)
            kp_array = features[0].keypoints.cpu().numpy()
            desc     = features[0].descriptors.cpu().numpy()
            kps = [cv2.KeyPoint(float(k[0]), float(k[1]), 1) for k in kp_array]
            return kps, desc

        elif self.kpt_type == 'GFTT':
            tensor = torch.from_numpy(gray).float().unsqueeze(0).unsqueeze(0).to(self.device) / 255.0
            with torch.no_grad():
                lafs, _, desc_t = self.keynet_model(tensor)
            kp_array = lafs[0, :, :, 2].cpu().numpy()
            desc     = desc_t[0].cpu().numpy()
            kps = [cv2.KeyPoint(float(k[0]), float(k[1]), 1) for k in kp_array]
            return kps, desc

        elif self.kpt_type == 'SUPERPOINT':
            tensor = torch.from_numpy(gray).float().unsqueeze(0).unsqueeze(0).to(self.device) / 255.0
            with torch.no_grad():
                out = self.superpoint_model({'image': tensor})
            kp_array = out['keypoints'][0].cpu().numpy()
            desc     = out['descriptors'][0].permute(1, 0).cpu().numpy()
            kps = [cv2.KeyPoint(float(k[0]), float(k[1]), 1) for k in kp_array]
            return kps, desc

        elif self.kpt_type == 'KEYNET':
            tensor = torch.from_numpy(gray).float().unsqueeze(0).unsqueeze(0).to(self.device) / 255.0
            with torch.no_grad():
                lafs, _ = self._keynet_model(tensor)
                lafs    = self._affnet(lafs, tensor)
                patches = KF.extract_patches_from_pyramid(tensor, lafs, PS=32)
                B, N, CH, H, W = patches.shape
                desc = self._hardnet(patches.view(B * N, CH, H, W)).view(B, N, -1)
            kp_array = lafs[0, :, :, 2].cpu().numpy()
            desc_np  = desc[0].cpu().numpy()
            kps = [cv2.KeyPoint(float(k[0]), float(k[1]), 1) for k in kp_array]
            return kps, desc_np

        elif self.kpt_type == 'ALIKE':
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if len(img.shape) == 3 \
                  else cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            with torch.no_grad():
                pred = self._alike_model(rgb, sub_pixel=True)
            kp_array = pred['keypoints']
            desc     = pred['descriptors']
            kps = [cv2.KeyPoint(float(k[0]), float(k[1]), 1) for k in kp_array]
            return kps, desc

        elif self.kpt_type == 'R2D2':
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(self.device)
            with torch.no_grad():
                res = self._r2d2_model(imgs=[tensor])
            xys  = res['keypoints'][0].cpu().numpy()
            desc = res['descriptors'][0].cpu().numpy()
            kps = [cv2.KeyPoint(float(k[0]), float(k[1]), 1) for k in xys]
            return kps, desc

        elif self.kpt_type == 'D2NET':
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
            with torch.no_grad():
                kp_array, desc, _ = self._d2net_model.detect_and_describe(rgb)
            kps = [cv2.KeyPoint(float(k[0]), float(k[1]), 1) for k in kp_array]
            return kps, desc.astype(np.float32)

        # ============================================================
        #  Dense matcher — no keypoints from extract_keypoints
        # ============================================================
        elif self.kpt_type == 'LOFTR':
            return [], None

        # XFeat -> detect_and_match
        elif self.kpt_type == 'XFEAT':
            raise RuntimeError("XFeat use detect_and_match(), not extract_keypoints()")

        else:
            raise ValueError(f"Unsupported method in extract_keypoints: {self.kpt_type}")

    # ------------------------------------------------------------------ #
    #  Dense matching (LoFTR)
    # ------------------------------------------------------------------ #
    def match_dense(self, img1, img2, conf_threshold=0.5):
        assert self.kpt_type == 'LOFTR', "match_dense for LOFTR use only"

        def _to_tensor(img):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
            return torch.from_numpy(gray).float().unsqueeze(0).unsqueeze(0).to(self.device) / 255.0

        with torch.no_grad():
            out = self.loftr_model({'image0': _to_tensor(img1), 'image1': _to_tensor(img2)})
        mask = out['confidence'].cpu().numpy() >= conf_threshold
        return out['keypoints0'].cpu().numpy()[mask], out['keypoints1'].cpu().numpy()[mask]

    # ------------------------------------------------------------------ #
    #  Detect + Match（XFeat only）
    # ------------------------------------------------------------------ #
    def detect_and_match(self, img1, img2):
        import time
        t_start = time.perf_counter()

        #  use extract_keypoints detector（ deep-learning and classical method）
        _EXTRACT_TYPES = {
            'ORB', 'SIFT', 'SURF', 'BRISK', 'AKAZE', 'KAZE',
            'HARRIS', 'FAST', 'AGAST', 'MSER',
            'DISK', 'GFTT', 'SUPERPOINT', 'KEYNET',
            'ALIKE', 'R2D2', 'D2NET'
        }

        if self.kpt_type == 'XFEAT':
            out1 = self._xfeat.detectAndCompute(img1, top_k=self.nFeatures)[0]
            out2 = self._xfeat.detectAndCompute(img2, top_k=self.nFeatures)[0]
            t_extract = time.perf_counter() - t_start

            t1 = time.perf_counter()
            idxs0, idxs1 = self._xfeat.match(out1['descriptors'], out2['descriptors'], min_cossim=0.82)
            t_match = time.perf_counter() - t1

            pts1   = out1['keypoints'][idxs0].cpu().numpy().astype(np.float32)
            pts2   = out2['keypoints'][idxs1].cpu().numpy().astype(np.float32)
            n_kp1  = len(out1['keypoints'])
            n_kp2  = len(out2['keypoints'])

        elif self.kpt_type == 'LOFTR':
            pts1, pts2 = self.match_dense(img1, img2)
            t_extract  = time.perf_counter() - t_start
            t_match    = 0.0
            n_kp1 = n_kp2 = len(pts1)

        elif self.kpt_type in _EXTRACT_TYPES:
            kp1, desc1 = self.extract_keypoints(img1)
            kp2, desc2 = self.extract_keypoints(img2)
            t_extract  = time.perf_counter() - t_start

            t1 = time.perf_counter()
            matches = self.match_keypoints(desc1, desc2)
            t_match = time.perf_counter() - t1

            pts1, pts2 = self.get_aligned_points(kp1, kp2, matches)
            n_kp1, n_kp2 = len(kp1), len(kp2)

        else:
            raise ValueError(f"detect_and_match: unsupported type {self.kpt_type}")

        return pts1, pts2, {
            'n_kp1': n_kp1, 'n_kp2': n_kp2, 'n_matches': len(pts1),
            't_extract_s': t_extract, 't_match_s': t_match,
        }

    # ------------------------------------------------------------------ #
    #  Descriptor Matching
    # ------------------------------------------------------------------ #
    def match_keypoints(self, desc1, desc2):
        if desc1 is None or desc2 is None:
            return []

        HAMMING_TYPES = {'ORB', 'BRISK', 'HARRIS', 'FAST', 'AGAST', 'MSER'}
        L2_KNN_TYPES  = {'SIFT', 'SURF', 'DISK', 'GFTT', 'SUPERPOINT',
                         'KEYNET', 'ALIKE', 'R2D2', 'D2NET', 'KAZE', 'AKAZE'}

        if self.kpt_type in HAMMING_TYPES:
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            return sorted(bf.match(desc1, desc2), key=lambda x: x.distance)

        elif self.kpt_type in L2_KNN_TYPES:
            index_params  = dict(algorithm=1, trees=5)
            search_params = dict(checks=50)
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            knn   = flann.knnMatch(desc1.astype(np.float32), desc2.astype(np.float32), k=2)
            return [m for m, n in knn if m.distance < 0.75 * n.distance]

        else:
            raise ValueError(f"match_keypoints: unsupported type {self.kpt_type}")

    # ------------------------------------------------------------------ #
    #  Helper
    # ------------------------------------------------------------------ #
    @staticmethod
    def get_aligned_points(kp1, kp2, matches):
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
        return pts1, pts2
