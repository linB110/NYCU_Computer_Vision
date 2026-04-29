import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
from scipy.spatial import KDTree


class Visualizer:
    @staticmethod
    def draw_epipolar_lines(img1, img2, pts1, pts2, F, num_lines=20, save_path=None, seed=0):
        """Draw points on each image and the corresponding epipolar lines on the other.
        Line in image 2 for point x in image 1: l' = F @ x.
        Line in image 1 for point x' in image 2: l  = F.T @ x'."""
        img1 = img1.copy()
        img2 = img2.copy()
        rng = np.random.default_rng(seed)
        if len(pts1) > num_lines:
            idx = rng.choice(len(pts1), num_lines, replace=False)
            pts1 = pts1[idx]
            pts2 = pts2[idx]

        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]

        def line_endpoints(line, w, h):
            a, b, c = line
            if abs(b) > 1e-6:
                return (0, int(round(-c / b))), (w - 1, int(round(-(a * (w - 1) + c) / b)))
            return (int(round(-c / a)), 0), (int(round(-c / a)), h - 1)

        for p1, p2 in zip(pts1, pts2):
            color = tuple(int(c) for c in rng.integers(0, 255, 3))
            l2 = F @ np.array([p1[0], p1[1], 1.0])
            p_a, p_b = line_endpoints(l2, w2, h2)
            cv2.line(img2, p_a, p_b, color, 1)
            cv2.circle(img2, (int(round(p2[0])), int(round(p2[1]))), 4, color, -1)
            cv2.circle(img1, (int(round(p1[0])), int(round(p1[1]))), 4, color, -1)

            l1 = F.T @ np.array([p2[0], p2[1], 1.0])
            p_a, p_b = line_endpoints(l1, w1, h1)
            cv2.line(img1, p_a, p_b, color, 1)

        # pad to same height for hstack
        h = max(h1, h2)
        if h1 != h:
            img1 = cv2.copyMakeBorder(img1, 0, h - h1, 0, 0, cv2.BORDER_CONSTANT)
        if h2 != h:
            img2 = cv2.copyMakeBorder(img2, 0, h - h2, 0, 0, cv2.BORDER_CONSTANT)
        combined = np.hstack([img1, img2])
        if save_path:
            cv2.imwrite(save_path, combined)
        return combined

    @staticmethod
    def clean_pointclouds(points_3d, colors=None, nb_neighbors=20, std_ratio=2.0):
        """Statistical outlier removal using KD-tree (numpy/scipy, no Open3D).
        Returns (cleaned_points, cleaned_colors, inlier_mask)."""
        tree = KDTree(points_3d)
        dists, _ = tree.query(points_3d, k=nb_neighbors + 1)
        mean_dists = dists[:, 1:].mean(axis=1)
        threshold = mean_dists.mean() + std_ratio * mean_dists.std()
        mask = mean_dists < threshold
        cleaned = points_3d[mask]
        cleaned_colors = colors[mask] if colors is not None else None
        return cleaned, cleaned_colors, mask

    @staticmethod
    def plot_pointcloud(points_3d, colors=None, filename='pointcloud.png', title='SfM Point Cloud'):
        """Save a 3D scatter plot of the point cloud as a PNG."""
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        c = colors if colors is not None else 'steelblue'
        ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2],
                   c=c, s=1, linewidths=0)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(title)
        plt.tight_layout()
        plt.savefig(filename, dpi=150)
        plt.close()
        print(f"point cloud plot saved to {filename}")

    @staticmethod
    def save_pointclouds(points_3d, colors=None, filename='SfM.ply'):
        """Write a binary-little-endian PLY file (no Open3D required)."""
        has_color = colors is not None
        header = [
            'ply',
            'format binary_little_endian 1.0',
            f'element vertex {len(points_3d)}',
            'property float x',
            'property float y',
            'property float z',
        ]
        if has_color:
            header += ['property uchar red', 'property uchar green', 'property uchar blue']
        header += ['end_header', '']
        header_bytes = '\n'.join(header).encode()

        xyz = points_3d.astype(np.float32)
        if has_color:
            rgb = (np.clip(colors, 0, 1) * 255).astype(np.uint8)
            data = np.hstack([xyz.view(np.uint8).reshape(-1, 12), rgb])
            dtype = np.dtype([('xyz', np.float32, (3,)), ('rgb', np.uint8, (3,))])
            vertices = np.empty(len(points_3d), dtype=dtype)
            vertices['xyz'] = xyz
            vertices['rgb'] = rgb
        else:
            dtype = np.dtype([('xyz', np.float32, (3,))])
            vertices = np.empty(len(points_3d), dtype=dtype)
            vertices['xyz'] = xyz

        with open(filename, 'wb') as f:
            f.write(header_bytes)
            vertices.tofile(f)
        print(f"saved {len(points_3d)} points to {filename}")
