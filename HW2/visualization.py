"""
Visualization utilities for SfM pipelines.
==========================================
Reusable drawing functions for:
  - SIFT feature matches (side-by-side)
  - 2D keypoint scatter plots
  - Epipolar lines on image pairs
  - 3D point cloud scatter plots

Usage example:
    from visualization import draw_sift_matches, draw_epipolar_lines, plot_3d_points
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


# ============================================================
# 1. SIFT matches — side-by-side image
# ============================================================
def draw_sift_matches(img1, img2, kp1, kp2, good_matches, output_dir,
                      filename="sift_matches.png"):
    """
    Draw SIFT feature matches between two images and save the result.

    Parameters
    ----------
    img1, img2 : np.ndarray
        BGR images (as read by cv2.imread).
    kp1, kp2 : list of cv2.KeyPoint
        Keypoints detected in img1 and img2.
    good_matches : list of cv2.DMatch
        Filtered matches to draw.
    output_dir : str
        Directory to save the output image.
    filename : str
        Output filename (default: "sift_matches.png").

    Returns
    -------
    match_img : np.ndarray
        The side-by-side match image (BGR).
    """
    os.makedirs(output_dir, exist_ok=True)

    match_img = cv2.drawMatches(
        img1, kp1, img2, kp2, good_matches, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    out_path = os.path.join(output_dir, filename)
    cv2.imwrite(out_path, match_img)
    print(f"[Visualization] Saved {filename} -> {out_path}")
    return match_img


# ============================================================
# 2. 2D keypoint coordinate scatter plots
# ============================================================
def draw_keypoints(img1, img2, pts1, pts2, output_dir,
                   filename="keypoints_2d.png", title1="Image 1", title2="Image 2"):
    """
    Draw matched 2D keypoint locations on both images (scatter overlay)
    and save a side-by-side figure.

    Parameters
    ----------
    img1, img2 : np.ndarray
        BGR images.
    pts1, pts2 : np.ndarray, shape (N, 2)
        Matched 2D point coordinates in each image.
    output_dir : str
        Directory to save the output figure.
    filename : str
        Output filename (default: "keypoints_2d.png").
    title1, title2 : str
        Subplot titles.

    Returns
    -------
    None
    """
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Image 1
    axes[0].imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    axes[0].scatter(pts1[:, 0], pts1[:, 1], s=8, c="lime", edgecolors="black",
                    linewidths=0.3, zorder=5)
    axes[0].set_title(f"{title1}  ({len(pts1)} points)")
    axes[0].axis("off")

    # Image 2
    axes[1].imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    axes[1].scatter(pts2[:, 0], pts2[:, 1], s=8, c="lime", edgecolors="black",
                    linewidths=0.3, zorder=5)
    axes[1].set_title(f"{title2}  ({len(pts2)} points)")
    axes[1].axis("off")

    plt.tight_layout()
    out_path = os.path.join(output_dir, filename)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[Visualization] Saved {filename} -> {out_path}")


# ============================================================
# 3. Epipolar lines
# ============================================================
def _draw_lines_on_image(img, lines, pts, colors_bgr):
    """
    Internal helper: draw epipolar lines and their corresponding points
    on a single image.
    """
    h, w = img.shape[:2]
    out = img.copy()
    for line, pt, c in zip(lines, pts, colors_bgr):
        a, b, c_coeff = line
        # Compute two boundary points of the line
        if abs(b) > 1e-8:
            x0, y0 = 0, int(-c_coeff / b)
            x1, y1 = w, int(-(a * w + c_coeff) / b)
        else:
            x0, y0 = int(-c_coeff / a), 0
            x1, y1 = int(-(b * h + c_coeff) / a), h
        c_int = tuple(int(v * 255) for v in c)
        cv2.line(out, (x0, y0), (x1, y1), c_int, 1)
        cv2.circle(out, (int(pt[0]), int(pt[1])), 5, c_int, -1)
    return out


def draw_epipolar_lines(img1, img2, pts1, pts2, F, output_dir,
                        num_lines=30, seed=42,
                        fname_img1="epipolar_img1.png",
                        fname_img2="epipolar_img2.png",
                        fname_combined="epipolar_lines.png"):
    """
    Draw epipolar lines on both images for a random subset of matches
    and save individual + side-by-side figures.

    Epipolar geometry:
        - Lines on img2 from points in img1:  l2 = F   @ x1
        - Lines on img1 from points in img2:  l1 = F^T @ x2

    Parameters
    ----------
    img1, img2 : np.ndarray
        BGR images.
    pts1, pts2 : np.ndarray, shape (N, 2)
        Matched inlier coordinates.
    F : np.ndarray, shape (3, 3)
        Fundamental matrix satisfying  x2^T F x1 = 0.
    output_dir : str
        Directory to save output images.
    num_lines : int
        Number of epipolar lines to draw (default: 30).
    seed : int
        Random seed for reproducible subset selection.
    fname_img1, fname_img2, fname_combined : str
        Output filenames.

    Returns
    -------
    out1, out2 : np.ndarray
        BGR images with epipolar lines drawn.
    """
    os.makedirs(output_dir, exist_ok=True)

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(pts1), size=min(num_lines, len(pts1)), replace=False)
    idx = np.sort(idx)

    colors = plt.cm.hsv(np.linspace(0, 1, len(idx)))[:, :3]   # RGB 0-1
    colors_bgr = colors[:, ::-1]                                # BGR for OpenCV

    sel_pts1 = pts1[idx]
    sel_pts2 = pts2[idx]
    ones = np.ones((len(idx), 1))

    # Epipolar lines on image2 from points in image1
    lines2 = (F @ np.hstack([sel_pts1, ones]).T).T              # Nx3
    # Epipolar lines on image1 from points in image2
    lines1 = (F.T @ np.hstack([sel_pts2, ones]).T).T            # Nx3

    out1 = _draw_lines_on_image(img1, lines1, sel_pts1, colors_bgr)
    out2 = _draw_lines_on_image(img2, lines2, sel_pts2, colors_bgr)

    # Save individual images
    cv2.imwrite(os.path.join(output_dir, fname_img1), out1)
    cv2.imwrite(os.path.join(output_dir, fname_img2), out2)
    print(f"[Visualization] Saved {fname_img1}  {fname_img2}")

    # Side-by-side matplotlib figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    axes[0].imshow(cv2.cvtColor(out1, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Epipolar lines on Image 1")
    axes[0].axis("off")
    axes[1].imshow(cv2.cvtColor(out2, cv2.COLOR_BGR2RGB))
    axes[1].set_title("Epipolar lines on Image 2")
    axes[1].axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, fname_combined), dpi=150)
    plt.close()
    print(f"[Visualization] Saved {fname_combined}")

    return out1, out2


# ============================================================
# 4. 3D point cloud scatter plot
# ============================================================
def plot_3d_points(pts3d, output_dir, filename="3d_points.png",
                   title="Triangulated 3D Points", color="steelblue",
                   point_size=1, elev=None, azim=None):
    """
    Create and save a 3D scatter plot of triangulated points.

    Parameters
    ----------
    pts3d : np.ndarray, shape (N, 3)
        3D point coordinates (X, Y, Z).
    output_dir : str
        Directory to save the output figure.
    filename : str
        Output filename (default: "3d_points.png").
    title : str
        Plot title.
    color : str
        Point colour.
    point_size : float
        Marker size.
    elev, azim : float or None
        Viewing angles for the 3D plot.

    Returns
    -------
    None
    """
    os.makedirs(output_dir, exist_ok=True)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(pts3d[:, 0], pts3d[:, 1], pts3d[:, 2],
               s=point_size, c=color)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)
    if elev is not None or azim is not None:
        ax.view_init(elev=elev, azim=azim)
    plt.tight_layout()
    out_path = os.path.join(output_dir, filename)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[Visualization] Saved {filename} -> {out_path}")
