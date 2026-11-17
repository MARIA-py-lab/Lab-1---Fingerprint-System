"""
fp_tools.py - helper functions for Lab 1 (Simple Fingerprint Biometric System)
CYSE304 Biometric Authentication

Put this file in the SAME folder as your notebook. You do not need to edit it.
Each function is short and commented so you can read how it works.
"""
import numpy as np
from skimage import io, filters, morphology, img_as_float
from scipy import ndimage as ndi
import matplotlib.pyplot as plt

# ------------------------------------------------------------------ 1. load
def load_image(path):
    """Read a fingerprint picture as numbers between 0 (black) and 1 (white)."""
    img = img_as_float(io.imread(path))
    if img.ndim == 3:              # colour picture -> grey
        img = img.mean(axis=2)
    return img

# ------------------------------------------------------------ 2. clean up
def enhance(img):
    """Reduce noise and even out the contrast (so ridges stand out)."""
    smooth = filters.gaussian(img, sigma=1.0)
    local_mean = filters.gaussian(smooth, sigma=12)
    local_std = np.sqrt(filters.gaussian((smooth - local_mean) ** 2, sigma=12)) + 1e-6
    return (smooth - local_mean) / local_std          # contrast-normalised image

def binarize(enh):
    """Black-and-white picture: True = ridge, False = valley."""
    return enh < 0                                    # ridges are darker than their surroundings

def thin(binary):
    """Shave every ridge down to a line one pixel wide (the 'skeleton')."""
    return morphology.skeletonize(binary)

# ------------------------------------------------------ 3. ridge direction
def ridge_orientation(img, blur=7):
    """Direction of the ridges at every pixel, in radians from 0 to pi."""
    gx = ndi.sobel(img, axis=1); gy = ndi.sobel(img, axis=0)
    gxx = filters.gaussian(gx * gx, blur); gyy = filters.gaussian(gy * gy, blur); gxy = filters.gaussian(gx * gy, blur)
    theta = 0.5 * np.arctan2(2 * gxy, gxx - gyy) + np.pi / 2
    return np.mod(theta, np.pi)

# ------------------------------------------------------- 4. find minutiae
def find_minutiae(skel, orientation, border=25, min_gap=9):
    """
    Walk over the skeleton and count the ridge neighbours of every ridge pixel:
        1 neighbour  -> ridge ENDING        (type 0)
        3 neighbours -> BIFURCATION (split) (type 1)
    Returns a table with one row per minutia: [x, y, angle, type].
    """
    h, w = skel.shape
    s = skel.astype(int)
    pts = []
    ys, xs = np.nonzero(skel)
    for y, x in zip(ys, xs):
        if y < border or x < border or y >= h - border or x >= w - border:
            continue                                  # ignore the picture edge (false minutiae live there)
        ring = [s[y-1, x], s[y-1, x+1], s[y, x+1], s[y+1, x+1], s[y+1, x], s[y+1, x-1], s[y, x-1], s[y-1, x-1]]
        crossing = sum(abs(ring[i] - ring[(i + 1) % 8]) for i in range(8)) // 2
        if crossing == 1:   pts.append((x, y, 0))
        elif crossing >= 3: pts.append((x, y, 1))
    # remove minutiae that are too close together (usually noise)
    keep = []
    for p in pts:
        if all((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 >= min_gap ** 2 for q in keep):
            keep.append(p)
    return np.array([(x, y, orientation[y, x], t) for x, y, t in keep], dtype=float).reshape(-1, 4)

# ----------------------------------------- 5. one call: image -> template
def extract_template(path_or_image):
    """Full pipeline: picture file (or picture array) -> list of minutiae (the 'template')."""
    img = load_image(path_or_image) if isinstance(path_or_image, str) else path_or_image
    enh = enhance(img)
    skel = thin(binarize(enh))
    return find_minutiae(skel, ridge_orientation(img))

# --------------------------------------------------------------- 6. match
def match(template_a, template_b, dist_tol=12, angle_tol=20, max_rotation=35):
    """
    Compare two templates. Returns (score, pairs).
    Step 1 ALIGN : pretend minutia i of A is the same point as minutia j of B, rotate and
                   shift B to fit, and do this for every sensible (i, j).
    Step 2 PAIR  : two minutiae are paired if they are close (dist_tol pixels), point the
                   same way (angle_tol degrees) and are the same type.
    Step 3 SCORE : keep the alignment with the most pairs.  score = pairs^2 / (nA * nB) * 100
    """
    A, B = np.asarray(template_a), np.asarray(template_b)
    if len(A) < 3 or len(B) < 3:
        return 0.0, []
    a_tol = np.radians(angle_tol); max_rot = np.radians(max_rotation)
    # all anchor choices (i from A, j from B) with the same type
    I, J = np.nonzero(A[:, 3][:, None] == B[:, 3][None, :])
    d = A[I, 2] - B[J, 2]
    d = (d + np.pi / 2) % np.pi - np.pi / 2           # rotation, folded into -90..+90 degrees
    ok = np.abs(d) <= max_rot
    I, J, d = I[ok], J[ok], d[ok]
    if len(I) == 0:
        return 0.0, []
    c, s = np.cos(d), np.sin(d)
    bx = B[:, 0][None, :] - B[J, 0][:, None]; by = B[:, 1][None, :] - B[J, 1][:, None]
    tx = c[:, None] * bx - s[:, None] * by + A[I, 0][:, None]        # B moved into A's coordinates
    ty = s[:, None] * bx + c[:, None] * by + A[I, 1][:, None]
    tang = np.mod(B[:, 2][None, :] + d[:, None], np.pi)
    dx = tx[:, :, None] - A[:, 0][None, None, :]; dy = ty[:, :, None] - A[:, 1][None, None, :]
    close = dx * dx + dy * dy <= dist_tol ** 2
    da = np.abs(tang[:, :, None] - A[:, 2][None, None, :]); da = np.minimum(da, np.pi - da)
    same = (A[:, 3][None, None, :] == B[:, 3][None, :, None])
    good = close & (da <= a_tol) & same                                # (anchors, nB, nA)
    count = np.minimum(good.any(axis=2).sum(axis=1), good.any(axis=1).sum(axis=1))
    k = int(np.argmax(count)); best = int(count[k])
    pairs = []
    used = set()
    for b in range(len(B)):
        cand = np.nonzero(good[k, b])[0]
        cand = [a for a in cand if a not in used]
        if cand:
            a = cand[int(np.argmin(dx[k, b, cand] ** 2 + dy[k, b, cand] ** 2))]
            used.add(a); pairs.append((a, b))
    score = 100.0 * len(pairs) ** 2 / (len(A) * len(B))
    return score, pairs

# ------------------------------------------------- 7. damage a picture (challenge)
def cover(img, fraction=0.5):
    """Hide the bottom part of the print (like a partial print): fraction=0.5 hides half."""
    out = img.copy(); out[int(img.shape[0] * (1 - fraction)):, :] = 1.0
    return out

# ------------------------------------------------------------- 8. pictures
def show_images(paths, titles=None, cols=4, size=3):
    rows = (len(paths) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * size, rows * size * 1.05))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes: ax.axis("off")
    for i, p in enumerate(paths):
        axes[i].imshow(load_image(p), cmap="gray"); axes[i].set_title(titles[i] if titles else p.split("/")[-1].split("\\")[-1], fontsize=10)
    plt.tight_layout(); plt.show()

def show_pipeline(path):
    """Show the four stages of feature extraction side by side."""
    img = load_image(path); enh = enhance(img); bw = binarize(enh); sk = thin(bw)
    mins = find_minutiae(sk, ridge_orientation(img))
    fig, ax = plt.subplots(1, 5, figsize=(16, 3.6))
    for a, im, t in zip(ax, [img, enh, bw, sk], ["1. Original", "2. Enhanced", "3. Black & white", "4. Thinned (1 pixel)"]):
        a.imshow(im, cmap="gray"); a.set_title(t, fontsize=11); a.axis("off")
    ax[4].imshow(img, cmap="gray"); ax[4].set_title("5. Minutiae found: %d" % len(mins), fontsize=11); ax[4].axis("off")
    for x, y, a_, t in mins:
        if t == 0: ax[4].plot(x, y, "o", mfc="none", mec="red", ms=9, mew=1.6)
        else:      ax[4].plot(x, y, "s", mfc="none", mec="blue", ms=9, mew=1.6)
    plt.tight_layout(); plt.show()
    return mins

def show_match(path_a, path_b, template_a, template_b, pairs, score):
    """Draw the two prints side by side and join the paired minutiae with lines (rough alignment by eye)."""
    ia, ib = load_image(path_a), load_image(path_b)
    h = max(ia.shape[0], ib.shape[0]); w = ia.shape[1]
    canvas = np.ones((h, ia.shape[1] + ib.shape[1] + 20)); canvas[:ia.shape[0], :ia.shape[1]] = ia; canvas[:ib.shape[0], ia.shape[1] + 20:] = ib
    plt.figure(figsize=(11, 5.5)); plt.imshow(canvas, cmap="gray"); plt.axis("off")
    off = ia.shape[1] + 20
    for a, b in pairs:
        plt.plot([template_a[a, 0], template_b[b, 0] + off], [template_a[a, 1], template_b[b, 1]], "-", color="lime", lw=1.2)
    plt.title("Paired minutiae: %d    score = %.1f" % (len(pairs), score), fontsize=13); plt.show()
