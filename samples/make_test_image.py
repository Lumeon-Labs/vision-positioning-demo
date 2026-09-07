"""
samples/make_test_image.py — build a synthetic "marker on a desk" test image
for `python demo.py --image ...` when no webcam or printed marker is on hand.

This does NOT photograph anything real. It renders a procedural desk-like
background (plain code, no external assets or network access) and
perspective-warps generate_marker.py's marker.png onto it at an angle, so
the composite exercises the same detect -> solvePnP -> draw pipeline that a
real camera frame would. Treat the resulting pose numbers as illustrative,
not a physical measurement — see README's "Honest limitations" section.

Run:
    python generate_marker.py          # writes marker.png (if missing)
    python samples/make_test_image.py  # writes samples/desk_with_marker.png
"""
import os

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
MARKER_PATH = os.path.join(_ROOT, "marker.png")
OUT_PATH = os.path.join(_HERE, "desk_with_marker.png")
BG_W, BG_H = 1600, 1200


def make_desk_background(w: int, h: int) -> np.ndarray:
    """Procedural wood-desk-ish gradient background (no external assets)."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    base = 92 + 30 * np.sin(yy / 26.0) + 0.01 * xx
    grain = np.sin(yy / 5.3) * 4.0
    gray = np.clip(base + grain, 0, 255).astype(np.float32)

    bg = np.zeros((h, w, 3), dtype=np.float32)
    bg[:, :, 0] = gray * 0.45   # B
    bg[:, :, 1] = gray * 0.65   # G
    bg[:, :, 2] = gray * 0.85   # R

    # Soft vignette so it reads as a lit desk surface, not a flat fill.
    cy, cx = h / 2.0, w / 2.0
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    vignette = 1.0 - 0.25 * (dist / dist.max())
    bg = bg * vignette[..., None]
    return np.clip(bg, 0, 255).astype(np.uint8)


def warp_marker_onto(bg: np.ndarray, marker: np.ndarray,
                      dst_quad: np.ndarray) -> np.ndarray:
    """Perspective-warp `marker` (full canvas incl. white quiet zone) onto
    `bg` at quadrilateral `dst_quad` (TL, TR, BR, BL corners), simulating a
    marker lying flat on a surface viewed from an angled camera."""
    mh, mw = marker.shape[:2]
    src = np.array([[0, 0], [mw - 1, 0], [mw - 1, mh - 1], [0, mh - 1]],
                    dtype=np.float32)
    M = cv2.getPerspectiveTransform(src, dst_quad.astype(np.float32))

    h, w = bg.shape[:2]
    warped = cv2.warpPerspective(marker, M, (w, h))
    mask = cv2.warpPerspective(
        np.full((mh, mw), 255, dtype=np.uint8), M, (w, h))
    mask3 = cv2.merge([mask, mask, mask]).astype(np.float32) / 255.0

    out = bg.astype(np.float32) * (1 - mask3) + warped.astype(np.float32) * mask3
    return np.clip(out, 0, 255).astype(np.uint8)


def main() -> None:
    if not os.path.exists(MARKER_PATH):
        raise SystemExit(
            f"[error] {MARKER_PATH} not found — run generate_marker.py first")
    marker = cv2.imread(MARKER_PATH, cv2.IMREAD_COLOR)

    bg = make_desk_background(BG_W, BG_H)

    # Tilted quad (TL, TR, BR, BL): top edge narrower/higher than the bottom
    # edge, as if a camera is looking down and across at a marker flat on a
    # desk — enough perspective skew for solvePnP to recover a real 6DoF pose.
    dst_quad = np.array([
        [520, 260],
        [1150, 210],
        [1240, 830],
        [470, 900],
    ], dtype=np.float32)

    composite = warp_marker_onto(bg, marker, dst_quad)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    cv2.imwrite(OUT_PATH, composite)
    print(f"[ok] wrote {OUT_PATH}  ({composite.shape[1]}x{composite.shape[0]} px)")
    print(f"     Try: python demo.py --image {OUT_PATH} --out docs/demo-static.png")


if __name__ == "__main__":
    main()
