"""
generate_marker.py — produce a printable ArUco marker.

Run once, print marker.png at any size, and show it to the webcam in demo.py.
The marker's physical printed size (edge length in mm) must match
MARKER_SIZE_MM in demo.py for the position readout to be metric.
"""
import cv2
import numpy as np

# --- config ---------------------------------------------------------------
ARUCO_DICT = cv2.aruco.DICT_4X4_50   # must match demo.py
MARKER_ID = 0                        # any id in the dictionary range
SIDE_PX = 800                        # marker resolution (not physical size)
QUIET_ZONE_PX = 120                  # white border (required for detection)
OUT_PATH = "marker.png"
# --------------------------------------------------------------------------


def make_marker(dictionary, marker_id: int, side_px: int) -> np.ndarray:
    """Render a marker bitmap, handling both modern and legacy OpenCV APIs."""
    if hasattr(cv2.aruco, "generateImageMarker"):       # OpenCV >= 4.7
        return cv2.aruco.generateImageMarker(dictionary, marker_id, side_px)
    return cv2.aruco.drawMarker(dictionary, marker_id, side_px)  # legacy


def main() -> None:
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    marker = make_marker(dictionary, MARKER_ID, SIDE_PX)

    # Add a white quiet zone — ArUco detection is unreliable without it.
    canvas = np.full(
        (SIDE_PX + 2 * QUIET_ZONE_PX, SIDE_PX + 2 * QUIET_ZONE_PX),
        255,
        dtype=np.uint8,
    )
    canvas[QUIET_ZONE_PX:QUIET_ZONE_PX + SIDE_PX,
           QUIET_ZONE_PX:QUIET_ZONE_PX + SIDE_PX] = marker

    # Caption so a printed sheet is self-documenting.
    label = f"ArUco DICT_4X4_50  id={MARKER_ID}"
    cv2.putText(canvas, label, (QUIET_ZONE_PX, QUIET_ZONE_PX - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, 0, 2, cv2.LINE_AA)

    cv2.imwrite(OUT_PATH, canvas)
    print(f"[ok] wrote {OUT_PATH}  ({canvas.shape[1]}x{canvas.shape[0]} px)")
    print("     Print it, measure the black square's edge in mm,")
    print("     and set MARKER_SIZE_MM in demo.py to that value.")


if __name__ == "__main__":
    main()
