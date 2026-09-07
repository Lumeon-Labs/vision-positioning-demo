"""
calibrate.py — camera intrinsic calibration (optional, for metric accuracy).

Print a standard chessboard (default 9x6 *inner* corners), hold it at varied
angles/distances, and press SPACE to capture each view. After >= 12 good
views press 'c' to calibrate. Saves camera_params.npz, which demo.py loads
automatically for true metric position readout.

Controls:  SPACE = capture view   c = calibrate & save   q = quit
"""
import cv2
import numpy as np

# --- config ---------------------------------------------------------------
CAMERA_INDEX = 0
BOARD_COLS = 9            # inner corners per row
BOARD_ROWS = 6           # inner corners per column
SQUARE_MM = 25.0         # printed chessboard square size (mm)
MIN_VIEWS = 12
OUT_PATH = "camera_params.npz"
# --------------------------------------------------------------------------

CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


def object_grid() -> np.ndarray:
    grid = np.zeros((BOARD_ROWS * BOARD_COLS, 3), np.float32)
    grid[:, :2] = np.mgrid[0:BOARD_COLS, 0:BOARD_ROWS].T.reshape(-1, 2)
    return grid * SQUARE_MM


def main() -> None:
    objp = object_grid()
    obj_points, img_points = [], []

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise SystemExit(f"[error] cannot open camera index {CAMERA_INDEX}")

    size = None
    print(f"[run] capture >= {MIN_VIEWS} views. SPACE=capture c=calibrate q=quit")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        size = gray.shape[::-1]
        found, corners = cv2.findChessboardCorners(
            gray, (BOARD_COLS, BOARD_ROWS), None)

        view = frame.copy()
        if found:
            cv2.drawChessboardCorners(
                view, (BOARD_COLS, BOARD_ROWS), corners, found)
        cv2.putText(view, f"views: {len(obj_points)}  (need {MIN_VIEWS})",
                    (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (120, 230, 120) if len(obj_points) >= MIN_VIEWS
                    else (80, 200, 255), 2, cv2.LINE_AA)
        cv2.imshow("calibrate", view)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord(" ") and found:
            refined = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1), CRITERIA)
            obj_points.append(objp.copy())
            img_points.append(refined)
            print(f"[capture] view {len(obj_points)}")
        if key == ord("c"):
            if len(obj_points) < MIN_VIEWS:
                print(f"[warn] need >= {MIN_VIEWS} views, have {len(obj_points)}")
                continue
            rms, cam, dist, _, _ = cv2.calibrateCamera(
                obj_points, img_points, size, None, None)
            np.savez(OUT_PATH, camera_matrix=cam, dist_coeffs=dist)
            print(f"[ok] saved {OUT_PATH}  (RMS reprojection error: {rms:.3f}px)")
            print("     demo.py will now use these intrinsics automatically.")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
