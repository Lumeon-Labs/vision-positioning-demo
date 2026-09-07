"""
demo.py — Real-time 6DoF visual positioning with ArUco markers.

Two entry points, one pipeline:

    python demo.py                                   # webcam, live loop
    python demo.py --image in.png --out result.png   # single static image

Both modes call the exact same detect -> solvePnP -> draw routine
(`estimate_pose_and_annotate`), so the static-image path is not a separate
reimplementation — it is the same code the webcam loop runs, frame by frame.
This is the same fiducial-positioning technique used for fab hand-eye
calibration and vision-guided pick-and-place.

Webcam controls:  q = quit   s = save snapshot to captures/

Accuracy note: without calibration the script uses *nominal* intrinsics
(focal length ~= frame width). The 3D axis and orientation are stable, but
metric position is approximate. Run calibrate.py to produce camera_params.npz
for true metric output. See README.
"""
import argparse
import math
import os
import time

import cv2
import numpy as np

# --- config ---------------------------------------------------------------
CAMERA_INDEX = 0
ARUCO_DICT = cv2.aruco.DICT_4X4_50      # must match generate_marker.py
MARKER_SIZE_MM = 50.0                   # printed black-square edge length (mm)
CALIB_PATH = "camera_params.npz"        # optional, from calibrate.py
CONSOLE_EVERY = 15                      # frames between console pose prints
CAPTURE_DIR = "captures"
# --------------------------------------------------------------------------

# Colors (BGR)
C_OK = (120, 230, 120)
C_ACCENT = (255, 200, 80)
C_TEXT = (240, 240, 240)
C_DIM = (170, 170, 170)


def load_intrinsics(frame_w: int, frame_h: int):
    """Load calibrated intrinsics if present, else fall back to nominal."""
    if os.path.exists(CALIB_PATH):
        data = np.load(CALIB_PATH)
        print(f"[calib] loaded {CALIB_PATH}")
        return data["camera_matrix"].astype(np.float32), \
            data["dist_coeffs"].astype(np.float32), True
    f = float(frame_w)  # rough focal length in pixels
    cam = np.array([[f, 0, frame_w / 2.0],
                    [0, f, frame_h / 2.0],
                    [0, 0, 1.0]], dtype=np.float32)
    dist = np.zeros((5, 1), dtype=np.float32)
    print("[calib] no camera_params.npz — using NOMINAL intrinsics "
          "(position approximate; run calibrate.py for metric accuracy)")
    return cam, dist, False


def make_detector(dictionary):
    """Return (detector, legacy_params). Handles OpenCV >=4.7 and legacy."""
    if hasattr(cv2.aruco, "ArucoDetector"):                 # OpenCV >= 4.7
        if hasattr(cv2.aruco, "DetectorParameters"):
            params = cv2.aruco.DetectorParameters()
        else:
            params = cv2.aruco.DetectorParameters_create()
        return cv2.aruco.ArucoDetector(dictionary, params), None
    # legacy free-function API
    params = cv2.aruco.DetectorParameters_create()
    return None, params


def detect(detector, legacy_params, dictionary, gray):
    if detector is not None:
        return detector.detectMarkers(gray)
    return cv2.aruco.detectMarkers(gray, dictionary, parameters=legacy_params)


def marker_object_points(size_mm: float) -> np.ndarray:
    """3D corners of the marker in its own frame, order matching ArUco."""
    h = size_mm / 2.0
    return np.array([[-h,  h, 0.0],
                     [ h,  h, 0.0],
                     [ h, -h, 0.0],
                     [-h, -h, 0.0]], dtype=np.float32)


def rvec_to_euler_deg(rvec) -> np.ndarray:
    """Rotation vector -> roll/pitch/yaw in degrees."""
    R, _ = cv2.Rodrigues(rvec)
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if sy > 1e-6:
        roll = math.atan2(R[2, 1], R[2, 2])
        pitch = math.atan2(-R[2, 0], sy)
        yaw = math.atan2(R[1, 0], R[0, 0])
    else:
        roll = math.atan2(-R[1, 2], R[1, 1])
        pitch = math.atan2(-R[2, 0], sy)
        yaw = 0.0
    return np.degrees([roll, pitch, yaw])


def draw_panel(frame, lines, calibrated):
    """Top-left HUD panel."""
    x, y = 12, 24
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 30), (20, 20, 20), -1)
    title = "Vision Positioning Demo  |  ArUco 6DoF"
    cv2.putText(frame, title, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                C_ACCENT, 2, cv2.LINE_AA)
    tag = "CALIBRATED" if calibrated else "NOMINAL (approx position)"
    cv2.putText(frame, tag, (frame.shape[1] - 320, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                C_OK if calibrated else C_DIM, 1, cv2.LINE_AA)
    yy = 56
    for ln in lines:
        cv2.putText(frame, ln, (x, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    C_TEXT, 1, cv2.LINE_AA)
        yy += 24


def estimate_pose_and_annotate(frame, detector, legacy_params, dictionary,
                                obj_pts, cam, dist, pnp_flag):
    """Core pipeline shared by webcam and --image modes.

    Detect ArUco markers in `frame`, solve each one's 6DoF pose, and draw
    the marker outline + 3D axis + id label on a copy of the frame.

    Returns (annotated_frame, poses, panel_lines) where `poses` is a list of
    dicts: {id, x, y, z, roll, pitch, yaw} (position mm, rotation deg).
    """
    annotated = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detect(detector, legacy_params, dictionary, gray)

    poses = []
    if ids is not None and len(ids) > 0:
        cv2.aruco.drawDetectedMarkers(annotated, corners, ids)
        for i, mid in enumerate(ids.flatten()):
            img_pts = corners[i][0].astype(np.float32)
            ok_pnp, rvec, tvec = cv2.solvePnP(
                obj_pts, img_pts, cam, dist, flags=pnp_flag)
            if not ok_pnp:
                continue
            cv2.drawFrameAxes(annotated, cam, dist, rvec, tvec,
                               MARKER_SIZE_MM * 0.5, 2)
            x, y, z = tvec.flatten()
            roll, pitch, yaw = rvec_to_euler_deg(rvec)

            # marker-side label
            cx, cy = img_pts.mean(axis=0).astype(int)
            cv2.putText(annotated, f"id{mid}", (cx - 14, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, C_OK, 2, cv2.LINE_AA)

            poses.append({
                "id": int(mid),
                "x": float(x), "y": float(y), "z": float(z),
                "roll": float(roll), "pitch": float(pitch), "yaw": float(yaw),
            })

    if poses:
        panel = []
        for p in poses:
            panel.append(
                f"id{p['id']}  pos(mm) X{p['x']:7.1f} Y{p['y']:7.1f} Z{p['z']:7.1f}")
            panel.append(
                f"      rot(deg) R{p['roll']:6.1f} P{p['pitch']:6.1f} Y{p['yaw']:6.1f}")
    else:
        panel = ["no marker — show marker.png to the camera"]

    return annotated, poses, panel


def build_pipeline(frame_w: int, frame_h: int):
    """Shared setup: dictionary/detector/object-points/intrinsics/PnP flag.

    Both run_webcam() and run_image() call this once with the first frame's
    dimensions, then reuse the same objects for every frame.
    """
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    detector, legacy_params = make_detector(dictionary)
    obj_pts = marker_object_points(MARKER_SIZE_MM)
    cam, dist, calibrated = load_intrinsics(frame_w, frame_h)
    pnp_flag = getattr(cv2, "SOLVEPNP_IPPE_SQUARE", cv2.SOLVEPNP_ITERATIVE)
    return dictionary, detector, legacy_params, obj_pts, cam, dist, calibrated, pnp_flag


def print_poses(poses):
    for p in poses:
        print(f"-> POSE id={p['id']} pos=({p['x']:.1f},{p['y']:.1f},{p['z']:.1f})mm "
              f"rot=({p['roll']:.1f},{p['pitch']:.1f},{p['yaw']:.1f})deg")


def run_webcam() -> None:
    """Live loop: read the webcam, run the pipeline on every frame, show it."""
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise SystemExit(f"[error] cannot open camera index {CAMERA_INDEX}")

    ok, frame = cap.read()
    if not ok:
        raise SystemExit("[error] camera opened but no frame")
    h, w = frame.shape[:2]
    dictionary, detector, legacy_params, obj_pts, cam, dist, calibrated, pnp_flag = \
        build_pipeline(w, h)

    os.makedirs(CAPTURE_DIR, exist_ok=True)
    frame_i = 0
    print("[run] press q to quit, s to snapshot")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_i += 1

        annotated, poses, panel = estimate_pose_and_annotate(
            frame, detector, legacy_params, dictionary, obj_pts, cam, dist, pnp_flag)

        if poses and frame_i % CONSOLE_EVERY == 0:
            print_poses(poses)

        draw_panel(annotated, panel, calibrated)
        cv2.imshow("Vision Positioning Demo", annotated)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            path = os.path.join(CAPTURE_DIR, f"snap_{int(time.time())}.png")
            cv2.imwrite(path, annotated)
            print(f"[snap] {path}")

    cap.release()
    cv2.destroyAllWindows()


def run_image(image_path: str, out_path: str) -> None:
    """Static-image mode: run the identical pipeline on one file, save a PNG.

    Useful when no webcam / printed marker is on hand — see
    samples/make_test_image.py for a synthetic "marker on a desk" input that
    exercises this same code path end-to-end without any hardware.
    """
    frame = cv2.imread(image_path)
    if frame is None:
        raise SystemExit(f"[error] cannot read image: {image_path}")

    h, w = frame.shape[:2]
    dictionary, detector, legacy_params, obj_pts, cam, dist, calibrated, pnp_flag = \
        build_pipeline(w, h)

    annotated, poses, panel = estimate_pose_and_annotate(
        frame, detector, legacy_params, dictionary, obj_pts, cam, dist, pnp_flag)
    draw_panel(annotated, panel, calibrated)

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(out_path, annotated)

    if poses:
        print_poses(poses)
    else:
        print("[warn] no marker detected in image")
    print(f"[ok] wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ArUco 6DoF pose demo — webcam by default, or a single "
                    "--image for a static run with no camera needed.")
    parser.add_argument("--image", metavar="PATH",
                        help="run the pipeline on one image instead of the webcam")
    parser.add_argument("--out", metavar="PATH", default="result.png",
                        help="output PNG path for --image mode (default: result.png)")
    args = parser.parse_args()

    if args.image:
        run_image(args.image, args.out)
    else:
        run_webcam()


if __name__ == "__main__":
    main()
