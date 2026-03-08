## Just for the wow factor, this version is slow, but tracks incredibly well.

import os
import csv
import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO
from collections import defaultdict, deque

# ----- VARIABLES ----------

# -- Paths -------------------------------------------------------------------
VIDEO_NAME  = "Road traffic video for object recognition_part_1.mp4"
OUTPUT_PATH = "../traffic_speed_output_supervision.mp4"  # annotated output video
CSV_PATH    = "../vehicle_speeds_supervision.csv"  # per-vehicle speed report

# -- YOLO class IDs (COCO dataset) -------------------------------------------
# 2 = car, 3 = motorcycle, 5 = bus, 7 = truck
VEHICLE_CLASS_IDS = [2, 3, 5, 7]

# Human-readable names for each class ID
CLASS_NAMES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

# GREEN-LINE TRIPWIRE POSITIONS  (pixel coordinates in the 1280x720 frame)
# Two virtual lines span the full road width.  Each line represents a
# real-world length of 25 m (given by the assignment specification).

GREEN_LINE_LENGTH_M = 25.0   # real-world length of each green line (metres)

# Upper green line (Line 1) — further from the camera
LINE_1_LEFT  = np.array([340, 400], dtype=np.float32)   # left endpoint  (x, y)
LINE_1_RIGHT = np.array([940, 400], dtype=np.float32)   # right endpoint (x, y)

# Lower green line (Line 2) — closer to the camera
LINE_2_LEFT  = np.array([100, 560], dtype=np.float32)
LINE_2_RIGHT = np.array([1180, 560], dtype=np.float32)

# Convenience: integer Y-coordinates for crossing detection
LINE_1_Y = int(LINE_1_LEFT[1])   # 400
LINE_2_Y = int(LINE_2_LEFT[1])   # 560

# Extra pixel margin around the zone for drawing (so we can display
# vehicles a few pixels before / after they cross the lines)
ZONE_MARGIN = 30

# PERSPECTIVE CALIBRATION
# A homography (3x3 matrix) that maps pixel coordinates in the camera
# frame to real-world metre coordinates on the flat road surface.
#
# SOURCE defines a trapezoid in the 1280x720 image that corresponds to
# a rectangular patch of road.  TARGET defines that rectangle in metres.
#
# These values were calibrated from lane-dash markings:
#   - Road width  ~28 m  (4 lanes + median + shoulders)
#   - Road length ~102 m  (6 dash-groups x 17 m each)

SOURCE = np.array([
    [ 490, 295],   # 0: top-left     (far left)
    [ 790, 295],   # 1: top-right    (far right)
    [1260, 700],   # 2: bottom-right (near right)
    [  20, 700],   # 3: bottom-left  (near left)
], dtype=np.float32)

TARGET = np.array([
    [ 0,   0],     # -> (0 m, 0 m)
    [28,   0],     # -> (28 m, 0 m)
    [28, 102],     # -> (28 m, 102 m)
    [ 0, 102],     # -> (0 m, 102 m)
], dtype=np.float32)

PERSPECTIVE_MATRIX = cv2.getPerspectiveTransform(SOURCE, TARGET)

# ============================================================================
# TRIPWIRE DISTANCE COMPUTATION
# ============================================================================
# Project the midpoint of each green line into BEV (metre space) and
# take the difference in Y to get the perspective-corrected inter-line
# distance.  A naive distance (uniform pixel scale) is also computed.

# ----- FUNCTIONS ----------

def _transform_pt(pt: np.ndarray) -> np.ndarray:
    """
    Transform a single 2-D point through the perspective matrix.
    
    Args:
        pt (numpy.ndarray): The 2-D pixel point to transform.
        
    Returns:
        numpy.ndarray: The transformed point in real-world meters.
    """
    src = np.array([[[float(pt[0]), float(pt[1])]]], dtype=np.float32)
    return cv2.perspectiveTransform(src, PERSPECTIVE_MATRIX)[0, 0]

# Midpoints of each green line (pixel space)
_centre_1 = (LINE_1_LEFT + LINE_1_RIGHT) / 2.0
_centre_2 = (LINE_2_LEFT + LINE_2_RIGHT) / 2.0

# Project midpoints to BEV
_bev_1 = _transform_pt(_centre_1)
_bev_2 = _transform_pt(_centre_2)

# Perspective-corrected distance (metres, along road)
TRIPWIRE_DISTANCE_M = abs(float(_bev_2[1]) - float(_bev_1[1]))

# Naive distance: assumes uniform scale from average pixel width
_L1 = float(np.linalg.norm(LINE_1_RIGHT - LINE_1_LEFT))  # px width of Line 1
_L2 = float(np.linalg.norm(LINE_2_RIGHT - LINE_2_LEFT))  # px width of Line 2
NAIVE_DISTANCE_M = (LINE_2_Y - LINE_1_Y) * GREEN_LINE_LENGTH_M / ((_L1 + _L2) / 2.0)

# Print calibration summary at startup
print(f"BEV Line 1 centre            : y = {_bev_1[1]:.1f} m")
print(f"BEV Line 2 centre            : y = {_bev_2[1]:.1f} m")
print(f"Perspective-corrected dist.  : {TRIPWIRE_DISTANCE_M:.2f} m")
print(f"Naive (avg-scale) dist.      : {NAIVE_DISTANCE_M:.2f} m")
print(f"Difference                   : {(TRIPWIRE_DISTANCE_M / NAIVE_DISTANCE_M - 1) * 100:+.1f} %")
print()


# TRIPWIRE CROSSING  (functional / dict-based)
# Records the frame at which a vehicle's anchor crosses Line 1, then
# Line 2.  The elapsed frame count yields a time, and dividing the known
# distance by that time gives speed.

def make_tripwire_state(line_1_y: int, line_2_y: int, corrected_dist_m: float, naive_dist_m: float, fps: float) -> dict:
    """
    Create a fresh tripwire state dictionary.

    Args:
        line_1_y (int): Pixel Y-coordinates of the first tripwire line.
        line_2_y (int): Pixel Y-coordinates of the second tripwire line.
        corrected_dist_m (float): Perspective-corrected real-world distance between the lines (m).
        naive_dist_m (float): Naive (uniform-scale) distance between the lines (m).
        fps (float): Video frame rate.

    Returns:
        dict: Mutable state used by ``update_tripwire()`` / ``get_tripwire_speed()``.
    """
    return {
        "line_1_y":         line_1_y,
        "line_2_y":         line_2_y,
        "corrected_dist_m": corrected_dist_m,
        "naive_dist_m":     naive_dist_m,
        "fps":              fps,
        "prev_y":           {},   # tid -> previous anchor Y
        "crossed_1":        {},   # tid -> frame index when Line 1 was crossed
        "speed_corr":       {},   # tid -> perspective-corrected speed (km/h)
        "speed_naive":      {},   # tid -> naive speed (km/h)
    }


def update_tripwire(state: dict, tracker_ids: list, anchor_ys: list, frame_idx: int) -> None:
    """
    Check every tracked vehicle for a tripwire line crossing.

    A vehicle must cross Line 1 first, then Line 2.  When the second
    crossing is detected, speeds are computed from elapsed time and
    the known inter-line distances.

    Args:
        state (dict): State dict from ``make_tripwire_state()``.
        tracker_ids (list): Iterable of integer track IDs.
        anchor_ys (list): Iterable of pixel-Y anchor positions (same order as tracker_ids).
        frame_idx (int): Current frame number (1-based).
    """
    for tid, cy in zip(tracker_ids, anchor_ys):
        tid = int(tid)
        cy  = float(cy)
        prev = state["prev_y"].get(tid)
        state["prev_y"][tid] = cy

        if prev is None:
            continue   # first observation; skip

        # --- Check crossing of Line 1 (either direction) ---
        if tid not in state["crossed_1"]:
            if prev <= state["line_1_y"] < cy or cy <= state["line_1_y"] < prev:
                state["crossed_1"][tid] = frame_idx

        # --- Check crossing of Line 2 (only after Line 1) ---
        elif tid not in state["speed_corr"]:
            if prev <= state["line_2_y"] < cy or cy <= state["line_2_y"] < prev:
                elapsed_frames = frame_idx - state["crossed_1"][tid]
                if elapsed_frames > 0:
                    elapsed_s = elapsed_frames / state["fps"]
                    # speed = distance / time, converted m/s -> km/h (* 3.6)
                    state["speed_corr"][tid]  = (state["corrected_dist_m"] / elapsed_s) * 3.6
                    state["speed_naive"][tid] = (state["naive_dist_m"]     / elapsed_s) * 3.6


def get_tripwire_speed(state: dict, tid: int) -> float:
    """
    Return the perspective-corrected tripwire speed for a track, or None.

    Args:
        state (dict): Tripwire state dictionary.
        tid (int): Track ID.

    Returns:
        float: Speed in km/h, or None if the vehicle hasn't completed both crossings.
    """
    return state["speed_corr"].get(int(tid))


# UTILITY: BATCH POINT TRANSFORM

def transform_points(points: np.ndarray) -> np.ndarray:
    """
    Project an array of 2-D pixel points through the perspective matrix.

    Args:
        points (numpy.ndarray): Pixel coordinates [[x, y], ...] of shape (N, 2).

    Returns:
        numpy.ndarray: Real-world coordinates in metres of shape (N, 2).
    """
    if points.size == 0:
        return points
    reshaped = points.reshape(-1, 1, 2).astype(np.float32)
    transformed = cv2.perspectiveTransform(reshaped, PERSPECTIVE_MATRIX)
    return transformed.reshape(-1, 2)


# DRAWING / ANNOTATION HELPERS

def draw_tripwires(frame: np.ndarray) -> None:
    """
    Draw the two green tripwire lines with labels on the frame.
    
    Args:
        frame (numpy.ndarray): The current video frame.
    """
    green = (0, 255, 0)
    cv2.line(frame, tuple(LINE_1_LEFT.astype(int)), tuple(LINE_1_RIGHT.astype(int)), green, 2)
    cv2.line(frame, tuple(LINE_2_LEFT.astype(int)), tuple(LINE_2_RIGHT.astype(int)), green, 2)
    cv2.putText(frame, "Line 1 (25 m)", (int(LINE_1_RIGHT[0]) + 10, int(LINE_1_RIGHT[1]) + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, green, 1, cv2.LINE_AA)
    cv2.putText(frame, "Line 2 (25 m)", (int(LINE_2_RIGHT[0]) + 10, int(LINE_2_RIGHT[1]) + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, green, 1, cv2.LINE_AA)


def draw_hud(frame: np.ndarray, vehicle_count: int, frame_idx: int, total: int) -> None:
    """
    Draw a semi-transparent heads-up display showing vehicle count,
    progress, and tripwire distance info.
    
    Args:
        frame (numpy.ndarray): The current video frame.
        vehicle_count (int): Number of vehicles currently in the zone.
        frame_idx (int): Current frame index.
        total (int): Total number of frames in the video.
    """
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (440, 100), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    cv2.putText(frame, f"Vehicles in zone: {vehicle_count}",
                (20, 37), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 220, 220), 2, cv2.LINE_AA)
    pct = frame_idx / total * 100 if total else 0
    cv2.putText(frame, f"Frame {frame_idx}/{total}  ({pct:.0f}%)",
                (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Tripwire dist: {TRIPWIRE_DISTANCE_M:.1f}m (corrected)"
                       f"  |  {NAIVE_DISTANCE_M:.1f}m (naive)",
                (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (100, 220, 100), 1, cv2.LINE_AA)


# ----- Main Script Logic ----------

def main() -> None:
    """
    Main function to execute the full YOLO-based traffic monitoring pipeline.

    This function handles:
    1. Load the YOLOv8x model and open the video via Supervision helpers.
    2. For each frame:
       a. Run YOLO inference to detect vehicles (car, motorcycle, bus, truck).
       b. Track detections across frames using ByteTrack.
       c. Compute BEV rolling-window speed for all tracked vehicles.
       d. Check for tripwire crossings and compute crossing speed.
       e. Filter detections to the tripwire zone for annotation only.
       f. Annotate the frame (boxes, labels, green lines, HUD).
    3. Write the annotated frame to the output video.
    4. After all frames: export per-vehicle CSV and print summary.
    """
    # -- Load YOLO model ------------------------------------------------------
    print("Loading YOLO model...")
    model = YOLO("Other/yolov8x.pt")

    # -- Open video via Supervision -------------------------------------------
    video_path = os.path.join("../Downloads", "Segments", VIDEO_NAME)
    video_info = sv.VideoInfo.from_video_path(video_path)
    fps = video_info.fps
    total_frames = video_info.total_frames or 0
    frame_generator = sv.get_video_frames_generator(video_path)

    print(f"Video  : {video_path}")
    print(f"Size   : {video_info.width}x{video_info.height}  |  FPS: {fps}")
    print(f"Output : {OUTPUT_PATH}  |  CSV: {CSV_PATH}")

    # -- Initialise tracker and annotators ------------------------------------
    byte_track = sv.ByteTrack(frame_rate=fps)
    bounding_box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)

    # BEV coordinate history: keep up to 1 second of positions per track
    coordinates = defaultdict(lambda: deque(maxlen=int(fps)))

    # Tripwire crossing detector
    tripwire = make_tripwire_state(
        LINE_1_Y, LINE_2_Y,
        TRIPWIRE_DISTANCE_M, NAIVE_DISTANCE_M,
        fps,
    )

    # Per-track metadata
    tid_to_class = {}                  # tracker_id -> "car" / "truck" / etc.
    bev_speeds   = defaultdict(list)   # tracker_id -> [speed samples (km/h)]
    frame_idx    = 0

    # FRAME LOOP
    with sv.VideoSink(OUTPUT_PATH, video_info=video_info) as sink:

        for frame in frame_generator:
            frame_idx += 1

            # --- Step 1: YOLO object detection --------------------------------
            result = model(frame, verbose=False)[0]
            detections = sv.Detections.from_ultralytics(result)

            # Keep only vehicle classes (car, motorcycle, bus, truck)
            detections = detections[np.isin(detections.class_id, VEHICLE_CLASS_IDS)]

            # --- Step 2: ByteTrack multi-object tracking ----------------------
            detections = byte_track.update_with_detections(detections=detections)

            # --- Step 3: Perspective transform on anchor points ---------------
            # Use the bottom-centre of each bounding box as the "foot" point
            anchors = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
            transformed = transform_points(anchors)

            # --- Step 4: Record YOLO class names per track -------------------
            if detections.tracker_id is not None:
                for tid, cid in zip(detections.tracker_id, detections.class_id):
                    tid_to_class[int(tid)] = CLASS_NAMES.get(int(cid), "vehicle")

            # --- Step 5: Speed calculation (ALL vehicles, full frame) --------
            all_labels = []

            if detections.tracker_id is not None and len(transformed) > 0:
                # 5a. Tripwire crossing update
                anchor_ys = anchors[:, 1]
                update_tripwire(tripwire, detections.tracker_id, anchor_ys, frame_idx)

                # 5b. BEV rolling-window speed (same formula as original script)
                for tracker_id, (_, bev_y) in zip(detections.tracker_id, transformed):
                    tracker_id = int(tracker_id)
                    coordinates[tracker_id].append(float(bev_y))

                    trip_spd = get_tripwire_speed(tripwire, tracker_id)
                    vtype = tid_to_class.get(tracker_id, "vehicle").upper()

                    if trip_spd is not None:
                        # Tripwire speed available -> show exact value
                        all_labels.append(f"{vtype} #{tracker_id}  {int(trip_spd)} km/h")
                    elif len(coordinates[tracker_id]) > fps / 2:
                        # Fallback: BEV displacement over the window
                        coord_start = coordinates[tracker_id][-1]   # newest
                        coord_end   = coordinates[tracker_id][0]    # oldest
                        distance    = abs(coord_start - coord_end)  # metres
                        elapsed     = len(coordinates[tracker_id]) / fps  # seconds
                        bev_spd     = distance / elapsed * 3.6 if elapsed > 0 else 0
                        bev_speeds[tracker_id].append(bev_spd)
                        all_labels.append(f"{vtype} #{tracker_id}  ~{int(bev_spd)} km/h")
                    else:
                        # Not enough data yet
                        all_labels.append(f"{vtype} #{tracker_id}")

            # --- Step 6: Filter to tripwire zone (drawing only) ---------------
            # Speed is computed for ALL vehicles (full frame), but bounding
            # boxes are only drawn for vehicles inside the zone.
            if len(anchors) > 0:
                in_zone_mask = ((anchors[:, 1] >= (LINE_1_Y - ZONE_MARGIN)) &
                                (anchors[:, 1] <= (LINE_2_Y + ZONE_MARGIN)))
                zone_detections = detections[in_zone_mask]
                zone_labels = [l for l, m in zip(all_labels, in_zone_mask) if m]
                zone_anchors = anchors[in_zone_mask]
            else:
                zone_detections = detections
                zone_labels = all_labels
                zone_anchors = anchors

            # --- Step 7: Annotate frame ---------------------------------------
            annotated_frame = frame.copy()
            draw_tripwires(annotated_frame)   # green lines (behind boxes)

            # Supervision annotators draw bounding boxes and labels
            annotated_frame = bounding_box_annotator.annotate(
                scene=annotated_frame, detections=zone_detections
            )
            annotated_frame = label_annotator.annotate(
                scene=annotated_frame, detections=zone_detections, labels=zone_labels
            )

            # Count vehicles strictly between the two lines for the HUD
            if len(zone_anchors) > 0:
                in_strict = int(np.sum(
                    (zone_anchors[:, 1] >= LINE_1_Y) & (zone_anchors[:, 1] <= LINE_2_Y)
                ))
            else:
                in_strict = 0

            draw_hud(annotated_frame, in_strict, frame_idx, total_frames)
            sink.write_frame(frame=annotated_frame)

            # Progress indicator
            if frame_idx % 25 == 0 or frame_idx == total_frames:
                pct = frame_idx / total_frames * 100 if total_frames else 0
                print(f"  Frame {frame_idx}/{total_frames}  ({pct:.0f}%)", end="\r")

    # CSV EXPORT
    # Merge track IDs from both speed methods
    all_tids = sorted(set(
        list(tripwire["speed_corr"].keys())
        + [tid for tid, samples in bev_speeds.items() if len(samples) > 0]
    ))

    with open(CSV_PATH, "w", newline="") as f:
        writer_csv = csv.writer(f)
        writer_csv.writerow([
            "vehicle_id", "vehicle_type",
            "tripwire_speed_corrected_kmh", "tripwire_speed_naive_kmh",
            "bev_avg_speed_kmh", "bev_min_speed_kmh", "bev_max_speed_kmh",
        ])
        for tid in all_tids:
            samples = bev_speeds.get(tid, [])
            writer_csv.writerow([
                tid,
                tid_to_class.get(tid, "unknown"),
                f"{tripwire['speed_corr'][tid]:.1f}" if tid in tripwire["speed_corr"] else "",
                f"{tripwire['speed_naive'][tid]:.1f}" if tid in tripwire["speed_naive"] else "",
                f"{np.mean(samples):.1f}" if samples else "",
                f"{np.min(samples):.1f}"  if samples else "",
                f"{np.max(samples):.1f}"  if samples else "",
            ])

    # SUMMARY
    n_trip = len(tripwire["speed_corr"])
    n_bev  = sum(1 for s in bev_speeds.values() if len(s) > 0)

    print(f"\nDone!")
    print(f"  Video : {OUTPUT_PATH}")
    print(f"  CSV   : {CSV_PATH}  ({n_trip} tripwire + {n_bev} BEV measurements)")

    if n_trip > 0:
        avg_corr  = np.mean(list(tripwire["speed_corr"].values()))
        avg_naive = np.mean(list(tripwire["speed_naive"].values()))
        print(f"\n  Tripwire avg (perspective-corrected): {avg_corr:.1f} km/h")
        print(f"  Tripwire avg (naive / no correction): {avg_naive:.1f} km/h")
        print(f"  Perspective correction effect:         {((avg_corr/avg_naive)-1)*100:+.1f} %")


if __name__ == "__main__":
    import time
    start_time = time.time()
    print(f'start time: {start_time}')
    main()
    print(f'total duration: {time.time() - start_time}')