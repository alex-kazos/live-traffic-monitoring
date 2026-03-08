import os
import re
import csv
import cv2
import time
import numpy as np
from collections import defaultdict, deque

# ---------------------------------------------------------------------------
# Kafka producer (imported lazily – silently disabled if Kafka is unavailable)
# ---------------------------------------------------------------------------
try:
    import sys

    # For local execution (../Kafka)
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Kafka"))
    # For Docker execution where producer.py is copied directly into /app/Kafka/
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "Kafka"))

    import producer as kafka_producer

    _KAFKA_ENABLED = True
except ImportError:
    _KAFKA_ENABLED = False

# ---------------------------------------------------------------------------
# Paths — driven by environment variables when running in Docker;
# fallback to the original local paths for standalone execution.
# ---------------------------------------------------------------------------

# SEGMENTS_DIR: container default is /data/segments
_SEGMENTS_DIR = os.getenv("SEGMENTS_DIR", os.path.join("..", "Downloads", "Segments"))

# VIDEO_PATH: if the env var is set use it directly.
# Otherwise, look in SEGMENTS_DIR, and find the first segment that hasn't
# already been processed (i.e. check if its corresponding output CSV exists).
_ENV_VIDEO = os.getenv("INPUT_VIDEO", "")
if _ENV_VIDEO:
    VIDEO_PATH = _ENV_VIDEO
    _base = os.path.splitext(os.path.basename(VIDEO_PATH))[0]
    OUTPUT_PATH = os.getenv(
        "OUTPUT_PATH", os.path.join(_SEGMENTS_DIR, f"{_base}_annotated.mp4")
    )
    CSV_PATH = os.getenv("CSV_PATH", os.path.join(_SEGMENTS_DIR, f"{_base}_speeds.csv"))
else:
    # Auto-discovery for parallel scaled workers
    _segments = sorted(
        [
            f
            for f in os.listdir(_SEGMENTS_DIR)
            if f.endswith(".mp4") and not f.endswith("_annotated.mp4")
        ]
    )
    if not _segments:
        raise FileNotFoundError(
            f"No unprocessed video segments found in {_SEGMENTS_DIR}"
        )

    _selected_segment = None
    for seg in _segments:
        _base = os.path.splitext(seg)[0]
        _csv_check = os.path.join(_SEGMENTS_DIR, f"{_base}_speeds.csv")
        # Check if a worker has already started processing this segment
        # (by checking if the CSV exists. As a basic lock, we'll create an empty file immediately)
        if not os.path.exists(_csv_check):
            # Claim it (very primitive lock for scaled docker workers)
            open(_csv_check, "a").close()
            _selected_segment = seg
            VIDEO_PATH = os.path.join(_SEGMENTS_DIR, _selected_segment)
            OUTPUT_PATH = os.path.join(_SEGMENTS_DIR, f"{_base}_annotated.mp4")
            CSV_PATH = _csv_check
            break

    if not _selected_segment:
        # All segments are either processed or currently being processed
        print("All segments appear to be assigned or processed. Exiting cleanly.")
        sys.exit(0)

# ---------------------------------------------------------------------------
# Derive segment number from the filename so we can compute the absolute
# timestamp within the full 32-minute video (e.g. part_3 → part index 3,
# offset = (3-1) * 120 = 240 seconds).
# ---------------------------------------------------------------------------
SEGMENT_SECONDS = int(os.getenv("SEGMENT_SECONDS", "120"))

_seg_match = re.search(r"part[_\s]*(\d+)", os.path.basename(VIDEO_PATH), re.I)
_SEGMENT_INDEX = int(_seg_match.group(1)) if _seg_match else 1
_SEGMENT_OFFSET_S = (_SEGMENT_INDEX - 1) * SEGMENT_SECONDS  # seconds from video start

_SEGMENT_NAME = os.path.basename(VIDEO_PATH)


# ----- VARIABLES ----------

FPS = 25.0
MIN_AREA = 1_500
TRUCK_MIN_AREA = 9_000
MAX_LOST = 12
MAX_DIST = 150

BEV_WINDOW = 8
BEV_MIN_FRAMES = 7
BEV_SPD_MIN = 5.0
BEV_SPD_MAX = 150.0  # UK motorway cap (km/h)

# Used for splitting detections into left vs right carriageway (based on bottom-center point)
SPLIT_X = 640

# TRIPWIRE LINES

LINE_1_Y = 400  # upper green line
LINE_2_Y = 560  # lower green line

# (10%) Perspective Calibration

# Start of left side tripwire (line 2), end of left side tripwire (line 1)
SOURCE_L = np.array(
    [
        [490, 295],  # top left
        [640, 295],  # top right
        [640, 700],  # bottom right
        [20, 700],  # bottom left
    ],
    dtype=np.float32,
)

# Start of right side tripwire (line 1), end of right side tripwire (line 2)
SOURCE_R = np.array(
    [
        [640, 295],  # top left
        [790, 295],  # top right
        [1260, 700],  # bottom right
        [640, 700],  # bottom left
    ],
    dtype=np.float32,
)

# The target Horizon
TARGET_H = np.array([[0, 0], [14, 0], [14, 102], [0, 102]], dtype=np.float32)

# Perspective transform matrices for left and right sides
PERSPECT_L = cv2.getPerspectiveTransform(SOURCE_L, TARGET_H)
PERSPECT_R = cv2.getPerspectiveTransform(SOURCE_R, TARGET_H)


# ----- FUNCTIONS ----------


def bev_calculation(pix_x: int, pix_y: int, perspective_array) -> float:
    """
    Calculate the Bird's Eye View (BEV) Y-coordinate in real-world meters.

    Args:
        pix_x (int): The x-coordinate in pixels.
        pix_y (int): The y-coordinate in pixels.
        perspective_array (numpy.ndarray): The 3x3 perspective transformation matrix.

    Returns:
        float: The BEV y-coordinate in meters.
    """
    input_point = np.array([[[float(pix_x), float(pix_y)]]], dtype=np.float32)

    bev = float(cv2.perspectiveTransform(input_point, perspective_array)[0, 0, 1])

    return bev


# Distance calculate for left & right tripwires
DIST_L = abs(
    bev_calculation(320, LINE_1_Y, PERSPECT_L)
    - bev_calculation(320, LINE_2_Y, PERSPECT_L)
)
DIST_R = abs(
    bev_calculation(960, LINE_1_Y, PERSPECT_R)
    - bev_calculation(960, LINE_2_Y, PERSPECT_R)
)


def initialize_tracker() -> dict:
    """
    Initialize a new tracker state dictionary.

    Returns:
        tracker_dict (dict): A dictionary containing next track ID, centroids, and lost frame counts.
    """
    tracker_dict = {"next_id": 0, "centroids": {}, "lost": defaultdict(int)}

    return tracker_dict


def update_tracker(tracker: dict, bounding_box: list) -> dict:
    """
    Update the tracker state with new bounding box detections for a frame.

    Matches new detections to existing tracks using Euclidean distance.
    Unmatched tracks are marked lost and removed if lost for too many frames.

    Args:
        tracker (dict): The tracker state dictionary.
        bounding_box (list of tuples): List of bounding boxes (x, y, w, h).

    Returns:
        centroids_dict (dict): The updated active centroids mapping (track ID -> (x, y)).
    """
    centroids = tracker["centroids"]
    lost = tracker["lost"]

    # Track point
    new_points = [(int(x + w / 2), int(y + h)) for x, y, w, h in bounding_box]

    # If no detections, mark all existing tracks as lost and remove if lost for too long
    if not bounding_box:
        # Mark all existing tracks as lost
        for track_id in list(centroids):
            lost[track_id] += 1
            if lost[track_id] > MAX_LOST:
                del centroids[track_id]
                del lost[track_id]
        return dict(centroids)

    # If no existing tracks, initialize new tracks for all detections
    if not centroids:
        for point in new_points:
            centroids[tracker["next_id"]] = point
            lost[tracker["next_id"]] = 0
            tracker["next_id"] += 1
        return dict(centroids)

    # Match new points to existing centroids based on distance
    existing_ids = list(centroids)
    existing_points = [centroids[i] for i in existing_ids]
    dist = np.linalg.norm(
        np.array(existing_points)[:, None] - np.array(new_points)[None, :], axis=-1
    )
    pairs = sorted(
        [
            (dist[i, j], i, j)
            for i in range(len(existing_points))
            for j in range(len(new_points))
        ],
        key=lambda x: x[0],
    )

    existing_indices = set()
    new_indices = set()

    # loop through sorted pairs and assign matches until distance exceeds threshold or all pairs are processed
    for distance, existing_index, new_index in pairs:
        if existing_index in existing_indices or new_index in new_indices:
            continue
        if distance > MAX_DIST:
            break

        track_id = existing_ids[existing_index]
        centroids[track_id] = new_points[new_index]
        lost[track_id] = 0
        existing_indices.add(existing_index)
        new_indices.add(new_index)

    # Mark unmatched existing tracks as lost and remove if lost for too long
    for existing_index, track_id in enumerate(existing_ids):
        if existing_index not in existing_indices:
            lost[track_id] += 1
            if lost[track_id] > MAX_LOST:
                del centroids[track_id]
                del lost[track_id]

    for new_index, point in enumerate(new_points):
        if new_index not in new_indices:
            centroids[tracker["next_id"]] = point
            lost[tracker["next_id"]] = 0
            tracker["next_id"] += 1

    centroids_dict = dict(centroids)

    return centroids_dict


def initialize_tripwire(first_line: int, second_line: int, dist_m: float) -> dict:
    """
    Initialize a new tripwire state dictionary for calculating vehicle speed.

    Args:
        first_line (int): The y-coordinate of the first tripwire line to cross.
        second_line (int): The y-coordinate of the second tripwire line to cross.
        dist_m (float): The actual real-world distance between the lines in meters.

    Returns:
        tripwire (dict): A dictionary tracking previous y-coords, crossings, and speeds.
    """
    tripwire = {
        "prev_y": {},
        "crossed_first": {},
        "speed": {},
        "line_first_y": first_line,
        "line_second_y": second_line,
        "dist_m": dist_m,
    }

    return tripwire


def crossed_line_check(prev_y: float, cur_y: float, line_y: int) -> bool:
    """
    Check if a line was crossed between the previous and current frame.

    Args:
        prev_y (float): The tracked object's previous bounding box bottom y-coordinate.
        cur_y (float): The tracked object's current bounding box bottom y-coordinate.
        line_y (int): The y-coordinate of the tripwire line.

    Returns:
        crossed_line (bool): True if the object crossed the line, False otherwise.
    """
    crossed_line = (prev_y < line_y <= cur_y) or (cur_y < line_y <= prev_y)

    return crossed_line


def update_tripwire(tripwire: dict, tracked_vehicles: dict, current_frame: int) -> None:
    """
    Update the tripwire state and calculate speeds for vehicles crossing both lines.

    Args:
        tripwire (dict): The tripwire state dictionary.
        tracked_vehicles (dict): The active tracked vehicles (track ID -> (x, y)).
        current_frame (int): The current frame index in the video.
    """
    # For each tracked vehicle, check if it has crossed the first line and then the second line.
    for track_id, (_, current_position_y) in tracked_vehicles.items():
        prev = tripwire["prev_y"].get(track_id)
        tripwire["prev_y"][track_id] = current_position_y
        if prev is None:
            continue
        if track_id not in tripwire["crossed_first"]:
            if crossed_line_check(prev, current_position_y, tripwire["line_first_y"]):
                tripwire["crossed_first"][track_id] = current_frame
        elif track_id not in tripwire["speed"]:
            if crossed_line_check(prev, current_position_y, tripwire["line_second_y"]):
                elapsed = (current_frame - tripwire["crossed_first"][track_id]) / FPS
                if elapsed > 0:
                    tripwire["speed"][track_id] = (tripwire["dist_m"] / elapsed) * 3.6


# ----- Main Script Logic ----------


def main():
    """
    Main function to execute the traffic tracking_speed monitoring pipeline.

    This function handles video loading, MOG2 background subtraction, vehicle tracking
    across two divided carriageways, tracking_speed calculation (via tripwires and BEV fallback),
    output video annotation, and results export to a CSV file.
    """
    print(f"Starting tracking on {VIDEO_PATH} ...", flush=True)
    # read the goddamn video
    capture_video = cv2.VideoCapture(VIDEO_PATH)
    if not capture_video.isOpened():
        raise FileNotFoundError(f"Cannot open: {VIDEO_PATH}")

    frame_width = int(capture_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(capture_video.get(cv2.CAP_PROP_FRAME_COUNT))

    # (DISABLED) print video stats, maybe a moving id can be assigned here and passed to the csv for tracking
    # print(f"Video: {VIDEO_PATH}  |  {frame_width}x{frame_height}  |  {total_frames} frames")

    background = cv2.createBackgroundSubtractorMOG2(
        history=150, varThreshold=40, detectShadows=True
    )
    # reduce noise for better tracking
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    kernel_large = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))

    # Two independent trackers & trackers
    tracker_left = initialize_tracker()
    tracker_right = initialize_tracker()

    tripwire_left = initialize_tripwire(LINE_2_Y, LINE_1_Y, DIST_L)
    tripwire_right = initialize_tripwire(LINE_1_Y, LINE_2_Y, DIST_R)

    vehicle_type = {}  # "car" // "truck"
    bev_hist = defaultdict(lambda: deque(maxlen=BEV_WINDOW))
    bev_speed = {}

    # Set up video writer for annotated output
    video_codec = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(
        OUTPUT_PATH, video_codec, FPS, (frame_width, frame_height)
    )

    # Process each frame
    for frame_id in range(1, total_frames + 1):
        result, frame = capture_video.read()
        if not result:
            break

        # Foreground mask
        foreground = background.apply(frame)
        _, foreground = cv2.threshold(foreground, 200, 255, cv2.THRESH_BINARY)
        foreground = cv2.morphologyEx(foreground, cv2.MORPH_OPEN, kernel_small)
        foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, kernel_medium)
        foreground = cv2.dilate(foreground, kernel_large)

        # Detections Split by carriageway
        centroids, _ = cv2.findContours(
            foreground, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        all_bounding_rect = [
            cv2.boundingRect(c) for c in centroids if cv2.contourArea(c) >= MIN_AREA
        ]
        rects_left = [r for r in all_bounding_rect if (r[0] + r[2] // 2) < SPLIT_X]
        rects_right = [r for r in all_bounding_rect if (r[0] + r[2] // 2) >= SPLIT_X]

        # Update trackers and tripwires
        active_left = update_tracker(tracker_left, rects_left)
        active_right = update_tracker(tracker_right, rects_right)

        update_tripwire(tripwire_left, active_left, frame_id)
        update_tripwire(tripwire_right, active_right, frame_id)

        # BEV + vehicle_type classification
        for side, active, matrix_side, rects_side in [
            ("L", active_left, PERSPECT_L, rects_left),
            ("R", active_right, PERSPECT_R, rects_right),
        ]:
            # For each active track, calculate BEV tracking_speed if enough history and classify vehicle type based on closest bounding box size
            for tracker_id, (current_position_x, current_position_y) in active.items():
                key = (side, tracker_id)
                bev_perspective = cv2.perspectiveTransform(
                    np.array(
                        [[[float(current_position_x), float(current_position_y)]]],
                        dtype=np.float32,
                    ),
                    matrix_side,
                )[0, 0]
                bev_hist[key].append(
                    (float(bev_perspective[0]), float(bev_perspective[1]))
                )
                h_pts = list(bev_hist[key])
                if len(h_pts) >= BEV_MIN_FRAMES:
                    diffs = [
                        (
                            (h_pts[i + 1][0] - h_pts[i][0]) ** 2
                            + (h_pts[i + 1][1] - h_pts[i][1]) ** 2
                        )
                        ** 0.5
                        for i in range(len(h_pts) - 1)
                    ]
                    tracking_speed = float(np.median(diffs)) * FPS * 3.6
                    if BEV_SPD_MIN < tracking_speed < BEV_SPD_MAX:
                        bev_speed[key] = tracking_speed

                # Classify vehicle type from the closest bounding box (using bottom-center)
                if rects_side:
                    rect = min(
                        rects_side,
                        key=lambda r: abs(current_position_x - (r[0] + r[2] // 2))
                        + abs(current_position_y - (r[1] + r[3])),
                    )
                    vehicle_type[key] = (
                        "truck" if rect[2] * rect[3] >= TRUCK_MIN_AREA else "car"
                    )

        # Annotate frame
        annotated_frame = frame.copy()

        # Draw the first tripwire line
        cv2.line(annotated_frame, (340, LINE_1_Y), (940, LINE_1_Y), (0, 255, 0), 2)
        # Draw the second tripwire line
        cv2.line(annotated_frame, (100, LINE_2_Y), (1180, LINE_2_Y), (0, 255, 0), 2)
        # Draw the vertical divider line
        cv2.line(
            annotated_frame, (SPLIT_X, 0), (SPLIT_X, frame_height), (80, 80, 80), 1
        )  # divider guide

        # Annotate each active track with its ID, type, and tracking_speed (from tripwire or BEV)
        for side, active, tripwire, rects_side, color in [
            ("L", active_left, tripwire_left, rects_left, (0, 200, 255)),  # orange
            ("R", active_right, tripwire_right, rects_right, (255, 180, 0)),  # blue
        ]:
            # Only annotate if the current position is between the tripwire lines to avoid clutter and misannotations
            for tracker_id, (current_position_x, current_position_y) in active.items():
                if not (LINE_1_Y <= current_position_y <= LINE_2_Y):
                    continue
                key = (side, tracker_id)
                if not rects_side:
                    continue
                rect = min(
                    rects_side,
                    key=lambda r: abs(current_position_x - (r[0] + r[2] // 2))
                    + abs(current_position_y - (r[1] + r[3])),
                )
                rect_x, rect_y, rect_w, rect_h = rect
                speed_tripwire = tripwire["speed"].get(tracker_id)
                speed_bev = bev_speed.get(key)
                if speed_tripwire is not None:
                    speed_label = f"{int(speed_tripwire)} km/h"
                elif speed_bev is not None:
                    speed_label = f"~{int(speed_bev)} km/h"
                else:
                    # we can later input the average tracking_speed of the video here if we want to give some indication of tracking_speed even when it can't be calculated
                    speed_label = "--"
                vehicle_label = vehicle_type.get(key, "car").upper()
                row = f"{vehicle_label} {side}{tracker_id}  {speed_label}"

                # Draw bounding box and label for the track
                cv2.rectangle(
                    annotated_frame,
                    (rect_x, rect_y),
                    (rect_x + rect_w, rect_y + rect_h),
                    color,
                    2,
                )
                cv2.putText(
                    annotated_frame,
                    row,
                    (rect_x, max(rect_y - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1,
                    cv2.LINE_AA,
                )

        video_writer.write(annotated_frame)

        # # (DISABLED) Progress update every 50 frames and on the last frame
        # if frame_id % 50 == 0 or frame_id == total_frames:
        #     print(f"  {frame_id}/{total_frames}", end="\r")

    # Release resources
    capture_video.release()
    video_writer.release()

    # -----------------------------------------------------------------------
    # CSV export + Kafka publishing
    # -----------------------------------------------------------------------
    # Map carriageway side code to a human-readable label.
    _SIDE_LABEL = {"L": "inbound", "R": "outbound"}

    with open(CSV_PATH, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(
            ["vehicle_id", "carriageway", "vehicle_type", "speed_kmh", "speed_source"]
        )
        # Combine the tripwire and BEV tracking_speed data and include all seen tracker IDs from both sources and vehicle type classifications.
        for side, tripwire, side_label in [
            ("L", tripwire_left, "inbound"),
            ("R", tripwire_right, "outbound"),
        ]:
            # union ids
            seen_track_ids = (
                set(tripwire["speed"])
                | {tracker_id for (s, tracker_id) in vehicle_type if s == side}
                | {tracker_id for (s, tracker_id) in bev_speed if s == side}
            )
            # Sort by tracker ID for consistent output
            for tracker_id in sorted(seen_track_ids):
                key = (side, tracker_id)
                speed_tripwire = tripwire["speed"].get(tracker_id)
                speed_bev = bev_speed.get(key)
                if speed_tripwire is not None:
                    tracking_speed, tracking_source = speed_tripwire, "tripwire"
                elif speed_bev is not None:
                    tracking_speed, tracking_source = speed_bev, "bev_avg"
                else:
                    tracking_speed, tracking_source = None, "none"

                vtype = vehicle_type.get(key, "unknown")
                vid = f"{side}{tracker_id}"

                wr.writerow(
                    [
                        vid,
                        side_label,
                        vtype,
                        f"{tracking_speed:.1f}" if tracking_speed is not None else "",
                        tracking_source,
                    ]
                )

                # ── Kafka: publish speed event ──────────────────────────────
                if _KAFKA_ENABLED and tracking_speed is not None:
                    # Estimate when this vehicle was observed (seconds into
                    # the FULL 32-min video) using the midpoint frame heuristic.
                    # The crossed_first frame gives us the entry time.
                    entry_frame = (
                        tripwire["crossed_first"].get(tracker_id, 0)
                        if tracking_source == "tripwire"
                        else 0
                    )
                    local_ts_s = entry_frame / FPS  # offset inside segment
                    global_ts_s = _SEGMENT_OFFSET_S + local_ts_s  # offset in full video

                    kafka_producer.publish_vehicle_event(
                        vehicle_id=vid,
                        vehicle_type=vtype,
                        speed_kmh=tracking_speed,
                        carriageway=side_label,
                        segment=_SEGMENT_NAME,
                        timestamp_s=global_ts_s,
                        speed_source=tracking_source,
                    )

    # Flush all buffered Kafka messages before the process exits
    if _KAFKA_ENABLED:
        kafka_producer.flush()
        kafka_producer.close()

    # Summary output
    n_trip = sum(len(tw["speed"]) for tw in (tripwire_left, tripwire_right))
    n_bev = sum(
        1
        for (s, tracker_id) in bev_speed
        if (s == "L" and tracker_id not in tripwire_left["speed"])
        or (s == "R" and tracker_id not in tripwire_right["speed"])
    )
    all_seen = (
        {("L", t) for t in tripwire_left["speed"]}
        | {("R", t) for t in tripwire_right["speed"]}
        | set(vehicle_type)
        | set(bev_speed)
    )

    print(f"\nDone.  Output: {OUTPUT_PATH}  |  CSV: {CSV_PATH}")
    print(f"  Tripwire speeds : {n_trip}")
    print(f"  BEV fallback    : {n_bev}")
    print(f"  No tracking_speed        : {len(all_seen) - n_trip - n_bev}")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"Total time: {time.time() - t0:.1f} s")
