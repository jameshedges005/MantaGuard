import argparse
import os
import shutil
import subprocess
import sys
import tempfile

from inference import get_model
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

EVENTS_CSV = "events.csv"
EVASION_EVENTS_CSV = r"D:\AAManta\Code\MantaGuard\evasion_events.csv"
APPROACHES_CSV = r"D:\AAManta\Code\MantaGuard\approaches.csv"
WINDOW_NAME = "MantaGuard"
MAX_DETECTION_SIDE = 640
MAX_TRACK_DISTANCE = 120
MAX_TRAIL_LENGTH = 60
EVASION_FLASH_FRAMES = 90
HEADING_CHANGE_THRESHOLD_DEG = 30.0
MIN_TRAIL_LENGTH_FOR_HEADING = 6
EVASION_HEADING_THRESHOLD_DEG = 45.0
EVASION_CONFIRMATION_FRAMES = 60
EVASION_LOCK_FRAMES = 90
EVASION_REFRACTORY_FRAMES = 150   # ~5 s; no new candidate until response ends (turn resets + timer clears)
PROX_GATE_M = 35.0
TURN_MIN_DEG = 45.0
CLOSING_LOOKBACK_FRAMES = 15
_HEADING_TRAILING_WIN = 150

_model = None


def _load_model():
    global _model
    if _model is None:
        _model = get_model(model_id="manta-detector/3", api_key="ADEO1giAIKoEYpkAWMFW")
    return _model


def parse_args():
    parser = argparse.ArgumentParser(description="MantaGuard desktop video analyzer")
    parser.add_argument("video_path", help="Path to the input video file")
    parser.add_argument("--altitude", type=float, default=80.0)
    parser.add_argument("--sensor-width", type=float, default=6.17)
    parser.add_argument("--focal-length", type=float, default=24.0)
    parser.add_argument("--boat-length", type=float, default=None)
    parser.add_argument("--frame-skip", type=int, default=1)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--uncertain", type=float, default=None,
                        help="Distance uncertainty as a fraction (e.g. 0.10 for +-10%%)")
    parser.add_argument("--save-plot-data", action="store_true",
                        help="Save per-frame plot data (frame, distance_m, heading_deg) to a CSV alongside each plot")
    parser.add_argument("--no-display", action="store_true",
                        help="Skip all cv2 window rendering (faster; use for batch processing)")
    parser.add_argument("--no-validate", action="store_true",
                        help="Skip human validation prompts; all evasions logged as unreviewed")
    parser.add_argument("--plot-display", action="store_true",
                        help="Show a live distance/heading plot alongside the video during processing")
    # ---- Configurable trigger parameters ----
    parser.add_argument("--turn-trigger", type=float, default=None,
                        help="Turn angle threshold (°) to activate evasion candidate (default: %(default)s → uses TURN_MIN_DEG constant)")
    parser.add_argument("--trigger-window", type=int, default=None,
                        help="Heading lookback window in frames for trigger baseline (default: %(default)s → uses _HEADING_TRAILING_WIN constant)")
    parser.add_argument("--trigger-distance", type=float, default=None,
                        help="Maximum boat distance (m) that must be met to trigger evasion candidate (default: %(default)s → uses PROX_GATE_M constant)")
    args, _ = parser.parse_known_args()
    return args


def normalize_class(raw_class):
    raw_class_lower = str(raw_class).strip().lower()
    raw_class_str = str(raw_class).strip()
    if raw_class_lower in {'boat', 'boats', 'speedboat', 'boat - dhoni'}:
        return raw_class_str
    if raw_class_lower in {'manta', 'manta-ray', 'mantaray', 'manta ray', 'manta_ray'}:
        return 'manta'
    return None


def is_boat_class(class_name):
    if class_name is None:
        return False
    class_lower = str(class_name).strip().lower()
    return class_lower in {'speedboat', 'boat - dhoni', 'boat', 'boats'}


def get_boat_type(class_name):
    if is_boat_class(class_name):
        return str(class_name).strip()
    return None


def get_boat_colour(class_name):
    if class_name is None:
        return (128, 128, 128)
    class_lower = str(class_name).strip().lower()
    if class_lower == 'speedboat':
        return (255, 0, 0)
    elif class_lower == 'boat - dhoni':
        return (0, 165, 255)
    return (128, 128, 128)


def roboflow_detect(frame):
    model = _load_model()
    results = model.infer(frame, confidence=0.3)[0]
    detections = []
    for pred in results.predictions:
        class_name = normalize_class(pred.class_name)
        if class_name is None:
            continue
        detections.append({
            'class': class_name,
            'confidence': float(pred.confidence),
            'x': float(pred.x),
            'y': float(pred.y),
            'width': float(pred.width),
            'height': float(pred.height),
        })
    return detections


def centroid_from_detection(det):
    return (float(det['x']), float(det['y']))


def compute_gsd(altitude_m, sensor_width_mm, focal_length_mm, image_width_px):
    if altitude_m <= 0 or sensor_width_mm <= 0 or focal_length_mm <= 0 or image_width_px <= 0:
        raise ValueError('Invalid parameters for GSD calculation')
    return (altitude_m * sensor_width_mm) / (focal_length_mm * image_width_px)


def euclidean_distance(a, b):
    return float(np.linalg.norm(np.array(a, dtype=np.float32) - np.array(b, dtype=np.float32)))


def angle_degrees_from_vector(vec):
    return float(np.degrees(np.arctan2(vec[1], vec[0])))


def angle_difference(a1, a2):
    diff = abs(a1 - a2) % 360
    return min(diff, 360 - diff)


def normalize_heading(degrees):
    return float(degrees % 360.0)


def dot_product(a, b):
    return float(np.dot(a, b))


def circular_mean_headings(headings):
    valid = [h for h in headings if h is not None and not np.isnan(float(h))]
    if not valid:
        return None
    rads = np.deg2rad(valid)
    return normalize_heading(float(np.degrees(np.arctan2(np.mean(np.sin(rads)), np.mean(np.cos(rads))))))


def is_evasion_candidate(turn_deg, boat_distance_m, closing):
    return turn_deg >= TURN_MIN_DEG and boat_distance_m <= PROX_GATE_M and closing


class Track:
    def __init__(self, track_id, class_name, centroid, bbox, frame_index, boat_type=None):
        self.id = track_id
        self.class_name = class_name
        self.boat_type = boat_type
        self.centroids = [centroid]
        self.bboxes = [bbox]
        self.last_frame = frame_index
        self.converging_frames = 0
        self.diverging_frames = 0
        self.in_approach = False
        self.approach_start_frame = None
        self.approach_start_heading = None
        self.status = 'TRACKING'
        self.evasion_flash = 0
        self.last_evasion_distance = 0.0
        self.manta_heading_at_evasion = None
        self.boat_heading_at_evasion = None
        self.heading_change_degrees = 0.0
        self.approach_duration = 0
        self.is_converging = False
        self.orientation_angle_deg = 0.0
        # Evasion candidate / confirmation window
        self.evasion_candidate_active = False
        self.evasion_candidate_onset_frame = None
        self.evasion_candidate_onset_distance_px = 0.0
        self.evasion_candidate_onset_manta_heading = None
        self.evasion_candidate_boat_heading = None
        self.evasion_candidate_heading_change = 0.0
        self.evasion_candidate_frames = 0
        self.in_post_evasion = False
        self.last_evasion_frame = None  # frame of most-recent confirmed evasion; drives refractory
        self._headings: list = []    # manta heading history (newest last), capped at 200
        self._distances: list = []   # boat distance in metres history, capped at 200

    def update(self, centroid, bbox, frame_index):
        self.centroids.append(centroid)
        self.bboxes.append(bbox)
        self.last_frame = frame_index
        if len(self.centroids) > MAX_TRAIL_LENGTH:
            self.centroids.pop(0)
            self.bboxes.pop(0)

    def latest_centroid(self):
        return self.centroids[-1]

    def has_trail(self):
        return len(self.centroids) >= 30

    def has_heading(self):
        return len(self.centroids) >= 16

    def current_heading_vector(self):
        if not self.has_heading():
            return None
        vec = np.array(self.centroids[-1], dtype=np.float32) - np.array(self.centroids[-16], dtype=np.float32)
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 1e-6 else None

    def heading_vector_over_n_frames(self, n=5):
        if len(self.centroids) < n + 1:
            return None
        vec = np.array(self.centroids[-1], dtype=np.float32) - np.array(self.centroids[-(n + 1)], dtype=np.float32)
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 1e-6 else None

    def instant_speed_m_per_s(self, gsd, fps):
        if len(self.centroids) < 2:
            return None
        return euclidean_distance(self.centroids[-1], self.centroids[-2]) * gsd * fps

    def heading_degrees(self, vector):
        if vector is None:
            return None
        return normalize_heading(angle_degrees_from_vector(vector))

    def heading_degrees_at_index(self, index):
        if index < 15 or index >= len(self.centroids):
            return None
        vec = np.array(self.centroids[index], dtype=np.float32) - np.array(self.centroids[index - 15], dtype=np.float32)
        norm = np.linalg.norm(vec)
        if norm <= 1e-6:
            return None
        return normalize_heading(angle_degrees_from_vector(vec))

    def current_speed_m_per_s(self, gsd, fps):
        if len(self.centroids) < 16:
            return None
        displacement_px = euclidean_distance(self.centroids[-1], self.centroids[-16])
        displacement_m = displacement_px * gsd
        elapsed_time_s = 15.0 / fps
        if elapsed_time_s <= 0:
            return None
        return displacement_m / elapsed_time_s

    def reset_approach(self):
        self.converging_frames = 0
        self.diverging_frames = 0
        self.in_approach = False
        self.approach_start_frame = None
        self.approach_start_heading = None
        self.status = 'TRACKING'
        self.is_converging = False
        self.evasion_candidate_active = False
        self.evasion_candidate_onset_frame = None
        self.evasion_candidate_frames = 0

    def update_heading_state(self, boat_track, frame_index, distance_m):
        if self.class_name != 'manta' or boat_track is None:
            self.reset_approach()
            return None

        if not self.has_trail() or not boat_track.has_trail():
            self.reset_approach()
            return None

        manta_vec = self.current_heading_vector()
        boat_vec = boat_track.current_heading_vector()

        if manta_vec is None or boat_vec is None:
            self.reset_approach()
            return None

        manta_pos = np.array(self.latest_centroid(), dtype=np.float32)
        boat_pos = np.array(boat_track.latest_centroid(), dtype=np.float32)
        to_boat = boat_pos - manta_pos
        norm_tb = np.linalg.norm(to_boat)
        if norm_tb <= 1e-6:
            self.reset_approach()
            return None
        manta_to_boat_vec = to_boat / norm_tb

        dot_val = float(np.dot(manta_vec, manta_to_boat_vec))
        angle_rad = float(np.arccos(np.clip(dot_val, -1.0, 1.0)))
        self.orientation_angle_deg = float(np.degrees(angle_rad))

        manta_heading = self.heading_degrees(manta_vec)

        # ================================================================
        # TRIGGER METRICS — computed every frame (before confirmation window
        # and refractory guard) so turn_deg is always current.
        # ================================================================

        # Maintain per-track heading and distance history.
        # Cap at WIN+70 so the baseline window is always reachable regardless of WIN.
        _history_cap = max(200, _HEADING_TRAILING_WIN + 70)
        self._headings.append(manta_heading)
        if len(self._headings) > _history_cap:
            self._headings.pop(0)
        self._distances.append(distance_m)
        if len(self._distances) > _history_cap:
            self._distances.pop(0)

        # ----------------------------------------------------------------
        # BASELINE HEADING CALCULATION
        # Take a 30-frame slice from (WIN-30) to (WIN-60) frames ago.
        # Example for WIN=150: average headings from 120 to 90 frames ago.
        # ----------------------------------------------------------------
        _n = len(self._headings)
        _bl_far_ago  = _HEADING_TRAILING_WIN - 30   # e.g. 120 for WIN=150
        _bl_near_ago = _HEADING_TRAILING_WIN - 60   # e.g.  90 for WIN=150
        _bl_start = _n - _bl_far_ago - 1
        _bl_end   = _n - _bl_near_ago
        _baseline_heading = None
        if _bl_near_ago >= 0 and _bl_start >= 0 and _bl_end > _bl_start:
            # Circular mean — wrap-safe across 0°/360°
            _baseline_heading = circular_mean_headings(self._headings[_bl_start:_bl_end])

        # Wrap-safe angular difference between current heading and circular-mean baseline
        _h_cur = self._headings[-1]
        turn_deg = (abs((_h_cur - _baseline_heading + 180) % 360 - 180)
                    if _h_cur is not None and _baseline_heading is not None else 0.0)

        # Closing: mean of first half of recent distance window > mean of second half
        closing = False
        _rd = [d for d in self._distances[-CLOSING_LOOKBACK_FRAMES:] if d is not None]
        if len(_rd) >= 4:
            _mid = len(_rd) // 2
            closing = float(np.mean(_rd[:_mid])) > float(np.mean(_rd[_mid:]))

        # ================================================================
        # END TRIGGER METRICS
        # ================================================================

        # ---- EVASION CANDIDATE CONFIRMATION WINDOW ----
        if self.evasion_candidate_active:
            # Count frames where heading remains above threshold (may be non-consecutive).
            if self.approach_start_heading is not None and manta_heading is not None:
                heading_change = angle_difference(self.approach_start_heading, manta_heading)
                if heading_change > EVASION_HEADING_THRESHOLD_DEG:
                    self.evasion_candidate_frames += 1

            # Phase is locked to EVASION_CANDIDATE for the full lock window.
            self.status = 'EVASION_CANDIDATE'

            frames_since_onset = frame_index - self.evasion_candidate_onset_frame
            if frames_since_onset < EVASION_LOCK_FRAMES:
                return None

            # Lock window expired — evaluate once.
            self.approach_duration = (
                self.evasion_candidate_onset_frame
                - (self.approach_start_frame or self.evasion_candidate_onset_frame) + 1
            )
            if self.evasion_candidate_frames >= EVASION_CONFIRMATION_FRAMES:
                self.status = 'EVADING'
                self.evasion_flash = EVASION_FLASH_FRAMES
                self.last_evasion_distance = self.evasion_candidate_onset_distance_px
                self.manta_heading_at_evasion = self.evasion_candidate_onset_manta_heading
                self.boat_heading_at_evasion = self.evasion_candidate_boat_heading
                self.heading_change_degrees = self.evasion_candidate_heading_change
                event = {
                    'type': 'confirmed',
                    'evasion_confirmed': True,
                    'distance': self.evasion_candidate_onset_distance_px,
                    'onset_frame': self.evasion_candidate_onset_frame,
                    'manta_heading_at_evasion': self.evasion_candidate_onset_manta_heading,
                    'boat_heading_at_evasion': self.evasion_candidate_boat_heading,
                    'heading_change_degrees': self.evasion_candidate_heading_change,
                    'approach_duration_frames': self.approach_duration,
                    'boat_centroids': list(boat_track.centroids[-15:]),
                }
                self.last_evasion_frame = frame_index  # start the refractory clock
                self.in_post_evasion = True
                self.reset_approach()
                return event
            else:
                event = {
                    'type': 'discarded',
                    'evasion_confirmed': False,
                    'distance': self.evasion_candidate_onset_distance_px,
                    'onset_frame': self.evasion_candidate_onset_frame,
                    'manta_heading_at_evasion': self.evasion_candidate_onset_manta_heading,
                    'boat_heading_at_evasion': self.evasion_candidate_boat_heading,
                    'heading_change_degrees': self.evasion_candidate_heading_change,
                    'approach_duration_frames': self.approach_duration,
                    'boat_centroids': list(boat_track.centroids[-15:]),
                }
                self.reset_approach()
                return event

        # ---- POST-EVASION REFRACTORY ----
        # Block a new candidate while the manta is still in its current response.
        # Two conditions must BOTH be met before re-arming:
        #   1. EVASION_REFRACTORY_FRAMES have elapsed since the last confirmation.
        #   2. The turn metric has dropped back below TURN_MIN_DEG (response is over).
        # Using both avoids splitting a slow sustained sweep (turn never drops while
        # ongoing) while still allowing a genuine second event once the manta settles.
        if self.in_post_evasion:
            _frames_since = (frame_index - self.last_evasion_frame
                             if self.last_evasion_frame is not None
                             else EVASION_REFRACTORY_FRAMES)
            if _frames_since < EVASION_REFRACTORY_FRAMES or turn_deg >= TURN_MIN_DEG:
                self.status = 'EVADING'
                return None
            # Both conditions cleared — allow normal triggering again
            self.in_post_evasion = False

        # ---- NORMAL APPROACH STATE MACHINE ----
        # (turn_deg and closing already computed above)

        # Display status (orientation-based, kept for overlay colours and phase labels)
        if self.orientation_angle_deg < 90.0:
            self.is_converging = True
            self.converging_frames += 1
            self.diverging_frames = 0
            if not self.in_approach and self.converging_frames >= 10:
                self.in_approach = True
                self.approach_start_frame = frame_index - 9
            self.status = 'APPROACHING' if self.in_approach else 'TRACKING'
        else:
            self.is_converging = False
            if self.in_approach:
                self.diverging_frames += 1
                self.status = 'APPROACHING'
            else:
                self.converging_frames = 0
                self.diverging_frames = 0
                self.status = 'TRACKING'

        # New trigger: fires from any approach status once all three conditions are met
        if is_evasion_candidate(turn_deg, distance_m, closing):
            self.evasion_candidate_active = True
            self.evasion_candidate_onset_frame = frame_index
            self.evasion_candidate_onset_distance_px = euclidean_distance(
                self.latest_centroid(), boat_track.latest_centroid()
            )
            self.evasion_candidate_onset_manta_heading = manta_heading
            self.evasion_candidate_boat_heading = boat_track.heading_degrees(boat_vec)
            self.evasion_candidate_heading_change = turn_deg
            self.evasion_candidate_frames = 1
            # Stable pre-evasion reference for the confirmation window:
            # same circular-mean baseline used to measure the triggering turn.
            self.approach_start_heading = _baseline_heading if _baseline_heading is not None else manta_heading
            self.status = 'EVASION_CANDIDATE'
        return None


class SimpleTracker:
    def __init__(self, max_distance=MAX_TRACK_DISTANCE):
        self.max_distance = max_distance
        self.tracks = []
        self.next_id = 1

    def update(self, detections, frame_index):
        assigned = set()
        track_matches = {}
        for tidx, track in enumerate(self.tracks):
            track_matches[tidx] = None
        for d_index, det in enumerate(detections):
            best_track = None
            best_distance = float('inf')
            for tidx, track in enumerate(self.tracks):
                if track.class_name != det['class']:
                    continue
                # ---- INCREASED dropout window from 5 to 90 frames (~3s at 30fps) ----
                if frame_index - track.last_frame > 90:
                    continue
                distance = euclidean_distance(track.latest_centroid(), centroid_from_detection(det))
                if distance < best_distance and distance <= self.max_distance and d_index not in assigned:
                    best_distance = distance
                    best_track = tidx
            if best_track is not None:
                assigned.add(d_index)
                track_matches[best_track] = d_index
        for tidx, d_index in track_matches.items():
            if d_index is not None:
                self.tracks[tidx].update(
                    centroid_from_detection(detections[d_index]),
                    detections[d_index],
                    frame_index,
                )
        for d_index, det in enumerate(detections):
            if d_index in assigned:
                continue
            boat_type = get_boat_type(det['class']) if is_boat_class(det['class']) else None
            self.tracks.append(
                Track(self.next_id, det['class'], centroid_from_detection(det), det, frame_index, boat_type=boat_type)
            )
            self.next_id += 1
        return [t for t in self.tracks if t.last_frame == frame_index]


def _log_manta_boat_interaction(pair_key, pair_data, gsd, fps, args):
    manta_id, boat_id = pair_key
    min_approach_distance_m = pair_data['min_distance_px'] * gsd

    if pair_data['evasion_event'] is not None:
        return

    boat_speed_ms_at_min = pair_data.get('boat_speed_ms_at_min_distance')
    boat_speed_knots_at_min = (boat_speed_ms_at_min * 1.94384) if boat_speed_ms_at_min is not None else None
    row = {
        'timestamp': None,
        'video_file': os.path.basename(args.video_path),
        'manta_id': manta_id,
        'distance_metres': None,
        'altitude_m': args.altitude,
        'gsd_m_per_px': round(gsd, 6),
        'manta_heading_at_evasion': None,
        'boat_heading_at_evasion': None,
        'heading_change_degrees': None,
        'approach_duration_frames': None,
        'boat_speed_m_per_s': round(boat_speed_ms_at_min, 3) if boat_speed_ms_at_min is not None else None,
        'boat_speed_knots': round(boat_speed_knots_at_min, 3) if boat_speed_knots_at_min is not None else None,
        'boat_type': None,
        'min_approach_distance_m': round(min_approach_distance_m, 2),
        'evasion_confirmed': None,
        'evasion_bearing_change_deg': None,
        'evasion_severity': '',
        'evasion_latency_frames': None,
    }
    speed_text = f"{boat_speed_knots_at_min:.2f} kts" if boat_speed_knots_at_min is not None else "N/A"
    print(f"_log_manta_boat_interaction: pair_key={pair_key}, min_approach_distance={round(min_approach_distance_m, 2)} m, boat_speed_at_min={speed_text}")
    append_event_csv(row)


def merge_and_log_pairs(all_pairs, gsd, fps, args):
    """
    Merge manta-boat pair entries that share the same boat_id and have
    similar min_approach_distances — these are almost certainly track
    fragments of the same animal.  Keeps only the record with the smallest
    minimum approach distance.  Evasion events are never collapsed.
    """
    # Group by boat_id
    by_boat = {}
    for (manta_id, boat_id), data in all_pairs.items():
        by_boat.setdefault(boat_id, []).append((manta_id, data))

    for boat_id, entries in by_boat.items():
        if len(entries) == 1:
            manta_id, data = entries[0]
            _log_manta_boat_interaction((manta_id, boat_id), data, gsd, fps, args)
            continue

        # Separate evasion events — always log individually
        evasion_entries = [(mid, d) for mid, d in entries if d['evasion_event'] is not None]
        non_evasion_entries = [(mid, d) for mid, d in entries if d['evasion_event'] is None]

        # Log evasion entries as-is
        for manta_id, data in evasion_entries:
            _log_manta_boat_interaction((manta_id, boat_id), data, gsd, fps, args)

        if not non_evasion_entries:
            continue

        # Sort non-evasion entries by min distance ascending
        non_evasion_entries.sort(key=lambda e: e[1]['min_distance_px'])
        best_manta_id, best_data = non_evasion_entries[0]
        base_dist = best_data['min_distance_px']

        stragglers = []
        for manta_id, data in non_evasion_entries[1:]:
            dist_diff_fraction = abs(data['min_distance_px'] - base_dist) / max(base_dist, 1)
            if dist_diff_fraction < 0.25:
                # Within 25% of the best — treat as a fragment of the same animal
                print(
                    f"[Merge] Absorbing manta {manta_id} into manta {best_manta_id} "
                    f"(min dist diff = {abs(data['min_distance_px'] - base_dist) * gsd:.2f} m, "
                    f"{dist_diff_fraction * 100:.1f}%)"
                )
            else:
                # Genuinely different approach distance — keep as separate record
                stragglers.append((manta_id, data))

        # Log the single best merged record
        _log_manta_boat_interaction((best_manta_id, boat_id), best_data, gsd, fps, args)

        # Log any genuinely separate records
        for manta_id, data in stragglers:
            _log_manta_boat_interaction((manta_id, boat_id), data, gsd, fps, args)


def append_event_csv(row):
    file_exists = os.path.isfile(EVENTS_CSV)
    df = pd.DataFrame([row])
    if file_exists:
        df.to_csv(EVENTS_CSV, mode='a', header=False, index=False)
    else:
        df.to_csv(EVENTS_CSV, mode='w', header=True, index=False)


def purge_video_from_csv(csv_path, video_basename):
    """Remove any existing rows for video_basename from csv_path before a re-run."""
    if not os.path.isfile(csv_path):
        return
    try:
        df = pd.read_csv(csv_path)
        if 'video_file' not in df.columns:
            return
        df = df[df['video_file'] != video_basename]
        df.to_csv(csv_path, index=False)
    except Exception as e:
        print(f"Warning: could not purge {csv_path} — {e}")


_EVASION_EVENTS_COLUMNS = [
    'video_file', 'manta_id', 'confirmation_frame', 'timestamp_s',
    'heading_change_5s_deg', 'heading_change_10s_deg', 'boat_speed_m_per_s',
    'distance_m', 'bearing_to_boat_deg', 'human_validated', 'human_notes',
]


def _ensure_evasion_csv_header():
    """Write header row if file is missing or empty."""
    if not os.path.isfile(EVASION_EVENTS_CSV) or os.path.getsize(EVASION_EVENTS_CSV) == 0:
        pd.DataFrame(columns=_EVASION_EVENTS_COLUMNS).to_csv(EVASION_EVENTS_CSV, index=False)


def append_evasion_event_csv(row):
    has_content = os.path.isfile(EVASION_EVENTS_CSV) and os.path.getsize(EVASION_EVENTS_CSV) > 0
    df = pd.DataFrame([row])
    if has_content:
        df.to_csv(EVASION_EVENTS_CSV, mode='a', header=False, index=False)
    else:
        df.to_csv(EVASION_EVENTS_CSV, mode='w', header=True, index=False)


_APPROACHES_COLUMNS = [
    'video_file', 'manta_id', 'closest_distance_m',
    'frame_at_closest', 'timestamp_s_at_closest', 'boat_speed_m_per_s_at_closest',
]


def _ensure_approaches_csv_header():
    if not os.path.isfile(APPROACHES_CSV) or os.path.getsize(APPROACHES_CSV) == 0:
        pd.DataFrame(columns=_APPROACHES_COLUMNS).to_csv(APPROACHES_CSV, index=False)


def _heading_change_at_frame(target_frame, lookback_frames, fn_list, hd_list):
    """Heading change between target_frame and target_frame - lookback_frames."""
    if not fn_list or not hd_list:
        return None
    fns = np.array(fn_list, dtype=float)
    cur_idx = int(np.searchsorted(fns, target_frame, side='right')) - 1
    if cur_idx < 0 or cur_idx >= len(hd_list) or hd_list[cur_idx] is None:
        return None
    past_target = target_frame - lookback_frames
    if past_target < fns[0]:
        return None
    past_idx = int(np.searchsorted(fns, past_target, side='left'))
    past_idx = min(past_idx, len(hd_list) - 1)
    if hd_list[past_idx] is None:
        return None
    return round(angle_difference(float(hd_list[cur_idx]), float(hd_list[past_idx])), 1)


def save_plot_data_csv(track_data, fps, output_path):
    frame_numbers = track_data['frame_numbers']
    n = len(frame_numbers)
    if n == 0:
        return

    def _to_arr(key, default=None):
        raw = track_data.get(key, [default] * n)
        return [default if v is None else v for v in raw]

    frames = np.array(frame_numbers, dtype=float)
    headings_arr = np.array([h if h is not None else np.nan for h in track_data.get('headings', [np.nan] * n)], dtype=float)

    # Heading change vs 5s and 10s ago
    lookback_5s = int(round(5.0 * fps))
    lookback_10s = int(round(10.0 * fps))
    heading_change_5s, heading_change_10s = [], []
    for i, fn in enumerate(frame_numbers):
        for lookback, out_list in [(lookback_5s, heading_change_5s), (lookback_10s, heading_change_10s)]:
            target = fn - lookback
            if target < frames[0] or np.isnan(headings_arr[i]):
                out_list.append(None)
            else:
                idx = min(int(np.searchsorted(frames, target, side='left')), n - 1)
                if np.isnan(headings_arr[idx]):
                    out_list.append(None)
                else:
                    out_list.append(round(angle_difference(float(headings_arr[i]), float(headings_arr[idx])), 1))

    # Manta swim speed — 5-frame rolling mean of instant speeds
    raw_speeds = pd.Series([s if s is not None else np.nan for s in track_data.get('instant_speeds', [np.nan] * n)])
    manta_speed_smooth = raw_speeds.rolling(5, min_periods=5).mean().tolist()

    # Tortuosity — 30-frame rolling window of cumulative path / straight-line displacement
    positions = track_data.get('manta_positions', [None] * n)
    tortuosity = [None] * n
    for i in range(n):
        if i < 30 or positions[i] is None:
            continue
        window = [p for p in positions[i - 30: i + 1] if p is not None]
        if len(window) < 2:
            continue
        pts = np.array(window, dtype=float)
        path_len = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
        straight = float(np.linalg.norm(pts[-1] - pts[0]))
        if straight < 1e-6:
            tortuosity[i] = None
        else:
            tortuosity[i] = round(path_len / straight, 4)

    df = pd.DataFrame({
        'frame': frame_numbers,
        'behavioural_phase': track_data.get('phases', [None] * n),
        'distance_m': track_data.get('distances', [None] * n),
        'bearing_to_boat_deg': track_data.get('bearings', [None] * n),
        'orientation_angle_deg': track_data.get('orientation_angles', [None] * n),
        'heading_deg': track_data.get('headings', [None] * n),
        'heading_change_5s_deg': heading_change_5s,
        'heading_change_10s_deg': heading_change_10s,
        'manta_speed_m_per_s': manta_speed_smooth,
        'tortuosity_index': tortuosity,
        'boat_speed_m_per_s': track_data.get('boat_speeds', [None] * n),
        'boat_approach_angle_deg': track_data.get('boat_approach_angles', [None] * n),
        'frames_since_boat_entered': track_data.get('frames_since_boat_entered', [None] * n),
        'manta_bbox_area_m2': track_data.get('bbox_areas', [None] * n),
    })
    df.to_csv(output_path, index=False)
    print(f"Plot data saved: {output_path}")


def draw_detection_box(frame, det, color):
    x = int(det['x'] - det['width'] / 2)
    y = int(det['y'] - det['height'] / 2)
    w = int(det['width'])
    h = int(det['height'])
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    label = f"{det['class']} {det['confidence']:.2f}"
    cv2.putText(frame, label, (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)


def draw_line_with_distance(frame, p1, p2, distance_m, color=(255, 255, 255)):
    pt1 = (int(p1[0]), int(p1[1]))
    pt2 = (int(p2[0]), int(p2[1]))
    cv2.line(frame, pt1, pt2, color, 2)
    mid = (int((pt1[0] + pt2[0]) / 2), int((pt1[1] + pt2[1]) / 2))
    text = f"{distance_m:.2f} m"
    cv2.putText(frame, text, mid, cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 3)


def draw_heading_arrow(frame, centroid, heading_deg, color=(255, 255, 255)):
    if heading_deg is None:
        return
    length = 30
    rad = np.deg2rad(heading_deg)
    dx = int(np.cos(rad) * length)
    dy = int(np.sin(rad) * length)
    pt1 = (int(centroid[0]), int(centroid[1]))
    pt2 = (pt1[0] + dx, pt1[1] + dy)
    cv2.arrowedLine(frame, pt1, pt2, color, 2, tipLength=0.2)


def _draw_validation_overlay(frame, display_manta_id, distance_m, heading_5s, boat_kts, timestamp_s):
    """Render semi-transparent confirmation box at top-centre of frame (in-place)."""
    w = frame.shape[1]
    box_w = max(700, w // 2)
    box_h = 240
    bx = (w - box_w) // 2
    by = 20

    overlay = frame.copy()
    cv2.rectangle(overlay, (bx, by), (bx + box_w, by + box_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)
    cv2.rectangle(frame, (bx, by), (bx + box_w, by + box_h), (0, 200, 255), 2)

    def _fmt(val, fmt, fallback='N/A'):
        try:
            return format(val, fmt)
        except (TypeError, ValueError):
            return fallback

    lines = [
        ("EVASION DETECTED -- CONFIRM?",                                   1.05, (0, 200, 255)),
        (f"Manta {display_manta_id}   |   t = {_fmt(timestamp_s, '.1f')} s", 0.75, (255, 255, 255)),
        (f"Distance: {_fmt(distance_m, '.1f')} m   |   "
         f"Heading change 5s: {_fmt(heading_5s, '.1f')} deg",              0.72, (255, 255, 255)),
        (f"Boat speed: {_fmt(boat_kts, '.1f')} kts",                       0.72, (255, 255, 255)),
        ("[Y] Confirm    [N] Reject    [U] Unsure    [S] Skip video",       0.68, (200, 200, 80)),
    ]

    y = by + 48
    for text, scale, color in lines:
        cv2.putText(frame, text, (bx + 18, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)
        y += int(scale * 58) + 4


def request_human_validation(frame, window_name, display_manta_id,
                             distance_m, heading_5s, boat_kts, timestamp_s):
    """
    Pause playback, show confirmation overlay, wait for Y/N/U/S keypress,
    flash a border, then return one of: 'confirmed', 'rejected', 'unsure', 'skip'.
    """
    display = frame.copy()
    _draw_validation_overlay(display, display_manta_id, distance_m, heading_5s, boat_kts, timestamp_s)
    cv2.imshow(window_name, display)

    decision = 'unsure'
    border_color = (0, 255, 255)
    while True:
        key = cv2.waitKey(100) & 0xFF
        if key == ord('y') or key == ord('Y'):
            decision, border_color = 'confirmed', (0, 255, 0)
            break
        elif key == ord('n') or key == ord('N'):
            decision, border_color = 'rejected', (0, 0, 255)
            break
        elif key == ord('u') or key == ord('U'):
            decision, border_color = 'unsure', (0, 255, 255)
            break
        elif key == ord('s') or key == ord('S'):
            decision, border_color = 'skip', (128, 128, 128)
            break

    # Flash coloured border for ~0.5 s
    h, w = display.shape[:2]
    flash = display.copy()
    cv2.rectangle(flash, (0, 0), (w - 1, h - 1), border_color, 18)
    cv2.imshow(window_name, flash)
    cv2.waitKey(int(500))

    return decision


def _on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        param['pos'] = (x, y)


def longest_consecutive_run(frame_numbers):
    if not frame_numbers:
        return 0
    longest = 1
    current = 1
    prev = frame_numbers[0]
    for frame in frame_numbers[1:]:
        if frame == prev + 1:
            current += 1
        else:
            current = 1
        prev = frame
        if current > longest:
            longest = current
    return longest


def stitch_manta_tracks(manta_plot_data, max_gap_frames=150):
    """
    Merges fragmented manta plot data entries that likely belong to the same
    animal.  Two tracks are stitched if the gap between them is within
    max_gap_frames AND their headings at the join point are within 90 degrees
    of each other.  Returns a new dict keyed by the lowest manta_id in each
    merged group.
    """
    manta_ids = sorted(manta_plot_data.keys())
    if not manta_ids:
        return manta_plot_data

    def track_summary(data):
        frames = data['frame_numbers']
        return {
            'first_frame': frames[0] if frames else None,
            'last_frame': frames[-1] if frames else None,
            'last_heading': data['headings'][-1] if data['headings'] else None,
            'first_heading': data['headings'][0] if data['headings'] else None,
        }

    summaries = {mid: track_summary(manta_plot_data[mid]) for mid in manta_ids}

    # Union-Find
    parent = {mid: mid for mid in manta_ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        parent[find(x)] = find(y)

    for i, id_a in enumerate(manta_ids):
        for id_b in manta_ids[i + 1:]:
            s_a = summaries[id_a]
            s_b = summaries[id_b]
            if s_a['last_frame'] is None or s_b['first_frame'] is None:
                continue
            # Ensure A ends before B starts
            if s_a['last_frame'] >= s_b['first_frame']:
                continue
            gap = s_b['first_frame'] - s_a['last_frame']
            if gap > max_gap_frames:
                continue
            # Heading continuity guard
            if s_a['last_heading'] is not None and s_b['first_heading'] is not None:
                heading_diff = angle_difference(s_a['last_heading'], s_b['first_heading'])
                if heading_diff > 90:
                    continue
            union(id_a, id_b)
            print(f"[Stitch] Merging manta track {id_b} into {id_a} (gap={gap} frames)")

    # Group by root
    groups = {}
    for mid in manta_ids:
        root = find(mid)
        groups.setdefault(root, []).append(mid)

    # Merge groups
    stitched = {}
    for root, members in groups.items():
        members_sorted = sorted(members, key=lambda m: summaries[m]['first_frame'] or 0)
        merged = {
            'frame_numbers': [],
            'distances': [],
            'headings': [],
            'boat_speeds': [],
            'bearings': [],
            'orientation_angles': [],
            'instant_speeds': [],
            'manta_positions': [],
            'boat_approach_angles': [],
            'frames_since_boat_entered': [],
            'bbox_areas': [],
            'phases': [],
            'evasion_frames': [],
            'distance_m_at_evasion': None,
            'boat_speed_ms_at_evasion': None,
            'boat_type_at_evasion': None,
            'heading_change_degrees': None,
        }
        for mid in members_sorted:
            d = manta_plot_data[mid]
            n = len(d['frame_numbers'])
            merged['frame_numbers'].extend(d['frame_numbers'])
            merged['distances'].extend(d['distances'])
            merged['headings'].extend(d['headings'])
            merged['boat_speeds'].extend(d.get('boat_speeds', [None] * n))
            merged['bearings'].extend(d.get('bearings', [None] * n))
            merged['orientation_angles'].extend(d.get('orientation_angles', [None] * n))
            merged['instant_speeds'].extend(d.get('instant_speeds', [None] * n))
            merged['manta_positions'].extend(d.get('manta_positions', [None] * n))
            merged['boat_approach_angles'].extend(d.get('boat_approach_angles', [None] * n))
            merged['frames_since_boat_entered'].extend(d.get('frames_since_boat_entered', [None] * n))
            merged['bbox_areas'].extend(d.get('bbox_areas', [None] * n))
            merged['phases'].extend(d.get('phases', [None] * n))
            merged['evasion_frames'].extend(d['evasion_frames'])
            if merged['distance_m_at_evasion'] is None and d['distance_m_at_evasion'] is not None:
                merged['distance_m_at_evasion'] = d['distance_m_at_evasion']
                merged['boat_speed_ms_at_evasion'] = d['boat_speed_ms_at_evasion']
                merged['boat_type_at_evasion'] = d['boat_type_at_evasion']
                merged['heading_change_degrees'] = d['heading_change_degrees']
        stitched[root] = merged

    return stitched


def save_plot(frame_numbers, distances, headings, evasion_frames, output_path,
              distance_m_at_evasion=None, boat_speed_ms_at_evasion=None, boat_type_at_evasion=None,
              heading_change_deg=None, uncertainty_fraction=None):
    if not frame_numbers:
        print(f"No valid frames for plotting. Skipping plot save: {output_path}")
        return

    df = pd.DataFrame({
        'frame': frame_numbers,
        'distance_m': distances,
        'heading_deg': headings,
    })

    first_valid_headings = df['heading_deg'].dropna().head(10).tolist()
    if first_valid_headings:
        baseline = circular_mean_headings(first_valid_headings) or 0.0
    else:
        baseline = 0.0

    def normalize_heading_change(x):
        return ((x - baseline + 180) % 360) - 180

    df['heading_change'] = df['heading_deg'].apply(normalize_heading_change).abs()
    df['heading_roll'] = df['heading_change'].rolling(10).mean()

    max_deviation = float(df['heading_change'].max()) if not df['heading_change'].empty else 0.0
    max_deviation = max(max_deviation, 45.0)

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(df['frame'], df['distance_m'], color='tab:blue', label='Distance (m)')

    if uncertainty_fraction is not None and uncertainty_fraction > 0:
        dist_array = np.array(df['distance_m'])
        ax1.fill_between(
            df['frame'],
            dist_array * (1 - uncertainty_fraction),
            dist_array * (1 + uncertainty_fraction),
            alpha=0.25,
            color='grey',
            label=f'±{uncertainty_fraction * 100:.0f}% uncertainty',
        )

    ax1.set_xlabel('Frame')
    ax1.set_ylabel('Distance (m)', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.grid(True, linestyle=':', alpha=0.5)

    ax2 = ax1.twinx()
    ax2.plot(df['frame'], df['heading_roll'], color='black', linestyle=':', label='Heading 10-frame avg')
    ax2.set_ylabel('Heading change from baseline (°)', color='black')
    ax2.tick_params(axis='y', labelcolor='black')
    ax2.set_ylim(0, max_deviation)

    for frame_num in evasion_frames:
        ax1.axvline(frame_num, color='red', linestyle='--', linewidth=1)
        ax1.text(frame_num, 1.01, 'evasion',
                 transform=ax1.get_xaxis_transform(),
                 rotation=90, va='bottom', ha='center',
                 fontsize=7, color='red', clip_on=False)

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2,
               loc='upper center', bbox_to_anchor=(0.5, -0.12),
               ncol=len(lines + lines2), frameon=True, fontsize=9)

    plt.subplots_adjust(bottom=0.3)

    if distance_m_at_evasion is not None or boat_speed_ms_at_evasion is not None or heading_change_deg is not None:
        distance_text = f"Evasion distance: {distance_m_at_evasion:.1f}m" if distance_m_at_evasion is not None else "Evasion distance: N/A"
        if boat_speed_ms_at_evasion is not None:
            boat_speed_knots = boat_speed_ms_at_evasion * 1.94384
            boat_type_label = boat_type_at_evasion if boat_type_at_evasion else "Boat"
            boat_speed_text = f"{boat_type_label} speed at evasion: {boat_speed_knots:.1f} kts"
        else:
            boat_type_label = boat_type_at_evasion if boat_type_at_evasion else "Boat"
            boat_speed_text = f"{boat_type_label} speed at evasion: N/A"
        heading_text = f"Heading change: {heading_change_deg:.1f}°" if heading_change_deg is not None else "Heading change: N/A"
        annotation_text = f"{distance_text}    |    {boat_speed_text}    |    {heading_text}"
        fig.text(0.5, 0.05, annotation_text, ha='center', fontsize=9, color='grey')

    fig.savefig(output_path)
    plt.close(fig)


def find_ffmpeg():
    return shutil.which("ffmpeg")


def convert_mov_to_mp4(input_path):
    ffmpeg_path = find_ffmpeg()
    if not ffmpeg_path:
        return None
    tmp_dir = tempfile.gettempdir()
    output_path = os.path.join(tmp_dir, f"manta_analyzer_temp_{os.getpid()}.mp4")
    command = [
        ffmpeg_path, "-y", "-i", input_path,
        "-c:v", "libx264", "-preset", "ultrafast",
        "-crf", "23", "-c:a", "aac", output_path,
    ]
    try:
        subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return output_path
    except subprocess.CalledProcessError:
        return None


def open_video_capture(video_path):
    if not os.path.isfile(video_path):
        return None, None
    cap = cv2.VideoCapture(video_path)
    if cap.isOpened():
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        fps = cap.get(cv2.CAP_PROP_FPS) or 0
        if width > 0 and height > 0 and fps > 0:
            return cap, None
    cap.release()

    if hasattr(cv2, 'CAP_FFMPEG'):
        cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
        if cap.isOpened():
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            fps = cap.get(cv2.CAP_PROP_FPS) or 0
            if width > 0 and height > 0 and fps > 0:
                return cap, None
        cap.release()

    if video_path.lower().endswith('.mov'):
        converted = convert_mov_to_mp4(input_path)
        if converted:
            cap = cv2.VideoCapture(converted)
            if cap.isOpened():
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
                fps = cap.get(cv2.CAP_PROP_FPS) or 0
                if width > 0 and height > 0 and fps > 0:
                    return cap, converted
                cap.release()
            try:
                os.remove(converted)
            except OSError:
                pass

    return None, None


def infer_altitude_from_boat(cap, total_frames, sensor_width_mm, focal_length_mm, image_width_px, boat_length_m):
    if total_frames <= 0 or image_width_px <= 0:
        return None

    frame_indices = [total_frames // 2]
    quarter_frame = int(total_frames * 0.25)
    three_quarter_frame = int(total_frames * 0.75)
    if quarter_frame not in frame_indices:
        frame_indices.append(quarter_frame)
    if three_quarter_frame not in frame_indices:
        frame_indices.append(three_quarter_frame)

    for frame_index in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        if not ret or frame is None:
            continue

        detections = roboflow_detect(frame)
        boats = [d for d in detections if is_boat_class(d['class'])]
        if not boats:
            continue

        largest_boat = max(boats, key=lambda d: d['width'])
        boat_length_px = float(largest_boat['width'])
        if boat_length_px <= 0:
            continue

        return (boat_length_m * focal_length_mm * image_width_px) / (boat_length_px * sensor_width_mm)

    return None


def main():
    args = parse_args()

    # ---- Apply CLI overrides to trigger constants (before any Track objects are created) ----
    global TURN_MIN_DEG, _HEADING_TRAILING_WIN, PROX_GATE_M
    if args.turn_trigger is not None:
        TURN_MIN_DEG = args.turn_trigger
    if args.trigger_window is not None:
        _HEADING_TRAILING_WIN = args.trigger_window
    if args.trigger_distance is not None:
        PROX_GATE_M = args.trigger_distance

    if not os.path.isfile(args.video_path):
        print(f"Video file not found: {args.video_path}")
        sys.exit(1)

    _basename = os.path.basename(args.video_path)
    purge_video_from_csv(EVENTS_CSV, _basename)
    purge_video_from_csv(EVASION_EVENTS_CSV, _basename)
    purge_video_from_csv(APPROACHES_CSV, _basename)
    _ensure_evasion_csv_header()
    _ensure_approaches_csv_header()

    cap, converted_path = open_video_capture(args.video_path)
    if cap is None:
        if args.video_path.lower().endswith('.mov'):
            print("Failed to open .mov file. Install ffmpeg or convert to mp4.")
        else:
            print(f"Unable to open video: {args.video_path}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    if args.boat_length is not None:
        inferred_altitude = infer_altitude_from_boat(
            cap, total_frames, args.sensor_width, args.focal_length, width, args.boat_length
        )
        if inferred_altitude is None:
            print("ERROR: --boat-length provided but no boat detected in middle frame. Cannot infer altitude.")
            sys.exit(1)
        args.altitude = inferred_altitude
        print(f"Altitude inferred from boat length: {inferred_altitude:.2f} m")

    gsd = compute_gsd(args.altitude, args.sensor_width, args.focal_length, width)
    print(f"Video: {args.video_path}")
    print(f"Resolution: {width}x{height} @ {fps:.2f} FPS")
    print(f"Altitude: {args.altitude} m, Sensor width: {args.sensor_width} mm, Focal length: {args.focal_length} mm")
    print(f"GSD: {gsd:.6f} m/pixel")
    print(
        "\nTrigger Settings\n"
        "----------------\n"
        f"Turn threshold : {TURN_MIN_DEG:.0f}°\n"
        f"History window : {_HEADING_TRAILING_WIN} frames\n"
        f"Distance gate  : {PROX_GATE_M:.0f} m\n"
        f"Confirmation   : {EVASION_CONFIRMATION_FRAMES} of {EVASION_LOCK_FRAMES} frames "
        f">{EVASION_HEADING_THRESHOLD_DEG:.0f}°"
    )

    tracker = SimpleTracker()
    frame_skip = max(1, args.frame_skip)
    last_detections = []
    manta_plot_data = {}
    manta_boat_pairs = {}
    # Holds pairs retired mid-video (boat swap or manta lost) — logged at end
    retired_pairs = {}
    # Tracks every frame each manta_id was detected — used for the 60-frame validity check
    manta_detection_frames = {}
    # Evasion events buffered until end-of-video validity check
    pending_events = []
    # First frame each boat_id was detected — used for evasion latency
    boat_first_frame = {}
    # Sequential display numbers for mantas (1, 2, 3…) independent of tracker IDs
    manta_display_ids = {}
    video_skipped = False       # set True when user presses S
    _mouse_click = {'pos': None}
    manually_flagged_pairs = set()  # (manta_id, boat_id) pairs already manually flagged

    if not args.no_display:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(WINDOW_NAME, _on_mouse, _mouse_click)
    frame_index = 0

    # ---- Live plot setup (renders into a numpy panel — no separate window) ----
    _LIVE_COLORS       = ['tab:blue', 'tab:orange', 'tab:green', 'tab:purple', 'tab:brown', 'tab:cyan']
    _LIVE_TARGET_H     = 900          # fixed combined-window height
    _LIVE_PLOT_W       = 700          # plot panel width in pixels
    _LIVE_DROPOUT_F    = 90           # frames absent before hiding a manta's line
    if args.plot_display:
        plt.switch_backend('Agg')   # off-screen renderer; no interactive window needed
        _live_fig, (_live_ax_dist, _live_ax_head) = plt.subplots(
            2, 1, figsize=(_LIVE_PLOT_W / 100, _LIVE_TARGET_H / 100), sharex=True)
        _live_fig.suptitle('Live Manta Tracking', fontsize=16)
        _live_ax_dist.set_ylabel('Distance to vessel (m)', fontsize=14)
        _live_ax_dist.tick_params(axis='both', labelsize=12)
        _live_ax_dist.grid(True, linestyle=':', alpha=0.5)
        _live_ax_head.set_ylabel('Heading change (°)', fontsize=14)
        _live_ax_head.set_xlabel('Frame', fontsize=14)
        _live_ax_head.tick_params(axis='both', labelsize=12)
        _live_ax_head.grid(True, linestyle=':', alpha=0.5)
        _live_lines     = {}   # manta_id -> (line_dist, line_head)
        _live_hc_state  = {}   # manta_id -> {'baseline': float|None, 'changes': list, 'roll': list}
        _live_last_seen = {}   # manta_id -> frame_index of last detection
        _live_ev_marked = set()  # onset frames already drawn as vlines
        _live_fig.tight_layout()
        # Dark blank panel shown until first render; sized to target height
        _live_plot_img = np.full((_LIVE_TARGET_H, _LIVE_PLOT_W, 3), 30, dtype=np.uint8)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_index % frame_skip == 0:
            try:
                detections = roboflow_detect(frame)
            except Exception as exc:
                print(f"Detection error at frame {frame_index}: {exc}")
                detections = []
            last_detections = detections
        else:
            detections = last_detections

        tracker.update(detections, frame_index)

        boat_tracks = [t for t in tracker.tracks if is_boat_class(t.class_name) and t.last_frame == frame_index]
        manta_tracks = [t for t in tracker.tracks if t.class_name == 'manta' and t.last_frame == frame_index]

        for det in detections:
            if is_boat_class(det['class']):
                draw_detection_box(frame, det, get_boat_colour(det['class']))

        for track in tracker.tracks:
            color = (0, 255, 0) if track.class_name == 'manta' else get_boat_colour(track.class_name) if is_boat_class(track.class_name) else (128, 128, 128)
            if track.class_name == 'manta' and track.bboxes:
                if track.id not in manta_display_ids:
                    manta_display_ids[track.id] = len(manta_display_ids) + 1
                display_num = manta_display_ids[track.id]
                bb = track.bboxes[-1]
                bx = int(bb['x'] - bb['width'] / 2)
                by = int(bb['y'] - bb['height'] / 2)
                cv2.rectangle(frame, (bx, by), (bx + int(bb['width']), by + int(bb['height'])), color, 2)
                cv2.putText(frame, f"Manta {display_num}", (bx, by - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
            for i, centroid in enumerate(track.centroids):
                alpha = (i + 1) / len(track.centroids)
                intensity = int(255 * alpha)
                dot_color = (
                    intensity if color[0] else 0,
                    intensity if color[1] else 0,
                    intensity if color[2] else 0,
                )
                cv2.circle(frame, (int(centroid[0]), int(centroid[1])), 3, dot_color, -1)
            heading_deg = track.heading_degrees(track.current_heading_vector())
            draw_heading_arrow(frame, track.latest_centroid(), heading_deg, color)

        for boat in boat_tracks:
            if boat.id not in boat_first_frame:
                boat_first_frame[boat.id] = frame_index
            boat_speed_ms = boat.current_speed_m_per_s(gsd, fps)
            boat_speed_knots = (boat_speed_ms * 1.94384) if boat_speed_ms is not None else None
            if boat_speed_ms is not None:
                speed_label = f"{boat_speed_ms:.2f} m/s ({boat_speed_knots:.2f} kts)"
                x = int(boat.bboxes[-1]['x'] - boat.bboxes[-1]['width'] / 2)
                top_y = int(boat.bboxes[-1]['y'] - boat.bboxes[-1]['height'] / 2)
                speed_y = top_y - 50
                if speed_y < 0:
                    speed_y = top_y + 30
                cv2.putText(frame, speed_label, (x, speed_y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 3)

        # Consume any pending mouse click once per frame so only one manta claims it
        frame_click = _mouse_click['pos']
        _mouse_click['pos'] = None

        for manta in manta_tracks:
            manta_detection_frames.setdefault(manta.id, []).append(frame_index)
            if boat_tracks:
                old_pair_key = None
                current_boat_id = None
                for pair_key in list(manta_boat_pairs.keys()):
                    if pair_key[0] == manta.id:
                        old_pair_key = pair_key
                        current_boat_id = pair_key[1]
                        break

                boat_ids_visible = {b.id for b in boat_tracks}

                if current_boat_id is not None and current_boat_id not in boat_ids_visible:
                    # Assigned boat not detected this frame — check how long it's been absent
                    assigned_track = next(
                        (t for t in tracker.tracks if t.id == current_boat_id and is_boat_class(t.class_name)),
                        None
                    )
                    frames_absent = (frame_index - assigned_track.last_frame) if assigned_track else 9999
                    if frames_absent <= 60:
                        # Brief dropout — hold the pairing, skip interaction this frame
                        cx, cy = manta.latest_centroid()
                        status_color = (0, 0, 255) if manta.status == 'EVADING' else \
                                       (0, 255, 255) if manta.status == 'APPROACHING' else \
                                       (255, 255, 255)
                        cv2.putText(
                            frame, f"{manta.status} {int(manta.orientation_angle_deg)}deg",
                            (int(cx) + 15, int(cy) - 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 3,
                        )
                        continue

                if current_boat_id is not None and current_boat_id in boat_ids_visible:
                    nearest_boat = next(b for b in boat_tracks if b.id == current_boat_id)
                else:
                    nearest_boat = min(
                        boat_tracks,
                        key=lambda b: euclidean_distance(manta.latest_centroid(), b.latest_centroid()),
                    )

                new_pair_key = (manta.id, nearest_boat.id)

                if old_pair_key is not None and old_pair_key != new_pair_key:
                    # Assigned boat permanently gone — retire old pair and re-assign to nearest
                    pair_data = manta_boat_pairs.pop(old_pair_key)
                    retired_pairs[old_pair_key] = pair_data

                current_distance_px = euclidean_distance(manta.latest_centroid(), nearest_boat.latest_centroid())
                current_boat_speed_ms = nearest_boat.current_speed_m_per_s(gsd, fps)
                if new_pair_key not in manta_boat_pairs:
                    manta_boat_pairs[new_pair_key] = {
                        'min_distance_px': current_distance_px,
                        'boat_speed_ms_at_min_distance': current_boat_speed_ms,
                        'evasion_event': None,
                    }
                else:
                    if current_distance_px < manta_boat_pairs[new_pair_key]['min_distance_px']:
                        manta_boat_pairs[new_pair_key]['min_distance_px'] = current_distance_px
                        manta_boat_pairs[new_pair_key]['boat_speed_ms_at_min_distance'] = current_boat_speed_ms

                distance_m = euclidean_distance(manta.latest_centroid(), nearest_boat.latest_centroid()) * gsd
                event = manta.update_heading_state(nearest_boat, frame_index, distance_m)
                manta_heading = manta.heading_degrees(manta.current_heading_vector())
                manta_cx, manta_cy = manta.latest_centroid()
                boat_cx, boat_cy = nearest_boat.latest_centroid()

                if manta_heading is not None:
                    track_data = manta_plot_data.setdefault(manta.id, {
                        'frame_numbers': [], 'distances': [], 'headings': [],
                        'boat_speeds': [], 'bearings': [], 'orientation_angles': [],
                        'instant_speeds': [], 'manta_positions': [],
                        'boat_approach_angles': [], 'frames_since_boat_entered': [],
                        'bbox_areas': [], 'phases': [],
                        'evasion_frames': [], 'distance_m_at_evasion': None,
                        'boat_speed_ms_at_evasion': None, 'boat_type_at_evasion': None,
                        'heading_change_degrees': None,
                    })

                    # Boat approach angle (requires boat speed >= 0.2 m/s)
                    boat_spd = nearest_boat.current_speed_m_per_s(gsd, fps)
                    if boat_spd is not None and boat_spd >= 0.2:
                        bv_short = nearest_boat.heading_vector_over_n_frames(5)
                        if bv_short is not None:
                            btm = np.array([manta_cx - boat_cx, manta_cy - boat_cy], dtype=np.float32)
                            n_btm = np.linalg.norm(btm)
                            if n_btm > 1e-6:
                                d_btm = float(np.dot(bv_short, btm / n_btm))
                                boat_approach_angle = float(np.degrees(np.arccos(np.clip(d_btm, -1.0, 1.0))))
                            else:
                                boat_approach_angle = None
                        else:
                            boat_approach_angle = None
                    else:
                        boat_approach_angle = None

                    # Manta bbox area
                    bbox = manta.bboxes[-1]
                    bbox_area_m2 = bbox['width'] * bbox['height'] * (gsd ** 2)

                    # Behavioural phase
                    if manta.in_post_evasion:
                        phase = 'post_evasion'
                    elif manta.status == 'EVASION_CANDIDATE':
                        phase = 'evasion_candidate'
                    elif distance_m <= 50.0:
                        phase = 'approach'
                    else:
                        phase = 'baseline'

                    track_data['frame_numbers'].append(frame_index)
                    track_data['distances'].append(distance_m)
                    track_data['headings'].append(manta_heading)
                    track_data['boat_speeds'].append(boat_spd)
                    track_data['bearings'].append(manta.orientation_angle_deg)
                    track_data['orientation_angles'].append(manta.orientation_angle_deg)
                    track_data['instant_speeds'].append(manta.instant_speed_m_per_s(gsd, fps))
                    track_data['manta_positions'].append((manta_cx, manta_cy))
                    track_data['boat_approach_angles'].append(boat_approach_angle)
                    track_data['frames_since_boat_entered'].append(
                        frame_index - boat_first_frame.get(nearest_boat.id, frame_index)
                    )
                    track_data['bbox_areas'].append(round(bbox_area_m2, 4))
                    track_data['phases'].append(phase)

                    if args.plot_display:
                        _mid_lv = manta.id
                        # Create a new line pair the first time this manta appears
                        if _mid_lv not in _live_lines:
                            _col = _LIVE_COLORS[len(_live_lines) % len(_LIVE_COLORS)]
                            _disp_lv = manta_display_ids.get(_mid_lv, _mid_lv)
                            _ld, = _live_ax_dist.plot([], [], color=_col, lw=2.5,
                                                      label=f'Manta {_disp_lv}',
                                                      visible=False)
                            _lh, = _live_ax_head.plot([], [], color=_col, lw=2.5,
                                                      visible=False)
                            _live_lines[_mid_lv] = (_ld, _lh)
                            _live_hc_state[_mid_lv] = {
                                'baseline': None, 'changes': [], 'roll': [],
                                'consecutive': 0}
                            _live_ax_dist.legend(fontsize=11, loc='upper right')
                        # Mark seen and increment consecutive-detection counter
                        _live_last_seen[_mid_lv] = frame_index
                        _hcs = _live_hc_state[_mid_lv]
                        _hcs['consecutive'] += 1
                        _ld, _lh = _live_lines[_mid_lv]
                        # Only show once 30 straight frames have been confirmed
                        if _hcs['consecutive'] >= 30:
                            _ld.set_visible(True)
                            _lh.set_visible(True)
                        # Incremental rolling heading change (10-frame window)
                        if _hcs['baseline'] is None:
                            _valid_hds = [h for h in track_data['headings'] if h is not None]
                            if len(_valid_hds) >= 5:
                                _hcs['baseline'] = float(np.mean(_valid_hds[:5]))
                        _bl = _hcs['baseline'] or 0.0
                        _hcs['changes'].append(
                            abs(((manta_heading - _bl + 180) % 360) - 180))
                        _hcs['roll'].append(float(np.mean(_hcs['changes'][-10:])))
                        # Push updated data to the line objects
                        _ld.set_data(track_data['frame_numbers'], track_data['distances'])
                        _lh.set_data(track_data['frame_numbers'], _hcs['roll'])

                if manta.evasion_flash > 0 or manta.in_post_evasion:
                    line_color = (0, 0, 255)
                elif manta.status == 'EVASION_CANDIDATE':
                    line_color = (0, 165, 255)
                elif manta.orientation_angle_deg < 90.0:
                    line_color = (0, 255, 255)
                else:
                    line_color = (255, 255, 255)

                draw_line_with_distance(
                    frame, manta.latest_centroid(), nearest_boat.latest_centroid(), distance_m, color=line_color
                )

                if event is not None:
                    onset_frame = event.get('onset_frame', frame_index)
                    evasion_confirmed = event.get('evasion_confirmed', True)
                    event_type = event.get('type', 'confirmed')

                    # evasion_bearing_change_deg: pre vs post heading means around onset
                    td = manta_plot_data.get(manta.id, {})
                    fn_list = td.get('frame_numbers', [])
                    hd_list = td.get('headings', [])
                    pre_h = [hd_list[i] for i, fn in enumerate(fn_list) if onset_frame - 30 <= fn < onset_frame]
                    post_h = [hd_list[i] for i, fn in enumerate(fn_list) if onset_frame + 5 <= fn <= onset_frame + 35]
                    pre_mean = circular_mean_headings(pre_h)
                    post_mean = circular_mean_headings(post_h)
                    evasion_bearing_change_deg = (
                        round(angle_difference(post_mean, pre_mean), 1)
                        if pre_mean is not None and post_mean is not None else None
                    )

                    # Severity
                    if evasion_confirmed and evasion_bearing_change_deg is not None:
                        if evasion_bearing_change_deg >= 120:
                            evasion_severity = 'strong'
                        elif evasion_bearing_change_deg >= 70:
                            evasion_severity = 'moderate'
                        else:
                            evasion_severity = 'mild'
                    else:
                        evasion_severity = ''

                    evasion_latency = (
                        onset_frame - boat_first_frame[nearest_boat.id]
                        if nearest_boat.id in boat_first_frame else None
                    )

                    # Use distance and boat speed recorded at evasion-candidate onset,
                    # not at the confirmation frame 90 frames later.
                    _d_list  = td.get('distances',   [])
                    _bs_list = td.get('boat_speeds', [])
                    if fn_list:
                        _oi = (fn_list.index(onset_frame) if onset_frame in fn_list
                               else int(np.argmin(np.abs(np.array(fn_list) - onset_frame))))
                        distance_metres = (_d_list[_oi]
                                           if _oi < len(_d_list) and _d_list[_oi] is not None
                                           else event['distance'] * gsd)
                        boat_speed_ms   = _bs_list[_oi] if _oi < len(_bs_list) else None
                    else:
                        distance_metres = event['distance'] * gsd
                        boat_speed_ms   = nearest_boat.current_speed_m_per_s(gsd, fps)
                    boat_speed_knots = (boat_speed_ms * 1.94384) if boat_speed_ms is not None else None
                    min_approach_distance_m = manta_boat_pairs[new_pair_key]['min_distance_px'] * gsd

                    row = {
                        'timestamp': round(onset_frame / fps, 2),
                        'video_file': os.path.basename(args.video_path),
                        'manta_id': manta.id,
                        'distance_metres': round(distance_metres, 2),
                        'altitude_m': args.altitude,
                        'gsd_m_per_px': round(gsd, 6),
                        'manta_heading_at_evasion': round(event['manta_heading_at_evasion'], 1),
                        'boat_heading_at_evasion': round(event['boat_heading_at_evasion'], 1),
                        'heading_change_degrees': round(event['heading_change_degrees'], 1),
                        'approach_duration_frames': event['approach_duration_frames'],
                        'boat_speed_m_per_s': round(boat_speed_ms, 3) if boat_speed_ms is not None else None,
                        'boat_speed_knots': round(boat_speed_knots, 3) if boat_speed_knots is not None else None,
                        'boat_type': nearest_boat.boat_type,
                        'min_approach_distance_m': round(min_approach_distance_m, 2),
                        'evasion_confirmed': evasion_confirmed,
                        'evasion_bearing_change_deg': evasion_bearing_change_deg,
                        'evasion_severity': evasion_severity,
                        'evasion_latency_frames': evasion_latency,
                    }

                    display_num = manta_display_ids.get(manta.id, manta.id)
                    hc5_at_confirm = None
                    hc10_at_confirm = None
                    if event_type == 'confirmed':
                        lookback_5s  = int(round(5.0  * fps))
                        lookback_10s = int(round(10.0 * fps))
                        hc5_at_confirm  = _heading_change_at_frame(frame_index, lookback_5s,  fn_list, hd_list)
                        hc10_at_confirm = _heading_change_at_frame(frame_index, lookback_10s, fn_list, hd_list)

                    if event_type == 'confirmed':
                        # ---- Human validation (runs before print so print shows true outcome) ----
                        can_validate = (
                            not args.no_display
                            and not args.no_validate
                            and not video_skipped
                        )
                        if can_validate:
                            decision = request_human_validation(
                                frame, WINDOW_NAME, display_num,
                                distance_metres,
                                hc5_at_confirm,
                                boat_speed_knots,
                                onset_frame / fps,
                            )
                            if decision == 'skip':
                                video_skipped = True
                                human_validated = 'unreviewed'
                            else:
                                human_validated = decision
                        else:
                            human_validated = 'unreviewed'

                        print(
                            f"EVASION CONFIRMED (machine) | human={human_validated.upper()} "
                            f"@ onset {onset_frame / fps:.1f}s | Manta {display_num} "
                            f"| distance={distance_metres:.1f} m "
                            f"| heading_change_5s={hc5_at_confirm if hc5_at_confirm is not None else 'N/A'} deg "
                            f"| boat={round(boat_speed_knots, 1) if boat_speed_knots is not None else 'N/A'} kts"
                        )

                        if human_validated != 'rejected':
                            # Retroactively label candidate frames as evasion_confirmed
                            if td:
                                phases_list = td.get('phases', [])
                                for i, fn in enumerate(fn_list):
                                    if onset_frame <= fn <= frame_index and i < len(phases_list):
                                        phases_list[i] = 'evasion_confirmed'
                            # Add plot marker for this confirmed evasion
                            _conf_td = manta_plot_data.setdefault(manta.id, {
                                'frame_numbers': [], 'distances': [], 'headings': [],
                                'boat_speeds': [], 'bearings': [], 'orientation_angles': [],
                                'instant_speeds': [], 'manta_positions': [],
                                'boat_approach_angles': [], 'frames_since_boat_entered': [],
                                'bbox_areas': [], 'phases': [],
                                'evasion_frames': [], 'distance_m_at_evasion': None,
                                'boat_speed_ms_at_evasion': None, 'boat_type_at_evasion': None,
                                'heading_change_degrees': None,
                            })
                            if not _conf_td['evasion_frames']:
                                _conf_td['distance_m_at_evasion'] = distance_metres
                                _conf_td['boat_speed_ms_at_evasion'] = boat_speed_ms
                                _conf_td['boat_type_at_evasion'] = nearest_boat.boat_type
                                _conf_td['heading_change_degrees'] = event['heading_change_degrees']
                            _conf_td['evasion_frames'].append(onset_frame)
                            try:
                                append_evasion_event_csv({
                                    'video_file': os.path.basename(args.video_path),
                                    'manta_id': manta_display_ids.get(manta.id, manta.id),
                                    'confirmation_frame': onset_frame,
                                    'timestamp_s': round(onset_frame / fps, 2),
                                    'heading_change_5s_deg': hc5_at_confirm,
                                    'heading_change_10s_deg': hc10_at_confirm,
                                    'boat_speed_m_per_s': round(boat_speed_ms, 3) if boat_speed_ms is not None else None,
                                    'distance_m': round(distance_metres, 2),
                                    'bearing_to_boat_deg': round(manta.orientation_angle_deg, 1),
                                    'human_validated': human_validated,
                                    'human_notes': '',
                                })
                                print(f"  -> Written to evasion_events.csv (human_validated={human_validated})")
                            except Exception as _csv_err:
                                print(f"  -> ERROR writing evasion_events.csv: {_csv_err}")
                        else:
                            print(f"  -> Rejected by human — not written to evasion_events.csv")
                    else:
                        print(
                            f"EVASION DISCARDED (machine) "
                            f"@ onset {onset_frame / fps:.1f}s | Manta {display_num} "
                            f"| heading_change={event['heading_change_degrees']:.1f} deg"
                        )
                        manta_boat_pairs[new_pair_key]['evasion_event'] = event

                    row['human_validated'] = human_validated if event_type == 'confirmed' else 'unreviewed'
                    row['human_notes'] = ''
                    pending_events.append(row)

                # ---- Manual evasion flag via bbox click ----
                if (frame_click is not None
                        and manta.bboxes
                        and (manta.id, nearest_boat.id) not in manually_flagged_pairs):
                    _bb = manta.bboxes[-1]
                    _bx1 = _bb['x'] - _bb['width'] / 2
                    _by1 = _bb['y'] - _bb['height'] / 2
                    _bx2 = _bb['x'] + _bb['width'] / 2
                    _by2 = _bb['y'] + _bb['height'] / 2
                    if _bx1 <= frame_click[0] <= _bx2 and _by1 <= frame_click[1] <= _by2:
                        frame_click = None  # consume — only one manta per click
                        manually_flagged_pairs.add((manta.id, nearest_boat.id))
                        _disp   = manta_display_ids.get(manta.id, manta.id)
                        _bs_ms  = nearest_boat.current_speed_m_per_s(gsd, fps)
                        _td_m   = manta_plot_data.get(manta.id, {})
                        _lb5    = int(round(5.0  * fps))
                        _lb10   = int(round(10.0 * fps))
                        _fns_m  = _td_m.get('frame_numbers', [])
                        _hds_m  = _td_m.get('headings', [])
                        try:
                            append_evasion_event_csv({
                                'video_file': os.path.basename(args.video_path),
                                'manta_id': _disp,
                                'confirmation_frame': frame_index,
                                'timestamp_s': round(frame_index / fps, 2),
                                'heading_change_5s_deg': _heading_change_at_frame(frame_index, _lb5,  _fns_m, _hds_m),
                                'heading_change_10s_deg': _heading_change_at_frame(frame_index, _lb10, _fns_m, _hds_m),
                                'boat_speed_m_per_s': round(_bs_ms, 3) if _bs_ms is not None else None,
                                'distance_m': round(distance_m, 2),
                                'bearing_to_boat_deg': round(manta.orientation_angle_deg, 1),
                                'human_validated': 'confirmed',
                                'human_notes': 'manually_flagged',
                            })
                            print(f"MANUAL EVASION flagged: Manta {_disp} @ {frame_index / fps:.1f}s "
                                  f"| distance={distance_m:.1f} m -> evasion_events.csv")
                        except Exception as _me:
                            print(f"ERROR writing manual evasion: {_me}")
                        # Prevent _log_manta_boat_interaction logging this as a non-evasion
                        if new_pair_key in manta_boat_pairs:
                            manta_boat_pairs[new_pair_key]['evasion_event'] = 'manually_flagged'
                        # Visual flash
                        if not args.no_display:
                            _fh, _fw = frame.shape[:2]
                            _fl = frame.copy()
                            cv2.rectangle(_fl, (0, 0), (_fw - 1, _fh - 1), (0, 255, 0), 18)
                            cv2.putText(_fl, f"MANUAL EVASION — Manta {_disp}",
                                        (_fw // 2 - 260, _fh // 2),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3)
                            cv2.imshow(WINDOW_NAME, _fl)
                            cv2.waitKey(700)
                        # Mark on the manta track for the plot
                        manta.evasion_flash = EVASION_FLASH_FRAMES
                        manta.last_evasion_distance = euclidean_distance(
                            manta.latest_centroid(), nearest_boat.latest_centroid()
                        )
                        _td_m2 = manta_plot_data.get(manta.id, {})
                        if _td_m2:
                            _td_m2.setdefault('evasion_frames', [])
                            if not _td_m2['evasion_frames']:
                                _td_m2['distance_m_at_evasion'] = distance_m
                                _td_m2['boat_speed_ms_at_evasion'] = _bs_ms
                                _td_m2['boat_type_at_evasion'] = nearest_boat.boat_type
                            _td_m2['evasion_frames'].append(frame_index)

                cx, cy = manta_cx, manta_cy
                status_color = (0, 0, 255) if manta.status == 'EVADING' else \
                               (0, 165, 255) if manta.status == 'EVASION_CANDIDATE' else \
                               (0, 255, 255) if manta.status == 'APPROACHING' else \
                               (255, 255, 255)
                cv2.putText(
                    frame, f"{manta.status} {int(manta.orientation_angle_deg)}deg",
                    (int(cx) + 15, int(cy) - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 3,
                )

                if manta.evasion_flash > 0:
                    evasion_distance_m = manta.last_evasion_distance * gsd
                    mid_x = int((cx + nearest_boat.latest_centroid()[0]) / 2)
                    mid_y = int((cy + nearest_boat.latest_centroid()[1]) / 2) + 40
                    cv2.putText(
                        frame, f"EVASION: {evasion_distance_m:.2f} m",
                        (mid_x, mid_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3,
                    )
                    manta.evasion_flash -= 1
            else:
                # No boat visible — retire any active pairs for this manta
                for pair_key in list(manta_boat_pairs.keys()):
                    if pair_key[0] == manta.id:
                        pair_data = manta_boat_pairs.pop(pair_key)
                        retired_pairs[pair_key] = pair_data
                manta.reset_approach()
                cx, cy = manta.latest_centroid()
                cv2.putText(
                    frame, f"{manta.status} {int(manta.orientation_angle_deg)}deg",
                    (int(cx) + 15, int(cy) - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3,
                )

        cv2.putText(frame, f"Frame: {frame_index}/{total_frames}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3)
        cv2.putText(frame, f"GSD: {gsd:.6f} m/px", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3)
        cv2.putText(frame, f"Altitude: {args.altitude} m", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3)
        cv2.putText(frame, "Q to quit", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (180, 180, 180), 3)

        if args.plot_display and frame_index % 5 == 0:
            # Hide lines for mantas absent more than the dropout window
            for _mid_lv, _ld_lh in _live_lines.items():
                if frame_index - _live_last_seen.get(_mid_lv, 0) > _LIVE_DROPOUT_F:
                    _ld_lh[0].set_visible(False)
                    _ld_lh[1].set_visible(False)
                    # Reset so it must re-qualify with 30 straight frames on reappearance
                    if _mid_lv in _live_hc_state:
                        _live_hc_state[_mid_lv]['consecutive'] = 0
            # Evasion vlines
            for _mid_ev, _edata in manta_plot_data.items():
                for _ef in _edata.get('evasion_frames', []):
                    if _ef not in _live_ev_marked:
                        _live_ev_marked.add(_ef)
                        _live_ax_dist.axvline(_ef, color='red', linestyle='--', lw=1.5)
                        _live_ax_head.axvline(_ef, color='red', linestyle='--', lw=1.5)
            _live_ax_dist.relim(); _live_ax_dist.autoscale_view()
            _live_ax_head.relim(); _live_ax_head.autoscale_view()
            _live_fig.canvas.draw()
            _cw, _ch = _live_fig.canvas.get_width_height()
            _plot_rgba = np.frombuffer(
                _live_fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(_ch, _cw, 4)
            _plot_bgr = cv2.cvtColor(_plot_rgba, cv2.COLOR_RGBA2BGR)
            # Pad to target height with black bars (no stretching)
            if _plot_bgr.shape[0] < _LIVE_TARGET_H:
                _pad_total = _LIVE_TARGET_H - _plot_bgr.shape[0]
                _pad_top   = _pad_total // 2
                _pad_bot   = _pad_total - _pad_top
                _plot_bgr  = np.vstack([
                    np.zeros((_pad_top, _LIVE_PLOT_W, 3), dtype=np.uint8),
                    _plot_bgr,
                    np.zeros((_pad_bot, _LIVE_PLOT_W, 3), dtype=np.uint8),
                ])
            elif _plot_bgr.shape[0] > _LIVE_TARGET_H:
                _plot_bgr = _plot_bgr[:_LIVE_TARGET_H]
            _live_plot_img = _plot_bgr

        if not args.no_display:
            if args.plot_display:
                # Scale video to target height, maintaining aspect ratio, with black bars if needed
                _vid_scale  = _LIVE_TARGET_H / height
                _vid_w_scaled = int(width * _vid_scale)
                _vid_scaled = cv2.resize(frame, (_vid_w_scaled, _LIVE_TARGET_H))
                _display = np.hstack([_live_plot_img, _vid_scaled])
            else:
                _display = frame
            cv2.imshow(WINDOW_NAME, _display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

        frame_index += 1

    # ---- End-of-video: validate manta tracks — must have 60 consecutive detected frames ----
    valid_manta_ids = {
        mid for mid, frames in manta_detection_frames.items()
        if longest_consecutive_run(frames) >= 30
    }
    invalid_manta_ids = set(manta_detection_frames.keys()) - valid_manta_ids
    print(f"Manta validity check (30 consecutive frames required):")
    print(f"  Valid:   {sorted(valid_manta_ids) if valid_manta_ids else 'none'}")
    if invalid_manta_ids:
        print(f"  Rejected: {sorted(invalid_manta_ids)} — too short to confirm as manta")

    # ---- Merge remaining active pairs into retired set, then log all at once ----
    for pair_key in list(manta_boat_pairs.keys()):
        retired_pairs[pair_key] = manta_boat_pairs.pop(pair_key)

    # Filter pairs to valid mantas only before logging
    retired_pairs = {k: v for k, v in retired_pairs.items() if k[0] in valid_manta_ids}

    # Write buffered evasion events for valid mantas only
    for row in pending_events:
        if row['manta_id'] in valid_manta_ids:
            append_event_csv(row)

    print(f"End-of-video cleanup: merging and logging {len(retired_pairs)} pair(s)")
    merge_and_log_pairs(retired_pairs, gsd, fps, args)

    cap.release()

    # ---- Write closest-approach records for all valid mantas ----
    _approach_rows = []
    for _mid, _data in manta_plot_data.items():
        if _mid not in valid_manta_ids:
            continue
        _dists  = _data.get('distances', [])
        _frames = _data.get('frame_numbers', [])
        _bspds  = _data.get('boat_speeds', [])
        _valid_triples = [
            (d, f, b)
            for d, f, b in zip(_dists, _frames, _bspds)
            if d is not None
        ]
        if not _valid_triples:
            continue
        _dv, _fv, _bv = zip(*_valid_triples)
        _min_idx = int(np.argmin(_dv))
        _disp_id = manta_display_ids.get(_mid, _mid)
        _approach_rows.append({
            'video_file':                    os.path.basename(args.video_path),
            'manta_id':                      _disp_id,
            'closest_distance_m':            round(_dv[_min_idx], 2),
            'frame_at_closest':              int(_fv[_min_idx]),
            'timestamp_s_at_closest':        round(_fv[_min_idx] / fps, 2),
            'boat_speed_m_per_s_at_closest': round(_bv[_min_idx], 3) if _bv[_min_idx] is not None else None,
        })
        print(f"Closest approach: Manta {_disp_id} = {_dv[_min_idx]:.1f} m "
              f"@ {_fv[_min_idx] / fps:.1f}s")
    if _approach_rows:
        _ap_has_content = os.path.isfile(APPROACHES_CSV) and os.path.getsize(APPROACHES_CSV) > 0
        pd.DataFrame(_approach_rows).to_csv(
            APPROACHES_CSV,
            mode='a' if _ap_has_content else 'w',
            header=not _ap_has_content,
            index=False,
        )
        print(f"  -> Written {len(_approach_rows)} approach row(s) to approaches.csv")

    # ---- Stitch fragmented plot tracks then save plots / CSVs ----
    if args.plot or args.save_plot_data:
        manta_plot_data = {mid: d for mid, d in manta_plot_data.items() if mid in valid_manta_ids}
        manta_plot_data = stitch_manta_tracks(manta_plot_data, max_gap_frames=150)
        video_dir = os.path.dirname(args.video_path)
        video_stem = os.path.splitext(os.path.basename(args.video_path))[0]
        for manta_id, data in manta_plot_data.items():
            if len(data['frame_numbers']) < 30:
                print(f"Skipping manta {manta_id}: only {len(data['frame_numbers'])} valid frames")
                continue
            if longest_consecutive_run(data['frame_numbers']) < 30:
                print(f"Skipping manta {manta_id}: fewer than 30 consecutive frames")
                continue

            # Use the display number shown in the video overlay, not the raw tracker ID.
            # Raw tracker IDs count all tracks (boats + mantas); display IDs count mantas only.
            _file_id = manta_display_ids.get(manta_id, manta_id)

            if args.plot:
                plot_path = os.path.join(video_dir, f"{video_stem}_manta{_file_id}_plot.png")
                save_plot(
                    data['frame_numbers'],
                    data['distances'],
                    data['headings'],
                    data['evasion_frames'],
                    plot_path,
                    distance_m_at_evasion=data.get('distance_m_at_evasion'),
                    boat_speed_ms_at_evasion=data.get('boat_speed_ms_at_evasion'),
                    boat_type_at_evasion=data.get('boat_type_at_evasion'),
                    heading_change_deg=data.get('heading_change_degrees'),
                    uncertainty_fraction=args.uncertain,
                )

            if args.save_plot_data:
                csv_path = os.path.join(video_dir, f"{video_stem}_manta{_file_id}_plotdata.csv")
                save_plot_data_csv(data, fps, csv_path)

    if converted_path and os.path.isfile(converted_path):
        try:
            os.remove(converted_path)
        except OSError:
            pass
    if not args.no_display:
        cv2.destroyAllWindows()

    if args.plot_display:
        plt.close(_live_fig)


if __name__ == '__main__':
    main()
