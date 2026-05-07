import numpy as np
from scipy.spatial.distance import cdist

def track_centroids(detections_over_frames, max_distance=50):
    """
    Simple centroid tracking using nearest neighbor.
    detections_over_frames: list of lists of detections per frame.
    Returns list of tracks: [{'id': int, 'positions': [(frame_id, centroid), ...], 'class': str}]
    """
    tracks = []
    track_id = 0
    prev_centroids = {}

    for frame_id, detections in enumerate(detections_over_frames):
        current_centroids = {}
        for det in detections:
            if det['class'] in ['manta', 'boat']:
                centroid = (det['x'], det['y'])
                current_centroids[centroid] = det

        # Match to previous
        if prev_centroids:
            prev_points = list(prev_centroids.keys())
            curr_points = list(current_centroids.keys())
            if prev_points and curr_points:
                distances = cdist(prev_points, curr_points)
                for i, j in zip(*np.unravel_index(np.argsort(distances, axis=None), distances.shape)):
                    if distances[i, j] < max_distance:
                        # Assign to existing track
                        track = next((t for t in tracks if t['id'] == list(prev_centroids.values())[i]['track_id']), None)
                        if track:
                            track['positions'].append((frame_id, curr_points[j]))
                        else:
                            # New track
                            tracks.append({
                                'id': track_id,
                                'class': current_centroids[curr_points[j]]['class'],
                                'positions': [(frame_id, curr_points[j])]
                            })
                            current_centroids[curr_points[j]]['track_id'] = track_id
                            track_id += 1
                        prev_centroids.pop(prev_points[i])
                        current_centroids.pop(curr_points[j])
                        break

        # New tracks for unmatched
        for cent, det in current_centroids.items():
            tracks.append({
                'id': track_id,
                'class': det['class'],
                'positions': [(frame_id, cent)]
            })
            track_id += 1

        prev_centroids = {k: v for k, v in current_centroids.items() if 'track_id' in v}

    return tracks

def detect_behavior_change(tracks, threshold_angle=30, flag_duration_frames=5):
    """
    Detect direction changes in manta tracks.
    Returns list of (track_id, frame_id, 'change', end_frame) for changes, with flag duration.
    """
    changes = []
    for track in tracks:
        if track['class'] == 'manta' and len(track['positions']) >= 3:
            positions = track['positions']
            for i in range(2, len(positions)):
                angle = calculate_turn_angle(positions[i-2][1], positions[i-1][1], positions[i][1])
                if angle > threshold_angle:
                    start_frame = positions[i][0]
                    end_frame = min(start_frame + flag_duration_frames, positions[-1][0])
                    changes.append((track['id'], start_frame, 'direction_change', end_frame))
    return changes

def calculate_turn_angle(p1, p2, p3):
    """
    Calculate the turn angle at p2.
    """
    v1 = np.array(p2) - np.array(p1)
    v2 = np.array(p3) - np.array(p2)
    if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
        return 0
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))