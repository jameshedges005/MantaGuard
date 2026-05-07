import cv2
import exifread
import numpy as np

def extract_metadata(video_path):
    """
    Extract metadata from video EXIF if available.
    Returns dict with altitude, sensor_size, focal_length.
    """
    try:
        with open(video_path, 'rb') as f:
            tags = exifread.process_file(f, details=False)
        altitude = float(tags.get('GPS GPSAltitude', 0)) if 'GPS GPSAltitude' in tags else None
        sensor_size = None  # Often not in EXIF
        focal_length = None
        return {'altitude': altitude, 'sensor_size': sensor_size, 'focal_length': focal_length}
    except Exception as e:
        print(f"Error extracting metadata: {e}")
        return {'altitude': None, 'sensor_size': None, 'focal_length': None}

def extract_frames(video_path, interval=30):
    """
    Extract frames from video at given interval.
    Returns list of frames.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open video file.")
    frames = []
    frame_count = 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % interval == 0:
            frames.append((frame_count / fps, frame))  # Include timestamp
        frame_count += 1
    cap.release()
    return frames

def get_video_info(video_path):
    """
    Get video properties: width, height, fps.
    """
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return width, height, fps