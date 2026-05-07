import streamlit as st
import cv2
import numpy as np
import pandas as pd
from src.video_processor import extract_frames, extract_metadata, get_video_info
from src.detector import detect_objects, get_centroid, filter_detections
from src.gsd_calculator import calculate_gsd, pixel_to_real_distance, euclidean_distance
from src.tracker import track_centroids, detect_behavior_change
import os

st.title("MantaGuard: GSD and Behavior Analysis App")

# Inputs
video_file = st.file_uploader("Upload Drone Video", type=['mp4', 'avi', 'mov'])
altitude = st.number_input("Altitude (m)", min_value=0.1, value=50.0)
sensor_size = st.number_input("Sensor Size (mm)", min_value=0.1, value=6.17)
focal_length = st.number_input("Focal Length (mm)", min_value=0.1, value=24.0)
confidence = st.slider("Detection Confidence", 0.1, 1.0, 0.4)
interval = st.slider("Frame Interval", 1, 100, 30)

if st.button("Analyze Video"):
    try:
        if video_file is None:
            st.error("Please upload a video.")
            st.stop()

        # Validate inputs
        if altitude <= 0 or sensor_size <= 0 or focal_length <= 0:
            st.error("All parameters must be positive numbers.")
            st.stop()

        # Save uploaded file temporarily
        temp_path = "data/temp_video.mp4"
        with open(temp_path, "wb") as f:
            f.write(video_file.getbuffer())

        # Extract metadata and frames
        metadata = extract_metadata(temp_path)
        if metadata['altitude'] is None:
            metadata['altitude'] = altitude
        if metadata['sensor_size'] is None:
            metadata['sensor_size'] = sensor_size
        if metadata['focal_length'] is None:
            metadata['focal_length'] = focal_length

        width, height, fps = get_video_info(temp_path)
        gsd = calculate_gsd(metadata['altitude'], metadata['sensor_size'], metadata['focal_length'], height)

        frames = extract_frames(temp_path, interval)
        if not frames:
            st.error("No frames extracted from video.")
            st.stop()
        st.write(f"Extracted {len(frames)} frames.")

        # Process detections
        detections_per_frame = []
        for timestamp, frame in frames:
            dets = detect_objects(frame, confidence=int(confidence*100))
            filtered = filter_detections(dets, min_conf=confidence)
            detections_per_frame.append(filtered)

        # Track
        tracks = track_centroids(detections_per_frame)

        # Separate boats and mantas
        boat_tracks = [t for t in tracks if t['class'] == 'boat']
        manta_tracks = [t for t in tracks if t['class'] == 'manta']

        if not boat_tracks:
            st.warning("No boat detected in the video.")
            boat_centroids = []  # No boats
        else:
            boat_centroids = [t['positions'][-1][1] for t in boat_tracks]  # Latest positions

        if not manta_tracks:
            st.warning("No manta rays detected.")
            st.stop()

        # Behavior changes
        changes = detect_behavior_change(manta_tracks, flag_duration_frames=int(5 * fps / interval))

        # Prepare output data
        output_data = []
        for change in changes:
            track_id, frame_id, _, _ = change
            manta_track = next((t for t in manta_tracks if t['id'] == track_id), None)
            if manta_track:
                manta_pos = dict(manta_track['positions']).get(frame_id)
                if manta_pos:
                    for boat_pos in boat_centroids:
                        dist = pixel_to_real_distance(euclidean_distance(manta_pos, boat_pos), gsd)
                        output_data.append({
                            'frame_id': frame_id,
                            'timestamp': frames[frame_id][0],
                            'manta_id': track_id,
                            'distance_m': dist,
                            'behavior_change': True
                        })

        # Save output file
        if output_data:
            df = pd.DataFrame(output_data)
            output_path = "output/behavior_changes.csv"
            df.to_csv(output_path, index=False)
            st.success(f"Output saved to {output_path}")
            st.dataframe(df)
        else:
            st.info("No behavior changes detected.")

        # Create annotated video
        annotated_frames = []
        change_flags = {}
        for c in changes:
            track_id, start, _, end = c
            for f in range(start, end + 1):
                change_flags[(track_id, f)] = True
        for frame_id, (timestamp, frame) in enumerate(frames):
            dets = detections_per_frame[frame_id]
            for det in dets:
                x, y, w, h = det['x'] - det['width']/2, det['y'] - det['height']/2, det['width'], det['height']
                color = (0, 255, 0)  # Green
                track_id = det.get('track_id')
                if det['class'] == 'manta' and track_id is not None and (track_id, frame_id) in change_flags:
                    color = (0, 0, 255)  # Red for change
                cv2.rectangle(frame, (int(x), int(y)), (int(x+w), int(y+h)), color, 2)
                cv2.putText(frame, f"{det['class']} {det['confidence']:.2f}", (int(x), int(y-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Draw lines between mantas and boats
            for det in dets:
                if det['class'] == 'manta':
                    m_cent = get_centroid(det)
                    for b_cent in boat_centroids:
                        cv2.line(frame, m_cent, b_cent, (255, 0, 0), 2)
                        dist = pixel_to_real_distance(euclidean_distance(m_cent, b_cent), gsd)
                        mid = ((m_cent[0] + b_cent[0])//2, (m_cent[1] + b_cent[1])//2)
                        cv2.putText(frame, f"{dist:.2f}m", mid, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            annotated_frames.append(frame)

        # Save annotated video
        out_path = "output/annotated_video.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(out_path, fourcc, fps/interval, (width, height))
        for f in annotated_frames:
            out.write(f)
        out.release()
        st.success(f"Annotated video saved to {out_path}")

        # Display sample frame
        st.image(annotated_frames[0], caption="Sample Annotated Frame", use_column_width=True)

        # Cleanup
        os.remove(temp_path)
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        # Cleanup on error
        if os.path.exists("data/temp_video.mp4"):
            os.remove("data/temp_video.mp4")