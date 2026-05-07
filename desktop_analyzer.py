"""
MantaGuard Desktop Analyzer - Real-time video analysis with live visualization
"""
import cv2
import numpy as np
import pandas as pd
import os
from pathlib import Path
from src.video_processor import extract_metadata, get_video_info
from src.detector import detect_objects, get_centroid, filter_detections
from src.gsd_calculator import calculate_gsd, pixel_to_real_distance, euclidean_distance
from src.tracker import track_centroids, detect_behavior_change


class DesktopAnalyzer:
    def __init__(self, video_path, altitude=None, sensor_size=None, focal_length=None):
        """Initialize the analyzer with video path and optional metadata."""
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        # Video properties
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Extract metadata
        metadata = extract_metadata(video_path)
        self.altitude = altitude if altitude else metadata['altitude']
        self.sensor_size = sensor_size if sensor_size else metadata['sensor_size']
        self.focal_length = focal_length if focal_length else metadata['focal_length']
        
        # Set defaults if still None
        if self.altitude is None:
            self.altitude = 50.0
        if self.sensor_size is None:
            self.sensor_size = 6.17
        if self.focal_length is None:
            self.focal_length = 24.0
        
        # Calculate GSD
        self.gsd = calculate_gsd(self.altitude, self.sensor_size, self.focal_length, self.frame_height)
        
        # State
        self.current_frame_idx = 0
        self.confidence_threshold = 0.4
        self.paused = False
        self.all_frames_cache = {}
        self.all_detections = {}
        self.tracks = None
        self.output_data = []
        
        print(f"Video loaded: {self.total_frames} frames @ {self.fps} FPS")
        print(f"Resolution: {self.frame_width}x{self.frame_height}")
        print(f"Altitude: {self.altitude}m, Sensor: {self.sensor_size}mm, Focal: {self.focal_length}mm")
        print(f"GSD: {self.gsd:.4f} m/pixel")
    
    def process_all_frames(self):
        """Process all frames for detection and tracking."""
        print("\nProcessing all frames for detection...")
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame_idx = 0
        detections_over_frames = []
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            self.all_frames_cache[frame_idx] = frame.copy()
            
            # Detect objects
            dets = detect_objects(frame, confidence=int(self.confidence_threshold * 100))
            filtered = filter_detections(dets, min_conf=self.confidence_threshold)
            self.all_detections[frame_idx] = filtered
            detections_over_frames.append(filtered)
            
            frame_idx += 1
            if frame_idx % 10 == 0:
                print(f"  Processed {frame_idx}/{self.total_frames} frames...")
        
        # Track centroids across frames
        print("Tracking objects across frames...")
        self.tracks = track_centroids(detections_over_frames)
        print(f"Found {len(self.tracks)} tracks")
        
        # Detect behavior changes
        print("Detecting behavior changes...")
        changes = detect_behavior_change(self.tracks, flag_duration_frames=int(5 * self.fps / 30))
        
        # Build output data
        boat_tracks = [t for t in self.tracks if t['class'] == 'boat']
        manta_tracks = [t for t in self.tracks if t['class'] == 'manta']
        
        if boat_tracks:
            for change in changes:
                track_id, frame_id, old_angle, new_angle = change
                manta_track = next((t for t in manta_tracks if t['id'] == track_id), None)
                if manta_track and frame_id in dict(manta_track['positions']):
                    manta_pos = dict(manta_track['positions'])[frame_id]
                    
                    # Calculate distance to each boat
                    for boat_track in boat_tracks:
                        boat_pos = dict(boat_track['positions']).get(frame_id)
                        if boat_pos:
                            pixel_dist = euclidean_distance(manta_pos, boat_pos)
                            real_dist = pixel_to_real_distance(pixel_dist, self.gsd)
                            
                            self.output_data.append({
                                'frame_id': frame_id,
                                'timestamp_s': frame_id / self.fps,
                                'manta_id': track_id,
                                'boat_id': boat_track['id'],
                                'pixel_distance': pixel_dist,
                                'real_distance_m': real_dist,
                                'angle_change_deg': new_angle - old_angle if old_angle else 0,
                                'behavior_change': True
                            })
        
        print(f"Extracted {len(self.output_data)} behavior change events")
    
    def draw_frame(self, frame, frame_idx):
        """Draw detections, tracks, and GSD info on frame."""
        display_frame = frame.copy()
        
        # Draw detections for current frame
        if frame_idx in self.all_detections:
            for det in self.all_detections[frame_idx]:
                x, y, w, h = det['x'], det['y'], det['width'], det['height']
                x1, y1 = int(x - w/2), int(y - h/2)
                x2, y2 = int(x + w/2), int(y + h/2)
                
                # Color based on class
                color = (0, 255, 0) if det['class'] == 'manta' else (255, 0, 0)
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                
                # Label
                label = f"{det['class']} {det['confidence']:.2f}"
                cv2.putText(display_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw GSD and metadata overlay
        overlay_y = 30
        cv2.putText(display_frame, f"Frame: {frame_idx}/{self.total_frames-1}", (10, overlay_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        overlay_y += 25
        cv2.putText(display_frame, f"Time: {frame_idx/self.fps:.2f}s", (10, overlay_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        overlay_y += 25
        cv2.putText(display_frame, f"Confidence: {self.confidence_threshold:.2f} (use +/- to adjust)", (10, overlay_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        overlay_y += 25
        cv2.putText(display_frame, f"GSD: {self.gsd:.4f} m/pixel | Alt: {self.altitude}m", (10, overlay_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        overlay_y += 25
        cv2.putText(display_frame, f"Sensor: {self.sensor_size}mm | Focal: {self.focal_length}mm", (10, overlay_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Status
        status = "PAUSED" if self.paused else "PLAYING"
        cv2.putText(display_frame, status, (10, self.frame_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Instructions
        cv2.putText(display_frame, "SPACE: Pause/Play | </>: Frame Step | +/-: Confidence | Q: Quit | S: Save", (10, self.frame_height - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        return display_frame
    
    def run(self):
        """Main interactive loop."""
        print("\n" + "="*60)
        print("MantaGuard Desktop Analyzer - Interactive Mode")
        print("="*60)
        print("\nControls:")
        print("  SPACE  - Pause/Play")
        print("  <      - Previous frame")
        print("  >      - Next frame")
        print("  +      - Increase confidence threshold")
        print("  -      - Decrease confidence threshold")
        print("  P      - Start full processing (detection, tracking, analysis)")
        print("  S      - Save results to CSV")
        print("  Q      - Quit")
        print("\nStarting playback... (Press P to analyze entire video)")
        print("-"*60 + "\n")
        
        while True:
            # Read frame
            if not self.paused or self.current_frame_idx not in self.all_frames_cache:
                ret, frame = self.cap.read()
                if not ret:
                    # Loop back to start
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self.current_frame_idx = 0
                    ret, frame = self.cap.read()
                
                if ret:
                    self.all_frames_cache[self.current_frame_idx] = frame.copy()
                    
                    # Detect on current frame if not cached
                    if self.current_frame_idx not in self.all_detections:
                        dets = detect_objects(frame, confidence=int(self.confidence_threshold * 100))
                        filtered = filter_detections(dets, min_conf=self.confidence_threshold)
                        self.all_detections[self.current_frame_idx] = filtered
            else:
                frame = self.all_frames_cache[self.current_frame_idx]
            
            # Draw frame
            display_frame = self.draw_frame(frame, self.current_frame_idx)
            
            # Resize if too large
            if display_frame.shape[1] > 1920:
                scale = 1920 / display_frame.shape[1]
                display_frame = cv2.resize(display_frame, (1920, int(display_frame.shape[0] * scale)))
            
            cv2.imshow("MantaGuard Desktop Analyzer", display_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(30 if not self.paused else 0) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord(' '):
                self.paused = not self.paused
                print(f"{'Paused' if self.paused else 'Resumed'}")
            elif key == ord('<') or key == ord(','):
                self.current_frame_idx = max(0, self.current_frame_idx - 1)
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx)
                print(f"Frame: {self.current_frame_idx}")
            elif key == ord('>') or key == ord('.'):
                self.current_frame_idx = min(self.total_frames - 1, self.current_frame_idx + 1)
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx)
                print(f"Frame: {self.current_frame_idx}")
            elif key == ord('+') or key == ord('='):
                self.confidence_threshold = min(1.0, self.confidence_threshold + 0.05)
                print(f"Confidence: {self.confidence_threshold:.2f}")
            elif key == ord('-') or key == ord('_'):
                self.confidence_threshold = max(0.1, self.confidence_threshold - 0.05)
                print(f"Confidence: {self.confidence_threshold:.2f}")
            elif key == ord('p') or key == ord('P'):
                self.process_all_frames()
            elif key == ord('s') or key == ord('S'):
                self.save_results()
            else:
                if not self.paused:
                    self.current_frame_idx += 1
                    if self.current_frame_idx >= self.total_frames:
                        self.current_frame_idx = 0
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        cv2.destroyAllWindows()
        self.cap.release()
        print("\nAnalyzer closed.")
    
    def save_results(self):
        """Save results to CSV."""
        if not self.output_data:
            print("No results to save. Run full analysis first (press P).")
            return
        
        df = pd.DataFrame(self.output_data)
        output_path = Path(self.video_path).stem + "_analysis.csv"
        df.to_csv(output_path, index=False)
        print(f"Results saved to: {output_path}")
        print(f"  - {len(df)} events analyzed")
        print(f"  - Columns: {', '.join(df.columns)}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python desktop_analyzer.py <video_path> [altitude] [sensor_size] [focal_length]")
        print("Example: python desktop_analyzer.py video.mp4 50 6.17 24")
        sys.exit(1)
    
    video_path = sys.argv[1]
    altitude = float(sys.argv[2]) if len(sys.argv) > 2 else None
    sensor_size = float(sys.argv[3]) if len(sys.argv) > 3 else None
    focal_length = float(sys.argv[4]) if len(sys.argv) > 4 else None
    
    analyzer = DesktopAnalyzer(video_path, altitude, sensor_size, focal_length)
    analyzer.run()
