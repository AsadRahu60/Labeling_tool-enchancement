# Import the necessary libraries at the top of app.py
import os
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from ultralytics import YOLO
import supervision as sv
import json
from PyQt5 import QtCore,QtGui,QtWidgets
from PyQt5.QtGui import QBrush ,QColor
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import *
from qtpy.QtCore import Qt

class VideoAnnotationModule:
    def __init__(self, main_window):
        """
        Initialize Video Annotation Module integrated with Labelme
        
        Args:
            main_window: Reference to the main Labelme application window
        """
        self.main_window = main_window
        
        # Initialize detection model (use a pre-trained YOLO model)
        self.detection_model = YOLO('yolov8n.pt')
        
        # Tracking configuration
        self.tracker = sv.ByteTrack()
        
        # Create UI elements for video annotation
        self._create_video_annotation_ui()
    
    def _create_video_annotation_ui(self):
        """
        Create UI elements for video annotation in the main window
        """
        # Add Video Annotation menu or button to existing Labelme UI
        self.video_annotate_action = QAction('Annotate Video', self.main_window)
        self.video_annotate_action.triggered.connect(self.start_video_annotation)
        
        # Add to existing menu (modify based on your current UI structure)
        self.main_window.menu.addAction(self.video_annotate_action)
    
    def start_video_annotation(self):
        """
        Main method to start video annotation process
        """
        # Open file dialog to select video
        video_path, _ = QFileDialog.getOpenFileName(
            self.main_window, 
            'Select Video for Annotation', 
            '', 
            'Video Files (*.mp4 *.avi *.mov)'
        )
        
        if not video_path:
            return
        
        # Show progress dialog
        progress_dialog = QProgressDialog(
            "Annotating Video...", 
            "Cancel", 0, 100, 
            self.main_window
        )
        progress_dialog.setWindowModality(Qt.WindowModal)
        progress_dialog.show()
        
        try:
            # Process video and generate annotations
            annotations = self.process_video(video_path, progress_dialog)
            
            # Save annotations
            self.save_video_annotations(video_path, annotations)
            
            # Notify user
            QMessageBox.information(
                self.main_window, 
                'Annotation Complete', 
                'Video annotation has been completed successfully!'
            )
        
        except Exception as e:
            QMessageBox.critical(
                self.main_window, 
                'Annotation Error', 
                f'An error occurred during video annotation: {str(e)}'
            )
        finally:
            progress_dialog.close()
    
    def process_video(self, video_path, progress_dialog=None):
        """
        Process entire video for automatic annotation
        
        Args:
            video_path (str): Path to input video
            progress_dialog (QProgressDialog): Optional progress dialog
        
        Returns:
            dict: Annotations for each frame
        """
        # Video Capture
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Annotation Storage
        video_annotations = {}
        
        # Frame Processing
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Update progress
            if progress_dialog:
                progress = int((frame_count / total_frames) * 100)
                progress_dialog.setValue(progress)
                QApplication.processEvents()
            
            # Detect and Track Objects
            detections = self._detect_objects(frame)
            
            # Process Detections
            frame_annotations = self._process_detections(frame, detections, frame_count)
            
            # Store Annotations
            video_annotations[frame_count] = frame_annotations
            
            frame_count += 1
            
            # Allow cancellation
            if progress_dialog and progress_dialog.wasCanceled():
                break
        
        cap.release()
        
        return video_annotations
    
    def _detect_objects(self, frame, confidence_threshold=0.5):
        """
        Detect objects in a single frame
        
        Args:
            frame (np.ndarray): Input frame
            confidence_threshold (float): Minimum detection confidence
        
        Returns:
            Detection results
        """
        # YOLO Detection for persons
        results = self.detection_model(frame)[0]
        
        # Filter for person class (typically class 0 in COCO dataset)
        person_detections = results.boxes[
            (results.boxes.cls == 0) & (results.boxes.conf > confidence_threshold)
        ]
        
        return person_detections
    
    def _process_detections(self, frame, detections, frame_number):
        """
        Process and track detected objects
        
        Args:
            frame (np.ndarray): Current frame
            detections: Detected objects
            frame_number (int): Current frame number
        
        Returns:
            Frame annotations
        """
        # Convert detections to supervision format
        detection_list = []
        confidence_list = []
        
        for det in detections:
            x1, y1, x2, y2 = det.xyxy[0].cpu().numpy()
            detection_list.append([x1, y1, x2, y2])
            confidence_list.append(det.conf.cpu().numpy()[0])
        
        # Create Detections object for tracking
        detections = sv.Detections(
            xyxy=np.array(detection_list),
            confidence=np.array(confidence_list)
        )
        
        # Apply Tracking
        tracked_detections = self.tracker.update(detections)
        
        # Annotation Storage
        frame_annotations = []
        
        # Process Tracked Detections
        for detection in tracked_detections:
            x1, y1, x2, y2 = detection.xyxy
            track_id = detection.track_id
            confidence = detection.confidence[0]
            
            # Prepare Annotation in Labelme format
            annotation = {
                "label": f"person_{track_id}",
                "points": [
                    [float(x1), float(y1)],
                    [float(x2), float(y2)]
                ],
                "group_id": None,
                "shape_type": "rectangle",
                "flags": {
                    "track_id": int(track_id),
                    "confidence": float(confidence)
                }
            }
            
            frame_annotations.append(annotation)
        
        return frame_annotations
    
    def save_video_annotations(self, video_path, annotations):
        """
        Save annotations for each frame
        
        Args:
            video_path (str): Source video path
            annotations (dict): Detected annotations
        """
        # Create output directory
        output_dir = os.path.join(
            os.path.dirname(video_path), 
            'video_annotations'
        )
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract video frames
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Generate frame annotation
            if frame_count in annotations:
                # Save frame image
                frame_path = os.path.join(
                    output_dir, 
                    f'frame_{frame_count:04d}.jpg'
                )
                cv2.imwrite(frame_path, frame)
                
                # Save annotation JSON
                annotation_path = os.path.join(
                    output_dir, 
                    f'frame_{frame_count:04d}.json'
                )
                
                annotation_data = {
                    "version": "5.1.1",
                    "flags": {},
                    "shapes": annotations[frame_count],
                    "imagePath": os.path.basename(frame_path),
                    "imageData": None,
                    "imageHeight": frame.shape[0],
                    "imageWidth": frame.shape[1]
                }
                
                with open(annotation_path, 'w') as f:
                    json.dump(annotation_data, f, indent=4)
            
            frame_count += 1
        
        cap.release()

# Integration Method in main Labelme App Class
def setup_video_annotation(self):
    """
    Method to set up video annotation module in Labelme
    
    Add this method to your main Labelme application class
    """
    self.video_annotation_module = VideoAnnotationModule(self)

# Modification in __init__ method of main app class
def __init__(self, ...):
    # Existing initialization code
    ...
    
    # Add video annotation setup
    self.setup_video_annotation()
