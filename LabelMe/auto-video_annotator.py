import os
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from ultralytics import YOLO
import supervision as sv
from sklearn.preprocessing import LabelEncoder
import json
import fastreid
print(fastreid.__file__)

from fastreid.config import get_cfg
from fastreid.engine import DefaultPredictor

class AutomaticVideoAnnotator:
    def __init__(self, 
                 detection_model='yolov8m.pt', 
                 reid_model=None, 
                 output_dir='./annotations'):
        """
        Initialize automatic video annotation pipeline
        
        Args:
            detection_model (str): Path to YOLO detection model
            reid_model (str, optional): Path to ReID model
            output_dir (str): Directory to save annotations
        """
        # Object Detection Model
        self.detection_model = YOLO(detection_model)
        
        # ReID Model (Optional)
        self.reid_model = None
        if reid_model:
           # Load FastReID model
                cfg = get_cfg()
                cfg.merge_from_file("A:/data/Project-Skills/Labeling_tool-enchancement/labelme/fastreid/fast-reid\configs/Market1501/bagtricks_R50.yml")
                cfg.MODEL.WEIGHTS = "A:/data/Project-Skills/Labeling_tool-enchancement/labelme/market_bot_R50.pth"  # Path to trained FastReID weights
                cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
                self.fastreid_model = DefaultPredictor(cfg)
                test_img = torch.randn(1, 3, 128, 256)  # Replace with appropriate input size
                reid_model= self.fastreid_model(test_img)
                print(reid_model.shape)  # Should output (1, expected_embedding_size)
        
        # Tracking Configuration
        self.tracker = sv.ByteTrack()
        
        # Output Configuration
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Label Encoding for consistent IDs
        self.label_encoder = LabelEncoder()
    
    def process_video(self, video_path, confidence_threshold=0.5):
        """
        Process entire video for automatic annotation
        
        Args:
            video_path (str): Path to input video
            confidence_threshold (float): Minimum detection confidence
        
        Returns:
            dict: Annotations for each frame
        """
        # Video Capture
        cap = cv2.VideoCapture(video_path)
        
        # Annotation Storage
        video_annotations = {}
        
        # Frame Processing
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect and Track Objects
            detections = self._detect_objects(frame, confidence_threshold)
            
            # Process Detections
            annotated_frame, frame_annotations = self._process_detections(frame, detections, frame_count)
            
            # Store Annotations
            video_annotations[frame_count] = frame_annotations
            
            frame_count += 1
        
        cap.release()
        
        # Save Annotations
        self._save_annotations(video_annotations, video_path)
        
        return video_annotations
    
    def _detect_objects(self, frame, confidence_threshold):
        """
        Detect objects in a single frame
        
        Args:
            frame (np.ndarray): Input frame
            confidence_threshold (float): Minimum detection confidence
        
        Returns:
            Detection results
        """
        # YOLO Detection
        results = self.detection_model(frame)[0]
        
        # Filter for person class (typically class 0 in COCO dataset)
        person_detections = results.boxes[results.boxes.cls == 0]
        
        return person_detections
    
    def _process_detections(self, frame, detections, frame_number):
        """
        Process and track detected objects
        
        Args:
            frame (np.ndarray): Current frame
            detections: Detected objects
            frame_number (int): Current frame number
        
        Returns:
            Annotated frame and frame annotations
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
        
        # Annotate Frame
        for detection in tracked_detections:
            x1, y1, x2, y2 = detection.xyxy
            track_id = detection.track_id
            confidence = detection.confidence[0]
            
            # Prepare Annotation
            annotation = {
                'frame': frame_number,
                'track_id': int(track_id),
                'bbox': {
                    'x1': float(x1),
                    'y1': float(y1),
                    'x2': float(x2),
                    'y2': float(y2)
                },
                'confidence': float(confidence)
            }
            
            frame_annotations.append(annotation)
        
        return frame, frame_annotations
    
    def _save_annotations(self, annotations, video_path):
        """
        Save annotations in Labelme compatible JSON format
        
        Args:
            annotations (dict): Detected annotations
            video_path (str): Source video path
        """
        output_filename = os.path.splitext(os.path.basename(video_path))[0] + '_annotations.json'
        output_path = os.path.join(self.output_dir, output_filename)
        
        with open(output_path, 'w') as f:
            json.dump(annotations, f, indent=4)
    
    def integrate_with_labelme(self, video_path):
        """
        Integration method for Labelme
        
        Args:
            video_path (str): Path to video for annotation
        
        Returns:
            Annotations compatible with Labelme
        """
        # Process Video
        annotations = self.process_video(video_path)
        
        # Convert to Labelme format
        labelme_annotations = self._convert_to_labelme_format(annotations)
        
        return labelme_annotations
    
    def _convert_to_labelme_format(self, annotations):
        """
        Convert annotations to Labelme-compatible format
        
        Args:
            annotations (dict): Detected annotations
        
        Returns:
            List of Labelme annotation dictionaries
        """
        labelme_annotations = []
        
        for frame_number, frame_annotations in annotations.items():
            for annotation in frame_annotations:
                labelme_annotation = {
                    "version": "5.1.1",
                    "flags": {},
                    "shapes": [
                        {
                            "label": f"person_{annotation['track_id']}",
                            "points": [
                                [annotation['bbox']['x1'], annotation['bbox']['y1']],
                                [annotation['bbox']['x2'], annotation['bbox']['y2']]
                            ],
                            "group_id": None,
                            "shape_type": "rectangle",
                            "flags": {
                                "confidence": annotation['confidence']
                            }
                        }
                    ],
                    "imagePath": f"frame_{frame_number}.jpg",
                    "imageData": None,
                    "imageHeight": None,
                    "imageWidth": None
                }
                labelme_annotations.append(labelme_annotation)
        
        return labelme_annotations

# Example Usage
def main():
     # Video path - replace with your video file
    video_path = '5198164-uhd_3840_2160_25fps.mp4'
    
    # Output directory for annotations
    output_dir = './labelme_annotations'
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize Annotator
    annotator = AutomaticVideoAnnotator(
        detection_model='yolov8n.pt',
        output_dir=output_dir
    )
    
    # Process Video
    annotations = annotator.integrate_with_labelme(video_path)
    
    # Extract Frames (Optional but recommended)
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Save each frame
        frame_path = os.path.join(output_dir, f'frame_{frame_count:04d}.jpg')
        cv2.imwrite(frame_path, frame)
        frame_count += 1
    cap.release()
    
    # Save Annotations
    for idx, annotation in enumerate(annotations):
        annotation_path = os.path.join(
            output_dir, 
            f'frame_{idx:04d}_annotation.json'
        )
        with open(annotation_path, 'w') as f:
            json.dump(annotation, f, indent=4)
    
    print(f"Annotations saved in {output_dir}")

if __name__ == '__main__':
    main()
