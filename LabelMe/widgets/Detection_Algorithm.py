"""In this py file we will add the detection algorithms which the user want to select in order to have the optimal 
results."""

import cv2
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO
import torch
from torchvision import transforms
from torchreid import models

class PersonDetection:
    
    def __init__(self):
        
            # Assuming you have the YOLO model and DeepSORT instance
        self.yolo_model = YOLO("yolov8n.pt")
        self.deepsort = DeepSort(
                max_age=30,        # Number of frames to keep tracks alive without detections
                nn_budget=100,     # Maximum number of features for the tracker
                max_iou_distance=0.4,  # Maximum IOU distance for matching
                n_init=3           # Number of consecutive detections before the track is considered valid
            )
        self.reid_model= models.build_model(
                name='osnet_x1_0',    # Use 'osnet_x1_0' for a balanced model size and performance
                num_classes=1000,     # Dummy number for classes (not used during feature extraction)
                pretrained=True,
                loss='softmax'# Load pre-trained weights
        )
        
        self.reid_model.eval()
        if torch.cuda.is_available():
            self.reid_model = self.reid_model.cuda()
        
        
        

        # Define transformation for ReID model input
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def detect_persons(self, frame):
        # Run YOLO detection on the frame
        results = self.yolo_model(frame)

        boxes = []
        confidences = []

        # Extract bounding boxes and confidence values
        for result in results:
            if hasattr(result, 'boxes'):
                for idx, cls in enumerate(result.boxes.cls):
                    if int(cls) == 0:  # Class ID 0 is for 'person'
                        x1, y1, x2, y2 = result.boxes.xyxy[idx].cpu().numpy().astype(int)
                        confidence = float(result.boxes.conf[idx].cpu().numpy())

                        # Convert bounding box to (x, y, width, height)
                        width, height = x2 - x1, y2 - y1
                        bbox_xywh = [x1, y1, width, height]
                        boxes.append(bbox_xywh)
                        confidences.append(confidence)

        # Extract ReID features for all detected boxes
        features = self.extract_reid_features(frame, boxes) if boxes else []
        return boxes, confidence, features  

    def track_persons(self,frame,boxes,features,confidences):
        
            # Ensure confidences is always a list
        if isinstance(confidences, float):
            confidences = [confidences]
        
        if len(boxes) > 0 and len(features) > 0:
            try:
                # Run DeepSORT
                tracks = self.deepsort.update_tracks(
                    raw_detections=list(zip(boxes, confidences)),
                    embeds=features
                )
                # Extract track IDs from the returned tracks
                ids = [track.track_id for track in tracks]

                # Draw bounding boxes and IDs on the frame
                for track in tracks:
                    track_id = track.track_id
                    bbox = track.to_tlbr()
                    x1, y1, x2, y2 = map(int, bbox)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            except AssertionError as e:
                print(f"Assertion error in DeepSORT: {e}")
                tracks=[]
            except TypeError as e:
                print(f"Type error in DeepSORT: {e}")
                tracks=[]    
        else:
              print("No valid bounding boxes or features for tracking.")
              tracks = []
        return frame, tracks



    def extract_reid_features(self, frame, boxes):
        """Extract features for person re-identification using the ReID model."""
        features = []

        for box in boxes:
            x, y, width, height = box
            # Crop the person image from the frame
            person_img = frame[y:y + height, x:x + width]

            if person_img.size == 0:
                continue  # Skip if the crop is invalid
            person_img = cv2.resize(person_img, (128, 256))

            # Transform the cropped image for ReID model input
            person_tensor = self.transform(person_img).unsqueeze(0)
            person_tensor = person_tensor.cuda() if torch.cuda.is_available() else person_tensor

            # Switch the ReID model to evaluation mode to avoid BatchNorm errors
            self.reid_model.eval()  # Ensure model is in evaluation mode

            # Extract features using the ReID model
            with torch.no_grad():
                feature = self.reid_model(person_tensor).cpu().numpy()
            feature= feature.flatten()
            features.append(feature)

        return features

  
# Use the following instance to call detection-related functions in other files
person_detection_instance = PersonDetection()
