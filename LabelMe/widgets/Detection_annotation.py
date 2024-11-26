# detection_annotation.py

import cv2
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO
import torch
from torchvision import transforms
from torchreid import models
import random


class PersonDetectionApp:
    def __init__(self, yolo_model, deepsort, reid_model):
        self.yolo_model = yolo_model
        self.deepsort = deepsort
        self.reid_model = reid_model  # ReID model for extracting features

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

        return boxes, confidences, features

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
            feature = feature.flatten()
            features.append(feature)

        return features

    def track_person(self, boxes, features, confidences):
        """Run DeepSORT tracking on the detected persons."""
        if len(boxes) > 0 and len(features) > 0:
            try:
                # Run DeepSORT
                tracks = self.deepsort.update_tracks(
                    raw_detections=list(zip(boxes, confidences)),
                    embeds=features
                )
                # Extract track IDs from the returned tracks
                ids = [track.track_id for track in tracks]

                return tracks, ids

            except AssertionError as e:
                print(f"Assertion error in DeepSORT: {e}")
            except TypeError as e:
                print(f"Type error in DeepSORT: {e}")

        return [], []

    def annotate_batch(self, frames, batch_size=2):
        """Process a batch of frames for annotation with YOLO, ReID, and DeepSORT."""
        all_boxes = []
        all_features = []
        all_ids = []
        all_confidences = []
        print(f"Batch size received: {batch_size}")
        # Loop through the frames in batches
        if batch_size <= 0:
            raise ValueError(" Batch size must be greator than zero")
        
        
        for idx in range(0, len(frames), batch_size):
            batch_frames = frames[idx: idx + batch_size]

            for frame in batch_frames:
                # Detect persons in the frame
                boxes, confidences, features = self.detect_persons(frame)

                # Track the detected persons
                tracks, ids = self.track_person(boxes, features, confidences)

                all_boxes.append(boxes)
                all_features.append(features)
                all_ids.append(ids)
                all_confidences.append(confidences)

                # Annotate the frame with detection and tracking info
                for track in tracks:
                    track_id = track.track_id
                    bbox = track.to_tlbr()
                    x1, y1, x2, y2 = map(int, bbox)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        return all_boxes, all_features, all_ids, all_confidences


