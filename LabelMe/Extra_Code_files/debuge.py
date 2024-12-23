from datetime import time
import random
import functools
import html
import math
import os
import os.path as osp
import re
import webbrowser
import sys
# print(sys.path)
sys.path.append('c:\\users\\aasad\\appdata\\local\\programs\\python\\python312\\lib\\site-packages')
PY3 = sys.version[0] == "3.12.4"

import imgviz
import natsort

from PyQt5 import QtCore,QtGui,QtWidgets
from PyQt5.QtGui import QBrush ,QColor
from PyQt5.QtCore import Qt ,pyqtSignal
from PyQt5.QtWidgets import *
from qtpy.QtCore import Qt


from labelme.ai import MODELS
from labelme.config import get_config
from labelme.label_file import LabelFile
from labelme.label_file import LabelFileError
from labelme.logger import logger
from labelme.shape import Shape
from labelme.widgets import BrightnessContrastDialog
from labelme.widgets import Canvas
from labelme.widgets import FileDialogPreview
from labelme.widgets import LabelDialog
from labelme.widgets import LabelListWidget
from labelme.widgets import LabelListWidgetItem
from labelme.widgets import ToolBar
from labelme.widgets import UniqueLabelQListWidget
from labelme.widgets import ZoomWidget

from labelme import utils
#from labelme.widgets.Detection_annotation import PersonDetectionApp
import cv2
import torch
import torchreid
from torchreid import models

import json
import numpy as np

from ultralytics import YOLO
from scipy.spatial.distance import euclidean
from deep_sort_realtime.deepsort_tracker import DeepSort
from deep_sort_realtime.deep_sort.nn_matching import NearestNeighborDistanceMetric
from deep_sort_realtime.deep_sort.detection import Detection
from deep_sort_realtime.deep_sort.tracker import Tracker

from shapely.geometry import box
from torchvision import transforms
import scipy.io as sio
from pathlib import Path
import h5py



# import torch.nn as nn
# from torchvision.models import resnet50, ResNet50_Weights
# yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

import fastreid
print(fastreid.__file__)

from fastreid.config import get_cfg
from fastreid.engine import DefaultPredictor
from sklearn.metrics.pairwise import cosine_similarity
from xml.etree.ElementTree import Element, SubElement, ElementTree

###############################################################################
import logging
from labelme.widgets.Dataset_Handler import CUHK03Handler, Market1501Handler


# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Set to DEBUG for more detailed logs
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),  # Logs to console
        logging.FileHandler("annotate_video.log"),  # Logs to file
    ]
)




def annotateVideo(self):
        """Annotate video using detection, ReID, and tracking."""
        try:
            # Ensure models are loaded
            if not self.detector or not self.reid_model:
                self.load_models()

            if not self.video_capture.isOpened():
                logger.warning("No video is loaded for annotation.")
                return

            logger.info("Starting video annotation...")
            
            # Initialize tracker
            if not hasattr(self, 'tracker') or self.tracker is None:
                success = self.initialize_tracker()
                if not success:
                    logger.error("Failed to initialize tracker")
                    return
            
            
            # Get video metadata
            total_frames = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = self.video_capture.get(cv2.CAP_PROP_FPS)
            
            logger.info(f"Video details: {total_frames} frames, {fps} FPS")
            
            # self.initialize_tracker() # Initialize DeepSORT tracker
            person_colors = {}
            all_annotations = []
            processed_frames = 0
            frames = []

            while self.video_capture.isOpened():
                ret, frame = self.video_capture.read()
                if not ret:
                    logger.info("End of video reached.")
                    break
                
                # Optional: Frame skip for performance
                if processed_frames % self.frame_skip != 0:
                    processed_frames += 1
                    continue
                # Store current frame for tracker
                self.current_frame = frame
                try:
                    # Step 1: Detect and Extract Features
                    frame_detections = self.process_image(frame)
                    
                    if frame_detections is not None:
                        
                        # Log the detections for debugging
                        logger.debug(f"Frame detections: {frame_detections}")

                        # Step 2: Update Tracker
                        self.update_tracker(frame_detections)

                        # Step 3: Process Tracks for Final Annotations
                        frame_annotations = self.process_tracks(frame, person_colors)

                        # Step 4: Draw Shapes and Update Canvas
                        self.canvas.drawShapesForFrame(frame_annotations)
                        self.canvas.update()
                        self.repaint()

                        if frame_annotations:
                            all_annotations.extend(frame_annotations)
                        frames.append(frame)
                    
                     # Update progress
                    processed_frames += 1
                    progress = int((processed_frames / total_frames) * 100)
                    self.progress_callback.emit(progress)
                    
                except Exception as frame_error:
                    logger.error(f"Error processing frame {processed_frames}: {frame_error}")
                    continue
                
                # Optional: Allow cancellation
                if self.is_cancelled:
                    logger.info("Video annotation cancelled by user.")
                    break
            
            if all_annotations:
                format_choice = self.choose_annotation_format()
                if format_choice:
                    self.save_reid_annotations(frames, all_annotations, format_choice)
                    self.actions.saveReIDAnnotations.setEnabled(True)
                    logger.info(f"Annotations saved successfully in {format_choice} format.")
                else:
                    logger.warning("Annotation saving canceled by user.")
            
            
            logger.info(f"Video annotation completed. Processed {processed_frames} frames.")
            self.progress_callback.emit(100)

        except Exception as e:
            logger.error(f"Critical error during video annotation: {e}", exc_info=True)
            self.progress_callback.emit(-1) 


        #############################################################################################################
    def run_yolo(self, frame):
        try:
            results = self.yolo_model(frame)
            boxes, confidences = [], []
            confidence_threshold = 0.4  # Adjusted threshold

            for det in results[0].boxes:
                if int(det.cls[0]) == 0 and float(det.conf[0]) > confidence_threshold:
                    x1, y1, x2, y2 = map(int, det.xyxy[0].tolist())
                    conf = float(det.conf[0])
                    if 0 <= x1 < x2 <= frame.shape[1] and 0 <= y1 < y2 <= frame.shape[0]:
                        boxes.append((x1, y1, x2, y2))
                        confidences.append(conf)
                    else:
                        logger.warning(f"Skipping invalid YOLO box: {[x1, y1, x2, y2]}")
                else:
                    logger.debug(f"Skipping non-person or low-confidence detection: Class={int(det.cls[0])}, Confidence={det.conf[0]}")

            logger.info(f"YOLO detected {len(boxes)} valid boxes.")
            return boxes, confidences

        except Exception as e:
            logger.error(f"Error during YOLO detection: {e}", exc_info=True)
            return [], []


    
    ####################################################################################################################################
    def load_fastreid_model(self):
        """
        Load the FastReID model with the specified configuration.
        """
        try:
            from fastreid.config import get_cfg
            from fastreid.engine.defaults import DefaultPredictor

            # Load FastReID model configuration
            cfg = get_cfg()
            cfg.merge_from_file("A:/data/Project-Skills/Labeling_tool-enchancement/labelme/fastreid/fast-reid/configs/Market1501/bagtricks_R50.yml")
            cfg.MODEL.WEIGHTS = "A:/data/Project-Skills/Labeling_tool-enchancement/labelme/market_bot_R50.pth"  # Path to trained FastReID weights
            cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

            # Initialize the FastReID model
            self.reid_model = DefaultPredictor(cfg)

            # Test the model with dummy data to ensure it's working
            test_img = torch.randn(1, 3, 128, 256).to(cfg.MODEL.DEVICE)  # Replace with actual input size
            feature = self.reid_model(test_img)
            logger.info(f"FastReID model loaded successfully. Test output shape: {feature.shape}")

        except Exception as e:
            logger.error(f"Error loading FastReID model: {e}", exc_info=True)
            raise RuntimeError("Failed to load the FastReID model.")


    ###############################################################################################################################
    def extract_reid_features(self, frame, bbox_xywh):
        """
        Extract ReID features for detected bounding boxes.

        Args:
            frame (np.ndarray): Current video frame (H, W, C).
            boxes (list of tuples): Detected bounding boxes [(x1, y1, x2, y2), ...].

        Returns:
            list: ReID feature vectors for each bounding box or None for invalid boxes.
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        """Extract ReID features for detected persons."""
        features = []
        
        try:
            if bbox_xywh is None or len(bbox_xywh) == 0:
                return []

            crops = []
            valid_boxes = []
            
            for box in bbox_xywh:
                x_center, y_center, w, h = box
                x1 = max(0, int(x_center - w/2))
                x2 = min(frame.shape[1], int(x_center + w/2))
                y1 = max(0, int(y_center - h/2))
                y2 = min(frame.shape[0], int(y_center + h/2))
                
                if x1 >= x2 or y1 >= y2:
                    continue
                    
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                    
                crop = cv2.resize(crop, (128, 256))
                crops.append(crop)
                valid_boxes.append(box)
                
            if not crops:
                return []
                
            # Convert to batch
            batch = np.stack(crops)
            batch = torch.from_numpy(batch).float()
            batch = batch.permute(0, 3, 1, 2)  # BHWC to BCHW
            
            # Extract features
            with torch.no_grad():
                if hasattr(self.reid_model, 'cuda') and torch.cuda.is_available():
                    batch = batch.cuda()
                    self.reid_model = self.reid_model.cuda()
                    
                features = self.reid_model(batch)
                if isinstance(features, dict):
                    features = features['features']
                features = features.cpu().numpy()
                
            logger.info(f"Extracted features for {len(crops)} bounding boxes.")
            return features
            
        except Exception as e:
            logger.error(f"Error extracting ReID features: {e}")
            return []
    ##########################################################################################
    def _preprocess_crop(self, frame, x1, y1, x2, y2, device):
        """
        Preprocess the cropped region for ReID model input using PyTorch transforms.
        """
        try:
            # Ensure bounding box coordinates are integers
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
            

            # Validate bounding box dimensions
            if x2 <= x1 or y2 <= y1:
                logger.warning(f"Invalid bounding box: ({x1}, {y1}, {x2}, {y2})")
                return None

            # Crop the bounding box
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                logger.warning(f"Empty crop for bbox: ({x1}, {y1}, {x2}, {y2})")
                return None

            # Preprocess the crop
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((256, 128)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            crop_tensor = transform(crop).unsqueeze(0)  # Add batch dimension
            return crop_tensor.to(device)

        except Exception as e:
            logger.warning(f"Error preprocessing crop ({x1}, {y1}, {x2}, {y2}): {e}")
            return None










    #######################################################################################################################################
    def update_tracker(self, detections):
        """Update tracker with new detections."""
        try:
            if not detections:
                logger.debug("No detections to update tracker")
                return

            bbox_xywh = detections.get('bbox_xywh')
            confidences = detections.get('confidence')
            features = detections.get('features')

            if bbox_xywh is None or len(bbox_xywh) == 0:
                logger.debug("No valid bounding boxes for tracking")
                return

            logger.debug(f"Updating tracker with {len(bbox_xywh)} detections")
            
            # Update tracker
            self.tracking_outputs = self.tracker.update(
                bbox_xywh,
                confidences,
                features
            )
            
            logger.debug(f"Tracker updated. Number of tracks: {len(self.tracker.tracks)}")
            
        except Exception as e:
            logger.error(f"Error updating tracker: {e}")
            self.tracking_outputs = []


    ############################################################################################################################
    def validate_detections(self, boxes, confidences, features, frame_shape):
        """
        Validate and prepare detections for the tracker.
        """
        frame_height, frame_width, _ = frame_shape
        detections = []

        for box, confidence, feature in zip(boxes, confidences, features):
            if feature is not None and len(feature) == self.expected_embedding_size and np.isfinite(feature).all():
                bbox_tuple = (
                    box[0] / frame_width,
                    box[1] / frame_height,
                    box[2] / frame_width,
                    box[3] / frame_height,
                )
                detections.append(Detection(bbox_tuple, confidence, feature))
            else:
                logger.warning(f"Invalid detection or feature for box {box}.")

        logger.info(f"Validated {len(detections)} detections out of {len(boxes)}.")
        logger.debug(f"YOLO bounding boxes: {detections}")

        return detections

    ############################################################################################################################
    
    def validate_bbox(self, bbox, frame_shape):
        frame_height, frame_width = frame_shape[:2]
        x1, y1, x2, y2 = bbox
        
        # Additional validation checks
        if x1 < 0 or y1 < 0 or x2 > frame_width or y2 > frame_height:
            logger.warning(f"Bbox exceeds frame dimensions: {bbox}")
            logger.warning(f"Frame dimensions: {frame_width}x{frame_height}")
            
            # Clamp values to frame dimensions
            x1 = max(0, min(x1, frame_width))
            y1 = max(0, min(y1, frame_height))
            x2 = max(0, min(x2, frame_width))
            y2 = max(0, min(y2, frame_height))
            
            return [x1, y1, x2, y2]
        
        return bbox

    #########################################################################################################################
    def process_image(self, frame):
        """Process frame with detection and ReID."""
        try:
            # YOLOv8 detection
            detections = self.detector(frame)
            bbox_xywh = []
            confidences = []
            
            height, width = frame.shape[:2]
            
            # Process only person detections with proper scaling
            for det in detections[0].boxes.data:
                if det[5] == 0:  # person class
                    x1, y1, x2, y2 = det[0:4].cpu().numpy()
                    
                    # Scale coordinates to frame size
                    x1 = int((x1 / 640) * width)
                    x2 = int((x2 / 640) * width)
                    y1 = int((y1 / 384) * height)
                    y2 = int((y2 / 384) * height)
                    
                    # Convert to center format with width/height
                    w = x2 - x1
                    h = y2 - y1
                    x_center = x1 + w/2
                    y_center = y1 + h/2
                    
                    bbox_xywh.append([x_center, y_center, w, h])
                    confidences.append(float(det[4].cpu().numpy()))
            
            if not bbox_xywh:
                return None
                
            bbox_xywh = np.array(bbox_xywh, dtype=np.float32)
            confidences = np.array(confidences, dtype=np.float32)
            
            # Extract ReID features
            features = self.extract_reid_features(frame, bbox_xywh)
            
            return {
                'bbox_xywh': bbox_xywh,
                'confidence': confidences,
                'features': features
            }
            
        except Exception as e:
            logger.error(f"Error in process_image: {e}")
            return None
##########################################################################################################################################################################
    def scale_bbox(self, bbox, width, height):
        """
        Scale bounding box coordinates to pixel coordinates.
        
        Args:
            bbox (list): Bounding box coordinates 
            width (int): Frame width in pixels
            height (int): Frame height in pixels
        
        Returns:
            list: Scaled bounding box coordinates [x1, y1, x2, y2]
        """
        # Ensure input bbox is a list with 4 coordinates
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise ValueError(f"Invalid bbox format. Expected list of 4 coordinates, got {bbox}")
        
        # Unpack coordinates
        x1, y1, x2, y2 = bbox
        
        # Normalize coordinates if they exceed 1
        if x1 > 1 or y1 > 1 or x2 > 1 or y2 > 1:
            # Find the maximum coordinate to determine scaling factor
            max_coord = max(x1, y1, x2, y2)
            
            # Normalize based on the maximum coordinate
            x1_norm = x1 / max_coord
            y1_norm = y1 / max_coord
            x2_norm = x2 / max_coord
            y2_norm = y2 / max_coord
        else:
            # Already normalized
            x1_norm, y1_norm, x2_norm, y2_norm = bbox
        
        # Clamp normalized coordinates between 0 and 1
        x1_norm = max(0, min(x1_norm, 1))
        y1_norm = max(0, min(y1_norm, 1))
        x2_norm = max(0, min(x2_norm, 1))
        y2_norm = max(0, min(y2_norm, 1))
        
        # Scale to pixel coordinates
        x1 = int(x1_norm * width)
        y1 = int(y1_norm * height)
        x2 = int(x2_norm * width)
        y2 = int(y2_norm * height)
        
        # Validate scaled coordinates
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(width, x2)
        y2 = min(height, y2)
        
        # Ensure bbox has valid dimensions
        if x2 <= x1 or y2 <= y1:
            # If dimensions are invalid, return an empty or default bbox
            return [0, 0, 0, 0]
        
        return [x1, y1, x2, y2]



    ###########################################################################################################
    def process_tracks(self, frame, person_colors):
        """Process tracks from the tracker and annotate frame."""
        frame_annotations = []
        frame_shape = frame.shape
        logging.info(f"frame shape: {frame_shape}")
        height, width, _ = frame.shape

        # Track IDs we've processed this frame to maintain consistency
        used_ids = set()

        # Use the tracking outputs from our enhanced tracker
        if hasattr(self.tracker, 'tracks'):
            tracks_to_process = self.tracker.tracks
        else:
            tracks_to_process = []
            logging.warning("No tracks available in tracker")
            return frame_annotations

        for track in tracks_to_process:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue

            # Get raw bounding box
            raw_bbox = track.to_tlbr().tolist()
            logging.info(f"Track {track.track_id} - Raw bbox: {raw_bbox}")

            try:
                # Scale bbox to pixel coordinates with improved validation
                scaled_bbox = self.scale_bbox(raw_bbox, width, height)
                if scaled_bbox[2] <= scaled_bbox[0] or scaled_bbox[3] <= scaled_bbox[1]:
                    continue

                # Improved ID management - keep IDs between 1-4
                if track.track_id not in self.id_mapping:
                    new_id = 1
                    while new_id in used_ids and new_id <= 4:
                        new_id += 1
                    if new_id <= 4:  # Only assign if within expected range
                        self.id_mapping[track.track_id] = new_id
                    else:
                        continue  # Skip if we can't assign a valid ID

                fixed_id = self.id_mapping[track.track_id]
                used_ids.add(fixed_id)

                # Get confidence
                track_confidence = self.tracker.tracks_confidence.get(track.track_id, None)
                confidence = track_confidence.confidence if track_confidence else 1.0
                label = f"Person ID: {fixed_id} ({confidence:.2f})"

                # Ensure coordinates are within frame bounds
                x1, y1, x2, y2 = map(int, scaled_bbox)
                x1 = max(0, min(x1, width-1))
                x2 = max(x1+1, min(x2, width))
                y1 = max(0, min(y1, height-1))
                y2 = max(y1+1, min(y2, height))

                color = person_colors.setdefault(fixed_id, self.get_random_color())
                
                # Prepare shape data
                shape_data = {
                    "bbox": [x1, y1, x2, y2],
                    "shape_type": "rectangle",
                    "shape_id": str(fixed_id),
                    "confidence": confidence,
                    "label": label  # Add label to shape data
                }

                # Create and add shape to canvas with improved error handling
                try:
                    shape = self.canvas.createShapeFromData(shape_data)
                    if shape and shape.boundingRect() and not self.canvas.is_shape_duplicate(shape.id):
                        self.canvas.addShape(shape)
                        logging.info(f"Shape added: ID {fixed_id}, Bbox: {[x1, y1, x2, y2]}")
                        
                        # Create and add UI labels only if shape was successfully added
                        label_shape = self.create_labelme_shape(fixed_id, x1, y1, x2, y2, color)
                        self.add_labels_to_UI(label_shape, fixed_id, color)
                        
                        # Add to annotations only if everything succeeded
                        frame_annotations.append(shape_data)
                except Exception as canvas_error:
                    logging.error(f"Canvas shape creation error for track {track.track_id}: {canvas_error}")

            except Exception as e:
                logging.error(f"Error processing track {track.track_id}: {e}")
                continue

        return frame_annotations

    def create_labelme_shape(self, track_id, x1, y1, x2, y2, color):
        """
        Create a LabelMe-compatible shape for the given track ID and bounding box coordinates.

        Args:
            track_id (int): Unique ID of the track.
            x1, y1, x2, y2 (float): Bounding box coordinates.
            color (tuple): Color associated with the track.

        Returns:
            Shape: A configured Shape object.
        """
        label = f"person{track_id}"
        shape = Shape(label=label, shape_id=track_id)
        
        # Define the bounding box points
        bounding_points = [
            QtCore.QPointF(x1, y1),  # Top-left
            QtCore.QPointF(x2, y1),  # Top-right
            QtCore.QPointF(x2, y2),  # Bottom-right
            QtCore.QPointF(x1, y2)   # Bottom-left
        ]
        
        # Add points to the shape
        for point in bounding_points:
            shape.addPoint(point)
        
        return shape

    def add_labels_to_UI(self, shape, track_id, color):
        """
        Add the created shape and associated labels to the LabelMe UI.

        Args:
            shape (Shape): The shape object to add.
            track_id (int): Unique track ID.
            color (tuple): Color associated with the track.
        """
        self.addLabel(shape)
        self.labelList.addPersonLabel(track_id, color)
        self.uniqLabelList.addUniquePersonLabel(f"person{track_id}", color)
   

        





    ########################################################################################################################
    def display_frame(self, frame):
        
        """
        Display the current frame with annotations in the LabelMe canvas.
        """
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height, width, channel = rgb_frame.shape
            bytes_per_line = channel * width
            q_img = QtGui.QImage(rgb_frame.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888)
            pixmap = QtGui.QPixmap.fromImage(q_img)
            self.canvas.loadPixmap(pixmap)
            self.canvas.update()
            logger.debug("Frame updated in the LabelMe canvas.")
        except Exception as e:
            logger.error(f"Error displaying frame: {e}", exc_info=True)



    
    


#################################################################################################################################
    def choose_annotation_format(self):
        """Allow the user to choose the annotation format."""
        formats = ["JSON", "XML", "COCO", "YOLO"]  # Add more formats if needed
        format_choice, ok = QtWidgets.QInputDialog.getItem(
            self,
            "Choose Annotation Format",
            "Select the format for saving annotations:",
            formats,
            0,
            False,
        )
        if ok and format_choice:
            return format_choice.lower()
        return None

    
    
    def get_next_available_id(self):
        used_ids = set(self.id_mapping.values())
        next_id = 1
        while next_id in used_ids:
            next_id += 1
        return min(next_id, 4)  # Force IDs to stay within 1-4 range

    def get_random_color(self):
        """Generate a random color for bounding boxes."""
        return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    
    def find_bbox_match(self, bbox, boxes, iou_threshold=0.5):
        """
        Find the best-matching bounding box for a given bbox using IoU.
        Args:
            bbox: The bounding box to match (x1, y1, x2, y2).
            boxes: List of detected boxes [(x1, y1, x2, y2), ...].
            iou_threshold: Minimum IoU threshold to consider a match.
        Returns:
            The best-matching box from `boxes`, or None if no match is found.
        """
        def iou(box1, box2):
            x1, y1, x2, y2 = box1
            x1_, y1_, x2_, y2_ = box2

            # Compute intersection
            inter_x1 = max(x1, x1_)
            inter_y1 = max(y1, y1_)
            inter_x2 = min(x2, x2_)
            inter_y2 = min(y2, y2_)
            inter_area = max(0, inter_x2 - inter_x1 + 1) * max(0, inter_y2 - inter_y1 + 1)

            # Compute union
            box1_area = (x2 - x1 + 1) * (y2 - y1 + 1)
            box2_area = (x2_ - x1_ + 1) * (y2_ - y1_ + 1)
            union_area = box1_area + box2_area - inter_area

            return inter_area / union_area if union_area > 0 else 0

        best_match = None
        best_iou = 0

        for detected_box in boxes:
            current_iou = iou(bbox, detected_box)
            if current_iou > best_iou and current_iou >= iou_threshold:
                best_match = detected_box
                best_iou = current_iou

        return best_match




        
        
    def enable_save_annotation_button(self):
        """Enable the saveReIDAnnotation button after video annotation is done."""
        self.saveReIDAnnotationsAction.setEnabled(True)  # Enable the saveReIDAnnotation button
 
    
    def choose_annotation_format(self):
        """Allow the user to choose the annotation format."""
        formats = ["JSON", "XML", "COCO", "YOLO"]  # Add more formats if needed
        format_choice, ok = QtWidgets.QInputDialog.getItem(
            self,
            "Choose Annotation Format",
            "Select the format for saving annotations:",
            formats,
            0,
            False,
        )
        if ok and format_choice:
            return format_choice.lower()
        return None

    
    
    
    
    
    
    def save_reid_annotations(self, frames, all_annotations, format_choice="json"):
        """Save the annotations in the chosen format."""
        options = QtWidgets.QFileDialog.Options()
        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Annotations", "", f"{format_choice.upper()} Files (*.{format_choice});;All Files (*)", options=options
        )

        if not file_path:
            print("Save canceled by user.")
            return

        try:
            if format_choice == "json":
                with open(file_path, "w") as f:
                    json.dump(all_annotations, f, indent=4)
            elif format_choice == "xml":
                self.save_annotations_as_xml(file_path, all_annotations)
            elif format_choice == "coco":
                self.save_annotations_as_coco(file_path, all_annotations)
            elif format_choice == "yolo":
                self.save_annotations_as_yolo(file_path, all_annotations)
            else:
                raise ValueError(f"Unsupported format: {format_choice}")

            QtWidgets.QMessageBox.information(self, "Success", f"Annotations saved as {format_choice.upper()}!")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to save annotations: {e}")
            print(f"Error saving annotations: {e}")



    def collect_reid_annotations(self, frames, all_annotations):
        """
        Collect the ReID annotations (bounding boxes, IDs, and confidences) for each frame.
        Each frame contains the following structure:
        - frame_index
        - detections: List of detected objects, each with track_id, bbox, confidence, class
        """
        annotations = []

        for frame_index, frame_annotations in enumerate(all_annotations):
            frame_data = {
                "frame": frame_index,
                "detections": []
            }

            for annotation in frame_annotations:
                # Validate the structure of each annotation
                if not isinstance(annotation, dict):
                    raise ValueError(f"Invalid annotation format at frame {frame_index}: {annotation}")

                # Ensure required keys exist
                required_keys = {"track_id", "bbox", "confidence", "class"}
                if not required_keys.issubset(annotation.keys()):
                    raise KeyError(f"Missing keys in annotation at frame {frame_index}: {annotation}")

                detection_data = {
                    "track_id": annotation["track_id"],
                    "bbox": annotation["bbox"],
                    "confidence": annotation["confidence"],
                    "class": annotation["class"]  # For example, "person"
                }
                frame_data["detections"].append(detection_data)

            annotations.append(frame_data)

        return annotations
    
    def save_annotations_as_coco(self,file_path, all_annotations):
        """Save annotations in COCO format."""
        # Convert annotations to COCO format
        coco_annotations = {
            "info": {"description": "Generated by LabelMe"},
            "images": [],
            "annotations": [],
            "categories": [{"id": 1, "name": "person"}],
        }
        for frame_idx, frame_anno in enumerate(all_annotations):
            coco_annotations["images"].append({"id": frame_idx, "file_name": f"frame_{frame_idx}.jpg"})
            for anno in frame_anno:
                coco_annotations["annotations"].append({
                    "id": anno["track_id"],
                    "image_id": frame_idx,
                    "bbox": anno["bbox"],
                    "category_id": 1,
                    "confidence": anno["confidence"]
                })
        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save as COCO", "", "COCO Files (*.json)")
        if file_path:
            with open(file_path, 'w') as f:
                json.dump(coco_annotations, f, indent=4)


    def save_annotations_as_xml(self,file_path, all_annotations):
        """Save annotations in Pascal VOC format."""
        for frame_idx, frame_anno in enumerate(all_annotations):
            root = Element('annotation')

            # Add the filename of the frame
            SubElement(root, 'filename').text = f"frame_{frame_idx}.jpg"

            # Loop through annotations for the current frame
            for anno in frame_anno:
                obj = SubElement(root, 'object')
                SubElement(obj, 'name').text = "person"  # Class name
                bndbox = SubElement(obj, 'bndbox')

                # Add bounding box coordinates
                SubElement(bndbox, 'xmin').text = str(anno["bbox"][0])
                SubElement(bndbox, 'ymin').text = str(anno["bbox"][1])
                SubElement(bndbox, 'xmax').text = str(anno["bbox"][2])
                SubElement(bndbox, 'ymax').text = str(anno["bbox"][3])

            # Save the XML file for the current frame
            tree = ElementTree(root)
            file_path = f"frame_{frame_idx}.xml"  # You can customize the file path if needed
            tree.write(file_path, encoding="utf-8", xml_declaration=True)
            print(f"Saved Pascal VOC annotations to {file_path}")
            
    def save_annotations_as_yolo(self, file_path, all_annotations):
        """
        Save annotations in YOLO format.
        Each line in a YOLO annotation file represents an object and has the format:
        <class> <x_center> <y_center> <width> <height>
        where values are normalized to [0, 1].
        """
        try:
            frame_width = self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)
            frame_height = self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)

            with open(file_path, 'w') as f:
                for frame_idx, frame_anno in enumerate(all_annotations):
                    for anno in frame_anno:
                        bbox = anno["bbox"]
                        x_center = ((bbox[0] + bbox[2]) / 2) / frame_width
                        y_center = ((bbox[1] + bbox[3]) / 2) / frame_height
                        width = (bbox[2] - bbox[0]) / frame_width
                        height = (bbox[3] - bbox[1]) / frame_height

                        # Write in YOLO format: <class> <x_center> <y_center> <width> <height>
                        f.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

            print(f"YOLO annotations saved to {file_path}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to save YOLO annotations: {e}")
            print(f"Error saving YOLO annotations: {e}")



    

    

  
   
    
    

            
    
    
    def display_reid_detections(self, detections):
        """Display ReID detections on the frame."""
        for detection in detections:
            # Ensure detection has the required number of elements (e.g., [x1, y1, x2, y2, confidence, class_id])
            if isinstance(detection, torch.Tensor):
                # Convert to a list and ensure there are at least 4 elements
                detection_list = detection.tolist()
                if len(detection_list) >= 4:
                    # Assuming YOLO returns [x1, y1, x2, y2, confidence, class_id]
                    x1, y1, x2, y2 = map(int, detection_list[:4])
                else:
                    # If not enough values, skip this detection
                    continue
            elif isinstance(detection, dict) and 'bbox' in detection:
                x1, y1, x2, y2 = map(int, detection['bbox'])
            else:
                # Handle unexpected data types if needed
                continue

            # Draw the bounding box on the frame
            painter = QtGui.QPainter(self.canvas.pixmap())
            painter.setPen(QtGui.QPen(QtGui.QColor(0, 255, 0), 2))
            painter.drawRect(x1, y1, x2 - x1, y2 - y1)
            painter.end()

        self.canvas.update()  # Refresh the canvas to show updated annotations
  # Refresh the canvas to show updated annotations
            
    


    def closeVideo(self):
        """Release video resources."""
        if hasattr(self, 'video_capture') and self.video_capture.isOpened():
            self.video_capture.release()
        print("Video capture resources released.")  
        # Enable the frame navigation buttons only if the video is loaded successfully
        if self.video_capture.isOpened():
            self.actions.openPrevFrame.setEnabled(False)
            self.actions.openNextFrame.setEnabled(False) 
            self.actions.AnnotateVideo.setEnabled(False)

  



    
    
    def load_annotations(video_name):
        with open(f"{video_name}_annotations.json", "r") as f:
            annotations = json.load(f)
        return annotations
    
    
    """Update the code and the UI
    """
    def _load_detection_model(self, model_name):
        """Load detection model dynamically."""
        if "yolov8" in model_name.lower():
            from ultralytics import YOLO
            return YOLO("yolov8m.pt")  # Replace with user-specified model path
        elif "yolov5" in model_name.lower():
            return torch.hub.load("ultralytics/yolov5", "custom", path="yolov5s.pt")
        elif "faster_rcnn" in model_name.lower():
            import torchvision.models.detection as detection
            model = detection.fasterrcnn_resnet50_fpn(pretrained=True)
            model.eval()
            return model
        else:
            logger.warning(f"Detection model {model_name} not recognized.")
            return None

    def _load_reid_model(self, model_name):
        """Load ReID model dynamically."""
        if isinstance(model_name, int):
            model_name = str(model_name)
        if "osnet" in model_name.lower():
            import torchreid
            model = torchreid.models.build_model(
                name="osnet_x1_0", num_classes=1000, pretrained=True)
            logging.debug("loading OSNet ReID model...")
            
            model.eval().to(self.device)
            return model
        elif "fastreid" in model_name.lower():
            from fastreid.config import get_cfg
            from fastreid.engine.defaults import DefaultPredictor
            cfg = get_cfg()
            cfg.merge_from_file("A:/data/Project-Skills/Labeling_tool-enchancement/labelme/fastreid/fast-reid/configs/Market1501/bagtricks_R50.yml")
            cfg.MODEL.WEIGHTS = "A:/data/Project-Skills/Labeling_tool-enchancement/labelme/market_bot_R50.pth"  # Path to trained FastReID weights
            cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
            return DefaultPredictor(cfg)
            logging.debug("loading Fastreid ReID model...")
        else:
            logger.warning(f"ReID model {model_name} not recognized.")
            return None
        
    def addModelSelectors(self):
        """Add dropdowns for selecting detection and ReID models."""
        self.tools = self.toolbar("Tools")
        # Detection Model Selector
        self.detectionModelSelector = QtWidgets.QComboBox(self)
        self.detectionModelSelector.addItems(["YOLOv8", "YOLOv5", "Faster R-CNN"])
        # detection_model_label = QLabel("Detection Model:")
        # detection_model_dropdown = QComboBox()
        # detection_model_dropdown.addItems(["YOLOv8", "YOLOv5","Faster R-CNN"])
        # detection_model_dropdown.setCurrentIndex(0)  # Default to YOLOv8
        self.tools.addWidget(QtWidgets.QLabel("Detection Model:"))
        self.tools.addWidget(self.detectionModelSelector)
        # self.tools.addWidget(detection_model_label, alignment=Qt.AlignLeft)
        # self.tools.addWidget(detection_model_dropdown, alignment=Qt.AlignLeft)
        # self.tools.addWidget(reid_model_label, alignment=Qt.AlignLeft)
        # self.tools.addWidget(reid_model_dropdown, alignment=Qt.AlignLeft)

        # ReID Model Selector
        self.reidModelSelector = QtWidgets.QComboBox(self)
        self.reidModelSelector.addItems(["OSNet", "FastReID"])
        self.reidModelSelector.setCurrentIndex(0)  # Default to OSNet
        self.tools.addWidget(QtWidgets.QLabel("ReID Model:"))
        self.tools.addWidget(self.reidModelSelector)

        # Connect selectors to model-loading logic
        self.detectionModelSelector.currentIndexChanged.connect(self.load_models)
        self.reidModelSelector.currentIndexChanged.connect(self.load_models)
    
    def log_message(self, message):
        self.log_area.append(message)

    def upload_dataset(self):
        # Select dataset folder
        dataset_path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Dataset Folder")
        if not dataset_path:
            self.log_message("No folder selected.")
            return

        # Select output folder
        output_path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if not output_path:
            self.log_message("No output folder selected.")
            return

        # Determine dataset type
        dataset_type = self.dataset_selector.currentText().lower()
        self.log_message(f"Selected Dataset: {dataset_type.capitalize()}")

        # Handle dataset based on selection
        try:
            if dataset_type == "cuhk03":
                handler = CUHK03Handler(dataset_path, output_path)
            elif dataset_type == "market1501":
                handler = Market1501Handler(dataset_path, output_path)
            else:
                raise ValueError(f"Unsupported dataset type: {dataset_type}")

            handler.prepare_dataset()
            self.log_message(f"{dataset_type.capitalize()} dataset processed successfully.")
        except Exception as e:
            self.log_message(f"Error: {e}")
#############################################################################################


"""Tracker and the confidence class """
class EnhancedDeepSORT:
    def __init__(self, reid_model):
        self.reid_model = reid_model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initialize metric and tracker with more robust parameters
        max_cosine_distance = 0.3
        nn_budget = 100
        metric = NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        
        max_iou_distance = 0.7
        max_age = 30
        n_init = 3
        
        self.tracker = Tracker(
            metric,
            max_iou_distance=max_iou_distance,
            max_age=max_age,
            n_init=n_init
        )
        self.tracks_confidence = {}
        
        logger.info("Enhanced DeepSORT tracker initialized")

    def update(self, bbox_xywh, confidences, features):
        # Predict first
        self.tracker.predict()
        
        # Prepare detections
        detections = []
        
        # Convert boxes to TLWH format
        bbox_tlwh = self._xywh_to_tlwh(bbox_xywh)
        
        # Create Detection objects
        for tlwh, conf, feat in zip(bbox_tlwh, confidences, features):
            if feat is not None:
                detection = Detection(tlwh, conf, feat)
                detections.append(detection)
        
        # Update tracker
        self.tracker.update(detections)
        
        # Get results
        outputs = []
        for track in self.tracker.tracks:
            if not track.is_confirmed():
                continue
                
            box = track.to_tlbr()
            track_id = track.track_id
            
            # Update confidence
            if track_id not in self.tracks_confidence:
                self.tracks_confidence[track_id] = ConfidenceTrack(track_id)
            
            self.tracks_confidence[track_id].update(track.confidence)
            
            if self.tracks_confidence[track_id].is_confirmed():
                outputs.append(np.append(box, track_id))
        
        return np.array(outputs)

    def _xywh_to_tlwh(self, bbox_xywh):
        """Convert [x_center, y_center, w, h] to [top_left_x, top_left_y, w, h]"""
        if bbox_xywh.size == 0:
            return np.array([])
            
        bbox_tlwh = bbox_xywh.copy()
        bbox_tlwh[:, 0] = bbox_xywh[:, 0] - bbox_xywh[:, 2]/2.
        bbox_tlwh[:, 1] = bbox_xywh[:, 1] - bbox_xywh[:, 3]/2.
        return bbox_tlwh

    @property
    def tracks(self):
        return self.tracker.tracks if hasattr(self.tracker, 'tracks') else []

class ConfidenceTrack:
    def __init__(self, track_id):
        self.track_id = track_id
        self.confidence = 0.0
        self.history = []
        self.missed_frames = 0
        
    def update(self, detection_conf):
        self.confidence = min(self.confidence + detection_conf * 0.2, 1.0)
        self.missed_frames = 0
        self.history.append(detection_conf)
        if len(self.history) > 30:
            self.history.pop(0)
            
    def mark_missed(self):
        self.missed_frames += 1
        self.confidence = max(self.confidence - 0.1, 0.0)
        
    def is_confirmed(self):
        return self.confidence > 0.5 and self.missed_frames < 5