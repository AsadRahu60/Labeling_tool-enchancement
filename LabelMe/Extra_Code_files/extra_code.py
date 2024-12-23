# def annotateVideo(self):
#         """Annotate and track persons in the video using YOLOv8 segmentation for detection and OSNet for ReID."""
#         if not hasattr(self, 'video_capture'):
#             QtWidgets.QMessageBox.warning(self, "Error", "No video loaded.")
#             return

#         person_id_map = {}  # Dictionary to store person IDs and their features
#         next_person_id = 1

#         while self.video_capture.isOpened():
#             ret, frame = self.video_capture.read()
#             if not ret:
#                 break

#             # Detect persons using YOLOv8 segmentation
#             boxes, masks = self.run_yolo_segmentation(frame)
#             annotated_frame = frame.copy()  # Copy frame for drawing

#             # Extract ReID features using OSNet
#             if boxes:
#                 features = self.extract_reid_features_with_masks(frame, boxes, masks)

#                 person_ids = []  # List to store IDs of persons in the current frame

#                 for feature in features:
#                     matched_id = None
#                     # Compare the extracted feature with the saved features in person_id_map
#                     for person_id, saved_feature in person_id_map.items():
#                         if self.is_same_person(feature, saved_feature):  # type: ignore
#                             matched_id = person_id  # Found a matching person, reuse the ID
#                             person_id_map[person_id] = feature  # Update the stored feature
#                             break

#                     if matched_id is None:
#                         # New person detected, assign a new ID
#                         matched_id = f"person_{next_person_id}"
#                         person_id_map[matched_id] = feature
#                         next_person_id += 1

#                     person_ids.append(matched_id)

#                 # Safeguard: Ensure we don't access more person_ids than there are bounding boxes
#                 if len(person_ids) > len(boxes):
#                     person_ids = person_ids[:len(boxes)]  # Truncate the IDs to match boxes
#                 elif len(person_ids) < len(boxes):
#                     # If fewer IDs than boxes, append 'Unknown' for missing IDs
#                     person_ids += ['Unknown'] * (len(boxes) - len(person_ids))

#                 # Draw bounding boxes and masks on the frame
#                 for i, box in enumerate(boxes):
#                     x1, y1, x2, y2 = box
#                     cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

#                     # Annotate person ID on the bounding box
#                     cv2.putText(annotated_frame, f"ID: {person_ids[i]}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

#                     # Draw the segmentation mask if available
#                     if i < len(masks):
#                         mask = masks[i] # Ensure the mask is converted to a numpy array
#                         mask = (mask > 0.5).astype(np.uint8)  # Binarize the mask (thresholding)
                        
#                         # Resize the mask to fit the bounding box size
#                         mask_resized = cv2.resize(mask, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)

#                         # Create a colored mask (apply a green color)
#                         colored_mask = np.zeros_like(frame, dtype=np.uint8)
#                         colored_mask[y1:y2, x1:x2, 1] = mask_resized * 255  # Apply green color on the mask

#                         # Blend the mask with the frame
#                         annotated_frame = cv2.addWeighted(annotated_frame, 1, colored_mask, 0.5, 0)

#                 print(f"Extracted ReID features and assigned IDs for {len(person_ids)} persons")

#             # Update the display with the annotated frame
#             self.update_display(annotated_frame)

#             cv2.waitKey(1)

#         # After the loop, release video capture
#         # self.video_capture.release()
        
        
#         ######################bbox code within the annotation_with_display
#          try:
#                 # Case 1: If bbox is a list of arrays, iterate through each array
#                 if isinstance(bbox, list) and all(isinstance(b, np.ndarray) for b in bbox):
#                     for b in bbox:
#                         # Check if each array is of correct size
#                         if len(b) == 4:
#                             x1, y1, x2, y2 = map(int, b.tolist())  # Convert array to list and unpack
#                         else:
#                             raise ValueError(f"Expected 4-element array, got {b}")

#                 # Case 2: If bbox is already a single array or list, handle it directly
#                 elif isinstance(bbox, (np.ndarray, list)) and len(bbox) == 4:
#                     x1, y1, x2, y2 = map(int, bbox)  # Safely unpack

#                 else:
#                     raise ValueError(f"Expected 4-element bounding box, got {bbox}")

#                 # Draw the bounding box and track ID on the frame
#                 farbe = self.get_random_color()
#                 cv2.rectangle(frame, (x1, y1), (x2, y2), farbe, 2)
#                 cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                
# ########################################################## run_ReID function#######################333
# # def run_reid_on_frame(self, frame,detections):
#     #     """Run Person ReID model on the frame and update tracking."""  
#     #     features = []

#     # # Iterate over each detection to extract features
#     #     for detection in detections:
#     #         x1, y1, x2, y2 = detection
#     #         # Crop the person region from the frame
#     #         person_img = frame[y1:y2, x1:x2]
            
#     #         # Preprocess the image for the ReID model
#     #         person_tensor = self.transform(person_img).unsqueeze(0)  # Assuming self.transform exists

#     #         # Extract features using the ReID model
#     #         with torch.no_grad(), torch.amp.autocast("cuda"):
#     #          # Assuming mixed precision for efficiency
#     #             feature = self.reid_model(person_tensor)
#     #         features.append(feature)

#     #     # Update person tracks using extracted features
#     #     for i, feature in enumerate(features):
#     #         # Here, we assume some logic to assign IDs to people based on extracted features.
#     #         # For simplicity, we can just generate unique IDs if no tracker is being used.
#     #         person_id = f"person_{i + 1}"  # Generate unique IDs for each person in the frame

#     #         # Update the person's track
#     #         if person_id not in self.person_tracks:
#     #             self.person_tracks[person_id] = []
#     #         self.person_tracks[person_id].append(detections[i])  # Update person's track with the bounding box

#     #     return features
    
#     ################################################ Run extract_mask##############################333333
#     def extract_reid_features_with_masks(self, frame, boxes, masks):
        
#         """Extract ReID features using OSNet for the segmented persons in the frame."""

#         persons = []
#         height, width, _ = frame.shape  # Frame dimensions for bounding box checks

#         for i, (x1, y1, x2, y2) in enumerate(boxes):
#             # Check if the bounding box is valid
#             if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
#                 print(f"Skipping invalid box: {x1, y1, x2, y2}")
#                 continue

#             # Crop the person image from the frame
#             person_img = frame[y1:y2, x1:x2]
#             if person_img.size == 0:
#                 print(f"Skipping invalid crop for box {i}")
#                 continue  # Skip if the crop is invalid

#             # Visualize the original cropped image for debugging
#             # cv2.imshow(f"Person_{i}_Original", person_img)
            
#             # Apply the mask if available
#             if masks[i] is not None:
#                 mask = masks[i]
#                 mask_resized = cv2.resize(mask, (person_img.shape[1], person_img.shape[0]), interpolation=cv2.INTER_NEAREST)

#                 # Ensure the mask is binary and in uint8 format
#                 mask_resized = (mask_resized > 0.5).astype(np.uint8) * 255  # Convert to 0 or 255

#                 # If mask is single-channel but person_img is 3-channel, adjust the mask
#                 if len(mask_resized.shape) == 2 and len(person_img.shape) == 3:
#                     mask_resized = cv2.merge([mask_resized, mask_resized, mask_resized])

#                 # Apply the mask to the person image
#                 person_img = cv2.bitwise_and(person_img, mask_resized)

#                 # Visualize the masked image for debugging
#                 # cv2.imshow(f"Person_{i}_Masked", person_img)
#                 # cv2.waitKey(0)  # Pause for debugging

#             # Resize to OSNet input size
#             person_img = cv2.resize(person_img, (128, 256))

#             # Preprocess for OSNet
#             person_tensor = self.transform(person_img).unsqueeze(0)
#             person_tensor = person_tensor.cuda() if torch.cuda.is_available() else person_tensor

#             persons.append(person_tensor)
#             torch.cuda.empty_cache()  # Clear GPU cache after processing
#         # Stack tensors and extract features
#         if persons:
#             person_batch = torch.cat(persons)
#             with torch.no_grad():
#                 features = self.reid_model(person_batch)
#             return features.cpu().numpy()
#         else:
#             return []
        
        
#         ##############################################################################
#     ####Code for the 
#     corners = tf.constant(boxes, tf.float32)
#   boxesList = box_list.BoxList(corners)
#   boxesList.add_field('scores', tf.constant(scores))
#   iou_thresh = 0.1
#   max_output_size = 100
#   sess = tf.Session()
#   nms = box_list_ops.non_max_suppression(
#       boxesList, iou_thresh, max_output_size)
#   boxes = sess.run(nms.get())
  
  
#   ##############################################################################################################
#   def box_area(self,box):
#         """Calculate the area of a bounding box given as (x1, y1, x2, y2)."""
#         x1, y1, x2, y2 = box
#         area = (x2 - x1) * (y2 - y1)
#         print(f"Box: {box}, Area: {area}")
#         return area
    
#     def run_yolo_segmentation(self, frame):
#         """Run YOLOv8 segmentation model to detect and segment people in the frame."""

#         # Set model thresholds
#         self.yolo_model.conf = 0.5  # Confidence threshold for detection
#         self.yolo_model.iou = 0.5   # IOU threshold for NMS

#         # Run YOLO inference
#         results = self.yolo_model(source=frame, stream=False)  # Batch inference on the frame

#         boxes = []
#         confidences = []  # Store confidence values
#         class_ids = []  # Store class IDs for detected objects

#         # Minimum and maximum area thresholds for bounding boxes
#         min_area_threshold = 500
#         max_area_threshold = 3000000

#         for result in results:
#             if hasattr(result, 'boxes'):
#                 for idx, cls in enumerate(result.boxes.cls):
#                     cls = int(cls)
#                     if cls == 0:  # Class 'person'
#                         # Extract bounding box
#                         x1, y1, x2, y2 = result.boxes.xyxy[idx].cpu().numpy().astype(int)

#                         # Calculate area and filter based on thresholds
#                         area = (x2 - x1) * (y2 - y1)
#                         if min_area_threshold < area < max_area_threshold:
#                             boxes.append([x1, y1, x2 - x1, y2 - y1])  # Convert to (x, y, w, h)
#                             confidences.append(float(result.boxes.conf[idx].cpu().numpy()))
#                             class_ids.append(cls)

#         # Apply Non-Maximum Suppression (NMS)
#         indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.4, nms_threshold=0.4)

#         filtered_boxes = []
#         filtered_confidences = []

#         for i in indices:
#             i = i[0]
#             filtered_boxes.append(boxes[i])
#             filtered_confidences.append(confidences[i])

#         print(f"Filtered boxes after NMS: {filtered_boxes}")
#         print(f"Number of filtered boxes: {len(filtered_boxes)}")
#         print(f"Confidences: {filtered_confidences}")

#         # Create masks if available
#         filtered_masks = [None] * len(filtered_boxes)  # Placeholder if masks are needed in the future

#         return filtered_boxes, filtered_masks, filtered_confidences
#     ###########################################################################################################
#     def run_batch_inference(self, frames, frame_indices):
       
#         """Run YOLOv8 segmentation and feature extraction in batch mode on multiple frames."""
#         all_bboxes = []
#         all_features = []
#         all_ids = []
#         all_confidences = []

#         results = self.yolo_model(frames)  # Batch inference on raw frames
        
#         min_area_threshold = 500  # Example minimum area threshold
#         max_area_threshold = 3000000  # Example maximum area threshold

#         for i, (result, frame) in enumerate(zip(results, frames)):
#             # YOLO result for the current frame
#             boxes, masks, confidences = self.run_yolo_segmentation(result, frame)
            
#             print(f"YOLO Detected Boxes: {boxes}, Masks: {masks}")

#             if not boxes:
#                 # No detections, skip processing
#                 continue

#             # Filter the bounding boxes and masks based on area
#             filtered_boxes = [box for box in boxes if min_area_threshold < self.box_area(box) < max_area_threshold]
#             filtered_masks = [mask for i, mask in enumerate(masks) if min_area_threshold < self.box_area(boxes[i]) < max_area_threshold]

#             print(f"Filtered Boxes: {filtered_boxes}")

#             # Extract ReID features
#             features = self.run_reid_with_or_without_masks(frame, filtered_boxes, filtered_masks)
#             print(f"Extracted Features: {features}")
#             features=np.array(features)

#             # Ensure both boxes and features are present
#             if not filtered_boxes or not features.any():
#                 continue  # Skip if there are no valid boxes or features

#             # Run DeepSORT with pre-extracted features
#             tracks = self.run_deepsort(filtered_boxes, features, confidences)
#             print(f"Tracks: {tracks}")

#             # Collect detected track IDs
#             ids = [track['track_id'] for track in tracks]

#             # Annotate and display tracks on the current frame
#             self.annotate_and_display_tracks(tracks, frame)

#             # Collect bounding boxes, features, track IDs, and confidences for saving
#             all_bboxes.extend(filtered_boxes)
#             all_features.extend(features)
#             all_ids.extend(ids)
#             all_confidences.extend(confidences)  # Store confidences

#             # Display the annotated frame and close windows after processing
#             self.update_display(frame)
#             cv2.destroyAllWindows()
#             cv2.waitKey(1)

#         return all_bboxes, all_features, all_ids, all_confidences
    
#     ##########################################################################################################################################
#     """class DeepSort:
#     def __init__(self, max_age=30, nn_budget=100, max_iou_distance=0.7, n_init=3):
#         self.max_age = max_age
#         self.nn_budget = nn_budget
#         self.max_iou_distance = max_iou_distance
#         self.n_init = n_init
#         self.tracks = []  # List to store tracked objects
#         self.next_track_id = 1  # ID to assign to new tracks

#     def update(self, bbox_xywh, confidences, reid_features):
#         """
#         Update the DeepSORT tracker with new detections and return updated tracks.
        
#         Parameters:
#         - bbox_xywh: Bounding boxes in (x, y, w, h) format
#         - confidences: Confidence scores for each detection
#         - reid_features: ReID features for each detection
        
#         Returns:
#         - List of updated tracks, each containing the track ID and bounding box
#         """
#         # 1. Predict new locations of existing tracks using Kalman filter
#         self._predict()

#         # 2. Match detections to existing tracks
#         matches, unmatched_detections, unmatched_tracks = self._data_association(bbox_xywh, reid_features)

#         # 3. Update matched tracks with new detection info
#         for track_idx, detection_idx in matches:
#             track = self.tracks[track_idx]
#             detection = bbox_xywh[detection_idx]
#             feature = reid_features[detection_idx]
#             track.update(detection, feature)  # Update the track with new data

#         # 4. Create new tracks for unmatched detections
#         for detection_idx in unmatched_detections:
#             self._create_track(bbox_xywh[detection_idx], reid_features[detection_idx])

#         # 5. Remove lost tracks
#         self._remove_lost_tracks()

#         # 6. Return the updated tracks (with IDs and bounding boxes)
#         return self._get_active_tracks()

#     def _predict(self):
#         """Predict the new locations of all tracks using the Kalman filter."""
#         for track in self.tracks:
#             track.predict()

#     def _data_association(self, bbox_xywh, reid_features):
#         """Match detections to existing tracks using bounding boxes and ReID features."""
#         # Placeholder: In a real implementation, you would match based on IoU and ReID feature similarity
#         # We'll assume a simple one-to-one match for demonstration purposes

#         matches = []  # List of (track_idx, detection_idx) pairs
#         unmatched_detections = list(range(len(bbox_xywh)))  # Assume all detections are unmatched
#         unmatched_tracks = list(range(len(self.tracks)))  # Assume all tracks are unmatched

#         # Lists to store items to be removed after iteration
#         detections_to_remove = []
#         tracks_to_remove = []
#         # Match based on some heuristic (e.g., IoU, feature similarity)
#         for i, track in enumerate(self.tracks):
#             for j, detection in enumerate(bbox_xywh):
#                 # Placeholder for IoU and feature matching logic
#                 iou = self._calculate_iou(track.bbox, detection)
#                 feature_similarity = self._calculate_feature_similarity(track.reid_feature, reid_features[j])

#                 if iou > 0.5 and feature_similarity > 0.5:  # Threshold values can be tuned
#                     matches.append((i, j))
#                     detections_to_remove.append(j)  # Collect detection index to remove later
#                     tracks_to_remove.append(i)  # Collect track index to remove later
#                     break
        
#             # Now remove matched detections and tracks outside the loop
#         unmatched_detections = [d for d in unmatched_detections if d not in detections_to_remove]
#         unmatched_tracks = [t for t in unmatched_tracks if t not in tracks_to_remove]

#         return matches, unmatched_detections, unmatched_tracks

#     def _create_track(self, bbox, reid_feature):
#         # """Create a new track for an unmatched detection."""
#         new_track = Track(self.next_track_id, bbox, reid_feature)
#         self.tracks.append(new_track)
#         self.next_track_id += 1

#     def _remove_lost_tracks(self):
#         """Remove tracks that have been lost for too long."""
#         self.tracks = [track for track in self.tracks if not track.is_lost()]

#     def _get_active_tracks(self):
#         """Return the currently active tracks (those that are not lost)."""
#         return [track for track in self.tracks if track.is_active()]

#     def _calculate_iou(self, bbox1, bbox2):
#         """Calculate the Intersection over Union (IoU) between two bounding boxes."""
#         # Implement IoU calculation here
#         return 0.7  # Placeholder value

#     def _calculate_feature_similarity(self, feature1, feature2):
#         """Calculate the similarity between two ReID features."""
#         # Implement feature similarity calculation here (e.g., cosine similarity)
#         return 0.9  # Placeholder value


# class Track:
#     def __init__(self, track_id, bbox, reid_feature):
#         self.track_id = track_id
#         self.bbox = bbox
#         self.reid_feature = reid_feature
#         self.kalman_filter = KalmanFilter()  # Placeholder for Kalman filter implementation
#         self.age = 0
#         self.time_since_update = 0

#     def predict(self):
#         # """Predict the next location using the Kalman filter."""
#         # Update the bounding box using the Kalman filter
#         self.bbox = self.kalman_filter.predict()
#         self.age += 1
#         self.time_since_update += 1

#     def update(self, bbox, reid_feature):
#         # """Update the track with new detection data."""
#         self.bbox = self.kalman_filter.update(bbox)  # Update the Kalman filter
#         self.reid_feature = reid_feature  # Update the ReID feature
#         self.time_since_update = 0

#     def is_active(self):
#         # """Return True if the track is still active."""
#         return self.time_since_update <= self.age

#     def is_lost(self):
#         # """Return True if the track is considered lost."""
#         return self.time_since_update > 30  # Can be adjusted based on max_age

#     def to_tlbr(self):
#         # """Return the bounding box in (top-left, bottom-right) format."""
#         print(self.bbox)
#         x, y, w, h = self.bbox
#         return [x, y, x + w, y + h]

# import numpy as np

# class KalmanFilter:
#     def __init__(self):
#         # Initialize state (x, y, dx, dy) and covariance matrix
#         self.state = np.zeros((4, 1))  # x, y, dx, dy
#         self.P = np.eye(4)  # Covariance matrix

#         # State transition matrix
#         self.F = np.array([
#             [1, 0, 1, 0],
#             [0, 1, 0, 1],
#             [0, 0, 1, 0],
#             [0, 0, 0, 1]
#         ])

#         # Measurement matrix
#         self.H = np.array([
#             [1, 0, 0, 0],
#             [0, 1, 0, 0]
#         ])

#         # Measurement noise covariance
#         self.R = np.eye(2)

#         # Process noise covariance
#         self.Q = np.eye(4) * 0.01

#     def predict(self):
#         # """Predict the next state based on the current state."""
#         self.state = np.dot(self.F, self.state)
#         self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
#         return self.state

#     def update(self, measurement):
#         # """Update the state based on a new measurement."""
#         # Measurement residual (innovation)
#         y = measurement - np.dot(self.H, self.state)

#         # Innovation covariance
#         S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R

#         # Kalman gain
#         K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))

#         # Update the state and covariance
#         self.state = self.state + np.dot(K, y)
#         self.P = np.dot(np.eye(4) - np.dot(K, self.H), self.P)
#         return self.state
# """ 

# """
# def detect_persons(self, frame):
#         # Run YOLO detection on the frame
#         results = self.yolo_model(frame)

#         boxes = []
#         confidences = []

#         # Extract bounding boxes and confidence values
#         for result in results:
#             if hasattr(result, 'boxes'):
#                 for idx, cls in enumerate(result.boxes.cls):
#                     if int(cls) == 0:  # Class ID 0 is for 'person'
#                         x1, y1, x2, y2 = result.boxes.xyxy[idx].cpu().numpy().astype(int)
#                         confidence = float(result.boxes.conf[idx].cpu().numpy())

#                         # Convert bounding box to (x, y, width, height)
#                         width, height = x2 - x1, y2 - y1
#                         bbox_xywh = [x1, y1, width, height]
#                         boxes.append(bbox_xywh)
#                         confidences.append(confidence)

#         # Extract ReID features for all detected boxes
#         features = self.extract_reid_features(frame, boxes) if boxes else []

#         if len(boxes) > 0 and len(features) > 0:
#             try:
#                 # Run DeepSORT
#                 tracks = self.deepsort.update_tracks(
#                     raw_detections=list(zip(boxes, confidences)),
#                     embeds=features
#                 )
#                 # Extract track IDs from the returned tracks
#                 ids = [track.track_id for track in tracks]

#                 # Draw bounding boxes and IDs on the frame
#                 for track in tracks:
#                     track_id = track.track_id
#                     bbox = track.to_tlbr()
#                     x1, y1, x2, y2 = map(int, bbox)
#                     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                     cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

#             except AssertionError as e:
#                 print(f"Assertion error in DeepSORT: {e}")
#             except TypeError as e:
#                 print(f"Type error in DeepSORT: {e}")

#         return boxes, confidences, ids, features



#     def extract_reid_features(self, frame, boxes):
#         """Extract features for person re-identification using the ReID model."""
#         features = []

#         for box in boxes:
#             x, y, width, height = box
#             # Crop the person image from the frame
#             person_img = frame[y:y + height, x:x + width]

#             if person_img.size == 0:
#                 continue  # Skip if the crop is invalid
#             person_img = cv2.resize(person_img, (128, 256))

#             # Transform the cropped image for ReID model input
#             person_tensor = self.transform(person_img).unsqueeze(0)
#             person_tensor = person_tensor.cuda() if torch.cuda.is_available() else person_tensor

#             # Switch the ReID model to evaluation mode to avoid BatchNorm errors
#             self.reid_model.eval()  # Ensure model is in evaluation mode

#             # Extract features using the ReID model
#             with torch.no_grad():
#                 feature = self.reid_model(person_tensor).cpu().numpy()
#             feature= feature.flatten()
#             features.append(feature)

#         return features
# """
# """
#  def process_video(self, video_path, output_txt_path="detections_output.txt"):
#         # Open the video
#         cap = cv2.VideoCapture(video_path)

#         # Open the output file for writing detection results
#         with open(output_txt_path, 'w') as output_file:
#             frame_count = 0
            
#             while cap.isOpened():
#                 ret, frame = cap.read()
#                 if not ret:
#                     break

#                 # Reduce frame size if needed for speed
#                 frame = cv2.resize(frame, (640, 384))
                
#                 # Run detection on the current frame
#                 boxes, confidences, ids, features = self.detect_persons(frame)

#                 # Save bounding box information to the file
#                 for i, box in enumerate(boxes):
#                     x, y, w, h = box
#                     confidence = confidences[i]
#                     track_id = ids[i] if ids else None

#                     # Write to the output file: frame number, track ID, bounding box coordinates, confidence
#                     output_file.write(f"Frame: {frame_count}, ID: {track_id}, BBox: [{x}, {y}, {w}, {h}], Confidence: {confidence:.2f}\n")
                
#                 frame_count += 1

#                 # Optional: display the frame for visualization
#                 for i, box in enumerate(boxes):
#                     x, y, w, h = box
#                     track_id = ids[i] if ids else "N/A"
#                     cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
#                     cv2.putText(frame, f"ID: {track_id}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

#                 # Show the processed frame
#                 cv2.imshow("Processed Video", frame)
                
#                 # Break if 'q' is pressed
#                 if cv2.waitKey(1) & 0xFF == ord('q'):
#                     break

#             # Release resources
#             cap.release()
#             cv2.destroyAllWindows()
# """

# # def load_models(self):
#     #     # Load YOLO model
#     #     self.yolo_model = YOLO("yolov8n.pt")

#     #     # Load OSNet model for ReID
#     #     self.reid_model = models.build_model(
#     #         name='osnet_x1_0',  # Use 'osnet_x1_0' for a balanced model size and performance
#     #         num_classes=1000,  # Dummy number for classes (not used during feature extraction)
#     #         pretrained=True,
#     #         loss='softmax'  # Load pre-trained weights
#     #     )
        
#     #     # Load DeepSORT
#     #     self.deepsort = DeepSort(
#     #         max_age=30,
#     #         nn_budget=100,
#     #         max_iou_distance=0.4,
#     #         n_init=3
#     #     )

#     #     # Define the transform for ReID
#     #     self.transform = transforms.Compose([
#     #         transforms.ToPILImage(),
#     #         transforms.Resize((256, 128)),
#     #         transforms.ToTensor(),
#     #         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     #     ])
        
#     #     # Move ReID model to GPU if available
#     #     self.reid_model = self.reid_model.cuda() if torch.cuda.is_available() else self.reid_model
#     #     self.reid_model.eval()  # Set the model to evaluation mode    
        
#     #     # Create an instance of PersonDetectionApp with the loaded models
#     #     self.person_detection_app = PersonDetectionApp(yolo_model=self.yolo_model, deepsort=self.deepsort, reid_model=self.reid_model)
    
    
    
# ##################################################ReID#########################################################

#     def run_reid_with_or_without_masks(self, frame, boxes, masks=None):
#         """
#         Run Person ReID model on the frame using bounding boxes and optional masks.
        
#         Args:
#             frame: The input frame from the video.
#             boxes: List of bounding boxes [(x1, y1, x2, y2), ...] for detected persons.
#             masks: Optional list of masks corresponding to the bounding boxes.
        
#         Returns:
#             features: Extracted ReID features for the detected persons.
#         """
#         persons = []
#         height, width, _ = frame.shape  # Frame dimensions for bounding box checks

#         # Iterate over each detection (bounding box)
#         for i, (x1, y1, x2, y2) in enumerate(boxes):
#             # Check if the bounding box is valid
#             if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
#                 print(f"Skipping invalid box: {x1, y1, x2, y2}")
#                 continue

#             # Crop the person image from the frame
#             person_img = frame[y1:y2, x1:x2]
#             if person_img.size == 0:
#                 print(f"Skipping invalid crop for box {i}")
#                 continue  # Skip if the crop is invalid

#             # Step 1: Apply the mask if available
#             if masks is not None and masks[i] is not None:
#                 mask = masks[i]
#                 mask_resized = cv2.resize(mask, (person_img.shape[1], person_img.shape[0]), interpolation=cv2.INTER_NEAREST)

#                 # Ensure the mask is binary and in uint8 format
#                 mask_resized = (mask_resized > 0.5).astype(np.uint8) * 255  # Convert to 0 or 255

#                 # If mask is single-channel but person_img is 3-channel, adjust the mask
#                 if len(mask_resized.shape) == 2 and len(person_img.shape) == 3:
#                     mask_resized = cv2.merge([mask_resized, mask_resized, mask_resized])

#                 # Apply the mask to the person image
#                 person_img = cv2.bitwise_and(person_img, mask_resized)

#             # Step 2: Resize to ReID model input size (e.g., OSNet uses 128x256)
#             person_img = cv2.resize(person_img, (128, 256))

#             # Step 3: Preprocess for the ReID model
#             person_tensor = self.transform(person_img).unsqueeze(0)  # Assuming self.transform exists
#             person_tensor = person_tensor.cuda() if torch.cuda.is_available() else person_tensor

#             # Collect the person tensor for feature extraction
#             persons.append(person_tensor)
#             torch.cuda.empty_cache()  # Clear GPU cache after processing

#         # Step 4: Extract ReID features using the model
#         if persons:
#             person_batch = torch.cat(persons)
#             with torch.no_grad():
#                 features = self.reid_model(person_batch)  # Run the ReID model
#             return features.cpu().numpy()
#         else:
#             return []  # Return empty list if no valid persons detected


# ################################################Deep Sort #########################################################3


#     # def run_deepsort(self, boxes, features, confidences):
#     #     """Run DeepSORT tracker and return bounding boxes and IDs."""
        
#     #     # Step 1: Debugging input sizes (Check the consistency in input sizes)
#     #     print(f"Running DeepSORT with {len(boxes)} boxes, {len(features)} features, {len(confidences)} confidences")

#     #     bbox_xywh = []
#     #     reid_features = []
        
#     #     # Step 2: Convert boxes and ensure there are corresponding features
#     #     for i, (x1, y1, x2, y2) in enumerate(boxes):
#     #         if i < len(features):  # Ensure there's a corresponding ReID feature
#     #             bbox_xywh.append([x1, y1, x2 - x1, y2 - y1])  # Convert to (x, y, w, h)
#     #             reid_features.append(features[i])
#     #         else:
#     #             print(f"Warning: No corresponding feature for box {i} (box: {[x1, y1, x2, y2]})")
        
#     #     # Step 3: Check the conversion results
#     #     print(f"Converted to DeepSORT format: {len(bbox_xywh)} bounding boxes and {len(reid_features)} features")

#     #     # Step 4: Convert lists to numpy arrays for DeepSORT input
#     #     bbox_xywh = np.array(bbox_xywh)
#     #     confidences = np.array(confidences)
#     #     reid_features = np.array(reid_features)

#     #     # Step 5: Check for invalid input (empty bounding boxes or features)
#     #     if len(bbox_xywh) == 0 or len(reid_features) == 0:
#     #         print("No valid bounding boxes or features to track.")
#     #         return []

#     #     # Step 6: Pass the inputs to DeepSORT and print the output tracks
#     #     tracks = self.deepsort.update(bbox_xywh, confidences, reid_features)
#     #     print(f"DeepSORT returned {len(tracks)} tracks")

#     #     # Step 7: Debugging the track information before assigning person IDs
#     #     person_tracks = []
#     #     for track in tracks:
#     #         track_id = track.track_id
#     #         print(f"Track ID: {track_id} | Bounding box: {track.to_tlbr()}")

#     #         # Step 8: Assign person IDs and store them in a map
#     #         if track_id not in self.person_id_map:
#     #             person_id = f"person_ID{len(self.person_id_map) + 1}"
#     #             self.person_id_map[track_id] = person_id

#     #         person_tracks.append({
#     #             "track_id": track_id,
#     #             "person_id": self.person_id_map[track_id],
#     #             "bbox": track.to_tlbr()
#     #         })

#     #     # Step 9: Print final mapping of person IDs to tracks
#     #     for pt in person_tracks:
#     #         print(f"Final Track -> Track ID: {pt['track_id']}, Person ID: {pt['person_id']}, BBox: {pt['bbox']}")

#     #     # Step 10: Additional check: Make sure the returned number of person tracks matches the input
#     #     if len(person_tracks) != len(bbox_xywh):
#     #         print(f"Warning: Returned {len(person_tracks)} person tracks but started with {len(bbox_xywh)} bounding boxes")

#     #     return person_tracks
    
#     def annotateVideo(self, batch_size=2):
#         """Annotate and track persons in the video using the imported detection class."""
#         self.load_models()
        
#         if not hasattr(self, 'video_capture'):
#             QtWidgets.QMessageBox.warning(self, "Error", "No video loaded.")
#             return

#         frames = []
#         while self.video_capture.isOpened():
#             ret, frame = self.video_capture.read()
#             if not ret:
#                 break
#             # Collect frames into a list for batch processing
#             frames.append(frame)

#         # Process collected frames in batches
#         all_boxes, all_features, all_ids, all_confidences = self.person_detection_app.annotate_batch(frames, batch_size)

#         # Display annotated frames as required
#         for frame in frames:
#             rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             height, width, channel = rgb_frame.shape
#             bytes_per_line = channel * width
#             q_img = QtGui.QImage(rgb_frame.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888)

#             # Update the canvas
#             if hasattr(self, 'canvas'):
#                 self.canvas.loadPixmap(QtGui.QPixmap.fromImage(q_img))
#                 self.setClean()
#                 self.canvas.update()
#                 self.repaint()

#         # Release resources
#         self.video_capture.release()
#         cv2.destroyAllWindows()
#         torch.cuda.empty_cache()  # Clean up GPU memory

#         # Inform user of processing completion
#         QtWidgets.QMessageBox.information(self, "Processing Complete", "The video has been processed successfully.")
        
        



    
    


    
     

    
#     def annotate_and_display_tracks(self, tracks, frame):
#         display_frame = frame.copy()
        
#         for track in tracks:
#             track_id = track['track_id']  # Access 'track_id' as a key in the dictionary
            
#             bbox = track['bbox']  # Access 'bbox' as a key in the dictionary
#             print(f"bbox before unpacking: {bbox}")
            
#             try:
#                 if isinstance(bbox, list) and all(isinstance(b, np.ndarray) for b in bbox):
#                     for b in bbox:
#                         if len(b) == 4:
#                             x1, y1, x2, y2 = map(int, b.tolist())
#                         else:
#                             raise ValueError(f"Expected 4-element array, got {b}")
#                 elif isinstance(bbox, (np.ndarray, list)) and len(bbox) == 4:
#                     x1, y1, x2, y2 = map(int, bbox)  # Safely unpack
#                 else:
#                     raise ValueError(f"Expected 4-element bounding box, got {bbox}")

#                 # Draw the bounding box and track ID on the frame
#                 farbe = self.get_random_color()
#                 cv2.rectangle(frame, (x1, y1), (x2, y2), farbe, 2)
#                 cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

#             except Exception as e:
#                 raise ValueError(f"Error unpacking bounding box {bbox}: {str(e)}")

#         # Update display and process key events (e.g., pause, exit)
#         self.update_display(display_frame)
#         cv2.destroyAllWindows()  # Ensure that all windows are closed
#         cv2.waitKey(1)


    
    
#     def extract_features_from_result(self, result, frame):
#         """Extract boxes, masks, and features from the YOLO result."""
#         # Check the type of 'result' before proceeding
#         if not hasattr(result, 'boxes'):
#             print(f"Unexpected 'result' object type: {type(result)}")
#             return [], [], []
#         boxes = []
#         masks = []
        
#         for idx, cls in enumerate(result.boxes.cls):
#             cls = int(cls)
#             if cls == 0:  # Class 'person'
#                 # Extract bounding box
#                 x1, y1, x2, y2 = result.boxes.xyxy[idx].cpu().numpy().astype(int)
#                 boxes.append((x1, y1, x2, y2))

#                 # Extract mask
#                 if result.masks is not None and len(result.masks.data) > idx:
#                     mask = result.masks.data[idx].cpu().numpy()
#                     masks.append(mask)
#                 else:
#                     masks.append(None)
        
#         # Extract features using your existing logic
#         features = self.extract_reid_features_with_masks(frame, boxes, masks)
        
        
#         return boxes, masks, features

#     def preprocess_frame(self, frame, target_size=(416, 416)):
#         """Resize the frame to the target size for consistent inference."""
#         if isinstance(frame, np.ndarray):
#             # Resize the frame
#             resized_frame = cv2.resize(frame, target_size)

#             # Optional: Normalize the frame (e.g., to [0, 1] or standardization)
#             resized_frame = resized_frame.astype(np.float32) / 255.0

#             # You can add more preprocessing steps (e.g., mean subtraction, etc.)
#             return resized_frame
#         else:
#             raise TypeError(f"Expected frame to be a numpy array, but got {type(frame)}")


# def update_display(self, frame):
#         """Display the updated frame in the LabelMe UI."""
#         try:
#             # Convert from BGR (OpenCV) to RGB (Qt format)
#             rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             height, width, channel = rgb_frame.shape
#             bytes_per_line = 3 * width
#             q_img = QtGui.QImage(rgb_frame.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888)

#             # Convert QImage to QPixmap
#             pixmap = QtGui.QPixmap.fromImage(q_img)

#             # Load the pixmap into the canvas
#             if hasattr(self, 'canvas'):
#                 if self.canvas:
#                     self.canvas.loadPixmap(pixmap)
#                 else:
#                     print("Canvas is initialized but not available for display.")
#             else:
#                 print("Canvas attribute not found. Unable to update display.")

#         except Exception as e:
#             print(f"Error updating display: {str(e)}")


# def saveVideoAnnotations(self, video_name, all_bboxes, all_features, all_ids, all_confidences):
#         """Save bounding boxes, features, IDs, and confidences into a JSON file."""
        
#         # Debugging: Print lengths of all lists before processing
#         print(f"Number of bounding boxes: {len(all_bboxes)}")
#         print(f"Number of features: {len(all_features)}")
#         print(f"Number of IDs: {len(all_ids)}")
#         print(f"Number of confidences: {len(all_confidences)}")

#         if not (len(all_bboxes) == len(all_features) == len(all_ids) == len(all_confidences)):
#             raise ValueError("Mismatch in the number of bounding boxes, features, IDs, and confidences")

#         annotations = []
#         for i, person_id in enumerate(all_ids):
#             annotations.append({
#                 "id": person_id,
#                 "bbox": all_bboxes[i],
#                 "features": all_features[i].tolist(),  # Convert feature vector to list for saving
#                 "confidence": all_confidences[i]  # Add the confidence score
#             })
        
#         # Save annotations to a JSON file
#         with open(f"{video_name}_annotations.json", "w") as f:
#             json.dump(annotations, f)
        
#         print(f"Annotations saved to {video_name}_annotations.json")
#######################################################################################################################################################
    # """This code down in cooporates the batch processing.
    # """
#     def annotateVideo(self, batch_size=2):
#     """Annotate video frames with YOLO and FastReID."""
#     self.load_models()  # Load YOLO and FastReID models

#     if not hasattr(self, 'video_capture'):
#         QtWidgets.QMessageBox.warning(self, "Error", "No video loaded.")
#         return

#     frames = []
#     previous_features = []
#     batch_frames = []  # To accumulate frames for batch processing

#     while self.video_capture.isOpened():
#         ret, frame = self.video_capture.read()
#         if not ret:
#             break

#         # Add the current frame to the batch
#         batch_frames.append(frame)

#         # If batch size is reached, process the batch
#         if len(batch_frames) == batch_size:
#             # Process the batch
#             current_features = self.process_batch(batch_frames, previous_features)
#             previous_features = current_features

#             # Add labels to the label list dialog box
#             self.process_and_annotate(current_features, batch_frames)

#             # Clear batch
#             batch_frames = []

#         # If the last few frames (less than batch_size) are left, process them
#         if len(batch_frames) > 0:
#             current_features = self.process_batch(batch_frames, previous_features)
#             previous_features = current_features
#             self.process_and_annotate(current_features, batch_frames)

#             batch_frames = []  # Clear after processing the remaining frames

#     self.save_reid_annotations(self)  # Save annotations
#     self.video_capture.release()
#     cv2.destroyAllWindows()

# def process_batch(self, batch_frames, previous_features):
#     """Process a batch of frames."""
#     all_features = []
#     all_boxes = []
    
#     for frame in batch_frames:
#         # Run YOLO for person detection
#         results = self.yolo_model(frame)
#         boxes = []
#         for det in results[0].boxes:
#             if int(det.cls[0]) == 0:  # Class 0 is 'person'
#                 x1, y1, x2, y2 = map(int, det.xyxy[0].tolist())
#                 boxes.append((x1, y1, x2, y2))

#         # Extract ReID features for detected persons
#         features = self.extract_reid_features(frame, boxes)

#         all_features.append(features)
#         all_boxes.append(boxes)

#     return all_features, all_boxes

# def process_and_annotate(self, current_features, batch_frames):
#     """Process and annotate frames with bounding boxes and labels."""
#     for i, frame in enumerate(batch_frames):
#         # Example of processing features for the i-th frame in the batch
#         features, boxes = current_features[i]

#         # Match IDs across frames
#         ids = self.match_ids(features, previous_features)
        
#         # Annotate the frame with bounding boxes and IDs
#         for i, box in enumerate(boxes):
#             x1, y1, x2, y2 = map(int, box)
#             label_text = f"ID: {ids[i]}"
#             self.addLabel(label_text)

#             cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#             cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

#         # Display frame in the canvas
#         self.update_display(frame)
# ####################################################################################################
#     """Match_ID
    
#     """
    # def match_ids(self, current_features, previous_features, threshold=0.7):
    #     """
    #     Match current features to previous features based on cosine similarity.

    #     Args:
    #         current_features: List of tuples [(feature_vector, bbox), ...].
    #         previous_features: List of tuples [(feature_vector, bbox), ...].
    #         threshold: Minimum cosine similarity to consider a match.

    #     Returns:
    #         List of tuples [(current_idx, matched_idx)].
    #     """
    #     matches = []
    #     for i, (curr_feat, curr_bbox) in enumerate(current_features):
    #         best_match = -1
    #         best_similarity = threshold

    #         # Extract only the feature vector (ignore bounding box in similarity calculation)
    #         curr_feat = np.array(curr_feat).flatten()

    #         for j, (prev_feat, prev_bbox) in enumerate(previous_features):
    #             # Extract only the feature vector (ignore bounding box in similarity calculation)
    #             prev_feat = np.array(prev_feat).flatten()

    #             # Compute cosine similarity
    #             similarity = cosine_similarity([curr_feat], [prev_feat])[0][0]
    #             if similarity > best_similarity:
    #                 best_similarity = similarity
    #                 best_match = j

    #         # If no match found, assign a new ID
    #         matches.append((i, best_match if best_match != -1 else i))
    #     return matches
# #######################################################################################
#         """AnnotationVideo
        
#         """
        
        #  def annotateVideo(self, batch_size=2):
        # """Annotate video frames with YOLO, FastReID, and tracking."""
        # self.load_models()  # Load YOLO and FastReID models

        # if not hasattr(self, 'video_capture'):
        #     QtWidgets.QMessageBox.warning(self, "Error", "No video loaded.")
        #     return

        # frames = []
        # previous_features = []

        # while self.video_capture.isOpened():
        #     ret, frame = self.video_capture.read()
        #     if not ret:
        #         break

        #     # Run YOLO for person detection
        #     results = self.yolo_model(frame)
        #     boxes = []
        #     for det in results[0].boxes:
        #         if int(det.cls[0]) == 0:  # Class 0 is 'person'
        #             x1, y1, x2, y2 = map(int, det.xyxy[0].tolist())
        #             boxes.append((x1, y1, x2, y2))

        #     # Extract ReID features for detected persons
        #     current_features = self.extract_reid_features(frame, boxes)

        #     # Match IDs across frames using ReID
        #     ids = self.match_ids(current_features, previous_features)
        #     previous_features = current_features

        #     # Annotate the frame with bounding boxes and IDs
        #     for i, box in enumerate(boxes):
        #         x1, y1, x2, y2 = map(int, box)
        #         label_text = f"ID: {ids[i]}"

        #         # Create a shape object for labeling
        #         shape = Shape(label="person", shape_id=id(self))
        #         self.addLabel(shape)

        #         # Annotate frame with bounding box and ID
        #         x1, y1, x2, y2 = map(int, box)
        #         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        #         cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        #     # Add frame to list for batch processing
        #     frames.append(frame)

        #     # Display the annotated frame
        #     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #     height, width, channel = rgb_frame.shape
        #     bytes_per_line = channel * width
        #     q_img = QtGui.QImage(rgb_frame.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888)
        #     self.canvas.loadPixmap(QtGui.QPixmap.fromImage(q_img))
        #     self.setClean()
        #     self.canvas.update()
        #     self.repaint()

        # # Save annotated frames as JSON
        # self.save_reid_annotations(frames, boxes, ids)
        # self.video_capture.release()
        # cv2.destroyAllWindows()
        
        #     """
        #     def annotateVideo(self):
        # """Annotate video with YOLO, FastReID, and tracking."""
        # self.load_models()

        # frames = []
        # previous_features = []
        # unique_id = 1  # Initialize unique ID counter

        # while self.video_capture.isOpened():
        #     ret, frame = self.video_capture.read()
        #     if not ret:
        #         break
            
        #     # Run YOLO for person detection
        #     results = self.yolo_model(frame)
        #     boxes = []
        #     confidences=[]
        #     for det in results[0].boxes:
        #         if int(det.cls[0]) == 0:  # Class 0 is 'person'
        #             x1, y1, x2, y2 = map(int, det.xyxy[0].tolist())
        #             boxes.append((x1, y1, x2, y2))
        #             confidences.append(float(det.conf[0]))# confidence score

            

        #     # Extract ReID features for detected persons
        #     current_features = self.extract_reid_features(frame, boxes)

        #     # Match IDs across frames using ReID
        #     ids = self.match_ids(current_features, previous_features)
        #     previous_features = current_features

        #     # Use DeepSORT for multi-frame tracking
        #     detections = []  # Convert box to a list and append confidence score
        #     embeds = []  # Store the flattened feature vectors
        #     # confidences = []  # Store confidence scores

        #     for box, confidence, feature in zip(boxes, confidences, current_features):
        #         bbox = box  # (x1, y1, x2, y2)
        #         feature_vector = np.array(feature[0]).flatten()  # Flatten the feature vector
        #         detections.append((bbox, confidence))  # Append the bounding box and confidence to detections
        #         embeds.append(feature_vector)  # Append the flattened feature vector to embeds

        #     # DeepSORT update
        #     tracks = self.deepsort.update_tracks(raw_detections=detections, embeds=embeds)

        #     # Annotate the frame with bounding boxes, IDs, and confidence
        #     for track, (bbox, confidence) in zip(tracks, detections):
        #         track_id = track.track_id
        #         x1, y1, x2, y2 = track.to_tlbr()  # Track coordinates (bounding box)

        #         # Draw the bounding box and label with the ID and confidence score
        #         label_text = f"ID: {track_id}, Conf: {confidence:.2f}"  # Use confidence from detections
        #         cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        #         cv2.putText(frame, label_text, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        #         # Create a shape object for labeling
        #         shape = Shape(label="person", shape_id=track_id)
        #         self.addLabel(shape)  # Add label to the label list dialog box

        #     frames.append(frame)
        #     # Update display (canvas in LabelMe UI)
        #     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #     height, width, channel = rgb_frame.shape
        #     bytes_per_line = channel * width
        #     q_img = QtGui.QImage(rgb_frame.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888)
        #     self.canvas.loadPixmap(QtGui.QPixmap.fromImage(q_img))

        # # Save annotated frames as JSON
        # self.save_reid_annotations(frames, boxes, ids)
        # self.video_capture.release()
        # cv2.destroyAllWindows()
        #     """
        
        
        # new_code for the PrevFrame    
         # Add buttons for video navigation
    #     self.prevFrameButton = QtWidgets.QPushButton(self.tr("Previous Frame"), self)
    #     self.nextFrameButton = QtWidgets.QPushButton(self.tr("Next Frame"), self)
    #     self.annotateVideoButton = QtWidgets.QPushButton(self.tr("Annotate Video"), self)
       

        

    #     # Define frameNumberInput here
    #     self.frameNumberInput = QtWidgets.QLineEdit(self)  # Define QLineEdit for frame input
    #     self.frameNumberInput.setPlaceholderText("Enter Frame #")  # Optional: Add placeholder text
        
    #     # Connect buttons to frame navigation
    #     self.annotateVideoButton.clicked.connect(self.annotateVideo)
    #     self.annotateVideoButton.clicked.connect(self.enable_save_annotation_button)

    #     """
    #       This layout manages frame navigation buttons like "Previous Frame", "Next Frame", and the input field for entering a frame number.
    #     """  
    #     # Create a layout for frame control
    #     frameControlLayout = QtWidgets.QHBoxLayout()
    #     frameControlLayout.addWidget(self.prevFrameButton)
    #     frameControlLayout.addWidget(self.frameNumberInput)
    #     frameControlLayout.addWidget(self.nextFrameButton)
       
        
        
        
    #     #Wrap frame Control in a widget
    #     self.frameControlWidget = QtWidgets.QWidget(self)
    #     self.frameControlWidget.setLayout(frameControlLayout)
        
    #     # Wrap the QWidget inside a QDockWidget for frame controls
    #     frameControlDock = QtWidgets.QDockWidget(self.tr("Frame Control"), self)
    #     frameControlDock.setObjectName("frameControlDock")  # Set object name
    #     frameControlDock.setWidget(self.frameControlWidget)
    #     self.addDockWidget(Qt.BottomDockWidgetArea, frameControlDock)
       
    #     """ This layout is added specifically for video-related buttons (in this case, the "Annotate Video" butto).
    #     """    
    
        
        
    #     """Both layouts are placed into separate widgets (frameControlWidget and videoControlWidget) and added to the QMainWindow using addDockWidget.
    #     """
        
    #     # Video control widget for annotation button
    #     self.videoControlWidget = QtWidgets.QWidget(self)
    #     videoControlLayout = QtWidgets.QHBoxLayout()
    #     self.annotateVideoButton = QtWidgets.QPushButton(self.tr("Annotate Video"), self)
    #     videoControlLayout.addWidget(self.annotateVideoButton)
    #     self.videoControlWidget.setLayout(videoControlLayout)
        
        
    #     # Wrap the QWidget inside a QDockWidget for video controls
    #     videoControlDock = QtWidgets.QDockWidget(self.tr("Video Control"), self)
    #     videoControlDock.setObjectName("VideoControlDock")#Set object name
    #     videoControlDock.setWidget(self.videoControlWidget)
    #     self.addDockWidget(QtCore.Qt.BottomDockWidgetArea, videoControlDock)

    #    # Define a widget for video controls
    #     self.videoControlWidget = QtWidgets.QWidget(self)
    #     self.videoControlsLayout = QtWidgets.QHBoxLayout(self.videoControlWidget)

    #     # Add Play, Pause, and Stop buttons
    #     self.playVideoButton = QtWidgets.QPushButton("Play", self)
    #     self.pauseVideoButton = QtWidgets.QPushButton("Pause", self)
    #     self.stopVideoButton = QtWidgets.QPushButton("Stop", self)

    #     # Add buttons to the layout
    #     self.videoControlsLayout.addWidget(self.playVideoButton)
    #     self.videoControlsLayout.addWidget(self.pauseVideoButton)
    #     self.videoControlsLayout.addWidget(self.stopVideoButton)

    #     # Connect buttons to their functionalities
    #     self.playVideoButton.clicked.connect(self.playVideo)
    #     self.pauseVideoButton.clicked.connect(self.pauseVideo)
    #     self.stopVideoButton.clicked.connect(self.stopVideo)

    #     # Add videoControlWidget to the main layout
    #     self.centralWidget = QtWidgets.QWidget(self)
    #     self.mainLayout = QtWidgets.QVBoxLayout(self.centralWidget)
    #     self.mainLayout.addWidget(self.videoControlWidget)
    #     self.setCentralWidget(self.centralWidget)
        
    #     # Initially disable buttons
    #     self.playVideoButton.setEnabled(False)
    #     self.pauseVideoButton.setEnabled(False)
    #     self.stopVideoButton.setEnabled(False)
    
#     To include these enhanced functions into your current LabelMe implementation, follow these steps. The enhancements will ensure dynamic and responsive interaction with bounding boxes, video frames, and annotations. Below is the modified code with the enhancements integrated:

# Enhanced Functions Integrated
# python
# Copy code
# def display_reid_detections(self, detections):
#     """Display ReID detections on the frame."""
#     if not hasattr(self, 'canvas') or not self.canvas:
#         print("Canvas is not available to display detections")
#         return

#     painter = QtGui.QPainter(self.canvas.pixmap())
#     for detection in detections:
#         # Handle detection formats (tensor, dictionary, etc.)
#         if isinstance(detection, torch.Tensor):
#             detection_list = detection.tolist()
#             if len(detection_list) >= 4:
#                 x1, y1, x2, y2 = map(int, detection_list[:4])
#             else:
#                 continue
#         elif isinstance(detection, dict) and 'bbox' in detection:
#             x1, y1, x2, y2 = map(int, detection['bbox'])
#         else:
#             continue

#         # Draw bounding box on the frame
#         painter.setPen(QtGui.QPen(QtGui.QColor(0, 255, 0), 2))
#         painter.drawRect(x1, y1, x2 - x1, y2 - y1)

#         # Optional: Display additional information (confidence, ID)
#         if 'confidence' in detection:
#             confidence = detection['confidence']
#             painter.drawText(x1, y1 - 10, f"Conf: {confidence:.2f}")

#     painter.end()
#     self.canvas.update()  # Refresh the canvas to show updated annotations


# def update_display(self, frame):
#     """Display the updated frame in the LabelMe UI."""
#     if frame is None:
#         print("No frame available for display")
#         return

#     # Convert OpenCV BGR image to RGB
#     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     height, width, channel = rgb_frame.shape
#     bytes_per_line = channel * width

#     # Create a QImage and load it into the canvas
#     q_img = QtGui.QImage(rgb_frame.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888)
#     pixmap = QtGui.QPixmap.fromImage(q_img)

#     if hasattr(self, 'canvas') and self.canvas:
#         self.canvas.loadPixmap(pixmap)
#     else:
#         print("Canvas is not initialized or available")


# def closeVideo(self):
#     """Release video resources and reset related UI components."""
#     if hasattr(self, 'video_capture') and self.video_capture.isOpened():
#         self.video_capture.release()
#         print("Video capture resources released.")

#     # Reset related UI elements
#     self.actions.openPrevFrame.setEnabled(False)
#     self.actions.openNextFrame.setEnabled(False)
#     self.actions.AnnotateVideo.setEnabled(False)
#     print("Video-related UI components disabled")


# def load_annotations(self, video_name):
#     """Load previously saved annotations for a video."""
#     try:
#         with open(f"{video_name}_annotations.json", "r") as f:
#             annotations = json.load(f)
#         print(f"Annotations loaded for video: {video_name}")
#         return annotations
#     except FileNotFoundError:
#         print(f"No saved annotations found for video: {video_name}")
#         return []
#     except json.JSONDecodeError:
#         print(f"Error decoding annotation file for video: {video_name}")
#         return []

# Steps to Integrate
# Add Functions to the Codebase: Copy and paste the above code into your LabelMe application's main script, ensuring the functions are under the class that handles UI interactions.

# Enhancements in annotateVideo: Update the annotateVideo method to call display_reid_detections and update_display after each frame is processed. For example:

# python
# Copy code
# detections = []  # Replace this with actual detection data
# self.display_reid_detections(detections)  # Display bounding boxes on the frame
# self.update_display(frame)  # Update the UI canvas
# UI Consistency: Modify closeVideo to ensure that all video-related UI components are properly reset when a video is closed. This prevents UI inconsistencies.

# Annotation Loading: Use load_annotations when a user loads a video to check for any previously saved annotations and load them into the system.

# Key Enhancements
# Real-Time Visualization: The display_reid_detections function ensures that users get immediate visual feedback, helping them validate the detection process.

# Session Continuity: The load_annotations function allows users to resume work on previously saved videos without redoing the annotation process.

# Error-Free Experience: closeVideo ensures that system resources are managed effectively and prevents issues like memory leaks or UI glitches.

# User-Friendly Display: The update_display function dynamically refreshes the UI, providing a seamless experience when navigating through frames.

# Testing
# Test the enhanced functions in isolation to ensure they work as expected.
# Integrate them into the full application and test the workflow:
# Load a video.
# Annotate a few frames.
# Save and reload annotations.
# Check if bounding boxes and labels appear correctly on the UI.
# Next Steps
# Once the integration is complete, focus on optimizing other areas, such as:

# Export Formats: Allow users to save annotations in multiple formats (e.g., JSON, XML, COCO, Pascal VOC).
# Bounding Box Refinement: Add options for users to refine or edit bounding boxes manually.
# Real-Time Confidence Threshold: Allow users to adjust the detection confidence threshold dynamically.
# Let me know if you need further help with integration or enhancements!

###########################################################################################################################################

    # def addLabel(self, shape):
    #     """
    #     Add a label to the Polygon Labels (labelList) and ensure it's unique.
    #     """
    #     # Add label to Polygon Labels
    #     label_text = f"{shape.label} - ID: {shape.shape_id}"  # Label with ID

    #     # Manually check for existing items
    #     existing_items = [
    #         self.labelList.item(i) 
    #         for i in range(self.labelList.count()) 
    #         if self.labelList.item(i).text() == label_text
    #     ]

    #     if not existing_items:
    #         # Add to Polygon Labels
    #         label_item = QtWidgets.QListWidgetItem(label_text)
    #         color = self.get_color_for_id(shape.shape_id)  # Assign unique color
    #         label_item.setBackground(QtGui.QColor(*color))  # Set background color
    #         self.labelList.addItem(label_item)

    #     # Add label to Unique Label List
    #     if self.uniqLabelList.findItemByLabel(shape.label) is None:
    #         uniq_item = self.uniqLabelList.createItemFromLabel(shape.label)
    #         self.uniqLabelList.addItem(uniq_item)
    #         rgb = self._get_rgb_by_label(shape.label)
    #         self.uniqLabelList.setItemLabel(uniq_item, shape.label, rgb)

    #     # Enable shape-related actions
    #     for action in self.actions.onShapesPresent:
    #         action.setEnabled(True)

    #     # Add label history for the dialog box
    #     self.labelDialog.addLabelHistory(shape.label)
    
    # def annotateVideo(self):
        
       
    #     """Annotate video with YOLO, FastReID, and DeepSORT."""
    #     self.load_models()

    #     frames = []
    #     all_annotations = []

    #     # Initialize the Tracker
    #     max_cosine_distance = 0.5
    #     max_age = 30
    #     n_init = 3
    #     metric = NearestNeighborDistanceMetric("cosine", max_cosine_distance, None)
    #     if not hasattr(self, 'tracker'):
    #         self.tracker = Tracker(metric=metric, max_age=max_age, n_init=n_init)

    #     person_colors = {}
    #     if not hasattr(self, 'track_id_manager'):
    #         self.track_id_manager = IDManager()

    #     # Set the expected embedding size (based on the model)
    #     if not hasattr(self, 'expected_embedding_size'):
    #         self.expected_embedding_size =2048   # Replace with your model's actual output size

    #     while self.video_capture.isOpened():
    #         ret, frame = self.video_capture.read()
    #         if not ret:
    #             break

    #         # Step 1: Run YOLO for person detection
    #         results = self.yolo_model(frame)
    #         boxes, confidences = [], []
    #         for det in results[0].boxes:
    #             if int(det.cls[0]) == 0 and float(det.conf[0]) > 0.5:
    #                 x1, y1, x2, y2 = map(int, det.xyxy[0].tolist())
    #                 boxes.append((x1, y1, x2, y2))
    #                 confidences.append(float(det.conf[0]))

    #         # Step 2: Extract ReID features
    #         current_features = self.extract_reid_features(frame, boxes)
    #         detections = []
    #         frame_height, frame_width, _ = frame.shape

    #         # Prepare detections
    #         for box, confidence, feature in zip(boxes, confidences, current_features):
    #             if isinstance(feature, tuple):
    #                 feature = feature[0]  # Extract feature vector if it's a tuple
    #             try:
    #                 feature_vector = np.array(feature).flatten()
    #                 # Validate feature size
    #                 if len(feature_vector) != self.expected_embedding_size:
    #                     raise ValueError(
    #                         f"Invalid feature size: Expected {self.expected_embedding_size}, got {len(feature_vector)}"
    #                     )

    #                 # Validate and normalize bounding box
    #                 x1, y1, x2, y2 = box
    #                 if not (0 <= x1 < x2 <= frame_width and 0 <= y1 < y2 <= frame_height):
    #                     print(f"Skipping invalid bounding box: {box}")
    #                     continue

    #                 bbox_tuple = self.normalize_bbox(box)
    #                 detection = Detection(bbox_tuple, confidence, feature_vector)
    #                 detections.append(detection)
    #             except Exception as e:
    #                 print(f"Error preparing detection for box {box}: {e}")

    #         # Step 3: Update the tracker
    #         try:
    #             self.tracker.predict()
    #             self.tracker.update(detections=detections)  # Pass the list of detections
    #         except Exception as e:
    #             print(f"Error during tracker update: {e}")



            
    #         frame_annotations = []
    #         for track in self.tracker.tracks:
    #             # Skip unconfirmed tracks or tracks that haven't been updated recently
    #             if not track.is_confirmed() or track.time_since_update > 1:
    #                 continue

    #             # Get track ID and bounding box
    #             track_id = track.track_id
    #             bbox = track.to_tlbr()  # Bounding box in (x1, y1, x2, y2) format

    #             # Assign unique colors to each track ID
    #             if track_id not in person_colors:
    #                 person_colors[track_id] = self.get_random_color()
    #             color = person_colors[track_id]

    #             # Annotate the frame with bounding box and label
    #             label_text = f"Person ID: {track_id}"
    #             cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
    #             cv2.putText(frame, label_text, (int(bbox[0]), int(bbox[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    #             # Match detection for confidence (optional, depending on detection-tracking pipeline)
    #             matched_detection = None
    #             for detection in detections:
    #                 if self.iou(detection.to_tlbr(), bbox) > 0.3:  # IoU threshold
    #                     matched_detection = detection
    #                     break
    #             detection_confidence = matched_detection.confidence if matched_detection else "No detection"

    #             # Debugging output
    #             print(f"Track ID: {track_id}, BBox: {bbox}, Confidence: {detection_confidence}")

    #             # Create a Shape object for LabelMe annotation
    #             shape = Shape(label=f"person{track_id}", shape_id=track_id)
    #             shape.addPoint(QtCore.QPointF(bbox[0], bbox[1]))
    #             shape.addPoint(QtCore.QPointF(bbox[2], bbox[1]))
    #             shape.addPoint(QtCore.QPointF(bbox[2], bbox[3]))
    #             shape.addPoint(QtCore.QPointF(bbox[0], bbox[3]))
    #             self.addLabel(shape)

    #             # Update label lists
    #             self.labelList.addPersonLabel(track_id, color)  # Update Label List
    #             self.uniqLabelList.addUniquePersonLabel(f"person{track_id}", color)  # Update Unique Label List

    #             # Append annotations for the current frame
    #             frame_annotations.append({
    #                 "track_id": track_id,
    #                 "bbox": list(bbox),
    #                 "confidence": detection_confidence,
    #                 "class": "person"
    #             })

    #         all_annotations.append(frame_annotations)
    #         frames.append(frame)

    #         # Update display
    #         rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #         height, width, channel = rgb_frame.shape
    #         bytes_per_line = channel * width
    #         q_img = QtGui.QImage(rgb_frame.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888)
    #         pixmap = QtGui.QPixmap.fromImage(q_img)
    #         self.canvas.loadPixmap(pixmap)
    #         self.canvas.update()
    #     self.save_reid_annotations(frames, all_annotations)
    #     self.video_capture.release()
    #     cv2.destroyAllWindows()
# file=open("Extra_Code_files\detections_output.txt")
# file.read()
# file.close()

# def annotateVideo(self):
#     """Annotate video with YOLO, FastReID, and DeepSORT."""
#     self.load_models()

#     frames = []
#     all_annotations = []

#     max_cosine_distance = 0.5
#     max_age = 30
#     n_init = 3
#     metric = NearestNeighborDistanceMetric("cosine", max_cosine_distance, None)
#     if not hasattr(self, 'tracker'):
#         self.tracker = Tracker(metric=metric, max_age=max_age, n_init=n_init)

#     person_colors = {}
#     if not hasattr(self, 'track_id_manager'):
#         self.track_id_manager = IDManager()

#     while self.video_capture.isOpened():
#         ret, frame = self.video_capture.read()
#         if not ret:
#             break

#         # Step 1: Run YOLO for person detection
#         results = self.yolo_model(frame)
#         boxes, confidences = [], []
#         for det in results[0].boxes:
#             if int(det.cls[0]) == 0 and float(det.conf[0]) > 0.5:
#                 x1, y1, x2, y2 = map(int, det.xyxy[0].tolist())
#                 boxes.append((x1, y1, x2, y2))
#                 confidences.append(float(det.conf[0]))

#         # Step 2: Extract ReID features
#         current_features = self.extract_reid_features(frame, boxes)
#         detections = []
#         frame_height, frame_width, _ = frame.shape

#         for box, confidence, feature in zip(boxes, confidences, current_features):
#             if isinstance(feature, tuple):
#                 feature = feature[0]
#             feature_vector = np.array(feature).flatten()
#             detection = Detection(box, confidence, feature_vector)
#             detections.append(detection)

#         # Update the tracker
#         self.tracker.predict()
#         self.tracker.update(detections)

#         frame_annotations = []
#         for track in self.tracker.tracks:
#             if not track.is_confirmed() or track.time_since_update > 1:
#                 continue

#             # Reuse or assign a new track ID
#             if track.track_id not in self.track_id_manager.used_ids:
#                 track.track_id = self.track_id_manager.get_new_id()

#             track_id = track.track_id
#             bbox = track.to_tlbr()  # Bounding box
#             bbox_tuple = self.normalize_bbox(bbox)

#             # Assign unique colors
#             if track_id not in person_colors:
#                 person_colors[track_id] = self.get_random_color()
#             color = person_colors[track_id]

#             # Annotate frame
#             label_text = f"Person ID: {track_id}"
#             cv2.rectangle(frame, (bbox_tuple[0], bbox_tuple[1]), (bbox_tuple[2], bbox_tuple[3]), color, 2)
#             cv2.putText(frame, label_text, (bbox_tuple[0], bbox_tuple[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

#             # Save annotations
#             frame_annotations.append({
#                 "track_id": track_id,
#                 "bbox": list(bbox_tuple),
#                 "confidence": track.confidence,  # Assuming tracker stores confidence
#                 "class": "person"
#             })

#         all_annotations.append(frame_annotations)
#         frames.append(frame)

#         # Display frame
#         rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         height, width, channel = rgb_frame.shape
#         bytes_per_line = channel * width
#         q_img = QtGui.QImage(rgb_frame.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888)
#         pixmap = QtGui.QPixmap.fromImage(q_img)
#         self.canvas.loadPixmap(pixmap)
#         self.canvas.update()

#     # Save annotations to JSON
#     self.save_reid_annotations(frames, all_annotations)
#     self.video_capture.release()
#     cv2.destroyAllWindows()


# def annotateVideo(self):
#     """Annotate video with YOLO, FastReID, and DeepSORT."""
#     self.load_models()

#     frames = []
#     all_annotations = []

#     # Initialize the Tracker
#     max_cosine_distance = 0.5
#     max_age = 30
#     n_init = 3
#     metric = NearestNeighborDistanceMetric("cosine", max_cosine_distance, None)
#     if not hasattr(self, 'tracker'):
#         self.tracker = Tracker(metric=metric, max_age=max_age, n_init=n_init)

#     person_colors = {}
#     if not hasattr(self, 'track_id_manager'):
#         self.track_id_manager = IDManager()

#     # Set the expected embedding size (based on the model)
#     if not hasattr(self, 'expected_embedding_size'):
#         self.expected_embedding_size = 2048  # Replace with your model's actual output size

#     while self.video_capture.isOpened():
#         ret, frame = self.video_capture.read()
#         if not ret:
#             break

#         # Step 1: Run YOLO for person detection
#         results = self.yolo_model(frame)
#         boxes, confidences = [], []
#         for det in results[0].boxes:
#             if int(det.cls[0]) == 0 and float(det.conf[0]) > 0.5:
#                 x1, y1, x2, y2 = map(int, det.xyxy[0].tolist())
#                 boxes.append((x1, y1, x2, y2))
#                 confidences.append(float(det.conf[0]))

#         # Step 2: Extract ReID features
#         current_features = self.extract_reid_features(frame, boxes)
#         detections = []
#         frame_height, frame_width, _ = frame.shape

#         # Prepare detections
#         for box, confidence, feature in zip(boxes, confidences, current_features):
#             if isinstance(feature, tuple):
#                 feature = feature[0]  # Extract feature vector if it's a tuple
#             try:
#                 feature_vector = np.array(feature).flatten()
#                 # Validate feature size
#                 if len(feature_vector) != self.expected_embedding_size:
#                     raise ValueError(
#                         f"Invalid feature size: Expected {self.expected_embedding_size}, got {len(feature_vector)}"
#                     )

#                 # Validate and normalize bounding box
#                 x1, y1, x2, y2 = box
#                 if not (0 <= x1 < x2 <= frame_width and 0 <= y1 < y2 <= frame_height):
#                     print(f"Skipping invalid bounding box: {box}")
#                     continue

#                 bbox_tuple = self.normalize_bbox(box)
#                 detection = Detection(bbox_tuple, confidence, feature_vector)
#                 detections.append(detection)
#             except Exception as e:
#                 print(f"Error preparing detection for box {box}: {e}")

#         # Step 3: Update the tracker
#         try:
#             self.tracker.predict()
#             self.tracker.update(detections=detections)  # Pass the list of detections
#         except Exception as e:
#             print(f"Error during tracker update: {e}")

#         frame_annotations = []
#         for track in self.tracker.tracks:
#             # Skip unconfirmed tracks or tracks that haven't been updated recently
#             if not track.is_confirmed() or track.time_since_update > 1:
#                 continue

#             # Get track ID and bounding box
#             track_id = track.track_id
#             bbox = track.to_tlbr()  # Bounding box in (x1, y1, x2, y2) format

#             # Assign unique colors to each track ID
#             if track_id not in person_colors:
#                 person_colors[track_id] = self.get_random_color()
#             color = person_colors[track_id]

#             # Annotate the frame with bounding box and label
#             label_text = f"Person ID: {track_id}"
#             cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
#             cv2.putText(frame, label_text, (int(bbox[0]), int(bbox[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

#             # Match detection for confidence (optional, depending on detection-tracking pipeline)
#             matched_detection = None
#             for detection in detections:
#                 if self.iou(detection.to_tlbr(), bbox) > 0.3:  # IoU threshold
#                     matched_detection = detection
#                     break
#             detection_confidence = matched_detection.confidence if matched_detection else "No detection"

#             # Debugging output
#             print(f"Track ID: {track_id}, BBox: {bbox}, Confidence: {detection_confidence}")

#             # Create a Shape object for LabelMe annotation
#             shape = Shape(label=f"person{track_id}", shape_id=track_id)
#             shape.addPoint(QtCore.QPointF(bbox[0], bbox[1]))
#             shape.addPoint(QtCore.QPointF(bbox[2], bbox[1]))
#             shape.addPoint(QtCore.QPointF(bbox[2], bbox[3]))
#             shape.addPoint(QtCore.QPointF(bbox[0], bbox[3]))
#             self.addLabel(shape)

#             # Update label lists
#             self.labelList.addPersonLabel(track_id, color)  # Update Label List
#             self.uniqLabelList.addUniquePersonLabel(f"person{track_id}", color)  # Update Unique Label List

#             # Append annotations for the current frame
#             frame_annotations.append({
#                 "track_id": track_id,
#                 "bbox": list(bbox),
#                 "confidence": detection_confidence,
#                 "class": "person"
#             })

#         all_annotations.append(frame_annotations)
#         frames.append(frame)

#         # Update display
#         rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         height, width, channel = rgb_frame.shape
#         bytes_per_line = channel * width
#         q_img = QtGui.QImage(rgb_frame.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888)
#         pixmap = QtGui.QPixmap.fromImage(q_img)
#         self.canvas.loadPixmap(pixmap)
#         self.canvas.update()

#     # Save annotations to JSON
#     self.save_reid_annotations(frames, all_annotations)
#     self.video_capture.release()
#     cv2.destroyAllWindows()

# # Step 4: Ask the user for the desired annotation format
#         format_choice = self.choose_annotation_format()
#         if format_choice:
#             self.save_reid_annotations(frames, all_annotations, format_choice)
#             self.actions.saveReIDAnnotations.setEnabled(True) 
#         else:
#             print("Annotation saving canceled by user.")

# def shapeSelectionChanged(self, selected_shapes):
#         self._noSelectionSlot = True

#         # Debug: Check the type and contents of selectedShapes
#         print(f"Type of selectedShapes: {type(self.canvas.selectedShapes)}")
#         print(f"Contents of selectedShapes: {self.canvas.selectedShapes}")

#         # Clear previous selection
#         for shape in self.canvas.selectedShapes:
#             if isinstance(shape, Shape):
#                 shape.selected = False
#             else:
#                 print(f"Warning: Found invalid shape object: {shape}")

#         self.labelList.clearSelection()

#         # Check if selectedShapes is defined and valid
#         if not hasattr(self.canvas, 'selectedShapes') or not isinstance(self.canvas.selectedShapes, list):
#             print("Warning: selectedShapes is not a list or undefined. Resetting to an empty list.")
#             self.canvas.selectedShapes = []

#         # Update selection
#         for shape in self.canvas.selectedShapes:
#             if isinstance(shape, Shape):
#                 shape.selected = True
#                 item = self.labelList.findItemByShape(shape)
#                 self.labelList.selectItemByShape(item)
#                 self.labelList.scrollToItem(item)
#             else:
#                 print(f"Skipping invalid shape object: {shape}")

#         self._noSelectionSlot = False

#         # Update action buttons
#         n_selected = len(selected_shapes)
#         self.actions.delete.setEnabled(n_selected)
#         self.actions.duplicate.setEnabled(n_selected)
#         self.actions.copy.setEnabled(n_selected)
#         self.actions.edit.setEnabled(n_selected)

# def annotateVideo(self):
        
       
#         """Annotate video with YOLO, FastReID, and DeepSORT."""
#         self.load_models()
#         self.track_id_manager = ImprovedIDManager()
#         frames = []
#         all_annotations = []

#         # Initialize the Tracker
#         max_cosine_distance = 0.5
#         max_age = 30
#         n_init = 3
#         metric = NearestNeighborDistanceMetric("cosine", max_cosine_distance)
#         if not hasattr(self, 'tracker'):
#             self.tracker = Tracker(metric=metric, max_age=max_age, n_init=n_init,max_iou_distance=0.7)

#         person_colors = {}
        
            

#         # Set the expected embedding size (based on the model)
#         if not hasattr(self, 'expected_embedding_size'):
#             self.expected_embedding_size =2048   # Replace with your model's actual output size

#         #Process video frame by frame
#         while self.video_capture.isOpened():
#             ret, frame = self.video_capture.read()
#             if not ret:
#                 print("End of video or no frame available.")
#                 break

#             # Step 1: Run YOLO for person detection
#             results = self.yolo_model(frame)
#             boxes, confidences = [], []
#             for det in results[0].boxes:
#                 if int(det.cls[0]) == 0 and float(det.conf[0]) > 0.5:  # Class 0 is person
#                     x1, y1, x2, y2 = map(int, det.xyxy[0].tolist())
#                     if not (0 <= x1 < x2 <= frame.shape[1] and 0 <= y1 < y2 <= frame.shape[0]):
#                         print(f"Skipping invalid YOLO box: {[x1, y1, x2, y2]}")
#                         continue
#                     boxes.append((x1, y1, x2, y2))
#                     confidences.append(float(det.conf[0]))
#                 print(f"Raw YOLO output box: {det.xyxy[0].tolist()}")
#                 print("Step 1 ends")

#             # Step 2: Extract ReID features for detected boxes
#             current_features = self.extract_reid_features(frame, boxes)
#             for i, feature in enumerate(current_features):
#                 if feature is None or not isinstance(feature, np.ndarray) or len(feature) != self.expected_embedding_size:
#                     print(f"Invalid feature for bounding box {boxes[i]}: {feature}")
#                     continue
#             detections = []
#             frame_height, frame_width, _ = frame.shape
#             print(f"Frame dimensions: Width={frame_width}, Height={frame_height}")
#             print(f"Bounding box: {boxes}")

#             for box, confidence, feature in zip(boxes, confidences, current_features):
#                 # Validate bounding box dimensions
#                 if not (0 <= box[0] < box[2] <= frame_width and 0 <= box[1] < box[3] <= frame_height):
#                     print(f"Skipping invalid YOLO bounding box: {box}")
#                     continue

#                 try:
#                     # Extract and flatten feature vector
#                     if isinstance(feature, tuple) or isinstance(feature, list):
#                         feature_vector = np.array(feature[0]).flatten()
#                     else:
#                         feature_vector = np.array(feature).flatten()

#                     # Validate feature size
#                     if len(np.array(feature[0]).flatten()) != self.expected_embedding_size:
#                         print(f"Invalid feature size: {len(feature_vector)} for box {box}")
#                         continue

#                     # Normalize bounding box
#                     bbox_tuple = (
#                         box[0] / frame_width,
#                         box[1] / frame_height,
#                         box[2] / frame_width,
#                         box[3] / frame_height,
#                     )

#                     # Create detection object
#                     detection = Detection(bbox_tuple, confidence, feature_vector)
#                     detections.append(detection)
#                     valid_detections=[]
#                     if (detection is not None and 
#                         len(detection.feature) == self.expected_embedding_size and 
#                         np.isfinite(detection.feature).all() and 
#                         0 <= detection.confidence <= 1):
#                         valid_detections.append(detection)
#                         print(f"Valid detection added: {detection}")
#                     else:
#                         print(f"Invalid detection filtered out: {detection}")

#                 except Exception as e:
#                     print(f"Error processing detection for box {box}: {e}")
#                     print(f"Error validating detection: {e}, Detection: {detection}")

#             print("Step 2 ends")
# ##################################################################################################################################
#              # Step 3: Update the tracker
#             if detections:
#                 try:
#                     print(f"Updating tracker with {len(detections)} detections.")
#                     self.tracker.predict()
#                     self.tracker.update(detections=detections)
#                     # print(self.tracker.update(detections=detections))# Pass the list of detections
#                 except Exception as e:
#                     print(f"Error during tracker update: {e}")
#             else:
#                 print("No valid detections for this frame, skipping tracker update.")

            
#             # Step 4: Process and annotate tracks
#             frame_annotations = []

#             for track in self.tracker.tracks:
#                 # Skip unconfirmed tracks or tracks that haven't been updated recently
#                 if not track.is_confirmed() or track.time_since_update > 1:
#                     print(f"Unconfirmed track, skipping: {track}")
#                     print(f"Releasing ID for unconfirmed track: {track.track_id}")
#                     track.track_id = self.track_id_manager.match_or_create_id(
#                         current_feature=track.get_feature(),
#                         similarity_threshold=0.7)
#                     continue
                    
                   

#                 # # Assign or create an ID for the track
               
#                 # if hasattr(track, 'get_feature') and track.get_feature() is not None:
#                 #     track_id = self.track_id_manager.match_or_create_id(
#                 #         current_feature=track.get_feature(),
#                 #         similarity_threshold=0.7
#                 #     )
#                 #     continue
                    
                    
                
                    

#                 track_id=track.track_id

#                 # Log the track details
#                 bbox = track.to_tlbr()
#                 print(f"Track ID: {track.track_id}, Bounding box: {bbox}")

            
                
                
                
#                 # if not (0 <= bbox[0] < bbox[2] <= frame_width and 0 <= bbox[1] < bbox[3] <= frame_height):
#                 #     print(f"Invalid bounding box for Track ID {track_id}: {bbox}")
#                 #     continue

#                 # Assign unique colors
#                 if track_id not in person_colors:
#                     person_colors[track_id] = self.get_random_color()
#                 color = person_colors[track_id]

#                 # Annotate the frame with bounding box and label
#                 label_text = f"Person ID: {track_id}"
#                 cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
#                 cv2.putText(frame, label_text, (int(bbox[0]), int(bbox[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

#                 # Create a Shape object for LabelMe annotation
#                 shape = Shape(label=f"person{track_id}", shape_id=track_id)
#                 shape.addPoint(QtCore.QPointF(bbox[0], bbox[1]))
#                 shape.addPoint(QtCore.QPointF(bbox[2], bbox[1]))
#                 shape.addPoint(QtCore.QPointF(bbox[2], bbox[3]))
#                 shape.addPoint(QtCore.QPointF(bbox[0], bbox[3]))
#                 self.addLabel(shape)

#                 # Update label lists
#                 self.labelList.addPersonLabel(track_id, color)  # Update Label List
#                 self.uniqLabelList.addUniquePersonLabel(f"person{track_id}", color)  # Update Unique Label List

#                     # Append annotations for the current frame
#                 frame_annotations.append({
#                     "track_id": track_id,
#                     "bbox": [x1, y1, x2, y2],
#                     "confidence": confidence,
#                     "class": "person"
#                 })
#                 print(f"Track ID: {track_id}, BBox: {bbox}")

#             if frame_annotations:
#                     all_annotations.append(frame_annotations)
#             frames.append(frame)

#             # Display the frame
#             try:
#                 rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#                 height, width, channel = rgb_frame.shape
#                 bytes_per_line = channel * width
#                 q_img = QtGui.QImage(rgb_frame.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888)
#                 pixmap = QtGui.QPixmap.fromImage(q_img)
#                 self.canvas.loadPixmap(pixmap)
#                 self.canvas.update()
#                 # cv2.imshow("Frame with Bounding Boxes", frame)
#                 cv2.waitKey(1)
#             except Exception as e:
#                 print(f"Error updating UI for frame: {e}")

#         # Step 5: Save annotations
#         format_choice = self.choose_annotation_format()
#         if format_choice:
#             self.save_reid_annotations(frames, all_annotations, format_choice)
#             self.actions.saveReIDAnnotations.setEnabled(True)
#         else:
#             print("Annotation saving canceled by user.")
#         self.video_capture.release()
#         cv2.destroyAllWindows()
# def load_models(self):
#         """Load YOLO and FastReID models."""
#         # Load YOLOv8 model (e.g., pretrained on COCO)
        
#         self.yolo_model = YOLO("yolov8m.pt")  # Adjust YOLO model variant as needed

#         # Load FastReID model
#         cfg = get_cfg()
#         cfg.merge_from_file("A:/data/Project-Skills/Labeling_tool-enchancement/labelme/fastreid/fast-reid\configs/Market1501/bagtricks_R50.yml")
#         cfg.MODEL.WEIGHTS = "A:/data/Project-Skills/Labeling_tool-enchancement/labelme/market_bot_R50.pth"  # Path to trained FastReID weights
#         cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
#         self.fastreid_model = DefaultPredictor(cfg)
#         test_img = torch.randn(1, 3, 128, 256)  # Replace with appropriate input size
#         feature = self.fastreid_model(test_img)
#         print(feature.shape)  # Should output (1, expected_embedding_size) 
        
#         # Initialize DeepSORT
#         self.deepsort = DeepSort(max_age=50, n_init=3)


# def process_tracks(self, frame, person_colors):
        
        
        
        
#         """
#         Process confirmed tracks, assign IDs, annotate the frame, and update the LabelMe UI.
#         """
#         frame_height, frame_width = frame.shape[:2]
#         frame_annotations = []

#         for track in self.tracker.tracks:
#             if not track.is_confirmed() or track.time_since_update > 1:
#                 logger.debug(f"Skipping unconfirmed or stale track: ID={track.track_id}")
#                 continue

#             try:
#                 # Get track ID, bounding box, and confidence
#                 track_id = track.track_id
#                 det_conf = getattr(track, "det_conf", None)  # Get confidence, if available
#                 bbox_tlbr = track.to_tlbr()
#                 logger.debug(f"Raw bbox: {bbox_tlbr}")
#                 logger.debug(f"Max bbox value: {max(bbox_tlbr)}")
#                 logger.debug(f"Min bbox value: {min(bbox_tlbr)}")

#                 logger.debug(f"Track ID {track_id}: to_tlbr={bbox_tlbr}, det_conf={det_conf}")

#                 def is_normalized_bbox(bbox):
#                     return all(0 <= coord <= 1.0 for coord in bbox)

#                 if is_normalized_bbox(bbox_tlbr):
#                     bbox_tlbr = [
#                         bbox_tlbr[0] * frame_width,
#                         bbox_tlbr[1] * frame_height,
#                         bbox_tlbr[2] * frame_width,
#                         bbox_tlbr[3] * frame_height,
#                     ]
#                     logger.debug(f"Track ID {track_id}: Normalized bbox scaled to pixel values: {bbox_tlbr}")
#                     continue
#                 # Validate and convert bounding box to integer
#                 abs_bbox = self.validate_bbox(bbox_tlbr, frame.shape)
#                 if abs_bbox is None:
#                     logger.warning(f"Skipping invalid bounding box for track ID {track_id}: {bbox_tlbr}")
#                     continue

#                 x1, y1, x2, y2 = map(int, abs_bbox)

#                 # Assign a unique color for the track
#                 if track_id not in person_colors:
#                     person_colors[track_id] = self.get_random_color()
#                 color = person_colors[track_id]

#                 # Annotate the frame with bounding box and label
#                 cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
#                 label_text = f"Person ID: {track_id}"
#                 cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

#                 # Add to LabelMe UI
#                 shape = Shape(label=f"person{track_id}", shape_id=track_id)
#                 shape.addPoint(QtCore.QPointF(x1, y1))
#                 shape.addPoint(QtCore.QPointF(x2, y1))
#                 shape.addPoint(QtCore.QPointF(x2, y2))
#                 shape.addPoint(QtCore.QPointF(x1, y2))
#                 self.addLabel(shape)

#                 # Update label lists
#                 self.labelList.addPersonLabel(track_id, color)
#                 self.uniqLabelList.addUniquePersonLabel(f"person{track_id}", color)

#                 # Prepare annotation data
#                 annotation = {
#                     "track_id": track_id,
#                     "bbox": [x1, y1, x2, y2],
#                     "confidence": det_conf,
#                     "class": "person",
#                 }
#                 frame_annotations.append(annotation)

#             except Exception as e:
#                 logger.warning(f"Error processing track ID {track_id}: {e}", exc_info=True)

#         logger.info(f"Processed {len(frame_annotations)} tracks for the current frame.")
#         return frame_annotations

# def annotateVideo(self):
#         """
#         Annotate video with YOLO for detection, ReID for identification, and DeepSORT for tracking.
#         """
#         try:
#             # Load models and initialize tracker
#             # Ensure models are loaded
#             if not self.detector or not self.reid_model:
#                 self.load_models()
#             self.initializeTracker()  # Initialize the tracker with the given configuration
#             self.track_id_manager = ImprovedIDManager()

#             frames = []
#             all_annotations = []
#             person_colors = {}

#             logger.info("Starting video annotation...")

#             while self.video_capture.isOpened():
#                 ret, frame = self.video_capture.read()
#                 if not ret:
#                     logger.info("End of video or no frame available.")
#                     break

#                 # Step 1: YOLO Detection
#                 boxes, confidences = self.run_yolo(frame)

#                 # Step 2: ReID Features
#                 features = self.extract_reid_features(frame, boxes)

#                 # Step 3: Validate Detections
#                 detections = self.validate_detections(boxes, confidences, features, frame.shape)

#                 # Step 4: Tracker Update
#                 self.update_tracker(detections)

#                 # Step 5: Process Tracks
#                 frame_annotations = self.process_tracks(frame, person_colors)

#                 if frame_annotations:
#                     all_annotations.append(frame_annotations)
#                 frames.append(frame)

#                 # Step 6: Display Frame
#                 self.display_frame(frame)

#             # Step 7: Save Annotations
#             format_choice = self.choose_annotation_format()
#             if format_choice:
#                 self.save_reid_annotations(frames, all_annotations, format_choice)
#                 self.actions.saveReIDAnnotations.setEnabled(True)
#                 logger.info(f"Annotations saved successfully in {format_choice} format.")
#             else:
#                 logger.warning("Annotation saving canceled by user.")

#         except Exception as e:
#             logger.error(f"Error during video annotation: {e}", exc_info=True)
#         finally:
#             # Release resources
#             self.video_capture.release()
#             cv2.destroyAllWindows()
#             logger.info("Video annotation process completed.")

# def extract_reid_features(self, frame, boxes):
#         """
#         Extract ReID features for the detected bounding boxes.
#         Args:
#             frame (np.ndarray): Current video frame (H, W, C).
#             boxes (list of tuples): Detected bounding boxes [(x1, y1, x2, y2), ...].

#         Returns:
#             list: ReID feature vectors for each bounding box or None for invalid boxes.
#         """
#         features = []
#         device = "cuda" if torch.cuda.is_available() else "cpu"

#         for box in boxes:
#             try:
#                 x1, y1, x2, y2 = box

#                 # Validate bounding box dimensions
#                 if x2 <= x1 or y2 <= y1:
#                     logger.warning(f"Skipping invalid box with dimensions {box}")
#                     features.append(None)
#                     continue

#                 # Extract and preprocess cropped region
#                 crop = frame[y1:y2, x1:x2]  # Crop the bounding box
#                 crop = cv2.resize(crop, (128, 256))  # Resize to ReID input size
#                 crop = crop.transpose(2, 0, 1)  # Convert (H, W, C) to (C, H, W)
#                 crop = torch.tensor(crop, dtype=torch.float32).unsqueeze(0) / 255.0  # Normalize and add batch dimension
#                 crop = crop.to(device)

#                 # Extract features using FastReID model
#                 output = self.reid_model(crop)  # Get model output
#                 logger.debug(f"Model output: {output}")

#                 if isinstance(output, dict) and "features" in output:
#                     feature = output["features"]
#                 else:
#                     feature = output

#                 # Handle flattened features if necessary
#                 if len(feature.shape) == 2:
#                     feature = feature[0]  # Access the first feature if it's a batch

#                 features.append(feature.cpu().detach().numpy())
#             except Exception as e:
#                 logger.warning(f"Error extracting feature for box {box}: {e}")
#                 features.append(None)

#         logger.info(f"Extracted features for {len(features)} boxes.")
#         return features

# def setLastLabel(self, text, flags):
#         assert text
#         self.shapes[-1].label = text
#         self.shapes[-1].flags = flags
#         self.shapesBackups.pop()
#         self.storeShapes()
#         return self.shapes[-1]


# def createShapeFromData(self, shape_data):
#         """
#         Robust method to convert annotation data into a Shape object.
        
#         Args:
#             shape_data (dict): Dictionary containing shape information
        
#         Returns:
#             Shape: A validated Shape object or None if creation fails
#         """
#         try:
#             # Extract and validate required fields
#             bbox = shape_data.get("bbox")
#             shape_type = shape_data.get("shape_type", "rectangle")
#             shape_id = shape_data.get("shape_id")
#             confidence = shape_data.get("confidence")

#             # Comprehensive bbox validation
#             if not bbox:
#                 raise ValueError("Missing bounding box coordinates")
            
#             # Ensure bbox is a list of numeric values
#             try:
#                 bbox = list(map(float, bbox))
#             except (TypeError, ValueError):
#                 raise ValueError(f"Invalid bbox format: {bbox}")
            
#             # Validate bbox length and dimensions
#             if len(bbox) != 4:
#                 raise ValueError(f"Bbox must have 4 coordinates, got {len(bbox)}")
            
#             # Use Shapely for robust geometric validation
#             try:
#                 geom = box(bbox[0], bbox[1], bbox[2], bbox[3])
#                 if not geom.is_valid:
#                     raise ValueError("Invalid geometric shape")
#             except Exception as geom_error:
#                 raise ValueError(f"Geometric validation failed: {geom_error}")
            
#             # Strict shape type validation with flexibility
#             ALLOWED_SHAPE_TYPES = {"rectangle", "bbox", "box"}
#             if shape_type.lower() not in ALLOWED_SHAPE_TYPES:
#                 logging.warning(f"Unsupported shape type '{shape_type}'. Defaulting to 'rectangle'")
#                 shape_type = "rectangle"
            
#             # Normalize coordinates to ensure positive dimensions
#             x1, y1, x2, y2 = map(int, [
#                 min(bbox[0], bbox[2]),
#                 min(bbox[1], bbox[3]),
#                 max(bbox[0], bbox[2]),
#                 max(bbox[1], bbox[3])
#             ])
            
#             # Create QRectF with normalized coordinates
#             rect = QtCore.QRectF(
#                 QtCore.QPointF(x1, y1), 
#                 QtCore.QPointF(x2, y2)
#             )
            
#             # Validate shape dimensions
#             if rect.width() < 1 or rect.height() < 1:
#                 raise ValueError(f"Shape dimensions too small: {rect.width()} x {rect.height()}")
            
#             # Optional: Add additional metadata validation
#             metadata = {
#                 "original_bbox": bbox,
#                 "shape_type": shape_type,
#                 "confidence": confidence
#             }
            
#             # Create and return Shape object with enhanced validation
#             shape = Shape(rect, shape_id, confidence)
#             shape.metadata = metadata
            
#             # Logging for tracking and debugging
#             logging.info(f"Successfully created shape: ID {shape_id}, Bbox {bbox}")
            
#             return shape
    
#         except Exception as e:
#             # Comprehensive error logging
#             logging.error(f"Shape creation failed: {e}")
#             logging.error(f"Problematic shape data: {shape_data}")
            
#             # Optional: You can choose to re-raise, return None, or handle differently
#             return None

# def createShapeFromData(self, shape_data):
#         """
#         Robust method to convert annotation data into a Shape object.

#         Args:
#             shape_data (dict): Dictionary containing shape information.

#         Returns:
#             Shape: A validated Shape object or None if creation fails.
#         """
#         try:
#             # Extract and validate required fields
#             bbox = shape_data.get("bbox")
#             shape_type = shape_data.get("shape_type", "rectangle")
#             shape_id = shape_data.get("shape_id")
#             confidence = shape_data.get("confidence")

#             # Validate bbox with utility method
#             if not self.is_bbox_valid(bbox, min_size=1):
#                 raise ValueError(f"Invalid bbox: {bbox}")

#             # Normalize bbox coordinates
#             x1, y1, x2, y2 = map(int, [
#                 min(bbox[0], bbox[2]),
#                 min(bbox[1], bbox[3]),
#                 max(bbox[0], bbox[2]),
#                 max(bbox[1], bbox[3])
#             ])

#             logging.debug(f"Creating QRectF with points: ({x1}, {y1}), ({x2}, {y2})")
#             rect = QtCore.QRectF(
#                 QtCore.QPointF(x1, y1),
#                 QtCore.QPointF(x2, y2)
#             )
#             logging.debug(f"Initialized QRectF: TopLeft ({rect.topLeft().x()}, {rect.topLeft().y()}), "
#                         f"BottomRight ({rect.bottomRight().x()}, {rect.bottomRight().y()}), "
#                         f"Width: {rect.width()}, Height: {rect.height()}")
            
#             if not (x2 > x1 and y2 > y1):
#                 raise ValueError(f"Invalid bbox dimensions: ({x1}, {y1}, {x2}, {y2})")


#             # Validate dimensions
#             if rect.width() <= 0 or rect.height() < 1:
#                 raise ValueError(f"Shape dimensions too small: {rect.width()} x {rect.height()}")

#             # Validate shape type
#             ALLOWED_SHAPE_TYPES = {"rectangle", "bbox", "box"}
#             if shape_type.lower() not in ALLOWED_SHAPE_TYPES:
#                 logging.warning(f"Unsupported shape type '{shape_type}'. Defaulting to 'rectangle'")
#                 shape_type = "rectangle"

#                 # Initialize Shape object
#             shape = Shape()
#             shape.label = shape_type
#             shape.id = shape_id
#             shape.confidence = confidence

#             # Add points for the bounding box
#             shape.addPoint(QtCore.QPointF(x1, y1))  # Top-left
#             shape.addPoint(QtCore.QPointF(x2, y2))  # Bottom-right

#             # Validate the bounding box dynamically using boundingRect()
#             if shape.boundingRect().isEmpty():
#                 raise ValueError(f"BoundingRect is empty for shape ID {shape_id}.")

#             # Attach metadata for debugging
#             shape.metadata = {
#                 "original_bbox": bbox,
#                 "shape_type": shape_type,
#                 "confidence": confidence
#             }

#             logging.info(f"Successfully created shape: ID {shape_id}, Bbox {bbox}")
#             return shape

#         except Exception as e:
#             logging.error(f"Shape creation failed: {e}")
#             logging.error(f"Problematic shape data: {shape_data}")
#             return None

# Create LabelMe UI shape
                # label_shape = Shape(label=f"person{track.track_id}", shape_id=track.track_id)
                # label_shape.addPoint(QtCore.QPointF(x1, y1))
                # label_shape.addPoint(QtCore.QPointF(x2, y1))
                # label_shape.addPoint(QtCore.QPointF(x2, y2))
                # label_shape.addPoint(QtCore.QPointF(x1, y2))
                
                # # Add labels
                # self.addLabel(label_shape)
                # self.labelList.addPersonLabel(track.track_id, color)
                # self.uniqLabelList.addUniquePersonLabel(f"person{track.track_id}", color)

                # # Append annotations for saving
                # frame_annotations.append(shape_data)
                
        #         def process_tracks(self, frame, person_colors):
        # """Process tracks from the tracker and annotate frame."""
        # frame_annotations = []
        # frame_shape = frame.shape
        # logging.info(f"frame shape: {frame_shape}")
        # height, width, _ = frame.shape

        # for track in self.tracker.tracks:
        #     # Skip unconfirmed or old tracks
        #     if not track.is_confirmed() or track.time_since_update > 1:
        #         continue

        #     # Get raw bounding box (normalized)
        #     raw_bbox = track.to_tlbr().tolist()
        #     logging.info(f"Track {track.track_id} - Raw bbox: {raw_bbox}")

        #     try:
        #         # Scale bbox to pixel coordinates
        #         scaled_bbox = self.scale_bbox(raw_bbox, width, height)
        #         # Skip invalid bounding boxes
        #         if scaled_bbox[2] <= scaled_bbox[0] or scaled_bbox[3] <= scaled_bbox[1]:
        #             continue
        #         # logging.info(f"Track {track.track_id} - Scaled bbox: {scaled_bbox}")
                
        #         # # Unpack scaled bbox
        #         # x1, y1, x2, y2 = scaled_bbox

        #         # # Validate bounding box dimensions
        #         # if x2 <= x1 or y2 <= y1:
        #         #     logging.warning(f"Invalid bbox dimensions for track {track.track_id}: {scaled_bbox}")
        #         #     continue

        #             # Fix and assign sequential ID
        #         if track.track_id not in self.id_mapping:
        #             self.id_mapping[track.track_id] = self.next_id
        #             self.next_id += 1

        #         fixed_id = self.id_mapping[track.track_id]

        #             # Confidence information
        #         confidence = getattr(track, "confidence", 1.0)  # Default to 1.0 if not provided
        #         label = f"Person ID: {fixed_id} ({confidence:.2f})"

        #         # Draw bounding box and label
        #         x1, y1, x2, y2 = map(int, scaled_bbox)
        #         color = person_colors.setdefault(fixed_id, self.get_random_color())
        #         cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        #         cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        #         # Prepare shape data for canvas
        #         shape_data = {
        #             "bbox": [x1, y1, x2, y2],
        #             "shape_type": "rectangle",
        #             "shape_id": str(fixed_id),
        #             "confidence": confidence
        #         }
        #         logging.debug(f"Constructed shape_data: {shape_data}")

        #         try:
        #             shape = self.canvas.createShapeFromData(shape_data)
        #             if shape:
        #                 # Validate and log the bounding rectangle
        #                 bbox = shape.boundingRect()
        #                 if bbox.isEmpty():
        #                     logging.warning(f"Shape boundingRect is empty for ID: {shape.id}")
        #                     continue

        #                 # Ensure shapes are not added twice
        #                 if not self.canvas.is_shape_duplicate(shape.id):
        #                     logging.debug(f"Shape bounding box before addition: {bbox}")
        #                     self.canvas.addShape(shape)
        #                     logging.info(f"Shape added: ID {fixed_id}, Bbox: {scaled_bbox}")
        #                 else:
        #                     logging.debug(f"Duplicate shape detected, skipping: {shape.id}")
        #             else:
        #                 logging.error(f"Failed to create valid shape for track ID: {track.track_id}")
        #         except Exception as canvas_error:
        #             logging.error(f"Canvas shape creation error for track {track.track_id}: {canvas_error}")


        #         # Create the shape and add it to the UI
        #         label_shape = self.create_labelme_shape(track.track_id, x1, y1, x2, y2, color)
        #         self.add_labels_to_UI(label_shape, track.track_id, color)


        #         # Append annotations for saving
        #         frame_annotations.append(shape_data)

        #     except Exception as e:
        #         logging.error(f"Error processing track {track.track_id}: {e}")
        #         continue

        # return frame_annotations
        
        # def annotateVideo(self):
        # """Annotate video using detection, ReID, and tracking."""
        # try:
        #     # Ensure models are loaded
        #     if not self.detector or not self.reid_model:
        #         self.load_models()

        #     if not self.video_capture.isOpened():
        #         logger.warning("No video is loaded for annotation.")
        #         return

        #     logger.info("Starting video annotation...")
            
        #     # Get video metadata
        #     total_frames = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        #     fps = self.video_capture.get(cv2.CAP_PROP_FPS)
            
        #     logger.info(f"Video details: {total_frames} frames, {fps} FPS")
            
        #     self.initializeTracker()  # Initialize DeepSORT tracker
        #     person_colors = {}
        #     all_annotations = []
        #     processed_frames = 0
        #     frames = []

        #     while self.video_capture.isOpened():
        #         ret, frame = self.video_capture.read()
        #         if not ret:
        #             logger.info("End of video reached.")
        #             break
                
        #             # Optional: Frame skip for performance
        #         if processed_frames % self.frame_skip != 0:
        #             processed_frames += 1
        #             continue
        #         try:
        #             # Step 1: Detect and Extract Features
        #             frame_detections = self.process_image(frame)

        #             # Step 2: Update Tracker
        #             self.update_tracker(frame_detections)

        #             # Step 3: Process Tracks for Final Annotations
        #             frame_annotations = self.process_tracks(frame, person_colors)

        #             # Step 4: Draw Shapes and Update Canvas
        #             self.canvas.drawShapesForFrame(frame_annotations)
        #             self.canvas.update()
        #             self.repaint()

        #             if frame_annotations:
        #                 all_annotations.extend(frame_annotations)
        #             frames.append(frame)
                    
        #              # Update progress
        #             processed_frames += 1
        #             progress = int((processed_frames / total_frames) * 100)
        #             self.progress_callback.emit(progress)
                    
        #             # Optional: Allow cancellation
        #             if self.is_cancelled:
        #                 logger.info("Video annotation cancelled by user.")
        #                 break
            
        #         except Exception as frame_error:
        #             logger.error(f"Error processing frame {processed_frames}: {frame_error}")
        #             continue
        #         # Save annotations
        #     format_choice = self.choose_annotation_format()
        #     if format_choice:
        #         self.save_reid_annotations(frames, all_annotations, format_choice)
        #         self.actions.saveReIDAnnotations.setEnabled(True)
        #         logger.info(f"Annotations saved successfully in {format_choice} format.")
        #     else:
        #         logger.warning("Annotation saving canceled by user.")
            
            
        #     logger.info(f"Video annotation completed. Processed {processed_frames} frames.")
        #     self.progress_callback.emit(100)

        # except Exception as e:
        #     logger.error(f"Critical error during video annotation: {e}", exc_info=True)
        #     self.progress_callback.emit(-1)
        
    #     class ImprovedTracker:
    # def __init__(self):
    #     self.unique_id_counter = 0
    #     self.active_tracks = {}  # Track active objects: {id: bbox}

    # def generate_unique_id(self):
    #     """Generate consistent and unique ID for new tracks."""
    #     self.unique_id_counter += 1
    #     return self.unique_id_counter

    # def normalize_bounding_box(self, raw_bbox, frame_shape):
    #     """
    #     Normalize and validate bounding box:
    #     - Clamp negative coordinates.
    #     - Ensure box is within frame boundaries.
    #     """
    #     x, y, w, h = raw_bbox
    #     frame_width, frame_height = frame_shape[1], frame_shape[0]

    #     x = max(0, x)
    #     y = max(0, y)
    #     w = min(w, frame_width - x)
    #     h = min(h, frame_height - y)
    #     return [x, y, w, h]

    # def calculate_iou(self, box1, box2):
    #     """Compute Intersection over Union (IoU) between two bounding boxes."""
    #     x1, y1, w1, h1 = box1
    #     x2, y2, w2, h2 = box2

    #     xi1 = max(x1, x2)
    #     yi1 = max(y1, y2)
    #     xi2 = min(x1 + w1, x2 + w2)
    #     yi2 = min(y1 + h1, y2 + h2)

    #     inter_width = max(0, xi2 - xi1)
    #     inter_height = max(0, yi2 - yi1)
    #     intersection = inter_width * inter_height

    #     area1 = w1 * h1
    #     area2 = w2 * h2
    #     union = area1 + area2 - intersection

    #     return intersection / union if union > 0 else 0

    # def find_matching_track(self, bbox, iou_threshold=0.5):
    #     """
    #     Find matching track using IoU (Intersection over Union).
    #     If no match, generate a new ID.
    #     """
    #     for track_id, track_bbox in self.active_tracks.items():
    #         iou = self.calculate_iou(track_bbox, bbox)
    #         if iou > iou_threshold:
    #             return track_id  # Return existing ID for matching track

    #     # If no match is found, create a new ID
    #     new_id = self.generate_unique_id()
    #     return new_id

    # def track_objects(self, detections, frame_shape):
    #     """
    #     Main tracking logic:
    #     - Normalize bounding boxes.
    #     - Assign consistent IDs using IoU matching.
    #     """
    #     new_tracks = {}
    #     for detection in detections:
    #         raw_bbox = detection['raw_bbox']
    #         confidence = detection['confidence']

    #         # Normalize bounding box
    #         normalized_bbox = self.normalize_bounding_box(raw_bbox, frame_shape)

    #         # Find or create track
    #         track_id = self.find_matching_track(normalized_bbox)
    #         new_tracks[track_id] = {
    #             'bbox': normalized_bbox,
    #             'confidence': confidence
    #         }

    #     self.active_tracks = new_tracks  # Update active tracks
    #     return new_tracks

# def initializeTracker(self):
#         """
#         Initialize the DeepSORT tracker with the specified parameters.
#         """
#         max_cosine_distance = 0.5
#         max_age = 30
#         n_init = 3
#         max_iou_distance = 0.7

#         # Metric for matching detections to existing tracks
#         metric = NearestNeighborDistanceMetric("cosine", max_cosine_distance)

#         # Initialize the tracker
#         if not hasattr(self, 'tracker'):
#             self.tracker = Tracker(metric=metric, max_age=max_age, n_init=n_init, max_iou_distance=max_iou_distance)

#         # Log tracker configuration
#         logger.info("Tracker initialized with parameters:")
#         logger.info(f"Max Cosine Distance: {max_cosine_distance}")
#         logger.info(f"Max Age: {max_age}")
#         logger.info(f"Initialization Frames (n_init): {n_init}")
#         logger.info(f"Max IOU Distance: {max_iou_distance}")

 
        # """
        # Load all necessary models (YOLO and ReID models).
        # """
        # try:
        #     logger.info("Loading YOLO model...")
        #     self.yolo_model = YOLO("yolov8m.pt")  # Replace with your YOLO model path

        #     logger.info("Loading FastReID model...")
        #     self.load_fastreid_model()
            
        #     logger.info("loading the DeepSort Tracker")
        #     self.deepsort = DeepSort(max_age=50, n_init=3)

        #     logger.info("All models loaded successfully.")
        # except Exception as e:
        #     logger.error(f"Error loading models: {e}", exc_info=True)
        #     raise RuntimeError("Failed to load necessary models.")
  
      
      
    #   # Should output (1, expected_embedding_size)
      
    #    def process_image(self, frame):
    #     """Detect persons, extract features, and prepare detections for tracking."""
    #     detections = []

    #     # Step 1: Run Detection
    #     results = self.detector(frame)
    #     boxes = []
    #     confidences = []

    #     # Extract bounding boxes and confidences
    #     for det in results[0].boxes:
    #         if int(det.cls[0]) == 0:  # Person class
    #             box = det.xyxy[0].tolist()
    #             confidence = float(det.conf[0])

    #             # Validate the bounding box
    #             if len(box) == 4:
    #                 boxes.append(box)
    #                 confidences.append(confidence)
    #             else:
    #                 logger.warning(f"Skipping invalid detection: {box}")

    #     if not boxes:
    #         logger.warning("No valid bounding boxes found.")
    #         return []

    #     # Step 2: Extract ReID Features
    #     features = self.extract_reid_features(frame, boxes)

    #     # Step 3: Prepare Detections
    #     for box, confidence, feature in zip(boxes, confidences, features):
    #         if feature is not None:
    #             x1, y1, x2, y2 = map(int, box)  # Ensure integer coordinates
    #             detection = Detection(
    #                 (x1 / frame.shape[1], y1 / frame.shape[0], x2 / frame.shape[1], y2 / frame.shape[0]),
    #                 confidence,
    #                 feature
    #             )
    #             detections.append(detection)

    #     logger.info(f"Processed {len(detections)} detections in the frame.")
    #     return detections
    
    # def update_tracker(self, detections):
    #     """Update tracker with new detections."""
    #     try:
    #         if not detections or not isinstance(detections, dict):
    #             return

    #         bbox_xywh = detections.get('bbox_xywh', [])
    #         if not bbox_xywh:
    #             return

    #         # Get confidences from detections
    #         confidences = detections.get('confidence', np.ones(len(bbox_xywh)))
            
    #         # Get current frame
    #         current_frame = self.video_capture.get(cv2.CAP_PROP_POS_FRAMES)
            
    #         # Predict next state
    #         self.tracker.predict()
            
    #         # Update tracker
    #         self.tracking_outputs = self.tracker.update(bbox_xywh, confidences, current_frame)
            
    #     except Exception as e:
    #         logger.error(f"Error updating tracker: {e}")
    #         self.tracking_outputs = []
    
    # class EnhancedDeepSORT:
    # def __init__(self, model_path):
    #     self.tracker = DeepSort(model_path)
    #     self.tracks_confidence = {}

    # @property
    # def tracks(self):
    #     """Access the base tracker's tracks."""
    #     return self.tracker.tracker.tracks if hasattr(self.tracker, 'tracker') else []

    # def predict(self):
    #     """Delegate predict to base tracker."""
    #     return self.tracker.predict()

    # def update(self, bbox_xywh, confidences, ori_img):
    #     # Get outputs from base tracker
    #     outputs = self.tracker.update(bbox_xywh, confidences, ori_img)
        
    #     # Update confidence tracking
    #     current_ids = []
    #     for track_idx, (x1, y1, x2, y2, track_id) in enumerate(outputs):
    #         if track_id not in self.tracks_confidence:
    #             self.tracks_confidence[track_id] = ConfidenceTrack(track_id)
            
    #         conf = confidences[track_idx] if track_idx < len(confidences) else 0.5
    #         self.tracks_confidence[track_id].update(conf)
    #         current_ids.append(track_id)
            
    #     for track_id in list(self.tracks_confidence.keys()):
    #         if track_id not in current_ids:
    #             self.tracks_confidence[track_id].mark_missed()
        
    #     filtered_outputs = []
    #     for output in outputs:
    #         track_id = output[4]
    #         if track_id in self.tracks_confidence and self.tracks_confidence[track_id].is_confirmed():
    #             filtered_outputs.append(output)
                
    #     return np.array(filtered_outputs) if filtered_outputs else np.array([])

    # def increment_ages(self):
    #     """Delegate increment_ages to base tracker if it exists."""
    #     if hasattr(self.tracker, 'increment_ages'):
    #         self.tracker.increment_ages()
    
    
    #  def process_image(self, frame):
    #     """Process frame with detection and ReID."""
    #     try:
    #         # YOLOv8 detection
    #         detections = self.detector(frame)
    #         person_detections = []
            
    #         for det in detections[0].boxes.data:
    #             cls, conf = det[5], det[4]
    #             if cls == 0:  # person class
    #                 x1, y1, x2, y2 = det[0:4]
    #                 person_detections.append({
    #                     'bbox': [x1, y1, x2, y2],
    #                     'confidence': conf
    #                 })
            
    #         if not person_detections:
    #             return None
                
    #         # Convert to formats needed for tracking
    #         bbox_xywh = []
    #         confidences = []
    #         for det in person_detections:
    #             x1, y1, x2, y2 = det['bbox']
    #             w = x2 - x1
    #             h = y2 - y1
    #             bbox_xywh.append([x1, y1, w, h])
    #             confidences.append(det['confidence'])
                
    #         # Extract ReID features
    #         features = self.extract_reid_features(frame, bbox_xywh)
            
    #         return {
    #             'bbox_xywh': np.array(bbox_xywh),
    #             'confidence': np.array(confidences),
    #             'features': features
    #         }
            
    #     except Exception as e:
    #         logger.error(f"Error in process_image: {e}")
    #         return None
    
    # def update(self, bbox_xywh, confidences, features):
    #     logger.debug(f"Updating with {len(bbox_xywh)} detections")
        
    #     # Update base tracker
    #     outputs = self.tracker.update(bbox_xywh, confidences, features)
    #     logger.debug(f"Base tracker returned {len(outputs)} outputs")
        
    #     # Update confidence tracking
    #     current_ids = []
    #     filtered_outputs = []
        
    #     for track_idx, output in enumerate(outputs):
    #         track_id = output[4]
            
    #         if track_id not in self.tracks_confidence:
    #             self.tracks_confidence[track_id] = ConfidenceTrack(track_id)
    #             logger.debug(f"Created new confidence track for ID {track_id}")
            
    #         conf = confidences[track_idx] if track_idx < len(confidences) else 0.5
    #         self.tracks_confidence[track_id].update(conf)
    #         current_ids.append(track_id)
            
    #         if self.tracks_confidence[track_id].is_confirmed():
    #             filtered_outputs.append(output)
        
    #     logger.debug(f"After confidence filtering: {len(filtered_outputs)} tracks")
    #     return np.array(filtered_outputs) if filtered_outputs else np.array([])

    # @property
    # def tracks(self):
    #     base_tracks = self.tracker.tracker.tracks if hasattr(self.tracker, 'tracker') else []
    #     logger.debug(f"Number of base tracks: {len(base_tracks)}")
    #     return base_tracks
    
    ####################################now######################################
    # self.tracker.predict()
        
    #     # Create detections with features
    #     detections = [Detection(bbox, conf, feat) 
    #                  for bbox, conf, feat in zip(bbox_xywh, confidences, features)]
        
    #     # Update tracker
    #     self.tracker.update(detections)
        
    #     # Get results
    #     outputs = []
    #     for track in self.tracker.tracks:
    #         if not track.is_confirmed():
    #             continue
                
    #         box = track.to_tlbr()
    #         track_id = track.track_id
            
    #         # Store detection confidence instead of trying to access track.confidence
    #         if track_id not in self.tracks_confidence:
    #             self.tracks_confidence[track_id] = ConfidenceTrack(track_id)
            
    #         # Use the latest detection confidence if available
    #         matched_det = next((det for det in detections 
    #                           if np.array_equal(det.tlwh, track.tlwh)), None)
    #         conf = matched_det.confidence if matched_det else 0.5
            
    #         self.tracks_confidence[track_id].update(conf)
            
    #         if self.tracks_confidence[track_id].is_confirmed():
    #             outputs.append([*box, track_id])
        
    #     return np.array(outputs) if outputs else np.array([])
    
    # def update_tracker(self, bbox_xywh, confidences, features=None):
    #     """
    #     Updates the tracker with bounding boxes, confidences, and optional features.
    #     """
    #     try:
    #         logger.debug(f"Updating tracker with {len(bbox_xywh)} detections.")
    #         detections = []
    #         for i, bbox in enumerate(bbox_xywh):
    #             detection = {
    #                 'bbox': bbox.tolist(),  # Ensure bbox is a list
    #                 'confidence': confidences[i]
    #             }
    #             if features is not None and len(features) > i:
    #                 detection['feature'] = features[i]
    #             detections.append(detection)
            
    #         # Perform the tracker update (e.g., DeepSORT)
    #         self.tracker.update(detections)
    #         logger.debug(f"Tracker updated successfully.")
    #     except Exception as e:
    #         logger.error(f"Error updating tracker: {e}")
    # def process_tracks(self, frame, person_colors):
    # """
    #     Process tracks and create annotations with robust tracking and ID management.
        
    #     Args:
    #         frame (numpy.ndarray): Current video frame
    #         person_colors (dict): Dictionary to store unique colors for each track
        
    #     Returns:
    #         list: Annotations for the current frame
    #     """
    #     frame_annotations = []
        
    #     # Validate frame and tracker
    #     if frame is None:
    #         logger.error("Received None frame in process_tracks")
    #         return frame_annotations
        
    #     # Extract frame dimensions
    #     try:
    #         height, width, _ = frame.shape
    #     except Exception as e:
    #         logger.error(f"Error extracting frame dimensions: {e}")
    #         return frame_annotations
        
    #     # Validate tracker
    #     if not hasattr(self, 'tracker') or self.tracker is None:
    #         logger.error("Tracker not initialized")
    #         return frame_annotations
        
    #     # Track management
    #     used_ids = set()
    #     max_tracks = 4  # Limit to 4 persons
        
    #     # Iterate through confirmed tracks
    #     for track in self.tracker.tracks:
    #         # Skip unconfirmed or old tracks
    #         if not track.is_confirmed() or track.time_since_update > 1:
    #             continue
            
    #         try:
    #             # Get raw bounding box
    #             raw_bbox = track.to_tlbr().tolist()
                
    #             # Validate bounding box
    #             if len(raw_bbox) != 4:
    #                 logger.warning(f"Invalid bounding box: {raw_bbox}")
    #                 continue
                
    #             # Scale bbox to pixel coordinates
    #             scaled_bbox = self.scale_bbox(raw_bbox, width, height)
                
    #             # Validate scaled bbox
    #             if scaled_bbox[2] <= scaled_bbox[0] or scaled_bbox[3] <= scaled_bbox[1]:
    #                 logger.debug(f"Invalid scaled bbox: {scaled_bbox}")
    #                 continue
                
    #             # Manage track IDs
    #             if track.track_id not in self.id_mapping:
    #                 # Assign new ID, ensuring it's unique and within 1-4 range
    #                 new_id = self.get_next_available_id()
    #                 if new_id is None:
    #                     logger.warning("Maximum number of tracks reached")
    #                     break
                    
    #                 self.id_mapping[track.track_id] = new_id
                
    #             # Get fixed ID
    #             fixed_id = self.id_mapping[track.track_id]
                
    #             # Prevent duplicate IDs
    #             if fixed_id in used_ids:
    #                 logger.debug(f"Skipping duplicate ID: {fixed_id}")
    #                 continue
                
    #             used_ids.add(fixed_id)
                
    #             # Ensure coordinates are within frame bounds
    #             x1, y1, x2, y2 = map(int, scaled_bbox)
    #             x1 = max(0, min(x1, width-1))
    #             x2 = max(x1+1, min(x2, width))
    #             y1 = max(0, min(y1, height-1))
    #             y2 = max(y1+1, min(y2, height))
                
    #             # Get track confidence
    #             try:
    #                 confidence = self.tracker.tracks_confidence.get(track.track_id, 1.0)
    #                 confidence = confidence.confidence if hasattr(confidence, 'confidence') else confidence
    #             except Exception:
    #                 confidence = 1.0
                
    #             # Generate or retrieve color for this track
    #             color = person_colors.setdefault(fixed_id, self.get_random_color())
                
    #             # Create annotation
    #             annotation = {
    #                 "track_id": fixed_id,
    #                 "bbox": [x1, y1, x2, y2],
    #                 "confidence": float(confidence),
    #                 "class": "person",
    #                 "color": color
    #             }
                
    #             # Add to frame annotations
    #             frame_annotations.append(annotation)

    #             # Create and add shape to canvas
    #             try:
                    
    #                 shape_data = {
    #                 "bbox": [x1, y1, x2, y2],
    #                 "shape_type": "rectangle",
    #                 "shape_id": str(fixed_id),
    #                 "confidence": confidence,
    #                 "label": f"Person ID: {fixed_id} ({confidence:.2f})"
    #             }
                    
                    
                    
                    
    #                 shape = self.canvas.createShapeFromData(shape_data)
    #                 if shape and shape.boundingRect() and not self.canvas.is_shape_duplicate(shape.id):
    #                     self.canvas.addShape(shape)
    #                     logging.info(f"Shape added: ID {fixed_id}, Bbox: {[x1, y1, x2, y2]}")
                        
    #                     # Create and add UI labels
    #                     label_shape = self.create_labelme_shape(fixed_id, x1, y1, x2, y2, color)
    #                     self.add_labels_to_UI(label_shape, fixed_id, color)
                        
    #                     # Add to annotations
    #                     frame_annotations.append(shape_data)
    #                 else:
    #                     logging.debug(f"Shape not added - duplicate or invalid: {shape.id if shape else 'None'}")
    #             except Exception as canvas_error:
    #                 logging.error(f"Canvas shape creation error for track {track.track_id}: {canvas_error}")

    #         except Exception as e:
    #             logging.error(f"Error processing track {track.track_id}: {e}")
    #             continue

    #     return frame_annotations
    #######################################################New ##########################################################################
    #     """ 
    #     The Method for annotating the video and automating the annotation process
    #     """
    # def annotateVideo(self):
    #     """
    #     Annotate video using detection, ReID, and tracking.
    #     Comprehensive method with robust error handling and logging.
    #     """
    #     try:
    #         # Validate video capture
    #         if not hasattr(self, 'video_capture') or not self.video_capture.isOpened():
    #             logger.error("No video is loaded or video capture is not opened.")
    #             QtWidgets.QMessageBox.warning(self, "Error", "Please load a video first.")
    #             return

    #         # Validate models
    #         if not self.load_models():
    #             logger.error("Failed to load detection or ReID models.")
    #             QtWidgets.QMessageBox.critical(self, "Model Error", "Could not load required models.")
    #             return
    #             # Initialize Tracker (CRITICAL STEP)
    #         if not self.initialize_tracker():
    #             logger.error("Failed to initialize tracker")
    #             QtWidgets.QMessageBox.critical(self, "Tracker Error", "Could not initialize tracking.")
    #             return

    #         # Reset tracking-related attributes
    #         self.id_mapping = {}  # Reset ID mapping
    #         self.person_tracks = {}  # Reset person tracks
    #         self.next_id = 1  # Reset next available ID

    #         # Video metadata
    #         total_frames = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    #         fps = self.video_capture.get(cv2.CAP_PROP_FPS)
    #         logger.info(f"Video Details: {total_frames} frames, {fps} FPS")

    #         # Tracking setup
    #         max_tracks = 4  # Limit to 4 persons
    #         person_colors = {}
    #         all_annotations = []
    #         processed_frames = []

    #         # Progress tracking
    #         self.is_cancelled = False
    #         self.progress_callback.emit(0)  # Initialize progress

    #         # Main video processing loop
    #         frame_number = 0
    #         while self.video_capture.isOpened() and not self.is_cancelled:
    #             ret, frame = self.video_capture.read()
    #             if not ret:
    #                 logger.info("Reached end of video.")
    #                 break

    #             # Optional: Frame skipping for performance
    #             if frame_number % self.frame_skip != 0:
    #                 frame_number += 1
    #                 continue

    #             try:
    #                 # Detect and process current frame
    #                 frame_detections = self.process_image(frame)

    #                 if frame_detections is None or len(frame_detections['bbox_xywh']) == 0:
    #                     logger.warning(f"No detections for frame {frame_number}")
    #                     frame_number += 1
    #                     continue

    #                 # Extract detection details
    #                 bbox_xywh = frame_detections['bbox_xywh']
    #                 confidences = frame_detections['confidence']
    #                 features = frame_detections.get('features', [])

    #                 # Update tracker
    #                 tracked_objects = self.update_tracker(bbox_xywh, confidences, features)

    #                 if len(tracked_objects) > 0:
    #                     # Process and annotate tracks
    #                     frame_annotations = self.process_tracks(frame, person_colors)
                        
    #                     if frame_annotations:
    #                         all_annotations.append({
    #                             'frame_number': frame_number,
    #                             'annotations': frame_annotations
    #                         })
    #                         processed_frames.append(frame)

    #                 # Update progress
    #                 progress = int((frame_number / total_frames) * 100)
    #                 self.progress_callback.emit(progress)

    #             except Exception as frame_error:
    #                 logger.error(f"Error processing frame {frame_number}: {frame_error}")
    #                 traceback.print_exc()

    #             frame_number += 1

    #             # Break if max tracks reached or cancelled
    #             if len(person_colors) >= max_tracks or self.is_cancelled:
    #                 break

    #         # Post-processing
    #         self.progress_callback.emit(100)

    #         # Save annotations if available
    #         if all_annotations:
    #             format_choice = self.choose_annotation_format()
    #             if format_choice:
    #                 self.save_reid_annotations(processed_frames, all_annotations, format_choice)
    #                 logger.info(f"Annotations saved in {format_choice} format")
    #             else:
    #                 logger.warning("Annotation saving cancelled")

    #         logger.info(f"Video annotation completed. Processed {frame_number} frames.")

    #     except Exception as e:
    #         logger.critical(f"Critical error during video annotation: {e}")
    #         traceback.print_exc()
    #         self.progress_callback.emit(-1)  # Indicate error
    #         QtWidgets.QMessageBox.critical(self, "Annotation Error", str(e)) 
    
    # def scale_bbox(self, bbox, frame_width, frame_height):
    #     """
    #     Scale bounding box coordinates to pixel coordinates and validate.
    #     """
    #     try:
    #         bbox = list(map(float, bbox))
            
    #         # Clamp bounding box values to frame dimensions
    #         x1 = max(0, min(bbox[0], frame_width - 1))
    #         y1 = max(0, min(bbox[1], frame_height - 1))
    #         x2 = max(x1 + 1, min(bbox[2], frame_width))  # Ensure x2 > x1
    #         y2 = max(y1 + 1, min(bbox[3], frame_height))  # Ensure y2 > y1
            
    #         if x1 >= x2 or y1 >= y2:
    #             logger.warning(f"Invalid scaled bbox: {bbox}")
    #             return [0, 0, frame_width // 2, frame_height // 2]  # Fallback to default bbox
            
    #         return [x1, y1, x2, y2]
        
    #     except Exception as e:
    #         logger.error(f"Bounding box scaling error: {e}")
    #         return [0, 0, frame_width // 2, frame_height // 2]  # Fallback to default bbox
############################## Now #############################################################
# "def createShapeFromData(self, shape_data):
#         """
#         Convert annotation data into a Shape object with robust error handling.
        
#         Args:
#             shape_data (dict): Dictionary containing shape information.

#         Returns:
#             Shape or None: A validated Shape object or None if creation fails.
#         """
#         try:
#             # Extract and validate required fields
#             bbox = shape_data.get("bbox")
#             shape_type = shape_data.get("shape_type", "rectangle")
#             shape_id = shape_data.get("shape_id")
#             confidence = shape_data.get("confidence", 1.0)

#             # Validate bbox presence and size
#             if not bbox or len(bbox) != 4:
#                 raise ValueError(f"Invalid bbox format: {bbox}")

#             # Normalize bbox coordinates
#             x1, y1, x2, y2 = map(int, [
#                 min(bbox[0], bbox[2]),
#                 min(bbox[1], bbox[3]),
#                 max(bbox[0], bbox[2]),
#                 max(bbox[1], bbox[3])
#             ])

#             # Validate dimensions are within frame bounds
#             frame_width, frame_height = self.pixmap.width(), self.pixmap.height()
#             x1 = max(0, min(x1, frame_width - 1))
#             x2 = max(x1 + 1, min(x2, frame_width))
#             y1 = max(0, min(y1, frame_height - 1))
#             y2 = max(y1 + 1, min(y2, frame_height))

#             # Check for valid rectangle dimensions
#             if x2 <= x1 or y2 <= y1:
#                 logger.warning(f"Invalid rectangle dimensions: ({x1}, {y1}, {x2}, {y2})")
#                 # Fallback to default bounding box
#                 x1, y1, x2, y2 = frame_width // 4, frame_height // 4, frame_width // 2, frame_height // 2
#                 logger.info(f"Using fallback bbox: ({x1}, {y1}, {x2}, {y2})")

#             # Create QRectF
#             rect = QtCore.QRectF(QtCore.QPointF(x1, y1), QtCore.QPointF(x2, y2))
#             if rect.isEmpty():
#                 raise ValueError(f"BoundingRect is empty for shape ID {shape_id}.")

#             # Initialize Shape object
#             shape = Shape(rect=rect, shape_id=shape_id, shape_type=shape_type, confidence=confidence)

#             # Add bounding box points
#             shape.addPoint(QtCore.QPointF(x1, y1))  # Top-left
#             shape.addPoint(QtCore.QPointF(x2, y1))  # Top-right
#             shape.addPoint(QtCore.QPointF(x2, y2))  # Bottom-right
#             shape.addPoint(QtCore.QPointF(x1, y2))  # Bottom-left

#             logger.info(f"Successfully created shape: ID {shape_id}, Bbox {bbox}, Type {shape_type}")
#             return shape

#         except Exception as e:
#             logger.error(f"Shape creation failed: {e}")
#             logger.error(f"Problematic shape data: {shape_data}")
#             return None"



####################################################12/23/2024###################################################################################################
# def process_tracks(self, frame, person_colors):
#         """
#         Process tracks from the tracker, assign IDs, annotate the frame, and update the LabelMe UI.
#         """
#         frame_annotations = []
#         frame_height, frame_width = frame.shape[:2]
#         logger.info(f"Frame dimensions: {frame_width}x{frame_height}")
#         logger.info(f"Processing tracks. Frame dimensions: {frame.shape[:2]}")
#         logger.info(f"Number of tracks: {len(self.tracker.tracks)}")
        
#         used_track_ids = set()  # Track processed track IDs
        
#         for track in self.tracker.tracks:
#              # Detailed track logging
#             logger.debug(f"Track details: ID={track.track_id}, "
#                          f"Confirmed={track.is_confirmed()}, "
#                          f"Time since update={track.time_since_update}")
            
#             if not track.is_confirmed() or track.time_since_update > 1:
#                 logger.warning(f"Skipping unconfirmed or stale track: {track.track_id}")
                    
#                 continue
            
#             # Skip if track ID already processed
#             if track.track_id in used_track_ids:
#                 logger.debug(f"Skipping already processed track ID: {track.track_id}")
#                 continue
            
            
#             raw_bbox = track.to_tlbr()
#             logger.debug(f"Raw bbox: {raw_bbox}")

#              # Scale bbox
#             try:
#                 scaled_bbox = self.scale_bbox(raw_bbox, frame_width, frame_height)
#                 logger.debug(f"Scaled bbox: {scaled_bbox}")
#             except Exception as scaling_error:
#                 logger.error(f"Bbox scaling error for track {track.track_id}: {scaling_error}")
#                 continue

#             try:   # Create a Shape object with the scaled bounding box
#                 shape = self.canvas.createShapeFromData({
#                     "bbox": scaled_bbox,
#                     "shape_type": "rectangle",
#                     "shape_id": str(track.track_id),
#                     "confidence": getattr(track, "confidence", 1.0)
#                 })

#                 if shape is None:
#                     logger.warning(f"Failed to create shape for track ID {track.track_id}")
#                     continue

#                 # Assign unique ID and color
#                 track_id = self.get_next_available_id(used_track_ids)
#                 color = person_colors.setdefault(track_id, self.get_random_color())
                

#                 # Prepare shape data
#                 shape_data = {
#                     "bbox": scaled_bbox,
#                     "shape_type": "rectangle",
#                     "shape_id": str(track_id),
#                     "confidence": getattr(track, "confidence", 1.0),
#                     "label": f"Person ID: {track_id}"
#                 }

#                 # Create and add shape to canvas
#                 if not self.canvas.is_shape_duplicate(shape.id):
#                     self.canvas.addShape(shape)
#                     logger.info(f"Shape added: ID {track_id}, Bbox: {scaled_bbox}")
#                      # Mark track ID as processed
#                     used_track_ids.add(track.track_id)
#                     frame_annotations.append(track,shape)
#                     logger.info(f"Successfully processed track: {track.track_id}")

#                     # Create LabelMe-compatible shape
#                     label_shape = self.create_labelme_shape(track_id, *scaled_bbox, color)
#                     self.add_labels_to_UI(label_shape, track_id, color)
#                 else:
#                     logger.warning(f"Failed to create shape for track ID: {track_id}")

#             except Exception as e:
#                 logger.error(f"Error processing track {track.track_id}: {e}")
#                 continue
#         logging.info(f"Processed {len(frame_annotations)} trucks succesfully")
#         return frame_annotations

# def process_tracks(self, frame, person_colors, tracked_objects):
#         """
#         Process tracks from the tracker, assign IDs, annotate the frame, and update the LabelMe UI.

#         Args:
#             frame (np.ndarray): The current frame being processed.
#             person_colors (dict): Dictionary of assigned colors for track IDs.
#             track_objects (list): List of tracked objects returned by self.update_tracker.
#         """
#         frame_annotations = []
#         frame_height, frame_width = frame.shape[:2]
#         logger.info(f"Frame dimensions: {frame_width}x{frame_height}")
#         logger.info(f"Number of tracked objects: {len(tracked_objects)}")

#         processed_track_ids = set()

#         for track in tracked_objects:
#             track_id = None  # Ensure track_id is always defined
#             try:
#                 # Ensure track has required attributes
#                 if not hasattr(track, 'is_confirmed') or not hasattr(track, 'track_id'):
#                     logger.error(f"Invalid track object structure: {track}")
#                     continue

#                 track_id = getattr(track, 'track_id', 'Unknown')
#                 if not track.is_confirmed():
#                     logger.warning(f"Skipping unconfirmed track: {track_id}")
#                     continue
#                 # Prevent duplicate processing
#                 if track_id in processed_track_ids:
#                     logger.debug(f"Skipping already processed track ID: {track_id}")
#                     continue
#                 # Get raw bounding box
#                 raw_bbox = getattr(track, 'to_tlbr', lambda: None)()
#                 if raw_bbox is None:
#                     logger.error(f"Track {track_id} does not have a valid bounding box; skipping.")
#                     continue

#                 logger.debug(f"Raw bbox for track {track_id}: {raw_bbox}")

                

#                 # # Extract bounding box
#                 # raw_bbox = track['bbox']  # [x1, y1, x2, y2]
#                 # logger.debug(f"Raw bbox for track {track_id}: {raw_bbox}")

                
                

               
                
                
#                 # Scale bbox with robust method
#                 try:
#                     scaled_bbox = self.scale_bbox(raw_bbox, frame_width, frame_height)
#                     logger.debug(f"Scaled bbox for track {track_id}: {scaled_bbox}")
#                 except Exception as scaling_error:
#                     logger.error(f"Bbox scaling error for track {track_id}: {scaling_error}")
#                     continue

#                 # Create shape from data
#                 shape = self.canvas.createShapeFromData({
#                     "bbox": scaled_bbox,
#                     "shape_type": "rectangle",
#                     "shape_id": str(track_id),
#                     "confidence": track.get("confidence", 1.0)
#                 })

#                 if shape is None:
#                     logger.warning(f"Failed to create shape for track ID: {track_id}")
#                     continue

#                 # Assign unique color
#                 color = person_colors.setdefault(track_id, self.get_random_color())

#                 # Prepare shape data
#                 shape_data = {
#                     "bbox": scaled_bbox,
#                     "shape_type": "rectangle",
#                     "shape_id": str(track_id),
#                     "confidence": track.get("confidence", 1.0),
#                     "label": f"Person ID: {track_id}"
#                 }

#                 # Add shape to canvas and process
#                 if not self.canvas.is_shape_duplicate(shape.id):
#                     self.canvas.addShape(shape)
#                     processed_track_ids.add(track_id)
#                     frame_annotations.append(shape_data)

#                     # Create LabelMe-compatible shape
#                     label_shape = self.create_labelme_shape(track_id, *scaled_bbox, color)
#                     self.add_labels_to_UI(label_shape, track_id, color)

#             except Exception as e:
#                 logger.error(f"Unexpected error processing track {track_id}: {e}")

#         # Log successful tracks
#         logger.info(f"Processed {len(processed_track_ids)} tracks successfully")
#         return frame_annotations