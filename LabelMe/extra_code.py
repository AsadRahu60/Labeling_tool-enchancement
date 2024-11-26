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
    """This code down in cooporates the batch processing.
    """
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
####################################################################################################
    """Match_ID
    
    """
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
