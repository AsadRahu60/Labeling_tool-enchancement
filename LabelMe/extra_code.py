def annotateVideo(self):
        """Annotate and track persons in the video using YOLOv8 segmentation for detection and OSNet for ReID."""
        if not hasattr(self, 'video_capture'):
            QtWidgets.QMessageBox.warning(self, "Error", "No video loaded.")
            return

        person_id_map = {}  # Dictionary to store person IDs and their features
        next_person_id = 1

        while self.video_capture.isOpened():
            ret, frame = self.video_capture.read()
            if not ret:
                break

            # Detect persons using YOLOv8 segmentation
            boxes, masks = self.run_yolo_segmentation(frame)
            annotated_frame = frame.copy()  # Copy frame for drawing

            # Extract ReID features using OSNet
            if boxes:
                features = self.extract_reid_features_with_masks(frame, boxes, masks)

                person_ids = []  # List to store IDs of persons in the current frame

                for feature in features:
                    matched_id = None
                    # Compare the extracted feature with the saved features in person_id_map
                    for person_id, saved_feature in person_id_map.items():
                        if self.is_same_person(feature, saved_feature):  # type: ignore
                            matched_id = person_id  # Found a matching person, reuse the ID
                            person_id_map[person_id] = feature  # Update the stored feature
                            break

                    if matched_id is None:
                        # New person detected, assign a new ID
                        matched_id = f"person_{next_person_id}"
                        person_id_map[matched_id] = feature
                        next_person_id += 1

                    person_ids.append(matched_id)

                # Safeguard: Ensure we don't access more person_ids than there are bounding boxes
                if len(person_ids) > len(boxes):
                    person_ids = person_ids[:len(boxes)]  # Truncate the IDs to match boxes
                elif len(person_ids) < len(boxes):
                    # If fewer IDs than boxes, append 'Unknown' for missing IDs
                    person_ids += ['Unknown'] * (len(boxes) - len(person_ids))

                # Draw bounding boxes and masks on the frame
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = box
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # Annotate person ID on the bounding box
                    cv2.putText(annotated_frame, f"ID: {person_ids[i]}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

                    # Draw the segmentation mask if available
                    if i < len(masks):
                        mask = masks[i] # Ensure the mask is converted to a numpy array
                        mask = (mask > 0.5).astype(np.uint8)  # Binarize the mask (thresholding)
                        
                        # Resize the mask to fit the bounding box size
                        mask_resized = cv2.resize(mask, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)

                        # Create a colored mask (apply a green color)
                        colored_mask = np.zeros_like(frame, dtype=np.uint8)
                        colored_mask[y1:y2, x1:x2, 1] = mask_resized * 255  # Apply green color on the mask

                        # Blend the mask with the frame
                        annotated_frame = cv2.addWeighted(annotated_frame, 1, colored_mask, 0.5, 0)

                print(f"Extracted ReID features and assigned IDs for {len(person_ids)} persons")

            # Update the display with the annotated frame
            self.update_display(annotated_frame)

            cv2.waitKey(1)

        # After the loop, release video capture
        # self.video_capture.release()
        
        
        ######################bbox code within the annotation_with_display
         try:
                # Case 1: If bbox is a list of arrays, iterate through each array
                if isinstance(bbox, list) and all(isinstance(b, np.ndarray) for b in bbox):
                    for b in bbox:
                        # Check if each array is of correct size
                        if len(b) == 4:
                            x1, y1, x2, y2 = map(int, b.tolist())  # Convert array to list and unpack
                        else:
                            raise ValueError(f"Expected 4-element array, got {b}")

                # Case 2: If bbox is already a single array or list, handle it directly
                elif isinstance(bbox, (np.ndarray, list)) and len(bbox) == 4:
                    x1, y1, x2, y2 = map(int, bbox)  # Safely unpack

                else:
                    raise ValueError(f"Expected 4-element bounding box, got {bbox}")

                # Draw the bounding box and track ID on the frame
                farbe = self.get_random_color()
                cv2.rectangle(frame, (x1, y1), (x2, y2), farbe, 2)
                cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                
                