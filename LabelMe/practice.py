# import torch
# import torchreid
# import cv2
# import numpy as np

# class PersonReID:
#     def __init__(self):
#         # Load OSNet model for ReID
#         self.reid_model = torchreid.models.build_model(
#             name='osnet_x1_0',    # Use 'osnet_x1_0' for a balanced model size and performance
#             num_classes=1000,     # Dummy number for classes (not used during feature extraction)
#             pretrained=True       # Load pre-trained weights
#         )
        
#         # Move the model to GPU if available
#         self.reid_model = self.reid_model.cuda() if torch.cuda.is_available() else self.reid_model
#         self.reid_model.eval()  # Set the model to evaluation mode

#     def extract_features(self, image):
#         # Preprocess the image for the model
#         image = cv2.resize(image, (256, 128))  # Resize to the input size of the model
#         image = image.transpose((2, 0, 1))  # Change to (C, H, W)
#         image = torch.from_numpy(image).float()  # Convert to tensor
#         image = image.unsqueeze(0)  # Add batch dimension
#         image = image.cuda() if torch.cuda.is_available() else image  # Move to GPU if available

#         with torch.no_grad():
#             features = self.reid_model(image)  # Extract features
#         return features.cpu().numpy()  # Move back to CPU and convert to numpy array

# def main():
#     # Initialize the ReID model
#     reid = PersonReID()

#     # Define the video capture device
#     cap = cv2.VideoCapture(0)

#     while True:
#         # Read a frame from the video capture device
#         ret, frame = cap.read()
#         if not ret:
#             break

#         # Extract features for the detected person (assuming you have a bounding box)
#         # Here, we simulate a bounding box for demonstration
#         bbox = (50, 50, 200, 400)  # Example bounding box (x1, y1, x2, y2)
#         x1, y1, x2, y2 = bbox
#         person_image = frame[y1:y2, x1:x2]  # Crop the person from the frame

#         # Extract features using OSNet
#         features = reid.extract_features(person_image)

#         # Here you would typically use these features for tracking or matching
#         # For demonstration, we just print the feature shape
#         print("Extracted features shape:", features.shape)

#         # Display the output
#         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw bounding box
#         cv2.imshow('Person Tracking', frame)

#         # Exit on key press
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     # Release the video capture device
#     cap.release()
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main()

import torch
import cv2
from deep_sort_realtime.deepsort_tracker import DeepSort
from torchreid import models

# Load the OSNet model
osnet_model = models.build_model(
    name='osnet_x1_0',  # OSNet model
    num_classes=751,  # Number of classes (adjust as needed)
    pretrained=True,
    loss='softmax'
)
osnet_model.eval()  # Set the model to evaluation mode

# Initialize DeepSORT
deepsort_model = DeepSort(
    max_age=30,  # Age for objects that are lost
    nn_budget=100,  # Maximum items for feature storage
    max_iou_distance=0.4,  # Maximum IOU for matches
    n_init=3  # Number of consecutive detections before initializing
)

# Define the video capture device
cap = cv2.VideoCapture("A:/data/Project-Skills/Labeling_tool-enchancement/labelme/5198164-uhd_3840_2160_25fps.mp4")

while True:
    # Read a frame from the video capture device
    ret, frame = cap.read()
    if not ret:
        break  # Exit if the video has ended

    # Preprocess the frame for OSNet
    frame_resized = cv2.resize(frame, (256, 128))  # Resize to the input size expected by OSNet
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)  # Convert to RGB
    frame_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).unsqueeze(0).float()  # Change to (1, C, H, W)
    frame_tensor /= 255.0  # Normalize to [0, 1]

    # Extract feature representations for the detected objects using OSNet
    with torch.no_grad():  # Disable gradient calculation
        features = osnet_model(frame_tensor)

    # Assuming you have a method to get bounding boxes (e.g., from an object detector)
    # Here, we will use dummy bounding boxes for demonstration
    # Replace this with actual detection logic
    bounding_boxes = [[50, 50, 100, 200, 1.0]]  # Example bounding box format: [x1, y1, x2, y2, confidence]

    # Format raw_detections correctly (excluding confidence for DeepSORT)
    raw_detections = [[[bounding_boxes[0][0], bounding_boxes[0][1], bounding_boxes[0][2], bounding_boxes[0][3]]]]  # Only x1, y1, x2, y2

    # Debugging: Print the structure of raw_detections
    print("Raw Detections:", raw_detections)

    # Ensure features are in the correct shape
    features = features.cpu().numpy()  # Convert features to numpy array
    features = features.reshape(features.shape[0], -1)  # Reshape if necessary

    # Update the DeepSORT tracker
    try:
        tracks = deepsort_model.update_tracks(raw_detections, features)  # Use the correct method name
    except AssertionError as e:
        print("AssertionError:", e)
        print("Raw Detections Structure:", raw_detections)
        print("Features Shape:", features.shape)
        continue  # Skip this iteration if there's an error

    # Draw bounding boxes and IDs for the tracked objects
    for track in tracks:
        x1, y1, x2, y2 = track.to_tlbr()  # Get bounding box coordinates
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, str(track.track_id), (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the output
    cv2.imshow('Person Tracking', frame)

    # Exit on key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture device
cap.release()

# Close all OpenCV windows
cv2.destroyAllWindows()