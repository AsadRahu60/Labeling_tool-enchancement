import cv2
from ultralytics import YOLO

class PersonDetector:
    def __init__(self, model_path="yolov8n.pt"):
        # Load the YOLOv8 model
        self.model = YOLO(model_path)

    def detect_persons(self, frame):
        # Perform inference
        results = self.model(frame)

        # Extract bounding boxes, confidences, and class IDs
        boxes = results[0].boxes.xyxy  # Get bounding boxes
        confidences = results[0].boxes.conf  # Get confidences
        class_ids = results[0].boxes.cls  # Get class IDs

        # Annotate the frame with bounding boxes and information
        for box, confidence, class_id in zip(boxes, confidences, class_ids):
            if class_id == 0:  # Class ID 0 corresponds to 'person'
                x1, y1, x2, y2 = map(int, box)  # Convert to integer coordinates
                # Draw rectangle around detected person
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # Prepare the label with class name and confidence
                label = f"Person: {confidence:.2f}"
                # Annotate the frame with the label
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return frame

if __name__ == "__main__":
    # Create an instance of the PersonDetector
    detector = PersonDetector()
    # Open the video file
    cap = cv2.VideoCapture("path/to/your/video.mp4")

    # Desired frame size
    frame_width = 640  # Set your desired width
    frame_height = 480  # Set your desired height

    while True:
        # Read a frame from the video
        ret, frame = cap.read()
        if not ret:
            break  # Exit if the video has ended

        # Resize the frame
        frame = cv2.resize(frame, (frame_width, frame_height))

        # Detect persons in the resized frame
        detected_frame = detector.detect_persons(frame)
        # Display the frame with detections
        cv2.imshow("Person Detection", detected_frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()