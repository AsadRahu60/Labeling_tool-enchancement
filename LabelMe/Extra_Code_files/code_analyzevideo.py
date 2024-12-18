# import cv2
# from ultralytics import YOLO

# class PersonDetector:
#     def __init__(self, model_path="yolov8n.pt"):
#         # Load the YOLOv8 model
#         self.model = YOLO(model_path)

#     def detect_persons(self, frame):
#         # Perform inference
#         results = self.model(frame)

#         # Extract bounding boxes, confidences, and class IDs
#         boxes = results[0].boxes.xyxy  # Get bounding boxes
#         confidences = results[0].boxes.conf  # Get confidences
#         class_ids = results[0].boxes.cls  # Get class IDs

#         # Annotate the frame with bounding boxes and information
#         for box, confidence, class_id in zip(boxes, confidences, class_ids):
#             if class_id == 0:  # Class ID 0 corresponds to 'person'
#                 x1, y1, x2, y2 = map(int, box)  # Convert to integer coordinates
#                 # Draw rectangle around detected person
#                 cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                 # Prepare the label with class name and confidence
#                 label = f"Person: {confidence:.2f}"
#                 # Annotate the frame with the label
#                 cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

#         return frame

# if __name__ == "__main__":
#     # Create an instance of the PersonDetector
#     detector = PersonDetector()
#     # Open the video file
#     cap = cv2.VideoCapture("path/to/your/video.mp4")

#     # Desired frame size
#     frame_width = 640  # Set your desired width
#     frame_height = 480  # Set your desired height

#     while True:
#         # Read a frame from the video
#         ret, frame = cap.read()
#         if not ret:
#             break  # Exit if the video has ended

#         # Resize the frame
#         frame = cv2.resize(frame, (frame_width, frame_height))

#         # Detect persons in the resized frame
#         detected_frame = detector.detect_persons(frame)
#         # Display the frame with detections
#         cv2.imshow("Person Detection", detected_frame)

#         # Exit on 'q' key press
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     # Release the video capture object and close all OpenCV windows
#     cap.release()
#     cv2.destroyAllWindows()

import os
import cv2
import numpy as np
import tensorflow as tf

import time
from ultralytics import YOLO

from deep_sort_realtime.deepsort_tracker import DeepSort
from deep_sort_realtime.deep_sort import nn_matching
from deep_sort_realtime.deep_sort.detection import Detection
from deep_sort_realtime.deep_sort.tracker import Tracker
from deep_sort_realtime.deep_sort import generate_detections as gdet



video_path   = "./IMAGES/test.mp4"

yolo = Create_Yolov3(input_size=input_size)


def Object_tracking(YoloV3, video_path, output_path, input_size=416, show=False, CLASSES=YOLO_COCO_CLASSES, score_threshold=0.3, iou_threshold=0.45, rectangle_colors='', Track_only = []):
    # Definition of the parameters
    max_cosine_distance = 0.5
    nn_budget = None
    
    #initialize deep sort object
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    times = []

    if video_path:
        vid = cv2.VideoCapture(video_path) # detect on video
    else:
        vid = cv2.VideoCapture(0) # detect from webcam

    # by default VideoCapture returns float instead of int
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    codec = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, codec, fps, (width, height)) # output_path must be .mp4

    NUM_CLASS = read_class_names(CLASSES)
    key_list = list(NUM_CLASS.keys()) 
    val_list = list(NUM_CLASS.values())
    while True:
        _, img = vid.read()

        try:
            original_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        except:
            break
        image_data = image_preprocess(np.copy(original_image), [input_size, input_size])
        image_data = tf.expand_dims(image_data, 0)
        
        t1 = time.time()
        pred_bbox = YoloV3.predict(image_data)
        t2 = time.time()

        times.append(t2-t1)
        times = times[-20:]
        
        pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
        pred_bbox = tf.concat(pred_bbox, axis=0)

        bboxes = postprocess_boxes(pred_bbox, original_image, input_size, score_threshold)
        bboxes = nms(bboxes, iou_threshold, method='nms')

        # extract bboxes to boxes (x, y, width, height), scores and names
        boxes, scores, names = [], [], []
        for bbox in bboxes:
            if len(Track_only) !=0 and NUM_CLASS[int(bbox[5])] in Track_only or len(Track_only) == 0:
                boxes.append([bbox[0].astype(int), bbox[1].astype(int), bbox[2].astype(int)-bbox[0].astype(int), bbox[3].astype(int)-bbox[1].astype(int)])
                scores.append(bbox[4])
                names.append(NUM_CLASS[int(bbox[5])])

        # Obtain all the detections for the given frame.
        boxes = np.array(boxes) 
        names = np.array(names)
        scores = np.array(scores)
        features = np.array(encoder(original_image, boxes))
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(boxes, scores, names, features)]

        # Pass detections to the deepsort object and obtain the track information.
        tracker.predict()
        tracker.update(detections)

        # Obtain info from the tracks
        tracked_bboxes = []
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue 
            bbox = track.to_tlbr() # Get the corrected/predicted bounding box
            class_name = track.get_class() #Get the class name of particular object
            tracking_id = track.track_id # Get the ID for the particular track
            index = key_list[val_list.index(class_name)] # Get predicted object index by object name
            tracked_bboxes.append(bbox.tolist() + [tracking_id, index]) # Structure data, that we could use it with our draw_bbox function

        ms = sum(times)/len(times)*1000
        fps = 1000 / ms

        # draw detection on frame
        image = draw_bbox(original_image, tracked_bboxes, CLASSES=CLASSES, tracking=True)
        image = cv2.putText(image, "Time: {:.1f} FPS".format(fps), (0, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)

        # draw original yolo detection
        #image = draw_bbox(image, bboxes, CLASSES=CLASSES, show_label=False, rectangle_colors=rectangle_colors, tracking=True)

        #print("Time: {:.2f}ms, {:.1f} FPS".format(ms, fps))

        if output_path != '': out.write(image)
        if show:
            cv2.imshow('output', image)
            if cv2.waitKey(25) & 0xFF == ord("q"):
                cv2.destroyAllWindows()
                break

    cv2.destroyAllWindows()

Object_tracking(yolo, video_path, '', input_size=input_size, show=True, iou_threshold=0.1, rectangle_colors=(255,0,0), Track_only = [])