import torch
import cv2
import numpy as np
from ultralytics import YOLO
from fastreid.config import get_cfg
from fastreid.engine import DefaultPredictor
from deep_sort_realtime.deep_sort import DeepSort

class ModelIntegration:
    def __init__(self):
        # Model Paths (adjust as needed)
        self.yolo_weights = "yolov8m.pt"
        self.fastreid_config = "configs/Market1501/bagtricks_R50.yml"
        self.fastreid_weights = "market_bot_R50.pth"

    def load_models(self):
        """
        Comprehensive model loading with advanced error handling and logging
        """
        # YOLO Model Loading
        try:
            self.yolo_model = YOLO(self.yolo_weights)
            print(f"YOLO model loaded successfully: {self.yolo_weights}")
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            self.yolo_model = None

        # FastReID Model Loading
        try:
            # Configure FastReID
            cfg = get_cfg()
            cfg.merge_from_file(self.fastreid_config)
            cfg.MODEL.WEIGHTS = self.fastreid_weights
            cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
            
            self.fastreid_model = DefaultPredictor(cfg)
            
            # Validate model by running a test inference
            test_img = torch.randn(1, 3, 128, 256)
            feature = self.fastreid_model(test_img)
            print(f"FastReID model loaded. Feature shape: {feature.shape}")
        except Exception as e:
            print(f"Error loading FastReID model: {e}")
            self.fastreid_model = None

        # DeepSORT Initialization
        try:
            self.deepsort = DeepSort(max_age=50, n_init=3)
            print("DeepSORT tracker initialized successfully")
        except Exception as e:
            print(f"Error initializing DeepSORT: {e}")
            self.deepsort = None

    def auto_annotate_video(self, video_path, confidence_threshold=0.5):
        """
        Automatic video annotation with multi-model integration
        
        Args:
            video_path (str): Path to input video
            confidence_threshold (float): Minimum detection confidence
        
        Returns:
            list: Annotations for each frame
        """
        if not all([self.yolo_model, self.fastreid_model, self.deepsort]):
            print("One or more models are not loaded. Cannot proceed with auto-annotation.")
            return []

        # Video Processing
        cap = cv2.VideoCapture(video_path)
        annotations = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # YOLO Detection
            results = self.yolo_model(frame)[0]
            
            # Filter for person class and confidence
            person_detections = results.boxes[
                (results.boxes.cls == 0) & 
                (results.boxes.conf > confidence_threshold)
            ]

            # Prepare detections for DeepSORT
            detection_list = []
            for det in person_detections:
                x1, y1, x2, y2 = det.xyxy[0].cpu().numpy()
                conf = det.conf.cpu().numpy()[0]
                
                # Extract ReID features
                person_crop = frame[int(y1):int(y2), int(x1):int(x2)]
                try:
                    reid_feature = self._extract_reid_feature(person_crop)
                except Exception as e:
                    print(f"ReID feature extraction error: {e}")
                    reid_feature = None

                detection_list.append({
                    'bbox': [x1, y1, x2-x1, y2-y1],
                    'confidence': conf,
                    'feature': reid_feature
                })

            # Track objects
            tracks = self.deepsort.update_tracks(detection_list, frame=frame)
            
            # Process tracks
            frame_annotations = []
            for track in tracks:
                if not track.is_confirmed():
                    continue

                track_id = track.track_id
                ltrb = track.to_ltrb()
                
                annotation = {
                    'track_id': track_id,
                    'bbox': {
                        'x1': ltrb[0],
                        'y1': ltrb[1],
                        'x2': ltrb[2],
                        'y2': ltrb[3]
                    }
                }
                frame_annotations.append(annotation)

            annotations.append(frame_annotations)

        cap.release()
        return annotations

    def _extract_reid_feature(self, person_image):
        """
        Extract ReID feature for a person image
        
        Args:
            person_image (np.ndarray): Cropped person image
        
        Returns:
            torch.Tensor: ReID feature embedding
        """
        # Preprocess image for ReID model
        preprocessed = self._preprocess_reid_image(person_image)
        
        # Extract features
        with torch.no_grad():
            features = self.fastreid_model(preprocessed)
        
        return features

    def _preprocess_reid_image(self, image):
        """
        Preprocess image for ReID feature extraction
        
        Args:
            image (np.ndarray): Input image
        
        Returns:
            torch.Tensor: Preprocessed image tensor
        """
        # Resize to model's expected input size
        image = cv2.resize(image, (256, 128))
        
        # Convert to tensor and normalize
        tensor = torch.from_numpy(image).float() / 255.0
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)
        
        return tensor

# Example Usage in Labelme App
class LabelmeApp:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Initialize Model Integration
        self.model_integration = ModelIntegration()
        self.model_integration.load_models()

    def start_auto_annotation(self, video_path):
        """
        Initiate automatic video annotation
        """
        annotations = self.model_integration.auto_annotate_video(video_path)
        self._save_annotations(annotations)

    def _save_annotations(self, annotations):
        """
        Save annotations in Labelme format
        """
        # Implement your annotation saving logic here
        pass
