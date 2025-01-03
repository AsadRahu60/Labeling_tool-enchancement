# dataset_processing.py
import logging
import os.path as osp
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt
import PyQt5.QtGui as QtGui
import labelme.utils
import cv2
from PIL import Image

logger = logging.getLogger(__name__)

class DatasetProcessor:
    """Handles image dataset processing and annotation."""
    
    def __init__(self, main_window):
        if not main_window:
            raise ValueError("Main window reference is  required")
        self.main_window = main_window  # Reference to MainWindow
        self.progress = None
        self._widget=None
        self._processing=  False

    def configure_processing(self):
       
        """Configure dataset processing options."""
        dialog = QtWidgets.QDialog(self.main_window)
        dialog.setWindowTitle("Dataset Processing Configuration")
        layout = QtWidgets.QVBoxLayout()

        # Batch size
        batch_layout = QtWidgets.QHBoxLayout()
        batch_layout.addWidget(QtWidgets.QLabel("Batch Size:"))
        batch_spinbox = QtWidgets.QSpinBox()
        batch_spinbox.setRange(10, 200)
        batch_spinbox.setValue(50)
        batch_layout.addWidget(batch_spinbox)
        layout.addLayout(batch_layout)

        # Confidence threshold
        conf_layout = QtWidgets.QHBoxLayout()
        conf_layout.addWidget(QtWidgets.QLabel("Confidence Threshold:"))
        conf_spinbox = QtWidgets.QDoubleSpinBox()
        conf_spinbox.setRange(0.1, 1.0)
        conf_spinbox.setValue(0.5)
        conf_spinbox.setSingleStep(0.1)
        conf_layout.addWidget(conf_spinbox)
        layout.addLayout(conf_layout)

        # Dataset splitting
        split_check = QtWidgets.QCheckBox("Split into Train/Test")
        layout.addWidget(split_check)

        split_layout = QtWidgets.QHBoxLayout()
        split_layout.addWidget(QtWidgets.QLabel("Train Split Ratio:"))
        split_spinbox = QtWidgets.QDoubleSpinBox()
        split_spinbox.setRange(0.1, 0.9)
        split_spinbox.setValue(0.8)
        split_spinbox.setSingleStep(0.1)
        split_layout.addWidget(split_spinbox)
        layout.addLayout(split_layout)

        # Save intermediate results
        save_intermediate = QtWidgets.QCheckBox("Save Intermediate Results")
        layout.addWidget(save_intermediate)

        # Export format
        format_layout = QtWidgets.QHBoxLayout()
        format_layout.addWidget(QtWidgets.QLabel("Export Format:"))
        format_combo = QtWidgets.QComboBox()
        format_combo.addItems(["COCO", "YOLO", "Pascal VOC", "JSON"])
        format_layout.addWidget(format_combo)
        layout.addLayout(format_layout)

        # Buttons
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)

        dialog.setLayout(layout)
        
        if dialog.exec_():
            return {
                'batch_size': batch_spinbox.value(),
                'confidence_threshold': conf_spinbox.value(),
                'split_dataset': split_check.isChecked(),
                'split_ratio': split_spinbox.value(),
                'save_intermediate': save_intermediate.isChecked(),
                'export_format': format_combo.currentText()
            }
        return None

    def process_dataset(self, directory_path):
        try:
            config = self.configure_processing()
            if not config:
                return
            
            
            # Initialize models before processing
            if not self.main_window.detectionModelSelector.currentText():
                QtWidgets.QMessageBox.critical(
                    self.main_window,
                    "Error",
                    "Please select a detection model first."
                )
                return
                
            if not self.main_window.reidModelSelector.currentText():
                QtWidgets.QMessageBox.critical(
                    self.main_window,
                    "Error",
                    "Please select a ReID model first."
                )
                return
                
            # Load models
            if not self.main_window.load_models():
                QtWidgets.QMessageBox.critical(
                    self.main_window,
                    "Error",
                    "Failed to initialize models. Please check model selections."
                )
                return
            
            
            
            

            self._widget = QtWidgets.QWidget()
            self.progress = QtWidgets.QProgressDialog(
                "Processing images...", "Cancel", 
                0, 100, self._widget
            )
            self.progress.setWindowModality(Qt.WindowModal)

            image_files = self.main_window.scanAllImages(directory_path)
            total_images = len(image_files)
            
            # Collect all annotations
            all_frames = []
            all_annotations = []

            for i, batch in enumerate(self.get_batches(image_files, config['batch_size'])):
                if self.progress.wasCanceled():
                    break
                    
                frames, annotations = self.process_batch(batch, config)
                if frames and annotations:
                    all_frames.extend(frames)
                    all_annotations.extend(annotations)
                    
                progress = int((i * config['batch_size'] / total_images) * 100)
                self.progress.setValue(progress)
                QtWidgets.QApplication.processEvents()

            # After all processing is done, save annotations
            if all_annotations:
                format_choice = self.main_window.choose_annotation_format()
                if format_choice:
                    try:
                        save_path = QtWidgets.QFileDialog.getSaveFileName(
                            self._widget,
                            "Save Annotations",
                            "",
                            f"{format_choice.upper()} Files (*.{format_choice})"
                        )[0]
                        
                        if save_path:
                            self.main_window.save_reid_annotations(
                                all_frames,
                                all_annotations,
                                format_choice
                                
                            )
                            QtWidgets.QMessageBox.information(
                                self._widget,
                                "Success",
                                "Annotations saved successfully!"
                            )
                    except Exception as save_error:
                        logger.error(f"Error saving annotations: {save_error}")

        except Exception as e:
            logger.error(f"Dataset processing error: {e}")
        finally:
            if self._widget:
                self._widget.deleteLater()

    def process_batch(self, image_batch, config):
        try:
                # Validate models
            if not self.main_window.load_models:
                QtWidgets.QMessageBox.critical(
                    self.main_window,
                    "Error",
                    "Detection or ReID model not initialized. Please select models first."
                )
                return None, None
                
            # Initialize person_colors if not exists
            if not hasattr(self.main_window, 'person_colors'):
                self.main_window.person_colors = {}

            # Collection for all annotations
            all_frames = []
            all_annotations = []

            for img_path in image_batch:
                try:
                        # Load image using cv2 directly
                    frame = cv2.imread(img_path)
                    if frame is None:
                        logger.error(f"Failed to load image: {img_path}")
                        continue
                        
                    # Convert BGR to RGB
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Load image in UI
                    qimage = QtGui.QImage(img_path)
                    if not qimage.isNull():
                        self.main_window.image = qimage
                        self.main_window.canvas.loadPixmap(QtGui.QPixmap.fromImage(qimage))
                        self.main_window.canvas.update()
                        
                    detections = self.main_window.process_image(frame)
                    if detections:
                        tracked_objects = self.main_window.update_tracker(
                            detections['bbox_xywh'],
                            detections['confidence'],
                            detections['features']
                            
                        )
                        
                        if tracked_objects:
                            frame_annotations = self.main_window.process_tracks(
                                frame, 
                                self.main_window.person_colors,
                                tracked_objects
                            )
                            
                            if frame_annotations:
                                # Store frame and its annotations
                                all_frames.append(frame)
                                all_annotations.append({
                                    'file_path': img_path,
                                    'annotations': frame_annotations
                                })
                    
                    QtWidgets.QApplication.processEvents()
                    
                except Exception as img_error:
                    logger.error(f"Error processing image {img_path}: {img_error}")
                    continue
                    
            # After processing all images, save annotations if required
            if all_annotations:
                return all_frames, all_annotations
                
        except Exception as e:
            logger.error(f"Batch processing error: {e}")
            return None, None

    def get_batches(self, items, batch_size):
        """Yield batches of items"""
        for i in range(0, len(items), batch_size):
            yield items[i:i + batch_size]

    def save_annotations(self, annotations, image_path):
        """Save annotations for an image"""
        try:
            save_path = osp.splitext(image_path)[0] + ".json"
            self.main_window.save_reid_annotations(
                None, 
                annotations, 
                "json", 
                save_path
            )
        except Exception as e:
            logger.error(f"Error saving annotations: {e}")
            
            
    
    def validate_image(file_path):
        try:
            with Image.open(file_path) as img:
                img.verify()  # Verify the file's integrity
            return True
        except Exception as e:
            logger.error(f"Invalid image file {file_path}: {e}")
            return False