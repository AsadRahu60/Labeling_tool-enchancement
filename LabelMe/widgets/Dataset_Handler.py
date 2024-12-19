import zipfile
import h5py
import os
import cv2
import numpy as np
from pathlib import Path
import sys
import os

# Add the labelme directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from widgets.Data_Handler import DatasetHandler




import logging


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class CUHK03Handler(DatasetHandler):
    def extract_files(self):
        """Extract .zip file for CUHK03 if necessary."""
        if self.dataset_path.suffix == '.zip':
            try:
                extract_dir = self.output_path / "cuhk03_release"
                extract_dir.mkdir(parents=True, exist_ok=True)
                
                with zipfile.ZipFile(self.dataset_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_dir)
                
                # Resolve redundant directory nesting
                nested_path = extract_dir / "cuhk03_release"
                if nested_path.exists() and nested_path.is_dir():
                    for item in nested_path.iterdir():
                        item.rename(extract_dir / item.name)
                    nested_path.rmdir()
                
                logging.info(f"Extracted CUHK03 dataset to: {extract_dir}")
                self.dataset_path = extract_dir  # Update dataset path to extracted location
            
            except Exception as e:
                logging.error(f"Failed to extract CUHK03 dataset: {e}")
                raise


    def validate_structure(self):
        """Ensure CUHK03 has required .mat and protocol files."""
        mat_file = self.dataset_path / "cuhk03_release" / "cuhk-03.mat"
        protocol_file = self.dataset_path / "cuhk03_new_protocol_config_detected.mat"
        if not mat_file.exists() or not protocol_file.exists():
            logging.error("CUHK03 .mat or protocol file is missing.")
            raise FileNotFoundError("CUHK03 .mat or protocol file missing.")
        logging.info("CUHK03 dataset structure validated successfully.")

    def process_annotations(self):
        """Extract train/query/gallery splits from CUHK03 protocol files."""
        protocol_file = self.dataset_path / "cuhk03_new_protocol_config_detected.mat"
        mat_file = self.dataset_path / "cuhk03_release" / "cuhk-03.mat"
        
        if not protocol_file.exists() or not mat_file.exists():
            logging.error("Protocol file or .mat file is missing.")
            raise FileNotFoundError("CUHK03 .mat or protocol file missing.")
        
        try:
            # Read protocol file
            with h5py.File(protocol_file, 'r') as f:
                self.train_idx = np.array(f['train_idx']).flatten() - 1
                self.query_idx = np.array(f['query_idx']).flatten() - 1
                self.gallery_idx = np.array(f['gallery_idx']).flatten() - 1
                logging.info("Loaded train/query/gallery indices from protocol file.")
            
            # Verify access to .mat file (if additional processing is needed)
            with h5py.File(mat_file, 'r') as f:
                if 'labeled' not in f and 'detected' not in f:
                    logging.error("CUHK03 .mat file does not have labeled or detected keys.")
                    raise ValueError("Invalid .mat file format.")
            logging.info("CUHK03 annotations processed successfully.")
        
        except Exception as e:
            logging.error(f"Failed to process CUHK03 annotations: {e}")
            raise


    def split_dataset(self):
        """Split CUHK03 dataset into train/query/gallery."""
        train_dir = self.output_path / "bounding_box_train"
        query_dir = self.output_path / "query"
        gallery_dir = self.output_path / "bounding_box_test"

        train_dir.mkdir(parents=True, exist_ok=True)
        query_dir.mkdir(parents=True, exist_ok=True)
        gallery_dir.mkdir(parents=True, exist_ok=True)

        mat_file = self.dataset_path / "cuhk03_release" / "cuhk-03.mat"
        try:
            with h5py.File(mat_file, 'r') as f:
                detected = [f[f['detected'][0][i]] for i in range(len(f['detected'][0]))]
                for idx_set, split_dir in zip(
                    [self.train_idx, self.query_idx, self.gallery_idx],
                    [train_dir, query_dir, gallery_dir]
                ):
                    for person_idx in idx_set:
                        person_images = detected[person_idx]
                        for img_idx, img_ref in enumerate(person_images):
                            image = np.array(f[img_ref]).transpose((2, 1, 0))
                            image_filename = split_dir / f"{person_idx:04d}_{img_idx:04d}.jpg"
                            cv2.imwrite(str(image_filename), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            logging.info("CUHK03 dataset split successfully into train/query/gallery.")
        except Exception as e:
            logging.error(f"Failed to split CUHK03 dataset: {e}")
            raise


class Market1501Handler(DatasetHandler):
    def extract_files(self):
        """Ensure Market1501 directory structure is correct."""
        required_dirs = ["bounding_box_train", "query", "bounding_box_test"]
        for dir_name in required_dirs:
            if not (self.dataset_path / dir_name).exists():
                raise FileNotFoundError(f"Directory {dir_name} is missing in {self.dataset_path}")

    def process_annotations(self):
        """No additional annotations processing needed for Market1501."""
        pass

    def split_dataset(self):
        """Validate Market1501 dataset structure."""
        print("Market1501 dataset is already split into train/query/gallery.")

# Add additional handlers for other datasets here as needed

