import os
import zipfile
from abc import ABC, abstractmethod
from pathlib import Path

class DatasetHandler(ABC):
    def __init__(self, dataset_path, output_path):
        """
        Base class for handling dataset preparation.

        Args:
            dataset_path (str): Path to the dataset source.
            output_path (str): Path to save processed dataset.
        """
        self.dataset_path = Path(dataset_path)
        self.output_path = Path(output_path)

    @abstractmethod
    def extract_files(self):
        """Extract files from the dataset source."""
        pass

    @abstractmethod
    def process_annotations(self):
        """Parse annotations specific to the dataset."""
        pass

    @abstractmethod
    def split_dataset(self):
        """Split the dataset into train/query/gallery."""
        pass

    def prepare_dataset(self):
        """Generalized workflow for dataset preparation."""
        print(f"Preparing dataset at: {self.dataset_path}")
        self.extract_files()
        self.process_annotations()
        self.split_dataset()
        print(f"Dataset prepared successfully at: {self.output_path}")
