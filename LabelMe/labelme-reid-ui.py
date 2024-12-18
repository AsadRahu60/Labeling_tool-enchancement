import sys
import os
from PyQt5.QtWidgets import (
    QMainWindow, QApplication, QWidget, QVBoxLayout, QHBoxLayout, 
    QLabel, QLineEdit, QComboBox, QCheckBox, QGroupBox, 
    QPushButton, QTabWidget, QTableWidget, QTableWidgetItem,
    QSplitter, QFrame, QScrollArea
)
from PyQt5.QtGui import QFont, QPixmap, QImage
from PyQt5.QtCore import Qt, QSize

class PersonReIDAnnotationTool(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
    
    def initUI(self):
        self.setWindowTitle('Person ReID Annotation Tool')
        self.setGeometry(100, 100, 1400, 800)
        
        # Main central widget and layout
        central_widget = QWidget()
        main_layout = QHBoxLayout()
        
        # Left Panel: Image and Annotation
        left_panel = QVBoxLayout()
        
        # Image Display
        self.image_display = QLabel('Image will be displayed here')
        self.image_display.setAlignment(Qt.AlignCenter)
        self.image_display.setMinimumSize(600, 400)
        self.image_display.setStyleSheet("""
            border: 2px solid #3498db;
            background-color: #f0f0f0;
        """)
        left_panel.addWidget(self.image_display)
        
        # Annotation Controls
        annotation_group = QGroupBox('Person Annotation')
        annotation_layout = QVBoxLayout()
        
        # Person ID Input
        person_id_layout = QHBoxLayout()
        person_id_label = QLabel('Person ID:')
        self.person_id_input = QLineEdit()
        person_id_layout.addWidget(person_id_label)
        person_id_layout.addWidget(self.person_id_input)
        annotation_layout.addLayout(person_id_layout)
        
        # Bounding Box Controls
        bbox_layout = QHBoxLayout()
        bbox_label = QLabel('Bounding Box:')
        self.bbox_x = QLineEdit('X')
        self.bbox_y = QLineEdit('Y')
        self.bbox_width = QLineEdit('Width')
        self.bbox_height = QLineEdit('Height')
        bbox_layout.addWidget(bbox_label)
        bbox_layout.addWidget(self.bbox_x)
        bbox_layout.addWidget(self.bbox_y)
        bbox_layout.addWidget(self.bbox_width)
        bbox_layout.addWidget(self.bbox_height)
        annotation_layout.addLayout(bbox_layout)
        
        # Additional Person Attributes
        attributes_group = QGroupBox('Person Attributes')
        attributes_layout = QHBoxLayout()
        
        # Gender Selection
        gender_layout = QVBoxLayout()
        gender_label = QLabel('Gender:')
        self.gender_combo = QComboBox()
        self.gender_combo.addItems(['Unknown', 'Male', 'Female'])
        gender_layout.addWidget(gender_label)
        gender_layout.addWidget(self.gender_combo)
        
        # Age Group
        age_layout = QVBoxLayout()
        age_label = QLabel('Age Group:')
        self.age_combo = QComboBox()
        self.age_combo.addItems(['Child', 'Young Adult', 'Middle-aged', 'Senior'])
        age_layout.addWidget(age_label)
        age_layout.addWidget(self.age_combo)
        
        # Clothing Color
        color_layout = QVBoxLayout()
        color_label = QLabel('Clothing Color:')
        self.color_combo = QComboBox()
        self.color_combo.addItems(['Black', 'White', 'Red', 'Blue', 'Green', 'Other'])
        color_layout.addWidget(color_label)
        color_layout.addWidget(self.color_combo)
        
        attributes_layout.addLayout(gender_layout)
        attributes_layout.addLayout(age_layout)
        attributes_layout.addLayout(color_layout)
        attributes_group.setLayout(attributes_layout)
        
        annotation_layout.addWidget(attributes_group)
        
        # Annotation Flags
        flags_layout = QHBoxLayout()
        self.occluded_check = QCheckBox('Occluded')
        self.truncated_check = QCheckBox('Truncated')
        self.difficult_check = QCheckBox('Difficult')
        flags_layout.addWidget(self.occluded_check)
        flags_layout.addWidget(self.truncated_check)
        flags_layout.addWidget(self.difficult_check)
        annotation_layout.addLayout(flags_layout)
        
        # Add Annotation Button
        self.add_annotation_btn = QPushButton('Add Annotation')
        annotation_layout.addWidget(self.add_annotation_btn)
        
        annotation_group.setLayout(annotation_layout)
        left_panel.addWidget(annotation_group)
        
        # Right Panel: Annotations and Gallery
        right_panel = QVBoxLayout()
        
        # Annotations Table
        self.annotations_table = QTableWidget()
        self.annotations_table.setColumnCount(8)
        self.annotations_table.setHorizontalHeaderLabels([
            'Person ID', 'X', 'Y', 'Width', 'Height', 
            'Gender', 'Age', 'Clothing Color'
        ])
        right_panel.addWidget(self.annotations_table)
        
        # Person Gallery
        gallery_group = QGroupBox('Detected Persons Gallery')
        gallery_layout = QHBoxLayout()
        self.gallery_scroll = QScrollArea()
        self.gallery_widget = QWidget()
        self.gallery_layout = QHBoxLayout(self.gallery_widget)
        self.gallery_scroll.setWidget(self.gallery_widget)
        self.gallery_scroll.setWidgetResizable(True)
        gallery_layout.addWidget(self.gallery_scroll)
        gallery_group.setLayout(gallery_layout)
        right_panel.addWidget(gallery_group)
        
        # Combine Layouts
        main_layout.addLayout(left_panel, 2)
        main_layout.addLayout(right_panel, 1)
        
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)
        
        # Connect Signals
        self.setup_connections()
    
    def setup_connections(self):
        # Example connections - you'd implement actual functionality
        self.add_annotation_btn.clicked.connect(self.add_annotation)
    
    def add_annotation(self):
        # Placeholder method for adding annotations
        row = self.annotations_table.rowCount()
        self.annotations_table.insertRow(row)
        
        # Add data to table
        self.annotations_table.setItem(row, 0, QTableWidgetItem(self.person_id_input.text()))
        self.annotations_table.setItem(row, 1, QTableWidgetItem(self.bbox_x.text()))
        self.annotations_table.setItem(row, 2, QTableWidgetItem(self.bbox_y.text()))
        self.annotations_table.setItem(row, 3, QTableWidgetItem(self.bbox_width.text()))
        self.annotations_table.setItem(row, 4, QTableWidgetItem(self.bbox_height.text()))
        self.annotations_table.setItem(row, 5, QTableWidgetItem(self.gender_combo.currentText()))
        self.annotations_table.setItem(row, 6, QTableWidgetItem(self.age_combo.currentText()))
        self.annotations_table.setItem(row, 7, QTableWidgetItem(self.color_combo.currentText()))

def main():
    app = QApplication(sys.argv)
    ex = PersonReIDAnnotationTool()
    ex.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
