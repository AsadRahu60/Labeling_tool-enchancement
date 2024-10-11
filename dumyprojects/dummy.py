import sys
from PyQt5.QtWidgets import QApplication, QFileDialog, QWidget, QVBoxLayout, QPushButton, QLabel
from PyQt5.QtGui import QImage, QPixmap
import cv2
import numpy as np

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        self.setWindowTitle("Video Frame Extractor")
        self.setGeometry(300, 300, 800, 600)

        layout = QVBoxLayout()
        self.setLayout(layout)

        # Video upload button
        uploadButton = QPushButton("Upload Video")
        uploadButton.clicked.connect(self.uploadVideo)
        layout.addWidget(uploadButton)

        # Frame display area
        self.frameLabel = QLabel()
        layout.addWidget(self.frameLabel)

        # Previous and next frame navigation buttons
        prevButton = QPushButton("Previous Frame")
        prevButton.clicked.connect(self.prevFrame)
        layout.addWidget(prevButton)

        nextButton = QPushButton("Next Frame")
        nextButton.clicked.connect(self.nextFrame)
        layout.addWidget(nextButton)

    def uploadVideo(self):
        # Open file dialog to select video file
        filename, _ = QFileDialog.getOpenFileName(self, "Select Video File", "", "Video Files (*.mp4 *.avi *.mov)")

        if filename:
            # Create a VideoUploader instance to extract frames
            self.videoUploader = VideoUploader(filename)
            self.videoUploader.extractFrames()

            # Display the first frame
            self.displayFrame(0)

    def prevFrame(self):
        # Get the current frame index
        currentFrame = self.videoUploader.getCurrentFrame()

        # Decrement the frame index and display the previous frame
        self.displayFrame(currentFrame - 1)

    def nextFrame(self):
        # Get the current frame index
        currentFrame = self.videoUploader.getCurrentFrame()

        # Increment the frame index and display the next frame
        self.displayFrame(currentFrame + 1)

    def displayFrame(self, frameIndex):
        
        # Get the frame at the specified index
        frame = self.videoUploader.getFrame(frameIndex)

        # Convert the frame to a QImage object
        height, width, channel = frame.shape
        bytesPerLine = 3 * width
        qImg = QImage(frame.data, width, height, bytesPerLine, QImage.Format_RGB888).rgbSwapped()

        # Convert the QImage to a QPixmap and display it
        pixmap = QPixmap.fromImage(qImg)
        self.frameLabel.setPixmap(pixmap)

class VideoUploader:
    def __init__(self, filename):
        self.filename = filename
        self.frames = []
        self.currentFrame = 0

    def extractFrames(self):
        # Open the video file using OpenCV
        cap = cv2.VideoCapture(self.filename)

        # Extract frames from the video
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            self.frames.append(frame)

        cap.release()

    def getCurrentFrame(self):
        return self.currentFrame

    def getFrame(self, frameIndex):
        return self.frames[frameIndex]

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())