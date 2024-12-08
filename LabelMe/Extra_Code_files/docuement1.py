from pathlib import Path

# Create the documentation file for all changes
documentation_content = """
# Changes Made to the LabelMe Project

## Overview
The following document outlines the major enhancements, bug fixes, and integrations implemented into the LabelMe project to support video annotation, tracking, and re-identification.

---

## Enhancements

### 1. Integration of IDManager Class
- Ensured track IDs are sequential and reused.
- Resolved issues of skipping IDs in the tracking process.
- Introduced methods:
  - `get_new_id`: Fetches a unique ID, reusing released ones.
  - `release_id`: Releases unused IDs for recycling.

### 2. Annotate Video Enhancements
- Integrated YOLO, FastReID, and DeepSORT tracking.
- Enhanced logic for handling:
  - Bounding box validation.
  - Feature vector size validation.
- Added support for:
  - Unique color assignment for each track ID.
  - Annotating frames with bounding boxes, IDs, and labels.
  - Releasing unused track IDs to avoid skipping.

### 3. Improved Label Management
- Enhanced `LabelListWidget` and `UniqueLabelQListWidget`:
  - Dynamically update label lists with IDs and colors.
  - Ensured no duplicate labels.

### 4. Error Handling
- Added error checks for:
  - Feature vector sizes (e.g., expected embedding size: 2048).
  - Invalid bounding boxes.
  - Tracker update failures.

---

## Key Changes in `app.py`
1. Added the `IDManager` class to manage track IDs.
2. Updated `annotateVideo` method to:
   - Integrate IDManager for consistent ID handling.
   - Annotate frames with unique colors and labels.
   - Save annotations as JSON.
3. Enhanced logic to release IDs for unconfirmed or inactive tracks.

---

## Error Tracking
All encountered errors and their resolutions are documented in the `error.txt` file.

---

## Integration Notes
- Ensure the following libraries are installed:
  - `torch`
  - `cv2`
  - `qtpy`
- Verify that YOLO and FastReID models are preloaded and accessible.
"""

# Write documentation file
Path("changes_documentation.txt").write_text(documentation_content)

# Create the error log file
error_log_content = """
# Error Log

## Error 1: Skipping IDs in Tracking
**Description:** Track IDs were not sequential, and some were skipped.
**Resolution:**
- Integrated `IDManager` class to manage and reuse IDs.
- Released unused IDs when tracks ended.

## Error 2: Invalid Feature Vector Size
**Description:** Mismatch in feature vector size (expected 2048).
**Resolution:**
- Added validation for feature vector size in the `annotateVideo` method.

## Error 3: Invalid Bounding Boxes
**Description:** Bounding boxes outside the frame dimensions were detected.
**Resolution:**
- Added checks to validate and normalize bounding boxes.

## Error 4: Tracker Update Failure
**Description:** DeepSORT tracker failed to update due to malformed detections.
**Resolution:**
- Added error handling and validation for detections before updating the tracker.

## Error 5: Duplicate Labels in Label List
**Description:** Duplicate labels appeared in the `LabelListWidget`.
**Resolution:**
- Checked for existing labels before adding new ones.

---

### Debugging Notes
- Ensure all models and dependencies are correctly configured.
- Validate the input video and annotations for consistency.
"""

# Write error log file
Path("error_log.txt").write_text(error_log_content)

# Paths of the created files
["changes_documentation.txt", "error_log.txt"]
