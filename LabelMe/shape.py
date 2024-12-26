import copy
import math

import numpy as np
import skimage.measure
from qtpy import QtCore
from qtpy.QtCore import QRectF
from qtpy import QtGui

import labelme.utils
from labelme.logger import logger
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Set to DEBUG for more detailed logs
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),  # Logs to console
        logging.FileHandler("annotate_video.log"),  # Logs to file
    ]
)

# TODO(unknown):
# - [opt] Store paths instead of creating new ones at each paint.


class Shape(object):
    # Render handles as squares
    P_SQUARE = 0

    # Render handles as circles
    P_ROUND = 1

    # Flag for the handles we would move if dragging
    MOVE_VERTEX = 0

    # Flag for all other handles on the current shape
    NEAR_VERTEX = 1

    # The following class variables influence the drawing of all shape objects.
    line_color = None
    fill_color = None
    select_line_color = None
    select_fill_color = None
    vertex_fill_color = None
    hvertex_fill_color = None
    point_type = P_ROUND
    point_size = 8
    scale = 1.0

    def __init__(
        self,
        label=None,
        line_color=None,
        shape_type=None,
        shape_id=None,
        flags=None,
        group_id=None,
        description=None,
        mask=None,
        rect=None,
        confidence=None,
        bbox=None
        
        
    ):
        # if rect is None:
        #     # raise ValueError("Shape cannot be initialized with a NoneType rect.")
        #     rect = QtCore.QRectF(0, 0, 1, 1)  # Default rectangle with minimum size
        #     logging.warning("Initializing Shape with default QRectF.")
        
        # if not isinstance(rect, QtCore.QRectF):
        #     raise ValueError(f"Expected QRectF, but got {type(rect)}.")
        # if rect.isEmpty():
        #     raise ValueError("QRectF is empty. Invalid bounding box dimensions.")
        self.label = label
        self.group_id = group_id
        self.id= shape_id
        self.rect= rect if rect else QtCore.QRectF()
        self.shape_id = shape_id  # Add this line
        self.confidence=confidence
        # Log shape creation
        logging.debug(f"Created Shape: ID {shape_id}, BoundingBox {self.rect}, Confidence {confidence}")
        self.points = []
        self.point_labels = []
        self.shape_type = shape_type 
        self._shape_raw = None
        self._points_raw = []
        self._shape_type_raw = None
        self.fill = False
        self.selected = False
        self.flags = flags or {}
        self.description = description
        self.other_data = {}
        self.mask = mask
        self._bbox = bbox  # Ensure this is set

        self._highlightIndex = None
        self._highlightMode = self.NEAR_VERTEX
        self._highlightSettings = {
            self.NEAR_VERTEX: (4, self.P_ROUND),
            self.MOVE_VERTEX: (1.5, self.P_SQUARE),
        }

        self._closed = False

        if line_color is not None:
            # Override the class line_color attribute
            # with an object attribute. Currently this
            # is used for drawing the pending line a different color.
            self.line_color = line_color

    def setShapeRefined(self, shape_type, points, point_labels, mask=None):
        self._shape_raw = (self.shape_type, self.points, self.point_labels)
        self.shape_type = shape_type
        self.points = points
        self.point_labels = point_labels
        self.mask = mask

    def restoreShapeRaw(self):
        if self._shape_raw is None:
            return
        self.shape_type, self.points, self.point_labels = self._shape_raw
        self._shape_raw = None

    def boundingBox(self):
        if self.rect and not self.rect.isEmpty():
            logging.debug(f"Bounding box valid: {self.rect}")
            return self.rect
        logging.warning("Bounding box is invalid or empty.")
        return None
    
    @property
    def bbox(self):
        return self._bbox
    @property
    def shape_type(self):
        return self._shape_type

    @shape_type.setter
    def shape_type(self, value):
        if value is None:
            value = "polygon"
        if value not in [
            "polygon",
            "rectangle",
            "point",
            "line",
            "circle",
            "linestrip",
            "points",
            "mask",
        ]:
            raise ValueError("Unexpected shape_type: {}".format(value))
        self._shape_type = value

    
    
    
    def close(self):
        self._closed = True

    def addPoint(self, point, label=1):
        if self.points and point == self.points[0]:
            self.close()
        else:
            self.points.append(point)
            self._updateBoundingRect()
            self.point_labels.append(label)

    def canAddPoint(self):
        return self.shape_type in ["polygon", "linestrip"]

    def popPoint(self):
        if self.points:
            if self.point_labels:
                self.point_labels.pop()
            return self.points.pop()
        return None

    def insertPoint(self, i, point, label=1):
        self.points.insert(i, point)
        self.point_labels.insert(i, label)

    def removePoint(self, i):
        if not self.canAddPoint():
            logger.warning(
                "Cannot remove point from: shape_type=%r",
                self.shape_type,
            )
            return

        if self.shape_type == "polygon" and len(self.points) <= 3:
            logger.warning(
                "Cannot remove point from: shape_type=%r, len(points)=%d",
                self.shape_type,
                len(self.points),
            )
            return

        if self.shape_type == "linestrip" and len(self.points) <= 2:
            logger.warning(
                "Cannot remove point from: shape_type=%r, len(points)=%d",
                self.shape_type,
                len(self.points),
            )
            return

        self.points.pop(i)
        self.point_labels.pop(i)

    def _updateBoundingRect(self):
        if len(self.points) < 2:
            return
        x_coords = [p.x() for p in self.points]
        y_coords = [p.y() for p in self.points]
        self.rect = QtCore.QRectF(min(x_coords), min(y_coords), 
                                  max(x_coords) - min(x_coords), 
                                  max(y_coords) - min(y_coords))
    
    
    def isClosed(self):
        return self._closed

    def setOpen(self):
        self._closed = False

    def getRectFromLine(self, pt1, pt2):
        x1, y1 = pt1.x(), pt1.y()
        x2, y2 = pt2.x(), pt2.y()
        return QtCore.QRectF(x1, y1, x2 - x1, y2 - y1)

    def paint(self, painter):
        if self.mask is None and not self.points:
            return

        color = self.select_line_color if self.selected else self.line_color
        pen = QtGui.QPen(color)
        # Try using integer sizes for smoother drawing(?)
        pen.setWidth(max(1, int(round(2.0 / self.scale))))
        painter.setPen(pen)

        if self.mask is not None:
            image_to_draw = np.zeros(self.mask.shape + (4,), dtype=np.uint8)
            fill_color = (
                self.select_fill_color.getRgb()
                if self.selected
                else self.fill_color.getRgb()
            )
            image_to_draw[self.mask] = fill_color
            qimage = QtGui.QImage.fromData(labelme.utils.img_arr_to_data(image_to_draw))
            painter.drawImage(
                int(round(self.points[0].x())),
                int(round(self.points[0].y())),
                qimage,
            )

            line_path = QtGui.QPainterPath()
            contours = skimage.measure.find_contours(np.pad(self.mask, pad_width=1))
            for contour in contours:
                contour += [self.points[0].y(), self.points[0].x()]
                line_path.moveTo(contour[0, 1], contour[0, 0])
                for point in contour[1:]:
                    line_path.lineTo(point[1], point[0])
            painter.drawPath(line_path)

        if self.points:
            line_path = QtGui.QPainterPath()
            vrtx_path = QtGui.QPainterPath()
            negative_vrtx_path = QtGui.QPainterPath()

            if self.shape_type in ["rectangle", "mask"]:
                assert len(self.points) in [1, 2]
                if len(self.points) == 2:
                    rectangle = self.getRectFromLine(*self.points)
                    line_path.addRect(rectangle)
                if self.shape_type == "rectangle":
                    for i in range(len(self.points)):
                        self.drawVertex(vrtx_path, i)
            elif self.shape_type == "circle":
                assert len(self.points) in [1, 2]
                if len(self.points) == 2:
                    rectangle = self.getCircleRectFromLine(self.points)
                    line_path.addEllipse(rectangle)
                for i in range(len(self.points)):
                    self.drawVertex(vrtx_path, i)
            elif self.shape_type == "linestrip":
                line_path.moveTo(self.points[0])
                for i, p in enumerate(self.points):
                    line_path.lineTo(p)
                    self.drawVertex(vrtx_path, i)
            elif self.shape_type == "points":
                assert len(self.points) == len(self.point_labels)
                for i, point_label in enumerate(self.point_labels):
                    if point_label == 1:
                        self.drawVertex(vrtx_path, i)
                    else:
                        self.drawVertex(negative_vrtx_path, i)
            else:
                line_path.moveTo(self.points[0])
                # Uncommenting the following line will draw 2 paths
                # for the 1st vertex, and make it non-filled, which
                # may be desirable.
                # self.drawVertex(vrtx_path, 0)

                for i, p in enumerate(self.points):
                    line_path.lineTo(p)
                    self.drawVertex(vrtx_path, i)
                if self.isClosed():
                    line_path.lineTo(self.points[0])

            painter.drawPath(line_path)
            if vrtx_path.length() > 0:
                painter.drawPath(vrtx_path)
                painter.fillPath(vrtx_path, self._vertex_fill_color)
            if self.fill and self.mask is None:
                color = self.select_fill_color if self.selected else self.fill_color
                painter.fillPath(line_path, color)

            pen.setColor(QtGui.QColor(255, 0, 0, 255))
            painter.setPen(pen)
            painter.drawPath(negative_vrtx_path)
            painter.fillPath(negative_vrtx_path, QtGui.QColor(255, 0, 0, 255))

    def drawVertex(self, path, i):
        d = self.point_size / self.scale
        shape = self.point_type
        point = self.points[i]
        if i == self._highlightIndex:
            size, shape = self._highlightSettings[self._highlightMode]
            d *= size
        if self._highlightIndex is not None:
            self._vertex_fill_color = self.hvertex_fill_color
        else:
            self._vertex_fill_color = self.vertex_fill_color
        if shape == self.P_SQUARE:
            path.addRect(point.x() - d / 2, point.y() - d / 2, d, d)
        elif shape == self.P_ROUND:
            path.addEllipse(point, d / 2.0, d / 2.0)
        else:
            assert False, "unsupported vertex shape"

    def nearestVertex(self, point, epsilon):
        min_distance = float("inf")
        min_i = None
        for i, p in enumerate(self.points):
            dist = labelme.utils.distance(p - point)
            if dist <= epsilon and dist < min_distance:
                min_distance = dist
                min_i = i
        return min_i

    def nearestEdge(self, point, epsilon):
        min_distance = float("inf")
        post_i = None
        for i in range(len(self.points)):
            line = [self.points[i - 1], self.points[i]]
            dist = labelme.utils.distancetoline(point, line)
            if dist <= epsilon and dist < min_distance:
                min_distance = dist
                post_i = i
        return post_i

    def containsPoint(self, point):
        if self.mask is not None:
            y = np.clip(
                int(round(point.y() - self.points[0].y())),
                0,
                self.mask.shape[0] - 1,
            )
            x = np.clip(
                int(round(point.x() - self.points[0].x())),
                0,
                self.mask.shape[1] - 1,
            )
            return self.mask[y, x]
        return self.makePath().contains(point)

    def getCircleRectFromLine(self, line):
        """Computes parameters to draw with `QPainterPath::addEllipse`"""
        if len(line) != 2:
            return None
        (c, point) = line
        r = line[0] - line[1]
        d = math.sqrt(math.pow(r.x(), 2) + math.pow(r.y(), 2))
        rectangle = QtCore.QRectF(c.x() - d, c.y() - d, 2 * d, 2 * d)
        return rectangle

    def makePath(self):
        if not self.points or len(self.points) < 2:
            logging.warning(f"Shape has insufficient points; returning empty path. Points: {self.points}")
            return QtGui.QPainterPath()

        path = QtGui.QPainterPath()
        try:
            if self.shape_type in ["rectangle", "mask"]:
                if len(self.points) == 2:
                    rectangle = self.getRectFromLine(*self.points)
                    if rectangle.isEmpty():
                        raise ValueError(f"Generated rectangle is empty for shape ID {self.shape_id}")
                    path.addRect(rectangle)
                else:
                    raise ValueError(f"Invalid number of points for rectangle/mask shape. Points: {self.points}")
            elif self.shape_type == "circle":
                if len(self.points) == 2:
                    rectangle = self.getCircleRectFromLine(self.points)
                    if rectangle.isEmpty():
                        raise ValueError(f"Generated ellipse rectangle is empty for shape ID {self.shape_id}")
                    path.addEllipse(rectangle)
                else:
                    raise ValueError(f"Invalid number of points for circle shape. Points: {self.points}")
            else:
                path.moveTo(self.points[0])
                for p in self.points[1:]:
                    path.lineTo(p)
                path.closeSubpath()
        except Exception as e:
            logging.error(f"Error generating path for shape ID {self.shape_id}: {e}")
            return QtGui.QPainterPath()

        if path.isEmpty():
            logging.error(f"Generated path is empty for shape ID {self.shape_id}. Points: {self.points}")

        return path



    


    def move(self, dx, dy):
        """
        Move the shape by dx and dy.
        """
        self.rect.translate(dx, dy)
    
    def boundingRect(self):
        """
        Return the bounding rectangle of the shape.
        Returns:
            QRectF: The bounding rectangle of the shape, or None if invalid.
        """
        path = self.makePath()
        if path.isEmpty():
            logging.error(f"BoundingRect is empty for shape ID {self.shape_id}.")
            return QtCore.QRectF()  # Return an empty QRectF

        rect = path.boundingRect()
        if rect.isEmpty():
            logging.error(f"Generated QRectF is empty for shape ID {self.shape_id}.")
        return rect

    def isValid(self):
        """Check if the shape is valid."""
        # if not self.rect or self.rect.isNull() or self.rect.isEmpty():
        #     return False
        # if len(self.points) != 4:
        #     return False
        # return all(isinstance(p, QtCore.QPointF) for p in self.points)
       
        return (len(self.points) >= 4 and
                not self.boundingRect().isEmpty() and
                self.points[0] != self.points[-1])
    
    def moveBy(self, offset):
        self.points = [p + offset for p in self.points]

    def moveVertexBy(self, i, offset):
        self.points[i] = self.points[i] + offset

    def highlightVertex(self, i, action):
        """Highlight a vertex appropriately based on the current action

        Args:
            i (int): The vertex index
            action (int): The action
            (see Shape.NEAR_VERTEX and Shape.MOVE_VERTEX)
        """
        self._highlightIndex = i
        self._highlightMode = action

    def highlightClear(self):
        """Clear the highlighted point"""
        self._highlightIndex = None

    def copy(self):
        return copy.deepcopy(self)

    def __len__(self):
        return len(self.points)

    def __getitem__(self, key):
        return self.points[key]

    def __setitem__(self, key, value):
        self.points[key] = value
