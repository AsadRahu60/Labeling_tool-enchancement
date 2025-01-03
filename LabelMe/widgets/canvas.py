import math
import imgviz
from qtpy import QtCore
from qtpy.QtCore import Qt
from qtpy import QtGui
from qtpy import QtWidgets
from shapely.geometry import box 
from PyQt5.QtCore import QRectF
import labelme.ai
import labelme.utils
from labelme import QT5
from labelme.logger import logger
from labelme.shape import Shape
import random
# TODO(unknown):
# - [maybe] Find optimal epsilon value.
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


logger = logging.getLogger(__name__)

CURSOR_DEFAULT = QtCore.Qt.ArrowCursor
CURSOR_POINT = QtCore.Qt.PointingHandCursor
CURSOR_DRAW = QtCore.Qt.CrossCursor
CURSOR_MOVE = QtCore.Qt.ClosedHandCursor
CURSOR_GRAB = QtCore.Qt.OpenHandCursor

MOVE_SPEED = 5.0


class Canvas(QtWidgets.QWidget):
    zoomRequest = QtCore.Signal(int, QtCore.QPoint)
    scrollRequest = QtCore.Signal(int, int)
    newShape = QtCore.Signal(object)
    selectionChanged = QtCore.Signal(bool)
    shapeMoved = QtCore.Signal()
    drawingPolygon = QtCore.Signal(bool)
    vertexSelected = QtCore.Signal(bool)
    mouseMoved = QtCore.Signal(QtCore.QPointF)

    CREATE, EDIT = 0, 1

    # polygon, rectangle, line, or point
    _createMode = "polygon"

    _fill_drawing = False

    def __init__(self, *args, **kwargs):
        self.epsilon = kwargs.pop("epsilon", 10.0)
        self.double_click = kwargs.pop("double_click", "close")
        if self.double_click not in [None, "close"]:
            raise ValueError(
                "Unexpected value for double_click event: {}".format(self.double_click)
            )
        self.num_backups = kwargs.pop("num_backups", 10)
        self._crosshair = kwargs.pop(
            "crosshair",
            {
                "polygon": False,
                "rectangle": True,
                "circle": False,
                "line": False,
                "point": False,
                "linestrip": False,
                "ai_polygon": False,
                "ai_mask": False,
            },
        )
        super(Canvas, self).__init__(*args, **kwargs)
        # Initialise local state.
        self.mode = self.EDIT
        self.shapes = []
        self.shapesBackups = []
        self.current = None
        self.selectedShape=None
        self.selectedShapes = []  # save the selected shapes here
        self.selectedShapesCopy = []
        
        # self.selectionChanged = QtCore.Signal(bool)  # Signal indicating change in selection
        # self.line represents:
        #   - createMode == 'polygon': edge from last point to current
        #   - createMode == 'rectangle': diagonal line of the rectangle
        #   - createMode == 'line': the line
        #   - createMode == 'point': the point
        self.line = Shape(QtCore.QRectF(0, 0, 1, 1))  # Default QRectF with minimum dimensions
        

        self.prevPoint = QtCore.QPoint()
        self.prevMovePoint = QtCore.QPoint()
        self.offsets = QtCore.QPoint(), QtCore.QPoint()
        self.scale = 1.0
        self.pixmap = QtGui.QPixmap()
        self.visible = {}
        self._hideBackround = False
        self.hideBackround = False
        self.hShape = None
        self.prevhShape = None
        self.hVertex = None
        self.prevhVertex = None
        self.hEdge = None
        self.prevhEdge = None
        self.movingShape = False
        self.snapping = True
        self.hShapeIsSelected = False
        self._painter = QtGui.QPainter()
        self._cursor = CURSOR_DEFAULT
        shape = Shape()
        print(type (shape))
        print(dir(shape))
        self.current_frame=0
        
        
        # Menus:
        # 0: right-click without selection and dragging of shapes
        # 1: right-click with selection and dragging of shapes
        self.menus = (QtWidgets.QMenu(), QtWidgets.QMenu())
        # Set widget options.
        self.setMouseTracking(True)
        self.setFocusPolicy(QtCore.Qt.WheelFocus)

        self._ai_model = None

        


    
    
    
    def setCurrentFrame(self, frame_number):
        """Set current frame number"""
        self.current_frame = frame_number

    def getCurrentFrame(self):
        """Get current frame number"""
        return self.current_frame
    
    
    
    def fillDrawing(self):
        return self._fill_drawing

    def setFillDrawing(self, value):
        self._fill_drawing = value

    @property
    def createMode(self):
        return self._createMode

    @createMode.setter
    def createMode(self, value):
        if value not in [
            "polygon",
            "rectangle",
            "circle",
            "line",
            "point",
            "linestrip",
            "ai_polygon",
            "ai_mask",
        ]:
            raise ValueError("Unsupported createMode: %s" % value)
        self._createMode = value

    def initializeAiModel(self, name):
        if name not in [model.name for model in labelme.ai.MODELS]:
            raise ValueError("Unsupported ai model: %s" % name)
        model = [model for model in labelme.ai.MODELS if model.name == name][0]

        if self._ai_model is not None and self._ai_model.name == model.name:
            logger.debug("AI model is already initialized: %r" % model.name)
        else:
            logger.debug("Initializing AI model: %r" % model.name)
            self._ai_model = model()

        if self.pixmap is None:
            logger.warning("Pixmap is not set yet")
            return

        self._ai_model.set_image(
            image=labelme.utils.img_qt_to_arr(self.pixmap.toImage())
        )
    def addShape(self, shape):
        if not shape or not shape.isValid() :
            logger.error(f"Cannot add invalid shape: {shape}")
            return
            # Add position validation
        if self.is_duplicate(shape.shape_id, shape.bbox):
            logger.warning(f"Duplicate shape rejected: ID={shape.shape_id}")
            return
        self.shapes.append(shape)
        logger.info(f"Shape added: ID={shape.shape_id}, bbox={shape.boundingRect()}")

        bbox = shape.boundingRect()
        if bbox is None or bbox.isEmpty():
            logging.error(f"Invalid bounding box for shape ID {shape.id}. Falling back to default size.")
            # Fallback: Create a minimal valid bounding rectangle
            fallback_rect = QtCore.QRectF(0, 0, 10, 10)
            shape.rect = fallback_rect
            shape.addPoint(fallback_rect.topLeft())
            shape.addPoint(fallback_rect.topRight())
            shape.addPoint(fallback_rect.bottomRight())
            shape.addPoint(fallback_rect.bottomLeft())
            logging.info(f"Fallback bounding box applied for shape ID {shape.id}. Rect: {fallback_rect}")

        self.shapes.append(shape)
        logging.info(f"Shape added to canvas: ID {shape.id}, BoundingBox {shape.boundingRect()}")




    def removeShape(self, shape):
        """
        Remove a shape from the canvas.
        """
        if shape in self.shapes:
            self.shapes.remove(shape)
            self.update()

    def clearShapes(self):
        """
        Clear all shapes from the canvas.
        """
        self.shapes = []
        self.selectedShape = None
        self.update()
    
    
    def storeShapes(self):
        shapesBackup = []
        for shape in self.shapes:
            shapesBackup.append(shape.copy())
        if len(self.shapesBackups) > self.num_backups:
            self.shapesBackups = self.shapesBackups[-self.num_backups - 1 :]
        self.shapesBackups.append(shapesBackup)

    @property
    def isShapeRestorable(self):
        # We save the state AFTER each edit (not before) so for an
        # edit to be undoable, we expect the CURRENT and the PREVIOUS state
        # to be in the undo stack.
        if len(self.shapesBackups) < 2:
            return False
        return True

    def restoreShape(self):
        # This does _part_ of the job of restoring shapes.
        # The complete process is also done in app.py::undoShapeEdit
        # and app.py::loadShapes and our own Canvas::loadShapes function.
        if not self.isShapeRestorable:
            return
        self.shapesBackups.pop()  # latest

        # The application will eventually call Canvas.loadShapes which will
        # push this right back onto the stack.
        shapesBackup = self.shapesBackups.pop()
        self.shapes = shapesBackup
        self.selectedShapes = []
        for shape in self.shapes:
            shape.selected = False
        self.update()

    def enterEvent(self, ev):
        self.overrideCursor(self._cursor)

    def leaveEvent(self, ev):
        self.unHighlight()
        self.restoreCursor()

    def focusOutEvent(self, ev):
        self.restoreCursor()

    def isVisible(self, shape):
        return self.visible.get(shape, True)

    def drawing(self):
        return self.mode == self.CREATE

    def editing(self):
        return self.mode == self.EDIT

    def setEditing(self, value=True):
        self.mode = self.EDIT if value else self.CREATE
        if self.mode == self.EDIT:
            # CREATE -> EDIT
            self.repaint()  # clear crosshair
        else:
            # EDIT -> CREATE
            self.unHighlight()
            self.deSelectShape()

    def unHighlight(self):
        if self.hShape:
            self.hShape.highlightClear()
            self.update()
        self.prevhShape = self.hShape
        self.prevhVertex = self.hVertex
        self.prevhEdge = self.hEdge
        self.hShape = self.hVertex = self.hEdge = None

    def selectedVertex(self):
        return self.hVertex is not None

    def selectedEdge(self):
        return self.hEdge is not None


    def assignColor(self, shape_id):
        """
        Assigns a unique color to a shape ID and returns the color.
        """
        if not hasattr(self, 'colors'):  # Ensure colors dictionary exists
            self.colors = {}
        if shape_id not in self.colors:
            self.colors[shape_id] = (
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255),
            )
        return self.colors[shape_id]

    def mouseMoveEvent(self, ev):
        """Update line with last point and current coordinates."""
        try:
            if QT5:
                pos = self.transformPos(ev.localPos())
            else:
                pos = self.transformPos(ev.posF())
        except AttributeError:
            return

        self.mouseMoved.emit(pos)

        self.prevMovePoint = pos
        self.restoreCursor()

        is_shift_pressed = ev.modifiers() & QtCore.Qt.ShiftModifier
        
        if self.selectedShapes and ev.buttons() == Qt.LeftButton:
            dx = ev.pos().x() - self.selectedShape.boundingBox().x()
            dy = ev.pos().y() - self.selectedShape.boundingBox().y()
            self.selectedShape.move(dx, dy)
            self.shapeMoved.emit()
        self.update()
           

        # Polygon drawing.
        if self.drawing():
            if self.createMode in ["ai_polygon", "ai_mask"]:
                self.line.shape_type = "points"
            else:
                self.line.shape_type = self.createMode

            self.overrideCursor(CURSOR_DRAW)
            if not self.current:
                self.repaint()  # draw crosshair
                return

            if self.outOfPixmap(pos):
                # Don't allow the user to draw outside the pixmap.
                # Project the point to the pixmap's edges.
                pos = self.intersectionPoint(self.current[-1], pos)
            elif (
                self.snapping
                and len(self.current) > 1
                and self.createMode == "polygon"
                and self.closeEnough(pos, self.current[0])
            ):
                # Attract line to starting point and
                # colorise to alert the user.
                pos = self.current[0]
                self.overrideCursor(CURSOR_POINT)
                self.current.highlightVertex(0, Shape.NEAR_VERTEX)
            if self.createMode in ["polygon", "linestrip"]:
                self.line.points = [self.current[-1], pos]
                self.line.point_labels = [1, 1]
            elif self.createMode in ["ai_polygon", "ai_mask"]:
                self.line.points = [self.current.points[-1], pos]
                self.line.point_labels = [
                    self.current.point_labels[-1],
                    0 if is_shift_pressed else 1,
                ]
            elif self.createMode == "rectangle":
                self.line.points = [self.current[0], pos]
                self.line.point_labels = [1, 1]
                self.line.close()
            elif self.createMode == "circle":
                self.line.points = [self.current[0], pos]
                self.line.point_labels = [1, 1]
                self.line.shape_type = "circle"
            elif self.createMode == "line":
                self.line.points = [self.current[0], pos]
                self.line.point_labels = [1, 1]
                self.line.close()
            elif self.createMode == "point":
                self.line.points = [self.current[0]]
                self.line.point_labels = [1]
                self.line.close()
            assert len(self.line.points) == len(self.line.point_labels)
            self.repaint()
            self.current.highlightClear()
            return

        # Polygon copy moving.
        if QtCore.Qt.RightButton & ev.buttons():
            if self.selectedShapesCopy and self.prevPoint:
                self.overrideCursor(CURSOR_MOVE)
                self.boundedMoveShapes(self.selectedShapesCopy, pos)
                self.repaint()
            elif self.selectedShapes:
                self.selectedShapesCopy = [s.copy() for s in self.selectedShapes]
                self.repaint()
            return

        # Polygon/Vertex moving.
        if QtCore.Qt.LeftButton & ev.buttons():
            if self.selectedVertex():
                self.boundedMoveVertex(pos)
                self.repaint()
                self.movingShape = True
            elif self.selectedShapes and self.prevPoint:
                self.overrideCursor(CURSOR_MOVE)
                self.boundedMoveShapes(self.selectedShapes, pos)
                self.repaint()
                self.movingShape = True
            return
        
        # Just hovering over the canvas, 2 possibilities:
        # - Highlight shapes
        # - Highlight vertex
        # Update shape/vertex fill and tooltip value accordingly.
        self.setToolTip(self.tr("Image"))
        for shape in reversed([s for s in self.shapes if self.isVisible(s)]):
            # Look for a nearby vertex to highlight. If that fails,
            # check if we happen to be inside a shape.
            index = shape.nearestVertex(pos, self.epsilon / self.scale)
            index_edge = shape.nearestEdge(pos, self.epsilon / self.scale)
            if index is not None:
                if self.selectedVertex():
                    self.hShape.highlightClear()
                self.prevhVertex = self.hVertex = index
                self.prevhShape = self.hShape = shape
                self.prevhEdge = self.hEdge
                self.hEdge = None
                shape.highlightVertex(index, shape.MOVE_VERTEX)
                self.overrideCursor(CURSOR_POINT)
                self.setToolTip(self.tr("Click & drag to move point"))
                self.setStatusTip(self.toolTip())
                self.update()
                break
            elif index_edge is not None and shape.canAddPoint():
                if self.selectedVertex():
                    self.hShape.highlightClear()
                self.prevhVertex = self.hVertex
                self.hVertex = None
                self.prevhShape = self.hShape = shape
                self.prevhEdge = self.hEdge = index_edge
                self.overrideCursor(CURSOR_POINT)
                self.setToolTip(self.tr("Click to create point"))
                self.setStatusTip(self.toolTip())
                self.update()
                break
            elif shape.containsPoint(pos):
                if self.selectedVertex():
                    self.hShape.highlightClear()
                self.prevhVertex = self.hVertex
                self.hVertex = None
                self.prevhShape = self.hShape = shape
                self.prevhEdge = self.hEdge
                self.hEdge = None
                self.setToolTip(
                    self.tr("Click & drag to move shape '%s'") % shape.label
                )
                self.setStatusTip(self.toolTip())
                self.overrideCursor(CURSOR_GRAB)
                self.update()
                break
        else:  # Nothing found, clear highlights, reset state.
            self.unHighlight()
        self.vertexSelected.emit(self.hVertex is not None)

    def addPointToEdge(self):
        shape = self.prevhShape
        index = self.prevhEdge
        point = self.prevMovePoint
        if shape is None or index is None or point is None:
            return
        shape.insertPoint(index, point)
        shape.highlightVertex(index, shape.MOVE_VERTEX)
        self.hShape = shape
        self.hVertex = index
        self.hEdge = None
        self.movingShape = True

    def removeSelectedPoint(self):
        shape = self.prevhShape
        index = self.prevhVertex
        if shape is None or index is None:
            return
        shape.removePoint(index)
        shape.highlightClear()
        self.hShape = shape
        self.prevhVertex = None
        self.movingShape = True  # Save changes

    def mousePressEvent(self, ev):
        if QT5:
            pos = self.transformPos(ev.localPos())
        else:
            pos = self.transformPos(ev.posF())

        is_shift_pressed = ev.modifiers() & QtCore.Qt.ShiftModifier
        clicked_shape = None  # Initialize clicked_shape
        for shape in self.shapes:
            if shape.boundingBox().contains(ev.pos()):
                clicked_shape = shape
                break

        if clicked_shape:
            self.selectedShape = clicked_shape
            self.selectionChanged.emit(True)
        else:
            self.selectedShape = None
            self.selectionChanged.emit(False)

        
        
        
        
        if ev.button() == QtCore.Qt.LeftButton:
            if self.drawing():
                if self.current:
                    # Add point to existing shape.
                    if self.createMode == "polygon":
                        self.current.addPoint(self.line[1])
                        self.line[0] = self.current[-1]
                        if self.current.isClosed():
                            self.finalise()
                    elif self.createMode in ["rectangle", "circle", "line"]:
                        assert len(self.current.points) == 1
                        self.current.points = self.line.points
                        self.finalise()
                    elif self.createMode == "linestrip":
                        self.current.addPoint(self.line[1])
                        self.line[0] = self.current[-1]
                        if int(ev.modifiers()) == QtCore.Qt.ControlModifier:
                            self.finalise()
                    elif self.createMode in ["ai_polygon", "ai_mask"]:
                        self.current.addPoint(
                            self.line.points[1],
                            label=self.line.point_labels[1],
                        )
                        self.line.points[0] = self.current.points[-1]
                        self.line.point_labels[0] = self.current.point_labels[-1]
                        if ev.modifiers() & QtCore.Qt.ControlModifier:
                            self.finalise()
                elif not self.outOfPixmap(pos):
                    # Create new shape.
                    self.current = Shape(
                        shape_type="points"
                        if self.createMode in ["ai_polygon", "ai_mask"]
                        else self.createMode
                    )
                    self.current.addPoint(pos, label=0 if is_shift_pressed else 1)
                    if self.createMode == "point":
                        self.finalise()
                    elif (
                        self.createMode in ["ai_polygon", "ai_mask"]
                        and ev.modifiers() & QtCore.Qt.ControlModifier
                    ):
                        self.finalise()
                    else:
                        if self.createMode == "circle":
                            self.current.shape_type = "circle"
                        self.line.points = [pos, pos]
                        if (
                            self.createMode in ["ai_polygon", "ai_mask"]
                            and is_shift_pressed
                        ):
                            self.line.point_labels = [0, 0]
                        else:
                            self.line.point_labels = [1, 1]
                        self.setHiding()
                        self.drawingPolygon.emit(True)
                        self.update()
            elif self.editing():
                if self.selectedEdge():
                    self.addPointToEdge()
                elif (
                    self.selectedVertex()
                    and int(ev.modifiers()) == QtCore.Qt.ShiftModifier
                ):
                    # Delete point if: left-click + SHIFT on a point
                    self.removeSelectedPoint()

                group_mode = int(ev.modifiers()) == QtCore.Qt.ControlModifier
                self.selectShapePoint(pos, multiple_selection_mode=group_mode)
                self.prevPoint = pos
                self.repaint()
        elif ev.button() == QtCore.Qt.RightButton and self.editing():
            group_mode = int(ev.modifiers()) == QtCore.Qt.ControlModifier
            if not self.selectedShapes or (
                self.hShape is not None and self.hShape not in self.selectedShapes
            ):
                self.selectShapePoint(pos, multiple_selection_mode=group_mode)
                self.repaint()
            self.prevPoint = pos
        self.update()

    def mouseReleaseEvent(self, ev):
        if ev.button() == QtCore.Qt.RightButton:
            menu = self.menus[len(self.selectedShapesCopy) > 0]
            self.restoreCursor()
            if not menu.exec_(self.mapToGlobal(ev.pos())) and self.selectedShapesCopy:
                # Cancel the move by deleting the shadow copy.
                self.selectedShapesCopy = []
                self.repaint()
        elif ev.button() == QtCore.Qt.LeftButton:
            if self.editing():
                if (
                    self.hShape is not None
                    and self.hShapeIsSelected
                    and not self.movingShape
                ):
                    self.selectionChanged.emit(
                        [x for x in self.selectedShapes if x != self.hShape]
                    )

        if self.movingShape and self.hShape:
            index = self.shapes.index(self.hShape)
            if self.shapesBackups[-1][index].points != self.shapes[index].points:
                self.storeShapes()
                self.shapeMoved.emit()

            self.movingShape = False

    def endMove(self, copy):
        assert self.selectedShapes and self.selectedShapesCopy
        assert len(self.selectedShapesCopy) == len(self.selectedShapes)
        if copy:
            for i, shape in enumerate(self.selectedShapesCopy):
                self.shapes.append(shape)
                self.selectedShapes[i].selected = False
                self.selectedShapes[i] = shape
        else:
            for i, shape in enumerate(self.selectedShapesCopy):
                self.selectedShapes[i].points = shape.points
        self.selectedShapesCopy = []
        self.repaint()
        self.storeShapes()
        return True

    def hideBackroundShapes(self, value):
        self.hideBackround = value
        if self.selectedShapes:
            # Only hide other shapes if there is a current selection.
            # Otherwise the user will not be able to select a shape.
            self.setHiding(True)
            self.update()

    def setHiding(self, enable=True):
        self._hideBackround = self.hideBackround if enable else False

    def canCloseShape(self):
        return self.drawing() and (
            (self.current and len(self.current) > 2)
            or self.createMode in ["ai_polygon", "ai_mask"]
        )

    def mouseDoubleClickEvent(self, ev):
        if self.double_click != "close":
            return

        if (
            self.createMode == "polygon" and self.canCloseShape()
        ) or self.createMode in ["ai_polygon", "ai_mask"]:
            self.finalise()

    def selectShapes(self, shapes):
        self.setHiding()
        if shapes not in self.selectedShapes:
            self.selectedShapes.append(shapes)
        self.selectionChanged.emit(True)
        self.update()

    def selectShapePoint(self, point, multiple_selection_mode):
        """Select the first shape created which contains this point."""
        if self.selectedVertex():  # A vertex is marked for selection.
            index, shape = self.hVertex, self.hShape
            shape.highlightVertex(index, shape.MOVE_VERTEX)
        else:
            for shape in reversed(self.shapes):
                if self.isVisible(shape) and shape.containsPoint(point):
                    self.setHiding()
                    if shape not in self.selectedShapes:
                        if multiple_selection_mode:
                            self.selectionChanged.emit(self.selectedShapes + [shape])
                        else:
                            self.selectionChanged.emit([shape])
                        self.hShapeIsSelected = False
                    else:
                        self.hShapeIsSelected = True
                    self.calculateOffsets(point)
                    return
        self.deSelectShape()

    def calculateOffsets(self, point):
        left = self.pixmap.width() - 1
        right = 0
        top = self.pixmap.height() - 1
        bottom = 0
        for s in self.selectedShapes:
            rect = s.boundingRect()
            if rect.left() < left:
                left = rect.left()
            if rect.right() > right:
                right = rect.right()
            if rect.top() < top:
                top = rect.top()
            if rect.bottom() > bottom:
                bottom = rect.bottom()

        x1 = left - point.x()
        y1 = top - point.y()
        x2 = right - point.x()
        y2 = bottom - point.y()
        self.offsets = QtCore.QPointF(x1, y1), QtCore.QPointF(x2, y2)

    def boundedMoveVertex(self, pos):
        index, shape = self.hVertex, self.hShape
        point = shape[index]
        if self.outOfPixmap(pos):
            pos = self.intersectionPoint(point, pos)
        shape.moveVertexBy(index, pos - point)

    def boundedMoveShapes(self, shapes, pos):
        if self.outOfPixmap(pos):
            return False  # No need to move
        o1 = pos + self.offsets[0]
        if self.outOfPixmap(o1):
            pos -= QtCore.QPointF(min(0, o1.x()), min(0, o1.y()))
        o2 = pos + self.offsets[1]
        if self.outOfPixmap(o2):
            pos += QtCore.QPointF(
                min(0, self.pixmap.width() - o2.x()),
                min(0, self.pixmap.height() - o2.y()),
            )
        # XXX: The next line tracks the new position of the cursor
        # relative to the shape, but also results in making it
        # a bit "shaky" when nearing the border and allows it to
        # go outside of the shape's area for some reason.
        # self.calculateOffsets(self.selectedShapes, pos)
        dp = pos - self.prevPoint
        if dp:
            for shape in shapes:
                shape.moveBy(dp)
            self.prevPoint = pos
            return True
        return False

    def deSelectShape(self):
        if self.selectedShapes:
            self.setHiding(False)
            self.selectionChanged.emit(bool(self.selectedShapes))
            self.hShapeIsSelected = False
            self.update()

    def deleteSelected(self):
        deleted_shapes = []
        if self.selectedShapes:
            for shape in self.selectedShapes:
                self.shapes.remove(shape)
                deleted_shapes.append(shape)
            self.storeShapes()
            self.selectedShapes = []
            self.update()
        return deleted_shapes

    def deleteShape(self, shape):
        if shape in self.selectedShapes:
            self.selectedShapes.remove(shape)
        if shape in self.shapes:
            self.shapes.remove(shape)
        self.storeShapes()
        self.update()

    def duplicateSelectedShapes(self):
        if self.selectedShapes:
            self.selectedShapesCopy = [s.copy() for s in self.selectedShapes]
            self.boundedShiftShapes(self.selectedShapesCopy)
            self.endMove(copy=True)
        return self.selectedShapes

    def boundedShiftShapes(self, shapes):
        # Try to move in one direction, and if it fails in another.
        # Give up if both fail.
        point = shapes[0][0]
        offset = QtCore.QPointF(2.0, 2.0)
        self.offsets = QtCore.QPoint(), QtCore.QPoint()
        self.prevPoint = point
        if not self.boundedMoveShapes(shapes, point - offset):
            self.boundedMoveShapes(shapes, point + offset)

    def paintEvent(self, event):
        """Override the paintEvent to draw shapes on the canvas."""
        if not self.pixmap:
            return super(Canvas, self).paintEvent(event)

        p = self._painter
        p.begin(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        p.setRenderHint(QtGui.QPainter.HighQualityAntialiasing)
        p.setRenderHint(QtGui.QPainter.SmoothPixmapTransform)

        # Adjust scaling and translation
        p.scale(self.scale, self.scale)
        p.translate(self.offsetToCenter())

        # Draw the main pixmap
        p.drawPixmap(0, 0, self.pixmap)

        # Draw crosshair if applicable
        if (
            self._crosshair[self._createMode]
            and self.drawing()
            and self.prevMovePoint
            and not self.outOfPixmap(self.prevMovePoint)
        ):
            p.setPen(QtGui.QColor(0, 0, 0))
            p.drawLine(
                0,
                int(self.prevMovePoint.y()),
                self.width() - 1,
                int(self.prevMovePoint.y()),
            )
            p.drawLine(
                int(self.prevMovePoint.x()),
                0,
                int(self.prevMovePoint.x()),
                self.height() - 1,
            )

        # Scale shapes
        Shape.scale = self.scale

        # Render each shape on the canvas
        try:
            for shape in self.shapes:
                # Validate the shape and bounding box
                if shape is None or shape.boundingBox() is None:
                    logging.warning(f"Skipping invalid shape: {shape}")
                    continue

                # Assign color and configure the pen
                color = self.assignColor(shape.id)
                pen = QtGui.QPen(QtGui.QColor(*color))
                pen.setWidth(2)
                p.setPen(pen)

                # Apply brush for selected shapes
                if shape == self.selectedShape:
                    brush = QtGui.QBrush(QtGui.QColor(255, 255, 255, 100))
                    p.setBrush(brush)
                else:
                    p.setBrush(Qt.NoBrush)

                # Draw the bounding box
                p.drawRect(shape.boundingBox())

                # Paint the shape if it's visible or selected
                if (shape.selected or not self._hideBackround) and self.isVisible(shape):
                    shape.fill = shape.selected or shape == self.hShape
                    shape.paint(p)

        except Exception as e:
            logging.error(f"Error while rendering shapes: {e}")

        # Render the current drawing shape
        if self.current:
            self.current.paint(p)
            assert len(self.line.points) == len(self.line.point_labels)
            self.line.paint(p)

        # Render copied selected shapes
        if self.selectedShapesCopy:
            for s in self.selectedShapesCopy:
                s.paint(p)

        # Render current drawing polygon or AI-assisted shapes
        try:
            if (
                self.fillDrawing()
                and self.createMode == "polygon"
                and self.current is not None
                and len(self.current.points) >= 2
            ):
                drawing_shape = self.current.copy()
                if drawing_shape.fill_color.getRgb()[3] == 0:
                    logging.warning(
                        "fill_drawing=true, but fill_color is transparent,"
                        " so forcing to be opaque."
                    )
                    drawing_shape.fill_color.setAlpha(64)
                drawing_shape.addPoint(self.line[1])
                drawing_shape.fill = True
                drawing_shape.paint(p)
            elif self.createMode == "ai_polygon" and self.current is not None:
                drawing_shape = self.current.copy()
                drawing_shape.addPoint(
                    point=self.line.points[1],
                    label=self.line.point_labels[1],
                )
                points = self._ai_model.predict_polygon_from_points(
                    points=[[point.x(), point.y()] for point in drawing_shape.points],
                    point_labels=drawing_shape.point_labels,
                )
                if len(points) > 2:
                    drawing_shape.setShapeRefined(
                        shape_type="polygon",
                        points=[QtCore.QPointF(point[0], point[1]) for point in points],
                        point_labels=[1] * len(points),
                    )
                    drawing_shape.fill = self.fillDrawing()
                    drawing_shape.selected = True
                    drawing_shape.paint(p)
            elif self.createMode == "ai_mask" and self.current is not None:
                drawing_shape = self.current.copy()
                drawing_shape.addPoint(
                    point=self.line.points[1],
                    label=self.line.point_labels[1],
                )
                mask = self._ai_model.predict_mask_from_points(
                    points=[[point.x(), point.y()] for point in drawing_shape.points],
                    point_labels=drawing_shape.point_labels,
                )
                y1, x1, y2, x2 = imgviz.instances.masks_to_bboxes([mask])[0].astype(int)
                drawing_shape.setShapeRefined(
                    shape_type="mask",
                    points=[QtCore.QPointF(x1, y1), QtCore.QPointF(x2, y2)],
                    point_labels=[1, 1],
                    mask=mask[y1 : y2 + 1, x1 : x2 + 1],
                )
                drawing_shape.selected = True
                drawing_shape.paint(p)
        except Exception as e:
            logging.error(f"Error while rendering current drawing shape: {e}")

        p.end()

        
    def drawShapesForFrame(self, frame_shapes):
        """
        Update the canvas with shapes for the current frame.
        """
        self.clearShapes()
        for shape_data in frame_shapes:
            shape = self.createShapeFromData(shape_data)
            self.addShape(shape)
        self.update()

    def createShapeFromData(self, shape_data):
        """
        Convert annotation data into a Shape object.
        Args:
            shape_data (dict): Dictionary containing shape information.
        Returns:
            Shape or None: A validated Shape object or None if creation fails.
        """
        try:
            logger.debug(f"Creating shape from data: {shape_data}")

            # Validate input and extract bbox
            bbox = self.validateInputData(shape_data)
            if bbox is None:
                return None

            # Process and validate coordinates
            processed_coords = self.processCoordinates(bbox)
            if processed_coords is None:
                return None

            # Create and validate rectangle
            rect = self.createValidRect(processed_coords, shape_data.get("shape_id"))
            if rect is None:
                return None

            # Create and configure shape
            shape = self.createAndConfigureShape(shape_data, processed_coords)
            if shape is None:
                return None

            logger.debug(f"Shape created successfully: ID={shape.shape_id}, bbox={shape.boundingRect()}")
            return shape

        except Exception as e:
            logger.error(f"Shape creation failed: {e}")
            return None

    def validateInputData(self, shape_data):
        """Validate input data and extract bbox."""
        try:
            bbox = shape_data.get("bbox")
            if not isinstance(bbox, list) or len(bbox) != 4:
                logger.error(f"Invalid bbox format: {bbox}")
                return None
            return bbox
        except Exception as e:
            logger.error(f"Input validation failed: {e}")
            return None

    def processCoordinates(self, bbox):
        """Process and validate coordinates within frame bounds."""
        try:
            if not self.pixmap:
                raise ValueError("No pixmap available for shape validation")

            # Normalize bbox coordinates
            x1, y1, x2, y2 = self.normalize_bbox(bbox)
            
            # Get frame dimensions
            frame_width = self.pixmap.width()
            frame_height = self.pixmap.height()
            
            # Clamp coordinates to frame bounds
            return self.clamp_coordinates(x1, y1, x2, y2, frame_width, frame_height)
        except Exception as e:
            logger.error(f"Coordinate processing failed: {e}")
            return None

    def createValidRect(self, coords, shape_id):
        """Create and validate QRectF from coordinates."""
        try:
            x1, y1, x2, y2 = coords
            rect = QtCore.QRectF(QtCore.QPointF(x1, y1), QtCore.QPointF(x2, y2))
            
            if rect.isNull() or rect.isEmpty():
                logger.error(f"Invalid QRectF for shape ID {shape_id}")
                return None
                
            return rect
        except Exception as e:
            logger.error(f"Rectangle creation failed: {e}")
            return None

    def createAndConfigureShape(self, shape_data, coords):
        """Improved shape creation with validation"""
        try:
            x1, y1, x2, y2 = coords
            confidence = max(0.0, min(1.0, shape_data.get("confidence", 1.0)))
            
            shape = Shape(
                label=shape_data.get("label", "person"),
                shape_type="rectangle",
                shape_id=shape_data["track_id"],
                confidence=confidence,
                frame_number=shape_data.get("frame_number", 0)
            )

            points = [QtCore.QPointF(x, y) for x, y in [(x1,y1), (x2,y1), (x2,y2), (x1,y2)]]
            for point in points:
                shape.addPoint(point)

            shape.isValid() # Add validation method to Shape class
            return shape

        except Exception as e:
            logger.error(f"Shape configuration failed: {e}")
            return None

    def normalize_bbox(self, bbox):
        """Normalize bbox coordinates."""
        x1, y1, x2, y2 = map(float, bbox)
        return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)

    def clamp_coordinates(self, x1, y1, x2, y2, width, height):
        """Clamp coordinates to frame bounds."""
        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(x1 + 1, min(x2, width))
        y2 = max(y1 + 1, min(y2, height))
        return x1, y1, x2, y2





    def validate_bbox(self, bbox):
        """Validate bbox format and values."""
        try:
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                return False
                
            x1, y1, x2, y2 = map(float, bbox)
            
            # Basic sanity checks
            if any(math.isnan(x) for x in [x1, y1, x2, y2]):
                return False
                
            # Ensure proper ordering
            if x1 >= x2 or y1 >= y2:
                return False
                
            # Ensure positive dimensions
            if (x2 - x1) <= 0 or (y2 - y1) <= 0:
                return False
                
            return True
        except (TypeError, ValueError):
            return False

    def is_duplicate(self, shape_id, bbox, threshold=0.5):
        """Check if shape with similar bbox exists"""
        try:
            if bbox is None:
                logger.warning("Bounding box is None, skipping duplicate check.")
                return False

            logger.debug(f"Checking duplicates for shape_id={shape_id}, bbox={bbox}, threshold={threshold}")
            
            for shape in self.shapes:
                current_bbox = shape.boundingRect()
                logger.debug(f"Comparing against shape_id={shape.shape_id}, current_bbox={current_bbox}")

                if shape.shape_id == shape_id:
                    logger.debug(f"Skipping shape_id={shape_id} as it matches the current track_id.")
                    continue
                
                overlap = self.calculate_iou(
                    [current_bbox.x(), current_bbox.y(),
                    current_bbox.x() + current_bbox.width(),
                    current_bbox.y() + current_bbox.height()],
                    bbox
                )
                logger.debug(f"IOU overlap={overlap} for shape_id={shape.shape_id}")
                
                if overlap > threshold:
                    logger.info(f"Duplicate detected for shape_id={shape_id} with overlap={overlap}")
                    return True
            
            return False
        except Exception as e:
            logger.error(f"Error checking shape duplicate: {e}")
            return False





    def calculate_iou(self, box1, box2):
        """Calculate Intersection over Union between two boxes."""
        try:
            x1 = max(box1[0], box2[0])
            y1 = max(box1[1], box2[1])
            x2 = min(box1[2], box2[2])
            y2 = min(box1[3], box2[3])

            intersection = max(0, x2 - x1) * max(0, y2 - y1)
            box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
            box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
            union = box1_area + box2_area - intersection

            return intersection / union if union > 0 else 0
        except Exception as e:
            logger.error(f"Error calculating IOU: {e}")
            return 0

    def predict_bbox_position(self, bbox, velocity, time_delta=1):
        """
        Predict future bbox position based on velocity.
        
        Args:
            bbox (list): Current bbox coordinates [x1,y1,x2,y2]
            velocity (tuple): (dx,dy) velocity components
            time_delta (int): Time steps to predict ahead
        """
        try:
            dx, dy = velocity
            x1, y1, x2, y2 = bbox
            
            # Predict new coordinates
            new_x1 = x1 + dx * time_delta
            new_y1 = y1 + dy * time_delta
            new_x2 = x2 + dx * time_delta
            new_y2 = y2 + dy * time_delta
            
            # Ensure coordinates stay within frame
            if hasattr(self, 'pixmap'):
                width = self.pixmap.width()
                height = self.pixmap.height()
                new_x1 = max(0, min(new_x1, width))
                new_x2 = max(0, min(new_x2, width))
                new_y1 = max(0, min(new_y1, height))
                new_y2 = max(0, min(new_y2, height))
                
            return [new_x1, new_y1, new_x2, new_y2]
            
        except Exception as e:
            logger.error(f"Error predicting bbox position: {e}")
            return bbox
    
    def calculate_velocity(self, track_data):
        """Calculate velocity from recent trajectory."""
        if len(track_data["trajectory"]) < 2:
            return (0, 0)
            
        recent = track_data["trajectory"][-2:]
        t1, t2 = recent[0]["frame"], recent[1]["frame"]
        b1, b2 = recent[0]["bbox"], recent[1]["bbox"]
        
        dt = max(t2 - t1, 1)
        dx = (b2[0] - b1[0]) / dt
        dy = (b2[1] - b1[1]) / dt
        
        return (dx, dy)
    
    
    def is_bbox_overlap(self, bbox1, bbox2, threshold=0.5):
        """Check if two bounding boxes overlap"""
        if bbox1 is None or bbox2 is None:
            return False
            
        try:
            x1, y1, x2, y2 = bbox1
            x1_, y1_, x2_, y2_ = bbox2
            
            # Calculate intersection
            inter_x1 = max(x1, x1_)
            inter_y1 = max(y1, y1_)
            inter_x2 = min(x2, x2_)
            inter_y2 = min(y2, y2_)
            
            # Check if boxes overlap
            if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
                return False
                
            inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
            bbox1_area = (x2 - x1) * (y2 - y1)
            bbox2_area = (x2_ - x1_) * (y2_ - y1_)
            union_area = bbox1_area + bbox2_area - inter_area
            
            return inter_area / union_area > threshold if union_area > 0 else False
            
        except (TypeError, ValueError) as e:
            logger.error(f"Error calculating bbox overlap: {e}")
            return False

    def removeTrackShapes(self, track_id):
        """Remove all shapes associated with a track ID."""
        try:
            shapes_to_remove = [shape for shape in self.shapes if shape.shape_id == track_id]
            for shape in shapes_to_remove:
                self.shapes.remove(shape)
            self.update()
            logger.debug(f"Removed {len(shapes_to_remove)} shapes for track {track_id}")
        except Exception as e:
            logger.error(f"Error removing shapes for track {track_id}: {e}")
    
    
    
    def getShapes(self):
        """
        Return all shapes on the canvas.
        """
        return self.shapes

    def transformPos(self, point):
        """Convert from widget-logical coordinates to painter-logical ones."""
        return point / self.scale - self.offsetToCenter()

    def offsetToCenter(self):
        s = self.scale
        area = super(Canvas, self).size()
        w, h = self.pixmap.width() * s, self.pixmap.height() * s
        aw, ah = area.width(), area.height()
        x = (aw - w) / (2 * s) if aw > w else 0
        y = (ah - h) / (2 * s) if ah > h else 0
        return QtCore.QPointF(x, y)

    def outOfPixmap(self, p):
        w, h = self.pixmap.width(), self.pixmap.height()
        return not (0 <= p.x() <= w - 1 and 0 <= p.y() <= h - 1)

    def finalise(self):
        assert self.current
        if self.createMode == "ai_polygon":
            # convert points to polygon by an AI model
            assert self.current.shape_type == "points"
            points = self._ai_model.predict_polygon_from_points(
                points=[[point.x(), point.y()] for point in self.current.points],
                point_labels=self.current.point_labels,
            )
            self.current.setShapeRefined(
                points=[QtCore.QPointF(point[0], point[1]) for point in points],
                point_labels=[1] * len(points),
                shape_type="polygon",
            )
        elif self.createMode == "ai_mask":
            # convert points to mask by an AI model
            assert self.current.shape_type == "points"
            mask = self._ai_model.predict_mask_from_points(
                points=[[point.x(), point.y()] for point in self.current.points],
                point_labels=self.current.point_labels,
            )
            y1, x1, y2, x2 = imgviz.instances.masks_to_bboxes([mask])[0].astype(int)
            self.current.setShapeRefined(
                shape_type="mask",
                points=[QtCore.QPointF(x1, y1), QtCore.QPointF(x2, y2)],
                point_labels=[1, 1],
                mask=mask[y1 : y2 + 1, x1 : x2 + 1],
            )
        self.current.close()

        self.shapes.append(self.current)
        self.storeShapes()
        self.current = None
        self.setHiding(False)
        self.newShape.emit()
        self.update()

    def closeEnough(self, p1, p2):
        # d = distance(p1 - p2)
        # m = (p1-p2).manhattanLength()
        # print "d %.2f, m %d, %.2f" % (d, m, d - m)
        # divide by scale to allow more precision when zoomed in
        return labelme.utils.distance(p1 - p2) < (self.epsilon / self.scale)

    def intersectionPoint(self, p1, p2):
        # Cycle through each image edge in clockwise fashion,
        # and find the one intersecting the current line segment.
        # http://paulbourke.net/geometry/lineline2d/
        size = self.pixmap.size()
        points = [
            (0, 0),
            (size.width() - 1, 0),
            (size.width() - 1, size.height() - 1),
            (0, size.height() - 1),
        ]
        # x1, y1 should be in the pixmap, x2, y2 should be out of the pixmap
        x1 = min(max(p1.x(), 0), size.width() - 1)
        y1 = min(max(p1.y(), 0), size.height() - 1)
        x2, y2 = p2.x(), p2.y()
        d, i, (x, y) = min(self.intersectingEdges((x1, y1), (x2, y2), points))
        x3, y3 = points[i]
        x4, y4 = points[(i + 1) % 4]
        if (x, y) == (x1, y1):
            # Handle cases where previous point is on one of the edges.
            if x3 == x4:
                return QtCore.QPointF(x3, min(max(0, y2), max(y3, y4)))
            else:  # y3 == y4
                return QtCore.QPointF(min(max(0, x2), max(x3, x4)), y3)
        return QtCore.QPointF(x, y)

    def intersectingEdges(self, point1, point2, points):
        """Find intersecting edges.

        For each edge formed by `points', yield the intersection
        with the line segment `(x1,y1) - (x2,y2)`, if it exists.
        Also return the distance of `(x2,y2)' to the middle of the
        edge along with its index, so that the one closest can be chosen.
        """
        (x1, y1) = point1
        (x2, y2) = point2
        for i in range(4):
            x3, y3 = points[i]
            x4, y4 = points[(i + 1) % 4]
            denom = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
            nua = (x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)
            nub = (x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)
            if denom == 0:
                # This covers two cases:
                #   nua == nub == 0: Coincident
                #   otherwise: Parallel
                continue
            ua, ub = nua / denom, nub / denom
            if 0 <= ua <= 1 and 0 <= ub <= 1:
                x = x1 + ua * (x2 - x1)
                y = y1 + ua * (y2 - y1)
                m = QtCore.QPointF((x3 + x4) / 2, (y3 + y4) / 2)
                d = labelme.utils.distance(m - QtCore.QPointF(x2, y2))
                yield d, i, (x, y)

    # These two, along with a call to adjustSize are required for the
    # scroll area.
    def sizeHint(self):
        return self.minimumSizeHint()

    def minimumSizeHint(self):
        if self.pixmap:
            return self.scale * self.pixmap.size()
        return super(Canvas, self).minimumSizeHint()

    def wheelEvent(self, ev):
        if QT5:
            mods = ev.modifiers()
            delta = ev.angleDelta()
            if QtCore.Qt.ControlModifier == int(mods):
                # with Ctrl/Command key
                # zoom
                self.zoomRequest.emit(delta.y(), ev.pos())
            else:
                # scroll
                self.scrollRequest.emit(delta.x(), QtCore.Qt.Horizontal)
                self.scrollRequest.emit(delta.y(), QtCore.Qt.Vertical)
        else:
            if ev.orientation() == QtCore.Qt.Vertical:
                mods = ev.modifiers()
                if QtCore.Qt.ControlModifier == int(mods):
                    # with Ctrl/Command key
                    self.zoomRequest.emit(ev.delta(), ev.pos())
                else:
                    self.scrollRequest.emit(
                        ev.delta(),
                        QtCore.Qt.Horizontal
                        if (QtCore.Qt.ShiftModifier == int(mods))
                        else QtCore.Qt.Vertical,
                    )
            else:
                self.scrollRequest.emit(ev.delta(), QtCore.Qt.Horizontal)
        ev.accept()
    
    def centerOnShape(self, shape):
        """Center the canvas view on the given shape."""
        if not self.pixmap or not shape:
            return

        x_min = min([point.x() for point in shape.points])
        y_min = min([point.y() for point in shape.points])
        x_max = max([point.x() for point in shape.points])
        y_max = max([point.y() for point in shape.points])

        # Calculate the center of the shape
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2

        # Scroll the canvas to center on the shape
        self.scrollRequest.emit(center_x, center_y)

    
    
    def moveByKeyboard(self, offset):
        if self.selectedShapes:
            self.boundedMoveShapes(self.selectedShapes, self.prevPoint + offset)
            self.repaint()
            self.movingShape = True

    def keyPressEvent(self, ev):
        modifiers = ev.modifiers()
        key = ev.key()
        if self.drawing():
            if key == QtCore.Qt.Key_Escape and self.current:
                self.current = None
                self.drawingPolygon.emit(False)
                self.update()
            elif key == QtCore.Qt.Key_Return and self.canCloseShape():
                self.finalise()
            elif modifiers == QtCore.Qt.AltModifier:
                self.snapping = False
        elif self.editing():
            if key == QtCore.Qt.Key_Up:
                self.moveByKeyboard(QtCore.QPointF(0.0, -MOVE_SPEED))
            elif key == QtCore.Qt.Key_Down:
                self.moveByKeyboard(QtCore.QPointF(0.0, MOVE_SPEED))
            elif key == QtCore.Qt.Key_Left:
                self.moveByKeyboard(QtCore.QPointF(-MOVE_SPEED, 0.0))
            elif key == QtCore.Qt.Key_Right:
                self.moveByKeyboard(QtCore.QPointF(MOVE_SPEED, 0.0))

    def keyReleaseEvent(self, ev):
        modifiers = ev.modifiers()
        if self.drawing():
            if int(modifiers) == 0:
                self.snapping = True
        elif self.editing():
            if self.movingShape and self.selectedShapes:
                index = self.shapes.index(self.selectedShapes[0])
                if self.shapesBackups[-1][index].points != self.shapes[index].points:
                    self.storeShapes()
                    self.shapeMoved.emit()

                self.movingShape = False
    
    def setLastLabel(self, text, flags):
        """
        Set the label and flags of the last added shape.
        Args:
            text (str): The label to assign to the last shape.
            flags (dict): Flags to assign to the last shape.
        Returns:
            Shape: The last shape with the updated label and flags.
        """
        assert text, "Label text cannot be empty."
        if not self.shapes:
            logging.warning("No shapes available to set label.")
            return None

        # Update label and flags of the last shape
        self.shapes[-1].label = text
        self.shapes[-1].flags = flags

        # Ensure shapesBackups is not empty before popping
        if self.shapesBackups:
            self.shapesBackups.pop()
        else:
            logging.warning("shapesBackups is empty; nothing to pop.")

        # Store shapes and return the last one
        self.storeShapes()
        return self.shapes[-1]

    
    

    

    def undoLastLine(self):
        assert self.shapes
        self.current = self.shapes.pop()
        self.current.setOpen()
        self.current.restoreShapeRaw()
        if self.createMode in ["polygon", "linestrip"]:
            self.line.points = [self.current[-1], self.current[0]]
        elif self.createMode in ["rectangle", "line", "circle"]:
            self.current.points = self.current.points[0:1]
        elif self.createMode == "point":
            self.current = None
        self.drawingPolygon.emit(True)

    def undoLastPoint(self):
        if not self.current or self.current.isClosed():
            return
        self.current.popPoint()
        if len(self.current) > 0:
            self.line[0] = self.current[-1]
        else:
            self.current = None
            self.drawingPolygon.emit(False)
        self.update()

    def loadPixmap(self, pixmap, clear_shapes=True):
        self.pixmap = pixmap
        if self._ai_model:
            self._ai_model.set_image(
                image=labelme.utils.img_qt_to_arr(self.pixmap.toImage())
            )
        if clear_shapes:
            self.shapes = []
        self.update()

    def loadShapes(self, shapes, replace=True):
        if replace:
            self.shapes = list(shapes)
        else:
            self.shapes.extend(shapes)
        self.storeShapes()
        self.current = None
        self.hShape = None
        self.hVertex = None
        self.hEdge = None
        self.update()

    ###########################################################################################################################
    def highlightShape(self, shape):
        """Highlight the given shape by changing its color temporarily."""
        if not shape:
            return

        # Save the original color
        original_color = shape.line_color

        # Set the highlight color (e.g., bright yellow)
        shape.line_color = QtGui.QColor(255, 255, 0)

        # Refresh the canvas to show the change
        self.update()

        # Optional: Restore the original color after a delay
        QtCore.QTimer.singleShot(1000, lambda: self._restoreShapeColor(shape, original_color))

    def _restoreShapeColor(self, shape, original_color):
        """Restore the original color of the shape."""
        if shape:
            shape.line_color = original_color
            self.update()

    
    
    def setShapeVisible(self, shape, value):
        self.visible[shape] = value
        self.update()

    def overrideCursor(self, cursor):
        self.restoreCursor()
        self._cursor = cursor
        QtWidgets.QApplication.setOverrideCursor(cursor)

    def restoreCursor(self):
        QtWidgets.QApplication.restoreOverrideCursor()

    def resetState(self):
        self.restoreCursor()
        self.pixmap = None
        self.shapesBackups = []
        self.update()

    def clearSelection(self):
        self.selectedShapes.clear()
        self.selectionChanged.emit(False)  # No shapes are selected


########################################################################
# class MockShape:
    
#     def __init__(self, shape_id, bbox):
#         self.shape_id = shape_id
#         self.bbox = bbox

#     @property
#     def boundingRect(self):
#         """Simulate the QRectF boundingRect of the shape."""
#         x1, y1, x2, y2 = self.bbox
#         return QRectF(x1, y1, x2 - x1, y2 - y1)


