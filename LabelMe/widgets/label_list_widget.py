from qtpy import QtCore
from qtpy import QtGui
from qtpy import QtWidgets
from qtpy.QtCore import Qt
from qtpy.QtGui import QPalette
from qtpy.QtWidgets import QStyle

import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('A:/data/Project-Skills/Labeling_tool-enchancement/labelme/reid_debug.log')
    ]
)

logger = logging.getLogger(__name__)


# https://stackoverflow.com/a/2039745/4158863
class HTMLDelegate(QtWidgets.QStyledItemDelegate):
    def __init__(self, parent=None):
        super(HTMLDelegate, self).__init__()
        self.doc = QtGui.QTextDocument(self)

    def paint(self, painter, option, index):
        painter.save()

        options = QtWidgets.QStyleOptionViewItem(option)

        self.initStyleOption(options, index)
        self.doc.setHtml(options.text)
        options.text = ""

        style = (
            QtWidgets.QApplication.style()
            if options.widget is None
            else options.widget.style()
        )
        style.drawControl(QStyle.CE_ItemViewItem, options, painter)

        ctx = QtGui.QAbstractTextDocumentLayout.PaintContext()

        if option.state & QStyle.State_Selected:
            ctx.palette.setColor(
                QPalette.Text,
                option.palette.color(QPalette.Active, QPalette.HighlightedText),
            )
        else:
            ctx.palette.setColor(
                QPalette.Text,
                option.palette.color(QPalette.Active, QPalette.Text),
            )

        textRect = style.subElementRect(QStyle.SE_ItemViewItemText, options)

        if index.column() != 0:
            textRect.adjust(5, 0, 0, 0)

        thefuckyourshitup_constant = 4
        margin = (option.rect.height() - options.fontMetrics.height()) // 2
        margin = margin - thefuckyourshitup_constant
        textRect.setTop(textRect.top() + margin)

        painter.translate(textRect.topLeft())
        painter.setClipRect(textRect.translated(-textRect.topLeft()))
        self.doc.documentLayout().draw(painter, ctx)

        painter.restore()

    def sizeHint(self, option, index):
        thefuckyourshitup_constant = 4
        return QtCore.QSize(
            int(self.doc.idealWidth()),
            int(self.doc.size().height() - thefuckyourshitup_constant),
        )


class LabelListWidgetItem(QtGui.QStandardItem):
    def __init__(self, text=None, shape=None):
        super(LabelListWidgetItem, self).__init__()
        self.setText(text or "")
        self.setShape(shape)
        

        self.setCheckable(True)
        self.setCheckState(Qt.Checked)
        self.setEditable(False)
        self.setTextAlignment(Qt.AlignBottom)
        self._shape=shape
        if shape:
            logger.debug(f"Created label item for shape {shape.shape_id}")
        else:
            logger.warning("Created label item with no shape")
    
    
    def clone(self):
        return LabelListWidgetItem(self.text(), self._shape())

    def setShape(self, shape):
        """Set shape with validation and logging."""
        if shape is None:
            logger.warning(f"Attempting to set None shape for item: {self.text()}")
            return False
            
        self._shape = shape
        self.setData(shape, Qt.UserRole)
        logger.debug(f"Shape set for label item: ID={shape.shape_id}")

    def shape(self):
        """Get shape with validation and error handling."""
        shape = self._shape or self.data(Qt.UserRole)
        if not shape:
            logger.warning(f"No shape associated with label item: {self.text()}")
        return shape

    def __hash__(self):
        return id(self)

    def __repr__(self):
        shape_id = self._shape.shape_id if self._shape else "None"
        return f'{self.__class__.__name__}("{self.text()}", shape_id={shape_id})'
    
    def isSelected(self):
        return self.checkState() == QtCore.Qt.CheckState.Checked



class StandardItemModel(QtGui.QStandardItemModel):
    itemDropped = QtCore.Signal()

    def removeRows(self, *args, **kwargs):
        ret = super().removeRows(*args, **kwargs)
        self.itemDropped.emit()
        return ret


class LabelListWidget(QtWidgets.QListView):
    itemDoubleClicked = QtCore.Signal(LabelListWidgetItem)
    itemSelectionChanged = QtCore.Signal(list, list)

    def __init__(self):
        super(LabelListWidget, self).__init__()
        self._selectedItems = []

        self.setWindowFlags(Qt.Window)
        self._model=StandardItemModel()
        self.setModel(self._model)
        self.model().setItemPrototype(LabelListWidgetItem())
        self.setItemDelegate(HTMLDelegate())
        self.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.setDragDropMode(QtWidgets.QAbstractItemView.InternalMove)
        self.setDefaultDropAction(Qt.MoveAction)

        self.doubleClicked.connect(self.itemDoubleClickedEvent)
        self.selectionModel().selectionChanged.connect(self.itemSelectionChangedEvent)

    def count(self):
        """Get number of items in list."""
        return self.model().rowCount()
    
    
    def __len__(self):
        return self.model().rowCount()

    def __getitem__(self, i):
        return self.model().item(i)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    @property
    def itemDropped(self):
        return self.model().itemDropped

    @property
    def itemChanged(self):
        return self.model().itemChanged

    def itemSelectionChangedEvent(self, selected, deselected):
        selected = [self.model().itemFromIndex(i) for i in selected.indexes()]
        deselected = [self.model().itemFromIndex(i) for i in deselected.indexes()]
        self.itemSelectionChanged.emit(selected, deselected)

    def itemDoubleClickedEvent(self, index):
        self.itemDoubleClicked.emit(self.model().itemFromIndex(index))

    def selectedItems(self):
        return [self.model().itemFromIndex(i) for i in self.selectedIndexes()]

    def scrollToItem(self, item):
        self.scrollTo(self.model().indexFromItem(item))

    def addItem(self, item):
        if not isinstance(item, LabelListWidgetItem):
            raise TypeError("item must be LabelListWidgetItem")
        self.model().setItem(self.model().rowCount(), 0, item)
        item.setSizeHint(self.itemDelegate().sizeHint(None, None))

    def removeItem(self, item):
        index = self.model().indexFromItem(item)
        self.model().removeRows(index.row(), 1)

    def selectItem(self, item):
        index = self.model().indexFromItem(item)
        self.selectionModel().select(index, QtCore.QItemSelectionModel.Select)

    def findItemByShape(self, shape):
        for row in range(self.model().rowCount()):
            item = self.model().item(row, 0)
            if item.shape() == shape:
                return item
        raise ValueError("cannot find shape: {}".format(shape))

    def findItems(self, text, matchFlag):
        """
        Find items in the list that match the given text.
        :param text: The text to search for.
        :param matchFlag: The matching criteria (e.g., QtCore.Qt.MatchExactly).
        :return: List of matching items.
        """
        matching_items = []
        for row in range(self.model().rowCount()):
            item = self.model().item(row, 0)
            if matchFlag == QtCore.Qt.MatchExactly and item.text() == text:
                matching_items.append(item)
            elif matchFlag == QtCore.Qt.MatchContains and text in item.text():
                matching_items.append(item)
        return matching_items
    
    def addPersonLabel(self, track_id, color, shape=None):
        """Add a person label with shape association."""
        try:
            label_text = f"Person ID {track_id}"
            
            # Create item with shape
            label_item = LabelListWidgetItem(label_text, shape)
            if not label_item.shape():
                logger.warning(f"Failed to set shape for {label_text}")
                return None
                
            # Set background color
            label_item.setBackground(QtGui.QColor(*color))
            
            # Add to list
            self.addItem(label_item)
            logger.debug(f"Added person label: {label_text} with shape")
            return label_item
            
        except Exception as e:
            logger.error(f"Error adding person label: {e}")
            return None

    def updatePersonLabel(self, person_id, color):
        """
        Update the color of an existing person label.
        Args:
            person_id: Unique ID for the person.
            color: New color for the label (R, G, B).
        """
        label_text = f"Person ID {person_id}"
        items = self.findItems(label_text, QtCore.Qt.MatchExactly)
        for item in items:
            item.setBackground(QtGui.QColor(*color))
    
    
    def clearPersonLabels(self):
        """Clear all person labels from the list."""
        self.clear()
    
    def clear(self):
        self.model().clear()
    
    
    def validateItems(self):
        """Validate all items have proper shape associations."""
        invalid_items = []
        for i in range(self.count()):
            item = self.model().item(i)
            if not item.shape():
                invalid_items.append(item.text())
                
        if invalid_items:
            logger.warning(f"Items missing shape associations: {invalid_items}")
        return len(invalid_items) == 0        
    
    def hasTrackID(self, track_id):
        """Check if track ID exists."""
        return any(item.shape() and item.shape().shape_id == track_id 
                for item in self.getAllItems())
              
    def removeTrackID(self, track_id):
        """Remove items with specific track ID."""
        items_to_remove = []
        for item in self.getAllItems():
            if item.shape() and item.shape().shape_id == track_id:
                items_to_remove.append(item)
                
        for item in items_to_remove:
            self.removeItem(item)
            
    def getAllTrackIDs(self):
        """Get set of all track IDs."""
        return {item.shape().shape_id for item in self.getAllItems() 
                if item.shape() and item.shape().shape_id}
    

    def getAllItems(self):
        """
        Retrieve all items from the LabelListWidget.
        Returns:
            List of all LabelListWidgetItem objects.
        """
        items = []
        for row in range(self.count()):
            item = self.model().item(row)
            if isinstance(item, LabelListWidgetItem):
                items.append(item)
        return items
