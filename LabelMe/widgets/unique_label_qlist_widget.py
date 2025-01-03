# -*- encoding: utf-8 -*-

import html

from qtpy import QtWidgets
from qtpy import QtCore ,QtGui
from qtpy.QtCore import Qt

from .escapable_qlist_widget import EscapableQListWidget


####################################################################################################

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
############################################################################################3

class UniqueLabelQListWidget(EscapableQListWidget):
    
    def __init__(self):
        super(UniqueLabelQListWidget, self).__init__()
        self._id_map = {}  # Track ID to item mapping
        
    def mousePressEvent(self, event):
        super(UniqueLabelQListWidget, self).mousePressEvent(event)
        if not self.indexAt(event.pos()).isValid():
            self.clearSelection()

    def findItemByLabel(self, label):
        """Enhanced to handle person IDs."""
        if isinstance(label, str) and label.startswith("Person ID"):
            person_id = label.split()[-1]
            return self._id_map.get(person_id)
            
        for row in range(self.count()):
            item = self.item(row)
            if item.data(Qt.UserRole) == label:
                return item
        return None

    def createItemFromLabel(self, label):
        if self.findItemByLabel(label):
            raise ValueError("Item for label '{}' already exists".format(label))

        item = QtWidgets.QListWidgetItem()
        item.setData(Qt.UserRole, label)
        return item

    def setItemLabel(self, item, label, color=None):
        qlabel = QtWidgets.QLabel()
        if color is None:
            qlabel.setText("{}".format(label))
        else:
            qlabel.setText(
                '{} <font color="#{:02x}{:02x}{:02x}">●</font>'.format(
                    html.escape(label), *color
                )
            )
        qlabel.setAlignment(Qt.AlignBottom)

        item.setSizeHint(qlabel.sizeHint())

        self.setItemWidget(item, qlabel)

    def addUniquePersonLabel(self, person_id, color, shape=None):
        """
        Add a unique person label with ID, color and shape association.
        Args:
            person_id: Unique ID for the person
            color: Tuple representing the color (R, G, B)
            shape: Associated Shape object
        """
        try:
            label_text = f"Person ID {person_id}"
            
            # Check if ID already exists
            if person_id in self._id_map:
                logger.debug(f"Person ID {person_id} already exists")
                return self._id_map[person_id]
            
            # Create new item with shape association
            item = QtWidgets.QListWidgetItem(label_text)
            item.setBackground(QtGui.QColor(*color))
            if shape:
                item.setData(Qt.UserRole + 1, shape)  # Store shape reference
                logger.debug(f"Added shape reference for Person ID {person_id}")
            
            self.addItem(item)
            self._id_map[person_id] = item
            
            # Create label widget with color indicator
            qlabel = QtWidgets.QLabel()
            qlabel.setText(
                '{} <font color="#{:02x}{:02x}{:02x}">●</font>'.format(
                    html.escape(label_text), *color
                )
            )
            qlabel.setAlignment(Qt.AlignBottom)
            self.setItemWidget(item, qlabel)
            
            logger.debug(f"Successfully added unique person label: {person_id}")
            return item
            
        except Exception as e:
            logger.error(f"Error adding unique person label: {e}")
            return None

    def getPersonShape(self, person_id):
        """Get shape associated with person ID."""
        item = self._id_map.get(person_id)
        if item:
            return item.data(Qt.UserRole + 1)
        return None

    def updateUniquePersonLabel(self, person_id, color, shape=None):
        """
        Update existing person label.
        """
        item = self._id_map.get(person_id)
        if item:
            if shape:
                item.setData(Qt.UserRole + 1, shape)
            item.setBackground(QtGui.QColor(*color))
            logger.debug(f"Updated person label: {person_id}")
            return True
        return False

    def clearUniquePersonLabels(self):
        """Clear all person labels and reset ID mapping."""
        self.clear()
        self._id_map.clear()
        logger.debug("Cleared all unique person labels")

    def validateIDs(self):
        """Validate ID consistency."""
        issues = []
        for person_id, item in self._id_map.items():
            if not self.findItems(f"Person ID {person_id}", Qt.MatchExactly):
                issues.append(person_id)
        
        if issues:
            logger.warning(f"Found inconsistent IDs: {issues}")
        return len(issues) == 0
    
    def hasPersonLabel(self, label):
        """Check if person label exists."""
        return self.findItemByLabel(label) is not None
    
    def removePersonLabel(self, label):
        """Remove person label."""
        item = self.findItemByLabel(label)
        if item:
            self.takeItem(self.row(item))