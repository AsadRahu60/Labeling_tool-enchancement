# -*- encoding: utf-8 -*-

import html

from qtpy import QtWidgets
from qtpy import QtCore ,QtGui
from qtpy.QtCore import Qt

from .escapable_qlist_widget import EscapableQListWidget


class UniqueLabelQListWidget(EscapableQListWidget):
    def mousePressEvent(self, event):
        super(UniqueLabelQListWidget, self).mousePressEvent(event)
        if not self.indexAt(event.pos()).isValid():
            self.clearSelection()

    def findItemByLabel(self, label):
        for row in range(self.count()):
            item = self.item(row)
            if item.data(Qt.UserRole) == label:
                return item

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

    def addUniquePersonLabel(self, person_id, color):
        """
        Add a unique person label with ID and color.
        Args:
            person_id: Unique ID for the person.
            color: Tuple representing the color (R, G, B) for the label.
        """
        label_text = f"Person ID {person_id}"
        existing_items = self.findItems(label_text, QtCore.Qt.MatchExactly)
        if not existing_items:  # Add only if it doesn't already exist
            item = QtWidgets.QListWidgetItem(label_text)
            item.setBackground(QtGui.QColor(*color))
            self.addItem(item)

    def updateUniquePersonLabel(self, person_id, color):
        """
        Update the color of an existing unique person label.
        Args:
            person_id: Unique ID for the person.
            color: New color for the label (R, G, B).
        """
        label_text = f"Person ID {person_id}"
        items = self.findItems(label_text, QtCore.Qt.MatchExactly)
        for item in items:
            item.setBackground(QtGui.QColor(*color))

    def clearUniquePersonLabels(self):
        """Clear all unique person labels from the list."""
        self.clear()