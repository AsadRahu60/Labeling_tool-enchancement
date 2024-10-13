from qtpy import QtCore
from qtpy import QtWidgets


class ToolBar(QtWidgets.QToolBar):
    def __init__(self, title):
        super(ToolBar, self).__init__(title)
        layout = self.layout()
        
        m = (5, 5, 5, 5)
        
        layout.setSpacing(5)
        
        layout.setContentsMargins(*m)
        
        self.setContentsMargins(*m)
        
        self.setWindowFlags(self.windowFlags() | QtCore.Qt.FramelessWindowHint)

        # Store references to actions and buttons
        self.actions_map = {}
        
    def addAction(self, action):
        if isinstance(action, QtWidgets.QWidgetAction):
            return super(ToolBar, self).addAction(action)
        # Create a tool button for the action
        btn = QtWidgets.QToolButton()
        btn.setDefaultAction(action)
        btn.setToolButtonStyle(self.toolButtonStyle())
        # Add the button widget to the toolbar
        self.addWidget(btn)
        
        # Store the action and button for future reference
        self.actions_map[action] = btn

        # Center align all tool buttons
        for i in range(self.layout().count()):
            if isinstance(self.layout().itemAt(i).widget(), QtWidgets.QToolButton):
                self.layout().itemAt(i).setAlignment(QtCore.Qt.AlignCenter)

    
    def setActionVisible(self, action, visible):
        """
        Toggle the visibility of a specific action in the toolbar.
        """
        if action in self.actions_map:
            self.actions_map[action].setVisible(visible)

    def setButtonStyle(self, style):
        """
        Set the tool button style (e.g., text under icon, icon only, etc.).
        """
        for action, button in self.actions_map.items():
            button.setToolButtonStyle(style)

    def clearActions(self):
        """
        Remove all actions from the toolbar.
        """
        for action, button in self.actions_map.items():
            self.removeAction(action)
            button.deleteLater()
        self.actions_map.clear()

    def getButtonForAction(self, action):
        """
        Get the button widget associated with an action.
        """
        return self.actions_map.get(action, None)

    def addSeparator(self):
        """
        Add a separator to the toolbar.
        """
        separator = QtWidgets.QToolButton()
        separator.setEnabled(False)
        separator.setFixedSize(2, self.sizeHint().height())
        separator.setStyleSheet("background-color: gray;")
        self.addWidget(separator)