import sys
import threading
from PyQt6.QtWidgets import (QApplication, QMainWindow, QPushButton, QVBoxLayout, 
                            QWidget, QLabel, QFileDialog, QLineEdit, QComboBox, 
                            QTextEdit, QMessageBox, QHBoxLayout, QGroupBox, QScrollArea,
                            QFrame)
from PyQt6.QtCore import QThread, pyqtSignal, QTimer
import uvicorn
import requests
import json

class ServerThread(QThread):
    error_signal = pyqtSignal(str)

    def run(self):
        try:
            uvicorn.run("main:app", host="127.0.0.1", port=8000, log_level="error")
        except Exception as e:
            self.error_signal.emit(str(e))

class ResultCard(QFrame):
    def __init__(self, result_data, parent=None):
        super().__init__(parent)
        self.setFrameStyle(QFrame.Shape.Box | QFrame.Shadow.Raised)
        self.setStyleSheet("""
            ResultCard {
                background-color: white;
                border-radius: 8px;
                border: 1px solid #ddd;
                margin: 8px;
                padding: 12px;
            }
            QLabel {
                margin: 4px 0;
            }
            .score-label {
                color: #0078D4;
                font-weight: bold;
            }
        """)
        
        layout = QVBoxLayout(self)
        
        # Score
        score_layout = QHBoxLayout()
        score_label = QLabel(f"Score: {result_data['score']:.2f}")
        score_label.setStyleSheet("color: #0078D4; font-weight: bold;")
        score_layout.addWidget(score_label)
        score_layout.addStretch()
        layout.addLayout(score_layout)
        
        # Text content
        text_label = QLabel(f"Text: {result_data['text']}")
        text_label.setWordWrap(True)
        layout.addWidget(text_label)
        
        # File path
        file_label = QLabel(f"File: {result_data['file_path']}")
        file_label.setStyleSheet("color: #666;")
        layout.addWidget(file_label)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Document Processing System")
        self.setGeometry(100, 100, 1000, 800)
        
        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        layout.setSpacing(10)
        layout.setContentsMargins(20, 20, 20, 20)

        # Collection management group
        collection_group = QGroupBox("Collection Management")
        collection_layout = QVBoxLayout()
        
        # Collection selector row
        selector_layout = QHBoxLayout()
        self.collection_combo = QComboBox()
        self.collection_combo.setMinimumWidth(300)
        self.collection_combo.setPlaceholderText("Select collection")
        selector_layout.addWidget(self.collection_combo)
        
        self.refresh_collections_btn = QPushButton("Refresh")
        self.refresh_collections_btn.setMaximumWidth(100)
        self.refresh_collections_btn.clicked.connect(self.refresh_collections)
        selector_layout.addWidget(self.refresh_collections_btn)
        collection_layout.addLayout(selector_layout)
        
        # New collection row
        new_collection_layout = QHBoxLayout()
        self.collection_name = QLineEdit()
        self.collection_name.setPlaceholderText("New collection name")
        new_collection_layout.addWidget(self.collection_name)
        
        self.create_collection_btn = QPushButton("Create Collection")
        self.create_collection_btn.setMaximumWidth(150)
        self.create_collection_btn.clicked.connect(self.create_collection)
        new_collection_layout.addWidget(self.create_collection_btn)
        collection_layout.addLayout(new_collection_layout)
        
        collection_group.setLayout(collection_layout)
        layout.addWidget(collection_group)

        # File processing group
        processing_group = QGroupBox("File Processing")
        processing_layout = QVBoxLayout()
        
        file_row_layout = QHBoxLayout()
        self.file_path = QLineEdit()
        self.file_path.setPlaceholderText("Select an image file")
        file_row_layout.addWidget(self.file_path)
        
        self.browse_btn = QPushButton("Browse")
        self.browse_btn.setMaximumWidth(100)
        self.browse_btn.clicked.connect(self.browse_file)
        file_row_layout.addWidget(self.browse_btn)
        processing_layout.addLayout(file_row_layout)
        
        self.process_btn = QPushButton("Process File")
        self.process_btn.clicked.connect(self.process_file)
        processing_layout.addWidget(self.process_btn)
        
        processing_group.setLayout(processing_layout)
        layout.addWidget(processing_group)

        # Search group
        search_group = QGroupBox("Search")
        search_layout = QVBoxLayout()
        
        search_row_layout = QHBoxLayout()
        self.search_query = QLineEdit()
        self.search_query.setPlaceholderText("Enter search query")
        search_row_layout.addWidget(self.search_query)
        
        self.search_btn = QPushButton("Search")
        self.search_btn.setMaximumWidth(100)
        self.search_btn.clicked.connect(self.search)
        search_row_layout.addWidget(self.search_btn)
        search_layout.addLayout(search_row_layout)
        
        search_group.setLayout(search_layout)
        layout.addWidget(search_group)

        # Results group
        results_group = QGroupBox("Results")
        results_layout = QVBoxLayout()
        
        # Create a widget to hold all result cards
        self.results_container = QWidget()
        self.results_layout = QVBoxLayout(self.results_container)
        self.results_layout.setSpacing(10)
        self.results_layout.addStretch()
        
        # Create a scroll area and add the container
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(self.results_container)
        scroll_area.setMinimumHeight(400)
        
        results_layout.addWidget(scroll_area)
        results_group.setLayout(results_layout)
        layout.addWidget(results_group)

        # Start the server and load collections
        self.server_thread = ServerThread()
        self.server_thread.error_signal.connect(self.show_error)
        self.server_thread.start()
        
        # Initial collection load
        self.retry_count = 0
        self.max_retries = 3
        QTimer.singleShot(5000, self.try_refresh_collections)

        # Set stylesheet for better appearance
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f0f0f0;
            }
            QGroupBox {
                font-weight: bold;
                border: 1px solid #cccccc;
                border-radius: 6px;
                margin-top: 6px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 7px;
                padding: 0px 5px 0px 5px;
            }
            QPushButton {
                background-color: #0078D4;
                color: white;
                border: none;
                padding: 5px 15px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #106EBE;
            }
            QPushButton:pressed {
                background-color: #005A9E;
            }
            QLineEdit, QComboBox, QTextEdit {
                padding: 5px;
                border: 1px solid #cccccc;
                border-radius: 4px;
                background-color: white;
            }
            QLineEdit:focus, QComboBox:focus, QTextEdit:focus {
                border: 1px solid #0078D4;
            }
            QScrollArea {
                border: none;
                background-color: transparent;
            }
            QScrollBar:vertical {
                border: none;
                background: #f0f0f0;
                width: 10px;
                margin: 0px;
            }
            QScrollBar::handle:vertical {
                background: #c1c1c1;
                min-height: 30px;
                border-radius: 5px;
            }
            QScrollBar::handle:vertical:hover {
                background: #a8a8a8;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
        """)

    def try_refresh_collections(self):
        try:
            self.refresh_collections()
        except Exception as e:
            if self.retry_count < self.max_retries:
                self.retry_count += 1
                QTimer.singleShot(2000, self.try_refresh_collections)  # Retry after 2 seconds
            else:
                QMessageBox.warning(
                    self, 
                    "Connection Error", 
                    "Could not connect to server. Please check if the server is running."
                )

    def show_error(self, error_message):
        QMessageBox.critical(self, "Error", error_message)

    def browse_file(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Image File", "", 
                                                 "Image Files (*.png *.jpg *.jpeg)")
        if file_name:
            self.file_path.setText(file_name)

    def refresh_collections(self):
        try:
            response = requests.get("http://127.0.0.1:8000/collections", timeout=5)
            if response.status_code == 200:
                collections = response.json()["collections"]
                current_text = self.collection_combo.currentText()
                
                self.collection_combo.clear()
                self.collection_combo.addItems(collections)
                
                # Restore previous selection if it still exists
                index = self.collection_combo.findText(current_text)
                if index >= 0:
                    self.collection_combo.setCurrentIndex(index)
            else:
                QMessageBox.warning(self, "Error", "Failed to fetch collections")
        except requests.exceptions.ConnectionError:
            raise Exception("Server not ready")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def get_current_collection(self):
        """Get the current collection name from either the combo box or text input"""
        return self.collection_combo.currentText() or self.collection_name.text()

    def create_collection(self):
        try:
            response = requests.post(
                "http://127.0.0.1:8000/collections/create",
                json={"collection_name": self.collection_name.text()}
            )
            if response.status_code == 200:
                QMessageBox.information(self, "Success", "Collection created successfully")
                self.refresh_collections()
            else:
                QMessageBox.warning(self, "Error", response.json()["detail"])
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def process_file(self):
        try:
            response = requests.post(
                "http://127.0.0.1:8000/process",
                json={
                    "file_path": self.file_path.text(),
                    "collection_name": self.get_current_collection()
                }
            )
            if response.status_code == 200:
                result = response.json()
                self.results_display.setText(f"Processed text: {result['text']}")
            else:
                QMessageBox.warning(self, "Error", response.json()["detail"])
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def search(self):
        try:
            response = requests.post(
                "http://127.0.0.1:8000/search",
                json={
                    "query": self.search_query.text(),
                    "collection_name": self.get_current_collection()
                }
            )
            if response.status_code == 200:
                results = response.json()["results"]
                
                # Clear previous results
                for i in reversed(range(self.results_layout.count())):
                    widget = self.results_layout.itemAt(i).widget()
                    if widget:
                        widget.deleteLater()
                
                # Add new result cards
                for result in results:
                    card = ResultCard(result)
                    self.results_layout.insertWidget(0, card)
                
                # Add stretch at the end to keep cards at the top
                self.results_layout.addStretch()
            else:
                QMessageBox.warning(self, "Error", response.json()["detail"])
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec()) 