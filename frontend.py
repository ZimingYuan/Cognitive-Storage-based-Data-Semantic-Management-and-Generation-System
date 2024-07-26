from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QFileDialog, QLabel,
    QLineEdit, QSpinBox, QPushButton, QListWidget, QGroupBox
)
from PySide6.QtCore import Qt, QThread, Signal
import sys
import generate_embeddings
import file_searching
import generate_slides

class GenerateEmbeddings(QThread):
    finished = Signal()
    def __init__(self, process_path, max_char_count, max_sequence_count):
        super().__init__()
        self.process_path = process_path
        self.max_char_count = max_char_count
        self.max_sequence_count = max_sequence_count
    def run(self):
        generate_embeddings.generate_embeddings(self.process_path, self.max_char_count, self.max_sequence_count, self.max_char_count // 2, 'aie')
        self.finished.emit()

class FileSearching(QThread):
    finished = Signal(list)
    def __init__(self, process_path, topk):
        super().__init__()
        self.process_path = process_path
        self.topk = topk
    def run(self):
        file_lists = file_searching.file_searching(self.process_path, self.topk, 'aie')
        self.finished.emit(file_lists)

class GenerateSlides(QThread):
    finished = Signal()
    def __init__(self, file_lists, process_path, max_char_count, max_sequence_count, language):
        super().__init__()
        self.file_lists = file_lists
        self.process_path = process_path
        self.max_char_count = max_char_count
        self.max_sequence_count = max_sequence_count
        self.language = language
    def run(self):
        generate_slides.generate_slides(self.file_lists, self.process_path, self.max_char_count, self.max_sequence_count, self.max_char_count // 2, self.language, 'aie')
        self.finished.emit()

class DirectoryProcessor(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Cognitive Storage-based Data Semantic Management and Generation System")
        self.initUI()
        self.file_lists = []

    def initUI(self):
        layout = QVBoxLayout()

        # Process Directory Section
        process_dir_layout = QHBoxLayout()
        process_dir_label = QLabel("Process directory")
        self.process_dir_input = QLineEdit()
        self.process_dir_input.setPlaceholderText("Select a directory")
        process_dir_button = QPushButton("Browse")
        process_dir_button.clicked.connect(self.browse_directory)

        process_dir_layout.addWidget(process_dir_label)
        process_dir_layout.addWidget(self.process_dir_input)
        process_dir_layout.addWidget(process_dir_button)
        layout.addLayout(process_dir_layout)

        max_seg_layout = QHBoxLayout()
        max_seg_label = QLabel("Max segment count per file")
        self.max_seg_input = QSpinBox()
        self.max_seg_input.setRange(2, 8192)
        self.max_seg_input.setValue(32)
        max_seg_layout.addWidget(max_seg_label)
        max_seg_layout.addWidget(self.max_seg_input)

        max_char_layout = QHBoxLayout()
        max_char_label = QLabel("Max character count per segment")
        self.max_char_input = QSpinBox()
        self.max_char_input.setRange(2, 8192)
        self.max_char_input.setValue(1024)
        max_char_layout.addWidget(max_char_label)
        max_char_layout.addWidget(self.max_char_input)

        self.generate_button = QPushButton("Generate")
        self.generate_button.clicked.connect(self.generate_embeddings)

        index_gen_group = QGroupBox('Index Generation')
        index_gen_layout = QVBoxLayout(index_gen_group)
        index_gen_layout.addLayout(max_seg_layout)
        index_gen_layout.addLayout(max_char_layout)
        index_gen_layout.addWidget(self.generate_button)
        layout.addWidget(index_gen_group)

        search_prompt_layout = QHBoxLayout()
        search_prompt_label = QLabel("Searching prompt")
        self.search_prompt_input = QLineEdit()
        self.search_prompt_input.setText('Some documents.')
        search_prompt_layout.addWidget(search_prompt_label)
        search_prompt_layout.addWidget(self.search_prompt_input)

        max_result_layout = QHBoxLayout()
        max_result_label = QLabel("Max result count")
        self.max_result_input = QSpinBox()
        self.max_result_input.setValue(10)
        self.max_result_input.setRange(1, 1000)
        self.search_button = QPushButton("Search")
        self.search_button.clicked.connect(self.file_search)
        max_result_layout.addWidget(max_result_label)
        max_result_layout.addWidget(self.max_result_input)
        max_result_layout.addWidget(self.search_button)

        self.search_result_list = QListWidget()

        file_search_group = QGroupBox('File Searching')
        file_search_layout = QVBoxLayout(file_search_group)
        file_search_layout.addLayout(search_prompt_layout)
        file_search_layout.addLayout(max_result_layout)
        file_search_layout.addWidget(self.search_result_list)
        layout.addWidget(file_search_group)

        self.search_result_list2 = QListWidget()

        max_seg_layout2 = QHBoxLayout()
        max_seg_label2 = QLabel("Max segment count per file")
        self.max_seg_input2 = QSpinBox()
        self.max_seg_input2.setRange(2, 8192)
        self.max_seg_input2.setValue(32)
        max_seg_layout2.addWidget(max_seg_label2)
        max_seg_layout2.addWidget(self.max_seg_input2)

        max_char_layout2 = QHBoxLayout()
        max_char_label2 = QLabel("Max character count per segment")
        self.max_char_input2 = QSpinBox()
        self.max_char_input2.setRange(2, 8192)
        self.max_char_input2.setValue(1024)
        max_char_layout2.addWidget(max_char_label2)
        max_char_layout2.addWidget(self.max_char_input2)

        generate_slides_layout = QHBoxLayout()
        language_label = QLabel("Generating language")
        self.language_input = QLineEdit()
        self.language_input.setText('English')
        self.generate_slides_button = QPushButton("Generate slides")
        self.generate_slides_button.clicked.connect(self.generate_slides)
        generate_slides_layout.addWidget(language_label)
        generate_slides_layout.addWidget(self.language_input)
        generate_slides_layout.addWidget(self.generate_slides_button)


        file_summarize_group = QGroupBox('File Summarizing')
        file_summarize_layout = QVBoxLayout(file_summarize_group)
        file_summarize_layout.addWidget(self.search_result_list2)
        file_summarize_layout.addLayout(max_seg_layout2)
        file_summarize_layout.addLayout(max_char_layout2)
        file_summarize_layout.addLayout(generate_slides_layout)
        layout.addWidget(file_summarize_group)

        self.state = QLabel('Idle')
        self.state.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.state)

        self.setLayout(layout)

    def browse_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Directory")
        self.process_dir_input.setText(directory)

    def generate_embeddings(self):
        self.state.setText("Generating embedding...")
        self.generate_button.setEnabled(False)
        self.search_button.setEnabled(False)
        self.generate_slides_button.setEnabled(False)
        self.thread = GenerateEmbeddings(self.process_dir_input.text(), self.max_char_input.value(), self.max_seg_input.value())
        self.thread.finished.connect(self.recover)
        self.thread.start()

    def file_search(self):
        self.state.setText("Searching files...")
        self.generate_button.setEnabled(False)
        self.search_button.setEnabled(False)
        self.generate_slides_button.setEnabled(False)
        self.thread = FileSearching(self.process_dir_input.text(), self.max_result_input.value())
        self.thread.finished.connect(self.get_filelists)
        self.thread.start()

    def generate_slides(self):
        self.state.setText("Generating slides...")
        self.generate_button.setEnabled(False)
        self.search_button.setEnabled(False)
        self.generate_slides_button.setEnabled(False)
        self.thread = GenerateSlides(self.file_lists, self.process_dir_input.text(), self.max_char_input2.value(), self.max_seg_input2.value(), self.language_input.text())
        self.thread.finished.connect(self.recover)
        self.thread.start()
        print("Generating slides...")

    def recover(self):
        self.generate_button.setEnabled(True)
        self.search_button.setEnabled(True)
        self.generate_slides_button.setEnabled(True)
        self.state.setText("Done")

    def get_filelists(self, file_lists):
        self.search_result_list.clear()
        self.search_result_list.addItems(file_lists)
        self.search_result_list2.clear()
        self.search_result_list2.addItems(file_lists)
        self.file_lists = file_lists
        self.recover()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DirectoryProcessor()
    window.show()
    sys.exit(app.exec())
