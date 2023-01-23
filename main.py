#==================================================
# Author: vinesmsuic
#+=================================================

from gui import DragDropListWidget, MyCanvas
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import * 
import sys
import os
from PIL import Image, ImageQt
from repainter import RePainter, ImgPath2Tensor

import random
import string

# creating class for window
class Window(QMainWindow):
	def __init__(self):
		super().__init__()
		title = "Intelligent Painter Demo V1.1"
		self.setWindowTitle(title)
		top = 0
		left = 0
		width = 1600
		height = 900
		self.setGeometry(top, left, width, height)
		self.canvas = MyCanvas()
		self.image_list = DragDropListWidget(folder_path = 'inject_photo')
		self.image_list.setFixedWidth(int(width*0.2))
		self.canvas.setFixedWidth(int(width*0.8))

		self.cwd = os.getcwd()

		btn_add_from_dia = QPushButton("Add extra image to list")
		btn_add_from_dia.clicked.connect(self.get_img_from_dialog)

		btn_add = QPushButton("Add selected image to canvas")
		btn_add.clicked.connect(self.add_img_to_canvas)

		btn_diffuse = QPushButton("Diffuse the Background")
		btn_diffuse.clicked.connect(self.generate)

		self.progress_label = QLabel(self)
		self.progress_label.setFrameStyle(QFrame.Sunken)
		self.progress_label.setText("When start inferencing, please wait for 5 ~ 30 minutes")
		self.progress_label.setAlignment(Qt.AlignLeft)

		wid = QFrame(self)
		self.setCentralWidget(wid)
		vbox1 = QVBoxLayout()

		box = QHBoxLayout()

		vbox2 = QVBoxLayout()
		vbox1.addWidget(btn_add)
		vbox1.addWidget(self.image_list)
		vbox1.addWidget(btn_add_from_dia)

		vbox2.addWidget(self.canvas)
		vbox2.addWidget(self.progress_label)
		vbox2.addWidget(btn_diffuse)

		box.addLayout(vbox1)
		box.addLayout(vbox2)
		wid.setLayout(box)

		#self._spawn_dummy_image()
		self.model = RePainter()
		self.helper = ImgPath2Tensor()

	def add_img_to_canvas(self):
		if len(self.image_list.selectedItems()) != 0:
			self.canvas.spawn_image(self.image_list.selectedItems()[0].statusTip())

	def get_img_from_dialog(self):
			fnames = QFileDialog.getOpenFileNames(self, 'Open file', self.cwd, "Image files (*.png *.jpg)")
			fnames = fnames[0] # its a list of image paths
			for fname in fnames:
					self.image_list.put_into_img_list(fname)

	def generate(self):
		out_dir = "out_inpaint"
		name = ''.join(random.choices(string.ascii_uppercase, k=6))
		filename = str(name + '.png')
		out_path = os.path.join(out_dir, filename)
		self.canvas.save_input(out_dir, filename)
		image = self.helper.get_tensor_from_img_path(out_path, is_mask=False)
		mask = self.canvas.get_mask_from_img_path(out_path, erode_kernel=-1, invert_mask=False)
		mask.save(os.path.join(out_dir, name + '_mask.png'))
		mask = self.helper.get_tensor_from_pil(mask, is_mask=True)
		result_path = self.model.infer_one_image(image, mask, out_path)
		self.canvas.plot_image(Image.open(result_path).convert("RGB"))
		self.progress_label.setText("Saved to " + str(out_path))
	
if __name__ == "__main__":
	app = QApplication(sys.argv)
	
	try:
		import qdarktheme
	except ImportError:
		pass
	else:
		# Apply dark theme to Qt application
		app.setStyleSheet(qdarktheme.load_stylesheet())

	window = Window()
	window.show()
	sys.exit(app.exec())
