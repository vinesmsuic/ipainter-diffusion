#==================================================
# Author: vinesmsuic
#+=================================================

from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import * 
import sys
import os
from PIL import Image, ImageQt
import cv2
import numpy as np

class DragDropListWidget(QListWidget):
	def __init__(self, folder_path):
		super().__init__()
		self.setAcceptDrops(True)
		self.setIconSize(QSize(72, 72))
		self.setMouseTracking(True)
		for root, dirs, files in os.walk(folder_path, topdown=False):
			list_files = self.get_img_list(files)
			for file in list_files:
				in_path = os.path.join(root, file)
				self.put_into_img_list(in_path)

	def put_into_img_list(self, path):
		picture = Image.open(path)
		if picture.mode == "RGBA" or picture.mode == "P":
			picture.thumbnail((71, 71), Image.Resampling.LANCZOS)
			icon = QIcon(QPixmap.fromImage(ImageQt.ImageQt(picture)))
			item = QListWidgetItem(os.path.basename(path), self)
			item.setStatusTip(path)
			item.setIcon(icon)
		else:
			print("=> Image " + str(path) + "is in mode " + picture.mode + " and it is not in RGBA format. Not adding into list.")

	def get_img_list(self, path_list):
		IMG_FORMATS = 'png', 'PNG'
		list_img = [img for img in path_list if (img.split(".")[-1] in IMG_FORMATS) ==True]
		return list_img

class MyCanvas(QWidget):
	def __init__(self):
		super().__init__()
		# Defining a scene rect of 400x200, with it's origin at 0,0.
		# If we don't set this on creation, we can set it later with .setSceneRect
		self.scene = QGraphicsScene()

		self.scene_out = QGraphicsScene()

		# Define our layout.
		box1 = QHBoxLayout()

		btn_up = QPushButton("Bring to Front")
		btn_up.clicked.connect(self.up)
		box1.addWidget(btn_up)

		btn_down = QPushButton("Bring to Back")
		btn_down.clicked.connect(self.down)
		box1.addWidget(btn_down)

		btn_scaleup = QPushButton("Scale Up")
		btn_scaleup.clicked.connect(self.scale_up)
		box1.addWidget(btn_scaleup)

		btn_scaledown = QPushButton("Scale Down")
		btn_scaledown.clicked.connect(self.scale_down)
		box1.addWidget(btn_scaledown)

		self.slider_rotate = QSlider(Qt.Horizontal)
		self.slider_rotate.setRange(0, 360)
		self.slider_rotate.valueChanged.connect(self.rotate)
		box1.addWidget(self.slider_rotate)

		self.view = QGraphicsView(self.scene)
		self.view.setRenderHint(QPainter.Antialiasing)
		self.view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
		self.view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

		self.view_out = QGraphicsView(self.scene_out)
		self.view_out.setRenderHint(QPainter.Antialiasing)
		self.view_out.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
		self.view_out.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

		btn_remove = QPushButton("Remove selected")
		btn_remove.clicked.connect(self.delete_single_item)
		box1.addWidget(btn_remove)

		btn_remove_all = QPushButton("Clear All")
		btn_remove_all.clicked.connect(self.clear)
		box1.addWidget(btn_remove_all)

		box2 = QHBoxLayout()
		box2.addWidget(self.view)
		box2.addWidget(self.view_out)

		large_box = QVBoxLayout(self)
		large_box.addLayout(box1)
		large_box.addLayout(box2)

		self.setLayout(large_box)

		self.set_size()
	
	def set_size(self):
		self.scene.setSceneRect(0,0,512,512)
		self.view.setFixedWidth(512)
		self.view.setFixedHeight(512)
		self.scene_out.setSceneRect(0,0,512,512)
		self.view_out.setFixedWidth(512)
		self.view_out.setFixedHeight(512)

	def spawn_image(self, path_to_file):
		self.image_qt = QImage(path_to_file)

		pic = QGraphicsPixmapItem()
		pixmap = QPixmap.fromImage(self.image_qt)
		pixmap = pixmap.scaled(512, 512, Qt.KeepAspectRatio) # automatically adjust to the largest size possible.
		pic.setPixmap(pixmap)
		self.scene.addItem(pic)

		self.set_moveable()

	def delete_single_item(self):
		""" Remove all images """
		items = self.scene.selectedItems()
		for item in items:
			self.scene.removeItem(item)

	def clear(self):
		""" Remove all images """
		for item in self.scene.items():
			self.scene.removeItem(item)
		self.scene.clear()
		self.scene_out.clear()

	def set_moveable(self, flag=True):
		# Set all items as moveable and selectable.
		for item in self.scene.items():
			item.setFlag(QGraphicsItem.ItemIsMovable, flag)
			item.setFlag(QGraphicsItem.ItemIsSelectable, flag)

	def up(self):
		""" Iterate all selected items in the view, moving them forward. """
		if len(self.scene.selectedItems()) != 0:
			items = self.scene.selectedItems()
			for item in items:
				z = item.zValue()
				item.setZValue(z + 1)

	def down(self):
		""" Iterate all selected items in the view, moving them backward. """
		if len(self.scene.selectedItems()) != 0:
			items = self.scene.selectedItems()
			for item in items:
				z = item.zValue()
				item.setZValue(z - 1)

	def rotate(self, value):
		""" Rotate the object by the received number of degrees """
		if len(self.scene.selectedItems()) != 0:
			items = self.scene.selectedItems()
			for item in items:
				item.setRotation(value)

	def scale_up(self):
		if len(self.scene.selectedItems()) != 0:
			items = self.scene.selectedItems()
			for item in items:
				scale = item.scale()
				new_scale = scale+0.1
				if int(new_scale*10) != 0 and new_scale>0:
					item.setScale(new_scale)

	def scale_down(self):
		if len(self.scene.selectedItems()) != 0:
			items = self.scene.selectedItems()
			for item in items:
				scale = item.scale()
				new_scale = scale-0.1
				if int(new_scale*10) != 0 and new_scale>0:
					item.setScale(new_scale)

	def save_input(self, out_dir="out_inpaint", filename="000000.png"):
		self.set_moveable(False)
		rect = self.view.scene().sceneRect()
		#print(rect.height())
		#print(rect.width())
		
		pixmap = QImage(rect.width(), rect.height(), QImage.Format_ARGB32_Premultiplied)
		painter = QPainter(pixmap)
		painter.setRenderHint(QPainter.Antialiasing)
		painter.setRenderHint(QPainter.SmoothPixmapTransform)
		pixmap.fill(Qt.transparent)

		#print(pixmap.rect().height())
		#print(pixmap.rect().width())
		rectf = QRectF(0,0,pixmap.rect().height(), pixmap.rect().width())
		self.view.scene().render(painter, rectf, rect)
		painter.end()
		os.makedirs(out_dir, exist_ok=True)
		pixmap.save(os.path.join(out_dir, filename))
		self.set_moveable(True)

	def plot_image(self, image_pil):
		image_pil = image_pil.resize((512, 512),Image.LANCZOS) # LANCZOS best for both upscaling and downscaling quality
		self.scene_out.clear()
		self.imgQ = ImageQt.ImageQt(image_pil)  # we need to hold reference to imgQ, or it will crash
		pixMap = QPixmap.fromImage(self.imgQ)
		self.scene_out.addPixmap(pixMap)
		self.view_out.fitInView(QRectF(0, 0, 512, 512), Qt.KeepAspectRatio)
		self.scene_out.update()

	def get_mask_from_img_path(self, img_path, erode_kernel=-1, invert_mask=False):

		img = cv2.imread(cv2.samples.findFile(img_path), cv2.IMREAD_UNCHANGED)
		# get mask from alpha channel
		mask = img[:,:,3]
		# Drop all pixels that has opacity below 100%
		mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)[1]
		
		if erode_kernel > 0:
			kernel = np.ones((erode_kernel,erode_kernel), np.uint8)
			mask = cv2.erode(mask, kernel, iterations = 1)

		if invert_mask:
			mask = 255 - mask
		
		# Convert back to PIL image
		mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
		mask_pil =Image.fromarray(mask)
		"""
		Convert from OpenCV img to PIL img will lost transparent channel. 
		While convert PIL img to OpenCV img will able to keep transparent channel, 
		although cv2.imshow not display it but save as png will gave result normally.
		"""
		return mask_pil