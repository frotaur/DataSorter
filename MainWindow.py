from tkinter import *


from PIL import Image
from MultiButFrame import MultiButFrame
from PicFrame import PicFrame
import tkinter.font as tkfont
import numpy as np
import glob
import os




class MainWindow(Tk):

	def __init__(self, **kwargs):
		Tk.__init__(self, **kwargs)
		fontana = tkfont.Font(family="Unispace", size=12, weight="bold")
		m_len = fontana.measure("M")
		width=(m_len+2*1)*15*4;
		height=600;
		
		self.geometry(str(width)+"x"+str(height))
		os.chdir(os.path.dirname(__file__))#Change to working directory
		# CREATE FOLDERS IF NOT DONE YET
		path = "Data\\"
		if(not os.path.exists("Data\\")):
			os.mkdir("Data")
			print("Heello ?")
		paths = ["WhaleInWater", "WhaleOutWater", "NotAWhale"]
		paths = ["Data\\"+paths[i] for i in range(len(paths))]

		for path in paths:
			if(not os.path.exists(path)):
				os.mkdir(path)

		upframe = Frame(self)
		self.picframe= PicFrame(upframe,"RawImages\\")
		downframe = MultiButFrame(self)


		yo = Button(upframe,text="am king",command=self.changeimg)
		yo.pack(side=TOP,fill=BOTH,expand=1)
		self.picframe.pack(side=TOP, fill=X)
		upframe.pack(side=TOP,fill=X,expand=1)
		downframe.pack(side=BOTTOM,fill=BOTH,expand=True)

	def changeimg(self):
		self.picframe.next_image()

	def mainloop(self):
		Tk.mainloop(self)