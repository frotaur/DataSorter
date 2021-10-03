from tkinter import *


from MultiButFrame import MultiButFrame
from PicFrame import PicFrame
from OptionPanel import OptionPanel
import tkinter.font as tkfont
import os




class MainWindow(Tk):

	def __init__(self, **kwargs):
		Tk.__init__(self, **kwargs)
		# fontana = tkfont.Font(family="Unispace", size=12, weight="bold")
		# m_len = fontana.measure("M")
		# width=(m_len+2*1)*15*4
		width=960
		height=620
		self.geometry(str(width)+"x"+str(height))
		os.chdir(os.path.dirname(__file__))  # Change to working directory

		self.top = OptionPanel(self)
		if("RawImages" not in os.listdir()):
			self.top.wait_window()
			raw_imgfold = self.top.getFolder()
		else:
			raw_imgfold="RawImages"

		if(raw_imgfold is None):
			raise ValueError("Raw image folder not found")
		self.lift()
		# CREATE FOLDERS IF NOT DONE YET
		path = "Data\\"
		if(not os.path.exists("Data\\")):
			os.mkdir("Data")
		paths=[path+fold for fold in os.listdir(path)]

		for path in paths:
			if(not os.path.exists(path)):
				os.mkdir(path)

		upframe = Frame(self)
		self.picframe= PicFrame(upframe,os.path.join(raw_imgfold,""))
		self.downframe = MultiButFrame(self,self.picframe)


		yo = Button(upframe,text="Options",command=self.openOption)
		yo.pack(side=TOP,fill=BOTH,expand=1)
		self.picframe.pack(side=TOP, fill=X)
		upframe.pack(side=TOP,fill=X,expand=1)
		self.downframe.pack(side=BOTTOM,fill=BOTH,expand=True)

	def changeimg(self):
		self.picframe.next_image()

	def getimgpath(self):
		return self.picframe.current_image()

	def mainloop(self):
		Tk.mainloop(self)

	def openOption(self):
		self.top=OptionPanel(self)
		self.top.wait_window()
		raw_imgfold=self.top.getFolder()
		if(raw_imgfold is None):
			raise ValueError("Raw image folder not found")
		self.lift()
		path="Data\\"
		paths=[path+fold for fold in os.listdir(path)]

		for path in paths:
			if(not os.path.exists(path)):
				os.mkdir(path)
		self.downframe.destroy()
		self.downframe=MultiButFrame(self,self.picframe)

		self.downframe.pack(side=BOTTOM,fill=BOTH,expand=True)
