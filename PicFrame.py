from tkinter import *
from PIL import ImageTk,Image

import os


class PicFrame(Frame):
	def __init__(self,fenetre,imgpath,**kwargs):
		Frame.__init__(self,fenetre,**kwargs)
		self.imgpath=imgpath
		self.canvas=Canvas(self,width=800,height=450)
		self.allimg=[path for path in os.listdir(imgpath)]
		self.imgnumber=0
		self.img = ImageTk.PhotoImage(Image.open(self.imgpath+self.allimg[0]))
		self.canvas.pack(anchor=CENTER)
		self.canvas.create_image(20,20,anchor=NW,image=self.img)

		

	def next_image(self):
		self.imgnumber=(self.imgnumber+1)%len(self.allimg)

		self.img = ImageTk.PhotoImage(Image.open(self.imgpath+self.allimg[self.imgnumber]))
		self.canvas.pack(anchor=CENTER)
		self.canvas.create_image(20,20,anchor=NW,image=self.img)
		
