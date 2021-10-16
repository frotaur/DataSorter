from tkinter import *
from PIL import ImageTk,Image
import os
import random


class PicFrame(Frame):
	def __init__(self,fenetre,imgpath,**kwargs):
		Frame.__init__(self,fenetre,**kwargs)
		self.CANVWIDTH = 960
		self.CANVHEIGHT = 520
		self.imgpath=imgpath
		self.canvas=Canvas(self,width=self.CANVWIDTH,height=self.CANVHEIGHT)
		self.allimg=[path for path in os.listdir(imgpath)]
		# random.shuffle(self.allimg)
		self.imgnumber=0
		self.img=None
		self.canvas.pack()
		self.showImg()


		

	def next_image(self):
		self.imgnumber=(self.imgnumber+1) % len(self.allimg)
		self.showImg()

	def previous_image(self):
		self.imgnumber = (self.imgnumber-1) % len(self.allimg)
		self.showImg()

	def reset_image(self):
		self.allimg=[path for path in os.listdir(self.imgpath)]
		# random.shuffle(self.allimg)
		self.imgnumber=0
		self.showImg()

	def showImg(self):
		pilimg = Image.open(self.imgpath+self.allimg[self.imgnumber])
		ratio = min(self.CANVWIDTH/pilimg.size[0],self.CANVHEIGHT/pilimg.size[1])
		
		pilimg=pilimg.resize((int(ratio*pilimg.size[0]),int(ratio*pilimg.size[1])))

		print("SIEZ IS : {}".format(pilimg.size))
		self.img = ImageTk.PhotoImage(pilimg)
		self.canvas.create_image(self.CANVWIDTH/2,self.CANVHEIGHT/2,anchor=CENTER,image=self.img)


	def currentImgPath(self):
		return os.path.join(self.imgpath+self.allimg[self.imgnumber])
