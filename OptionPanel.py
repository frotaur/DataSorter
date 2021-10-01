from tkinter import *

import os
import time

class OptionPanel(Toplevel):

	def __init__(self, fenetre, **kwargs):
		Toplevel.__init__(self, fenetre, **kwargs)

		self.imgfolder=None
		self.quittime=False

		self.questext = StringVar()
		self.questext.set("Name of raw-image folder ?:")
		self.attributes("-topmost","true")
		self.questlabel = Label(self, textvariable=self.questext)

		self.entry = Entry(self, width=30)
		self.entry.insert(0, "Folder name (case sensitive)")

		self.execMessage = StringVar(value="Hello there !")
		self.execButton = Button(self, command=self.submit, text="Submit")

		self.execLabel= Label(self,textvariable=self.execMessage,font=("Unispace",10,"bold"))
		self.questlabel.pack(side=TOP)
		self.entry.pack(side=TOP, padx=2, pady=2)
		self.execButton.pack(side=TOP)
		self.execLabel.pack(fill=X,expand=1)

		self.after(1000,self.delaycloser)

	def submit(self):
		foldername=self.entry.get()
		if(foldername not in os.listdir()):
			self.execLabel.configure(foreground="black",background="red",font=("Unispace",10,"bold"))
			self.execMessage.set("Non-existing directory")
		else:
			self.imgfolder=foldername
			self.execLabel.configure(foreground="black",background="green",font=("Unispace",10,"bold"))
			self.execMessage.set("Folder set !")
			self.quittime=True

	def getFolder(self):
		return self.imgfolder

	def delaycloser(self):
		if(self.quittime is True):
			self.after(1000,self.destroy)
		else:
			self.after(200,self.delaycloser)
