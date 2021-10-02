from tkinter import *

import os
import shutil


class OptionPanel(Toplevel):

	def __init__(self, fenetre, **kwargs):
		Toplevel.__init__(self, fenetre, **kwargs)

		self.attributes("-topmost","true")  # Always on top
		self.imgfolder=None
		self.quittime=False
		self.kenobi = True
		self.msgtimer = 0
		self.confirmColor= False
		self.bind('<Button-1>', self.onClick)
		self.msglist = ["Hello there !", "General Kenobi...", "You are a bold one","Kill him!",
		"Back away !", "I will deal with this", "Jedi slime myself","Your move","You fool!",
		"I've been trained", "In the jedi arts","by Count Dooku", "Attack, Kenobi!"]

		# Set geometry
		self.geometry("300x100")
		self.minsize(300, 100)  # set minimum window size value
		self.maxsize(300, 100)  # set maximum window size value

		# Define widgets
		self.buttonFrame=Frame(self)
		self.questext = StringVar()
		self.questext.set("Name of raw-image folder ?:")
		
		self.questlabel = Label(self, textvariable=self.questext)

		self.entrytext=StringVar()
		self.entrytext.set("Folder name (case sensitive)")
		self.entry = Entry(self, width=30, textvariable=self.entrytext)

		self.execMessage = StringVar(value="Hello there !")
		self.execButton = Button(self.buttonFrame, command=self.submit, text="Submit",width=-5,
			font=("Consolas",8,"bold"))
		self.execLabel= Label(self,textvariable=self.execMessage,font=("Unispace",10,"bold"))

		self.delButton = Button(self.buttonFrame, command=self.delAll, text="Delete classes",width=-5,
			background="red",activebackground="red",font=("Consolas",8,"bold"))

		# Pack everything
		self.questlabel.pack(side=TOP)
		self.entry.pack(side=TOP, padx=2, pady=2)
		self.buttonFrame.pack(side=TOP,expand=1,fill=BOTH)
		self.execButton.pack(side=LEFT,padx=2,fill=X,expand=1)
		self.delButton.pack(side=LEFT,padx=2,fill=X,expand=1)
		self.execLabel.pack(fill=X,expand=1)

		# Fast-forward in case folder here
		if("RawImages" in os.listdir()):
			self.entrytext.set("RawImages")
			self.submit()
			self.execMessage.set("Raw image folder found !")
			self.entrytext.set("Class name")

		# Launch kenobi sequence after 5 seconds
		self.after(5000,self.revertMessage)

	def onClick(self,event):
		""" If there is a click, deactivate Kenobi easter-egg"""
		self.kenobi = False
		self.unbind("<Button-1>")

	def submit(self):
		""" Submit the raw-image folder """
		foldername=self.entrytext.get()
		if(foldername not in os.listdir()):
			self.execLabel.configure(foreground="black",background="red",font=("Unispace",10,"bold"))
			self.execMessage.set("Non-existing directory")
		else:
			self.imgfolder=foldername
			self.execLabel.configure(foreground="black",background="green",font=("Unispace",10,"bold"))
			self.execMessage.set("Raw images folder set !")

			self.questext.set("Add a class folder ?")
			self.execButton.configure(text="Add Class",command=self.createClass)

	def createClass(self):
		""" Create class folder using name given"""
		classname=self.entry.get()
		if(not os.path.exists(os.path.join("Data",classname))):
			os.mkdir(os.path.join("Data",classname))
			if(self.confirmColor):
				bg="green"
			else:
				bg="lawn green"
			self.confirmColor=not self.confirmColor
			self.execLabel.configure(foreground="black",background=bg,font=("Unispace",10,"bold"))
			self.execMessage.set("Class Added !")

	def delAll(self):
		""" Delete all class folders and files"""
		top = WarningPanel(self)
		top.wait_window()
		doDelete = top.getAnswer()
		if(doDelete):
			if(len(os.listdir("Data//"))==0):
				self.execLabel.configure(foreground="black",background="orange",font=("Unispace",10,"bold"))
				self.execMessage.set("Class folder already empty")
			else:
				for classfold in os.listdir("Data//"):
					shutil.rmtree(os.path.join("Data//"+classfold))
					self.execLabel.configure(foreground="black",background="orange",font=("Unispace",10,"bold"))
					self.execMessage.set("Deleted all class folders!")

	def getFolder(self):
		return self.imgfolder

	def revertMessage(self):
		if(self.kenobi):
			self.msgtimer+=1
			self.execMessage.set(self.msglist[self.msgtimer % (len(self.msglist))])
			self.execLabel.configure(foreground="black",background="SystemButtonFace",font=("Unispace",10,"bold"))
			self.after(5000,self.revertMessage)

class WarningPanel(Toplevel):

	def __init__(self,fenetre,**kwargs):
		Toplevel.__init__(self,fenetre,**kwargs)
		self.attributes("-topmost","true")
		self.warningText = StringVar()
		self.warningText.set("Are you sure ?:")

		width=150
		height=50

		self.geometry("{}x{}".format(width,height))
		self.minsize(width, height)  # set minimum window size value
		self.maxsize(width, height)  # set maximum window size value
		self.answer = False
		self.butFrame=Frame(self)
		
		self.warninglabel = Label(self, textvariable=self.warningText,font=("Unispace",10,"bold"),fg="Red")
		self.cancelBut = Button(self.butFrame, command=self.cancel, text="Cancel",width=-5,
			font=("Consolas",8,"bold"))
		self.oksure = Button(self.butFrame, command=self.ok, text="Yes sure",width=-5,
			background="red",activebackground="red",font=("Consolas",8,"bold"))

		self.cancelBut.pack(side=LEFT,padx=5)
		self.oksure.pack(padx=5)

		self.warninglabel.pack(side=TOP)
		self.butFrame.pack(side=TOP,pady=5,padx=2)

	def cancel(self):
		self.answer=False
		self.destroy()

	def ok(self):
		self.answer = True
		self.destroy()

	def getAnswer(self):
		return self.answer
