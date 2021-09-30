from tkinter import *

import os
class MultiButFrame(Frame) :
	"""
		Frame with as many buttons as there are classes, as determined
		by the number of folders in Data/
	"""
	def __init__(self, fenetre,**kwargs):
		Frame.__init__(self,fenetre,**kwargs)
		datapath = "Data\\"

		classnames = [i for i in os.listdir(datapath) if os.path.isdir(datapath+i)]
		print(os.path.isdir(os.listdir(datapath)[0]))
		nbclass = len(classnames)
		self.buttons=[]

		nbrows = nbclass//4+1
		self.rows=[]
		for i in range(nbrows):
			self.rows.append(Frame(self))
		rownbr=-1
		for i,name in enumerate(classnames):
			if(i%4==0):
				rownbr+=1
			self.buttons.append(Button(self.rows[rownbr], command=self.buttonfunc(datapath+name),text=name,font=("Unispace", 12, "bold"),
				activebackground="sky blue",bg="blue",foreground="sky blue",width=-15))

		for but in self.buttons:
			but.pack(side=LEFT,fill=X,padx=1,expand=1)

		for row in self.rows:
			row.pack(side=TOP,fill=BOTH)

	def buttonfunc(self,targetfolder):
		print("I AM THE ALMIGHTY",targetfolder)
