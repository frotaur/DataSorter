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
        height=820
        self.geometry(str(width)+"x"+str(height))
        os.chdir(os.path.dirname(__file__))  # Change to working directory

        # CREATE FOLDERS IF NOT DONE YET
        path = "Data\\"
        os.makedirs(path,exist_ok=True)
        
        if("RawImages" not in os.listdir()):
            self.top = OptionPanel(self)
            self.top.wait_window()
            raw_imgfold = self.top.getFolder()
        else:
            raw_imgfold="RawImages"

        if(raw_imgfold is None):
            raise ValueError("Raw image folder not found")
        self.lift()

        upframe = Frame(self)
        self.picframe= PicFrame(upframe,os.path.join(raw_imgfold,""))
        yo = Button(upframe,text="Options",command=self.openOption,font=("Unispace", 12, "bold"),
            activebackground="sky blue",bg="blue",foreground="sky blue",width=-15)

        paths=[path+fold for fold in os.listdir(path)]
        for path in paths:
            if(not os.path.exists(path)):
                os.mkdir(path)

        
        self.downframe = MultiButFrame(self,self.picframe)


        yo.pack(side=TOP,fill=BOTH,expand=1)
        self.picframe.pack(side=TOP, fill=X)
        upframe.pack(side=TOP,fill=X,expand=1)
        self.downframe.pack(side=BOTTOM,fill=BOTH,expand=True)

        self.bind("<KeyRelease>",self.downframe.keyReleased)

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
        resetimg = self.top.isReset()

        if(raw_imgfold is None):
            raise ValueError("Raw image folder not found")
        self.lift()
        path="Data\\"
        paths=[path+fold for fold in os.listdir(path)]

        for path in paths:
            if(not os.path.exists(path)):
                os.mkdir(path)
        self.downframe.destroyButtons()
        self.downframe.createButtons()

        self.downframe.pack(side=BOTTOM,fill=BOTH,expand=True)

        if(resetimg):
            self.picframe.reset_image()
