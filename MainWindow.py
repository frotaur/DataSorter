from tkinter import *


from MultiButFrame import MultiButFrame
from ViewFrame import ViewFrame
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
        path = "Data"
        os.makedirs(path,exist_ok=True)
        
        # Check if "RawData" folder exists, otherwise prompt user with OptionPanel
        if("RawData" not in os.listdir()):
            self.top = OptionPanel(self)
            self.top.wait_window()
            raw_imgfold = self.top.getFolder()
        else:
            raw_imgfold="RawData"

        if(raw_imgfold is None):
            raise ValueError("Raw data folder not found")
        self.lift()

        # Get data type :
        self.datatype = self.get_datatype(raw_imgfold)

        # Create an upper frame and a button in the frame
        upframe = Frame(self)

        self.picframe= ViewFrame(upframe,raw_imgfold,self.datatype)

        options_button = Button(upframe,text="Options",command=self.openOption,font=("Unispace", 12, "bold"),
            activebackground="sky blue",bg="blue",foreground="sky blue",width=-15)

        # paths=[os.path.join(path,fold) for fold in os.listdir(path)]

        # for path in paths:
        #     os.makedirs(path,exist_ok=True)

        
        self.downframe = MultiButFrame(self,self.picframe)


        options_button.pack(side=TOP,fill=BOTH,expand=1)
        self.picframe.pack(side=TOP, fill=X)
        upframe.pack(side=TOP,fill=X,expand=1)
        self.downframe.pack(side=BOTTOM,fill=BOTH,expand=True)

        self.bind("<KeyRelease>",self.downframe.keyReleased)

    def changeimg(self):
        self.picframe.next_data()

    def getimgpath(self):
        return self.picframe.current_data()

    def mainloop(self):
        Tk.mainloop(self)

    def get_datatype(self,raw_imgfold):
        """ Get data type from raw image folder"""
        file = os.listdir(raw_imgfold)[0].lower()

        if(file.endswith(".jpg") or file.endswith(".jpeg") or file.endswith(".png")):
            print('detected datatype : imgs')
            return "img"
        elif(file.endswith(".mp4") or file.endswith(".avi") or file.endswith(".mov")):
            print('detected datatype : vids')
            return "vid"
        else:
            raise ValueError(f"Un-supported data type : {file.split('.')[-1]}")

    def openOption(self):
        self.top=OptionPanel(self)
        self.top.wait_window()

        raw_imgfold=self.top.getFolder()
        resetimg = self.top.isReset()

        if(raw_imgfold is None):
            raise ValueError("Raw image folder not found")
        self.lift()
        path="Data"
        paths=[os.path.join(path,fold) for fold in os.listdir(path)]

        for path in paths:
            if(not os.path.exists(path)):
                os.mkdir(path)
        self.downframe.destroyButtons()
        self.downframe.createButtons()

        self.downframe.pack(side=BOTTOM,fill=BOTH,expand=True)

        if(resetimg):
            self.picframe.reset_image()
