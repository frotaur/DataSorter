from tkinter import *


from ButFrame import BestButFrame
from ViewFrame import ViewFrame
from OptionPanel import  ResetOptionPanel, DownloadPanel, center_geometry
import tkinter.font as tkfont
import os, sys





class MainWindow(Tk):

    def __init__(self, **kwargs):
        Tk.__init__(self, **kwargs)
        # fontana = tkfont.Font(family="Unispace", size=12, weight="bold")
        # m_len = fontana.measure("M")
        # width=(m_len+2*1)*15*4
        width=960
        height=820

        # Get the screen width and height
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()

        # Calculate the x and y coordinates to center the window
        x = (screen_width - width) // 2
        y = (screen_height - height) // 2

        # Set the position of the window
        self.geometry(f'{width}x{height}+{x}+{y}')
        if getattr(sys, 'frozen', False):
            self.application_path = os.path.dirname(sys.executable)
        elif __file__:
            self.application_path = os.path.dirname(__file__)
        # os.chdir(os.path.dirname(__file__))  # Change to working directory
        
        # CREATE FOLDERS IF NOT DONE YET
        self.datapath = os.path.join(self.application_path,"Data")
        os.makedirs(self.datapath,exist_ok=True)
        self.raw_imgfold = None

        # Check if "RawData" folder exists, (removed prompt for datafolder)
        self.raw_imgfold = os.path.join(self.application_path,"RawData")
        os.makedirs(self.raw_imgfold,exist_ok=True)
        

        
        if(len(os.listdir(self.raw_imgfold))==0):
            self.top=DownloadPanel(self,raw_img_folder=self.raw_imgfold,root_folder=self.application_path)
            self.top.wait_window()
        self.lift()
        # show 'preparing data' label while preparing data
        label = Label(self, text="Preparing data...", font=("Unispace", 36, "bold"))
        label.pack(side=TOP, fill=BOTH, expand=1)
        self.update()
        # Create an upper frame and a button in the frame
        upframe = Frame(self)

        self.picframe= ViewFrame(upframe,self.raw_imgfold,self.datapath)

        options_button = Button(upframe,text="Options",command=self.openOption,font=("Unispace", 12, "bold"),
            activebackground="sky blue",bg="blue",foreground="sky blue",width=-15)

        # paths=[os.path.join(path,fold) for fold in os.listdir(path)]

        # for path in paths:
        #     os.makedirs(path,exist_ok=True)

        
        self.downframe = BestButFrame(self,self.picframe,self.datapath)

        label.destroy()
        options_button.pack(side=TOP,fill=BOTH,expand=1)
        self.picframe.pack(side=TOP, fill=BOTH)
        upframe.pack(side=TOP,fill=X,expand=1)
        self.downframe.pack(side=BOTTOM,fill=BOTH,expand=True)

        self.bind("<KeyRelease>",self.downframe.keyReleased)

    def changeimg(self):
        self.picframe.next_data()

    def mainloop(self):
        Tk.mainloop(self)

    def openOption(self):
        self.top=ResetOptionPanel(self,datafolder=self.datapath,raw_img_folder=self.raw_imgfold)
        self.top.wait_window()

        self.raw_imgfold=self.top.getFolder()
        resetimg = self.top.isReset()

        if(not os.path.exists(self.raw_imgfold)):
            raise ValueError("Raw image folder not found")
        self.lift()

        # paths=[os.path.join(path,fold) for fold in os.listdir(self.datapath)]

        # for path in paths:
        #     if(not os.path.exists(path)):
        #         os.mkdir(path)
        self.downframe.destroyButtons()
        self.downframe.createButtons()

        self.downframe.pack(side=BOTTOM,fill=BOTH,expand=True)

        if(resetimg):
            self.picframe.reset_data()
            self.downframe.reset_data()
