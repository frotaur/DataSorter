from tkinter import *


from ButFrame import BestButFrame
from ViewFrame import ViewFrame
from OptionPanel import OptionPanel, ResetOptionPanel, DownloadPanel, center_geometry
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

        # Get the screen width and height
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()

        # Calculate the x and y coordinates to center the window
        x = (screen_width - width) // 2
        y = (screen_height - height) // 2

        # Set the position of the window
        self.geometry(f'{width}x{height}+{x}+{y}')

        os.chdir(os.path.dirname(__file__))  # Change to working directory
        
        # CREATE FOLDERS IF NOT DONE YET
        path = "Data"
        os.makedirs(path,exist_ok=True)
        self.raw_imgfold = None

        # Check if "RawData" folder exists, (removed prompt for datafolder)
        os.makedirs("RawData",exist_ok=True)
        self.raw_imgfold = "RawData"

        
        if(len(os.listdir(self.raw_imgfold))==0):
            self.top=DownloadPanel(self,raw_img_folder=self.raw_imgfold)
            self.top.wait_window()
        self.lift()

        # Get data type :
        self.datatype = self.get_datatype(self.raw_imgfold)
        assert self.datatype == "vid" # Limit ourselves to videos for now
        # Create an upper frame and a button in the frame
        upframe = Frame(self)

        self.picframe= ViewFrame(upframe,self.raw_imgfold,self.datatype)

        options_button = Button(upframe,text="Options",command=self.openOption,font=("Unispace", 12, "bold"),
            activebackground="sky blue",bg="blue",foreground="sky blue",width=-15)

        # paths=[os.path.join(path,fold) for fold in os.listdir(path)]

        # for path in paths:
        #     os.makedirs(path,exist_ok=True)

        
        self.downframe = BestButFrame(self,self.picframe)


        options_button.pack(side=TOP,fill=BOTH,expand=1)
        self.picframe.pack(side=TOP, fill=X)
        upframe.pack(side=TOP,fill=X,expand=1)
        self.downframe.pack(side=BOTTOM,fill=BOTH,expand=True)

        self.bind("<KeyRelease>",self.downframe.keyReleased)

    def changeimg(self):
        self.picframe.next_data()

    # def getimgpath(self):
    #     return self.picframe.current_data()

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
        self.top=ResetOptionPanel(self,raw_img_folder=self.raw_imgfold)
        self.top.wait_window()

        self.raw_imgfold=self.top.getFolder()
        resetimg = self.top.isReset()

        if(not os.path.exists(self.raw_imgfold)):
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
            self.picframe.reset_data()
            self.downframe.reset_data()
