from tkinter import *
from PIL import ImageTk,Image
import os, cv2


class ViewFrame(Frame):
    """
        Frame to display either a picture or a video
    """
    def __init__(self,fenetre,datapath,datatype="img",**kwargs):
        Frame.__init__(self,fenetre,**kwargs)
        self.CANVWIDTH = 960
        self.CANVHEIGHT = 520
        self.datapath=datapath
        self.canvas=Canvas(self,width=self.CANVWIDTH,height=self.CANVHEIGHT)
        self.datatype = datatype

        self.all_data=[path for path in os.listdir(datapath)]
        # random.shuffle(self.all_data)
        self.datanumber=0
        self.photo=None
        self.canvas.pack()

        self.fenetre=fenetre
        self._after_id = None
        self.vid = None
        self.showData()
        

    def showData(self):
        if(self.datatype=="img"):
            self.showImg()
        elif(self.datatype=="vid"):
            self.showVid()
    
    def next_data(self):
        self.datanumber=(self.datanumber+1) % len(self.all_data)
        self.showData()

    def previous_data(self):
        self.datanumber = (self.datanumber-1) % len(self.all_data)
        self.showData()

    def reset_data(self):
        self.all_data=[path for path in os.listdir(self.datapath)]
        # random.shuffle(self.all_data)
        self.datanumber=0
        self.showData()

    def showImg(self):
        pilimg = Image.open(os.path.join(self.datapathm,self.all_data[self.datanumber]))
        ratio = min(self.CANVWIDTH/pilimg.size[0],self.CANVHEIGHT/pilimg.size[1])
        
        pilimg=pilimg.resize((int(ratio*pilimg.size[0]),int(ratio*pilimg.size[1])))

        print("SIEZ IS : {}".format(pilimg.size))
        self.photo = ImageTk.PhotoImage(pilimg)
        self.canvas.create_image(self.CANVWIDTH/2,self.CANVHEIGHT/2,anchor=CENTER,image=self.photo)

    def showVid(self):
        if(self.vid is not None and self.vid.isOpened()):
            self.vid.release()

        self.vid= cv2.VideoCapture(os.path.join(self.datapath,self.all_data[self.datanumber]))
        if not self.vid.isOpened():
            raise Exception("Error: Couldn't open video file.")
        if(self._after_id):
            self.fenetre.after_cancel(self._after_id)
            self._after_id = None
        width = int(self.vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT))

        ratio = min(self.CANVWIDTH/width,self.CANVHEIGHT/height)

        re_size  = (int(ratio*width),int(ratio*height))
        self.update(re_size)
    
    def update(self,size):
        ret, frame = self.vid.read()
        if ret:
            frame = cv2.resize(frame, size)
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            self.canvas.create_image(self.CANVWIDTH/2,self.CANVHEIGHT/2,anchor=CENTER,image=self.photo)

        self._after_id = self.fenetre.after(10, lambda : self.update(size))  # ref  resh every 10ms

    def currentPath(self):
        return os.path.join(self.datapath,self.all_data[self.datanumber])
    
    def __del__(self):
        if(self.vid.isOpened()):
            self.vid.release()