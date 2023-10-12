from tkinter import *
from PIL import ImageTk,Image
import os, cv2


class ViewFrame(Frame):
    """
        Frame to display either a picture or a video
    """
    def __init__(self,fenetre,datapath,datatype="img",**kwargs):
        Frame.__init__(self,fenetre,**kwargs)
        self.WIDTH = 960

        self.CANVWIDTH = self.WIDTH//2
        self.CANVHEIGHT = 520
        self.datapath=datapath
        self.canvas = {'left':Canvas(self,width=self.CANVWIDTH,height=self.CANVHEIGHT),
                       'right':Canvas(self,width=self.CANVWIDTH,height=self.CANVHEIGHT)}
        self.datatype = datatype

        self.all_data=[path for path in os.listdir(datapath)]
        # random.shuffle(self.all_data)
        self.datanumber=0
        self.photo={'right':None,'left':None}
        self.canvas['left'].pack(side=LEFT, padx=5)
        self.canvas['right'].pack(side=RIGHT, padx=5)

        self.fenetre=fenetre
        self._after_id = {'right' :None, 'left': None}
        self.vid = {'right':None,'left':None}

        self.showData('left')
        self.fenetre.after(100, lambda: self.showData('right'))
        

    def showData(self,position='left'):
        if(self.datatype=="img"):
            self.showImg(position)
        elif(self.datatype=="vid"):
            self.showVid(position)
    
    def showAll(self):
        self.showData('left')
        self.showData('right')

    def next_data(self):
        self.datanumber=(self.datanumber+1) % len(self.all_data)
        self.showAll()

    def previous_data(self):
        self.datanumber = (self.datanumber-1) % len(self.all_data)
        self.showAll()

    def reset_data(self):
        self.all_data=[path for path in os.listdir(self.datapath)]
        # random.shuffle(self.all_data)
        self.datanumber=0
        self.showAll()

    def showImg(self, position='left'):
        pilimg = Image.open(os.path.join(self.datapathm,self.all_data[self.datanumber]))
        ratio = min(self.CANVWIDTH/pilimg.size[0],self.CANVHEIGHT/pilimg.size[1])
        
        pilimg=pilimg.resize((int(ratio*pilimg.size[0]),int(ratio*pilimg.size[1])))

        print("SIEZ IS : {}".format(pilimg.size))
        self.photo[position] = ImageTk.PhotoImage(pilimg)
        self.canvas[position].create_image(self.CANVWIDTH/2,self.CANVHEIGHT/2,anchor=CENTER,image=self.photo[position])

    
    def showVid(self,position='left'):
        if(self.vid[position] is not None and self.vid[position].isOpened()):
            self.vid[position].release()

        self.vid[position]= cv2.VideoCapture(os.path.join(self.datapath,self.all_data[self.datanumber]))
        if not self.vid[position].isOpened():
            raise Exception("Error: Couldn't open video file.")
        if(self._after_id[position]):
            self.fenetre.after_cancel(self._after_id[position])
            self._after_id[position] = None
        width = int(self.vid[position].get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.vid[position].get(cv2.CAP_PROP_FRAME_HEIGHT))

        ratio = min(self.CANVWIDTH/width,self.CANVHEIGHT/height)

        re_size  = (int(ratio*width),int(ratio*height))
        self.update(re_size,position)
    
    def update(self,size,position):
        ret, frame = self.vid[position].read()
        if ret:
            frame = cv2.resize(frame, size)
            self.photo[position] = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))

            self.canvas[position].create_image(self.CANVWIDTH/2,self.CANVHEIGHT/2,anchor=CENTER,image=self.photo[position])

        self._after_id[position] = self.fenetre.after(20, lambda : self.update(size,position))  # ref  resh every 10ms

    def currentPath(self):
        return os.path.join(self.datapath,self.all_data[self.datanumber])
    
    def __del__(self):
        for vid in self.vid.values():
            if(vid.isOpened()):
                vid.release()