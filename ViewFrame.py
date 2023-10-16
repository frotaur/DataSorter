from tkinter import *
from PIL import ImageTk,Image
import os, cv2, json
import shutil

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
        self.pairpath="pairs"
        self.canvas = {'left':Canvas(self,width=self.CANVWIDTH,height=self.CANVHEIGHT),
                       'right':Canvas(self,width=self.CANVWIDTH,height=self.CANVHEIGHT)}
        self.datatype = datatype

        self.error_text = StringVar(value="")
        self.error_msg = Label(self,textvariable=self.error_text,font=("Unispace", 12, "bold"),fg="red")

        self.DONE = False
        
        self.fenetre=fenetre
        self._after_id = {'right' :None, 'left': None}
        self.vid = {'right':None,'left':None}

        self.all_data=[path for path in os.listdir(datapath)]
        if(not os.path.exists(self.pairpath)):
            self.make_pair_files()
        elif(len(os.listdir(self.pairpath))==0):
            self.set_done(True)    
        self.pairs = [path for path in os.listdir(self.pairpath)]

        # random.shuffle(self.all_data)
        self.datanumber=0
        self.photo={'right':None,'left':None}

        self.error_msg.pack(side=BOTTOM)
        self.canvas['left'].pack(side=LEFT, padx=5)
        self.canvas['right'].pack(side=RIGHT, padx=5)

    
        self.showPair()
        

    
    def make_pair_files(self,purge=False):
        """
            Creates and save a list of .json files which contain simply the path to a pair of images.

        """
        if(purge and os.path.exists(self.pairpath)):
            shutil.rmtree(self.pairpath)

        os.makedirs(self.pairpath,exist_ok=True)
        
        for i in range(len(self.all_data)) :
            for j in range(i+1,len(self.all_data)): 
                with open(os.path.join(self.pairpath,f"pair_{i}_{j}.json"),'w') as f:
                    json.dump({"left":self.all_data[i],"right":self.all_data[j]},f)
        

    def next_data(self):
        if(self.datanumber+1 >=len(self.pairs)):
            print("ESTOP")
            self.set_done(True)
        self.datanumber=(self.datanumber+1) % len(self.pairs)
        self.showPair()

    def previous_data(self):
        self.datanumber = (self.datanumber-1) % len(self.pairs)
        self.showPair()

    def reset_data(self):
        self.make_pair_files(purge=True)
        self.set_done(False)

        self.pairs=[path for path in os.listdir(self.pairpath)]
        # random.shuffle(self.all_data)
        self.datanumber=0
        print("Before showpair, done is :",self.DONE)
        self.showPair()
    
    def showPair(self):
        if(self.DONE):
            return
        else:
            vidpaths = self.getPathsFromPair(os.path.join(self.pairpath,self.pairs[self.datanumber]))
            for pos in ['right','left']:
                self.showVid(vidpaths[pos],pos)

    def set_done(self,value):
        if(value):
            self.DONE=value
            for direc in ['left','right']:
                self.canvas[direc].delete("all")
                self.stopVid(direc)
            self.error_text.set("No more data to label (pairs folder is empty). To restart -> Options -> Delete all data")

        else:
            self.DONE=value
            self.error_text.set("")
    
    def getPathsFromPair(self,pair_file):
        with open(pair_file,'r') as f:
            pair = json.load(f)
        return {'left':os.path.join(self.datapath,pair['left']),'right':os.path.join(self.datapath,pair['right'])}

    def stopVid(self,position='left'):
        if(self.vid[position] is not None and self.vid[position].isOpened()):
            self.vid[position].release()
        if(self._after_id[position]):
            self.fenetre.after_cancel(self._after_id[position])
            self._after_id[position] = None
    def showVid(self,vidpath,position='left'):
        self.stopVid(position)

        self.vid[position]= cv2.VideoCapture(vidpath)
        if not self.vid[position].isOpened():
            raise Exception(f"Error: Couldn't open video file {vidpath}.")
        
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
        else:
            # Reset the video to the beginning
            self.vid[position].set(cv2.CAP_PROP_POS_FRAMES, 0)
        self._after_id[position] = self.fenetre.after(20, lambda : self.update(size,position))  # ref  resh every 10ms

    def currentPair(self):
        if(not self.DONE):
            return os.path.join(self.pairpath,self.pairs[self.datanumber])
        else :
            return None
    
    def currentPairDict(self):
        if(not self.DONE):
            with open(self.currentPair(),'r') as f:
                return json.load(f)
        else :
            return None
        

    def __del__(self):
        for vid in self.vid.values():
            if(vid.isOpened()):
                vid.release()