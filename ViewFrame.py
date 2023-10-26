from tkinter import *
from PIL import ImageTk,Image
import os, cv2, json
import shutil, random
from tqdm import tqdm
import pandas as pd


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
        self.pairpath=os.path.join("pairs","pairs.csv")

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
        self.vid_num = len(self.all_data)

        if(not os.path.exists(self.pairpath)):
            self.make_pair_data()

        self.pairs = pd.read_csv(self.pairpath)
        self.pairs.to_csv(self.pairpath,index=False)

        if(len(self.pairs)==0):
            self.set_done(True)    
        
        


        self.datanumber=0
        self.photo={'right':None,'left':None}

        self.error_msg.pack(side=BOTTOM)
        self.canvas['left'].pack(side=LEFT, padx=5)
        self.canvas['right'].pack(side=RIGHT, padx=5)

        self.update_cur_pair()
        self.showPair()
        

    
    def make_pair_data(self):
        """
            Creates and save a set of panda dataframes containing all possible pairs
            of files. Will overwrite any existing pairs.
        """
        if(os.path.exists(os.path.dirname(self.pairpath))):
            shutil.rmtree(os.path.dirname(self.pairpath))
    
        os.makedirs(os.path.dirname(self.pairpath),exist_ok=True)

        data = {'left':[],'right':[]}
        for i in tqdm(range(len(self.all_data))) :
            for j in range(i+1,len(self.all_data)): 
                data['left'].append(self.all_data[i])
                data['right'].append(self.all_data[j])

        df = pd.DataFrame(data)
        df.to_csv(self.pairpath,index=False)

    def next_data(self):
        if(self.pairs.shape[0]<=0):
            print("ESTOP")
            self.set_done(True)
        else:
            self.datanumber=(self.datanumber+1) % self.pairs.shape[0]
            self.update_cur_pair()
            self.showPair()

    def previous_data(self, pair_dict):
        """
            Reduces datanumber by 1, adds cur_pair back to dataframe, and displays previous pair
        """
        if not self.DONE:
            self.datanumber = (self.datanumber-1)
            to_append = pd.DataFrame({key:[value] for key,value in pair_dict.items()})
            self.pairs = pd.concat([self.pairs,to_append],ignore_index=True)
            self.pairs.to_csv(self.pairpath,index=False)

            self.cur_pair = pair_dict
            self.showPair()
    
    def update_cur_pair(self):
        """
            Gets current pair according to datanumber
        """
        if not self.DONE:
            rando_from = self.pairs[self.pairs['left']==self.all_data[self.datanumber%self.vid_num]] # group with left equal to specified file
            if(len(rando_from)==0):
                # No pair with such key, just give a random one
                sampled = self.pairs.sample(1)
            else:
                sampled = rando_from.sample(1)

            self.pairs.drop(sampled.index,inplace=True)
            self.cur_pair = sampled.to_dict(orient='records')[0]
            self.pairs.to_csv(self.pairpath,index=False)
        else:
            self.cur_pair=None
        

    def reset_data(self):
        self.make_pair_data()
        self.set_done(False)

        self.pairs=pd.read_csv(self.pairpath)

        self.datanumber=0
        print("Before showpair, done is :",self.DONE)

        self.update_cur_pair()
        self.showPair()
    
    def currentPair(self):
        return self.cur_pair
    
    def showPair(self):
        if(self.DONE):
            return
        else:
            vidpaths = self.cur_pair

            for pos in ['right','left']: 
                self.showVid(os.path.join(self.datapath,vidpaths[pos]),pos)

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
            raise Exception(f"Error: Couldn't open video file {vidpath}")
        
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
        

    def __del__(self):
        for vid in self.vid.values():
            if(vid.isOpened()):
                vid.release()