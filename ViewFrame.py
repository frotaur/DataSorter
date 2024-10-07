from tkinter import *
from PIL import ImageTk,Image
import os, cv2, json
import shutil, random
from tqdm import tqdm
import pandas as pd
from tkinter import ttk
import torch, torchvision
from threading import Thread, Event
from reward_training import RewardTrainer
import time

class ViewFrame(Frame):
    """
        Frame to display either a picture or a video
    """
    def __init__(self,fenetre,rawdatapath,datapath,reward_trainer : RewardTrainer,**kwargs):
        """
            Args:
            fenetre : Tk parent window
            rawdatapath : str, path to the raw data folder
            datapath : str, path to the data folder
            reward_trainer : RewardTrainer, trainer used to rank the videos
        """
        Frame.__init__(self,fenetre,**kwargs)
        self.WIDTH = 960

        self.CANVWIDTH = self.WIDTH//2
        self.CANVHEIGHT = 520-20

        self.datapath = datapath
        self.rawdatapath = rawdatapath

        self.vidpath = os.path.join(rawdatapath,"Videos")
        self.pairpath=os.path.join(datapath,"pairs","pairs.csv")
        self.reward_trainer = reward_trainer


        self.canFrame = {'left' : Frame(self),'right' : Frame(self)}

        self.canvas = {'left':Canvas(self.canFrame['left'],width=self.CANVWIDTH,height=self.CANVHEIGHT),
                       'right':Canvas(self.canFrame['right'],width=self.CANVWIDTH,height=self.CANVHEIGHT)}
        
        self.vidCaption = {'left':Label(self.canFrame['left'],text="Left",font=("Unispace", 12, "bold")),
                           'right':Label(self.canFrame['right'],text="Right",font=("Unispace", 12, "bold"))}


        self.error_text = StringVar(value="")
        self.error_msg = Label(self,textvariable=self.error_text,font=("Unispace", 12, "bold"),fg="red")

        self.DONE = False
        
        self.fenetre=fenetre
        self._after_id = {'right' :None, 'left': None}
        self.vid = {'right':None,'left':None}
        self.tens_vid = {'right':None,'left':None}

        self.loaded = {'right':Event(),'left':Event()}

        self.all_data=[path for path in os.listdir(self.vidpath) if os.path.isfile(os.path.join(self.vidpath,path))]

        self.vid_num = len(self.all_data)
        self.datamixer = [i for i in range(self.vid_num)] # mixes left column of sequential data when restarting
        random.shuffle(self.datamixer)

        if(not os.path.exists(self.pairpath)):
            self.make_pair_data()

        self.pairs = pd.read_csv(self.pairpath)
        self.pairs.to_csv(self.pairpath,index=False)

        if(len(self.pairs)==0):
            self.set_done(True)    

        self.datanumber=0
        self.ranked_data_since_last = reward_trainer.num_datapoints # Number of datapoint already there
        self.photo={'right':None,'left':None}


# +++++++++++++++++++++++++++++++ LAYOUT ++++++++++++++++++++++++++++++++++++++++
        style = ttk.Style(self)
        style.theme_use('clam')
        # Define the properties of the style
        style.configure("youtube.Horizontal.TProgressbar",
                        thickness=10,  # Height of the progress bar
                        troughcolor="lightgray",  # Background color
                        background="red",  # Foreground color (color of the progress)
                        borderwidth=0,  # Remove the border
                        relief="flat"  # Ensure the progress bar is flat
                        )
        
        # Adjust the layout
        style.layout("youtube.Horizontal.TProgressbar", 
             [('Horizontal.Progressbar.trough',
               {'children': [('Horizontal.Progressbar.pbar',
                              {'side': 'left', 'sticky': 'nswe', 'expand': True})],
                'sticky': 'nswe'})])
        # pack the error message
        self.error_msg.pack(side=BOTTOM)
        # red progress bar for video progress
        self.video_progress = ttk.Progressbar(self, orient = HORIZONTAL, length = 2*self.CANVWIDTH-5, mode = 'determinate',style="youtube.Horizontal.TProgressbar")
        # self.video_progress = Button(self,text="Restart",command=self.restart_video,font=("Unispace", 12, "bold"))
        self.video_progress['value']=0
        # self.error_msg.pack(side=BOTTOM)
        self.video_progress.pack(side=BOTTOM)

        self.canFrame['left'].pack(side=LEFT, padx=5)
        self.canFrame['right'].pack(side=RIGHT, padx=5)

        self.canvas['left'].pack(side=BOTTOM,fill=BOTH,expand=1,padx=0,pady=0)
        self.canvas['right'].pack(side=BOTTOM,fill=BOTH,expand=1)
        self.vidCaption['left'].pack(side=TOP,fill=X,pady=5)
        self.vidCaption['right'].pack(side=TOP,fill=X,pady=5)
        

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
        self.train_check()
        if(self.pairs.shape[0]<=0):
            print("ESTOP")
            self.set_done(True)
        else:
            self.datanumber=(self.datanumber+1) % self.pairs.shape[0]
            self.ranked_data_since_last+=1
            self.update_cur_pair()
            self.showPair()

    def previous_data(self, pair_dict):
        """
            Reduces datanumber by 1, adds cur_pair back to dataframe, and displays previous pair
        """
        if not self.DONE:
            self.datanumber = (self.datanumber-1)
            self.ranked_data_since_last-=1
            to_append = pd.DataFrame({key:[value] for key,value in pair_dict.items()})
            self.pairs = pd.concat([self.pairs,to_append],ignore_index=True)
            self.pairs.to_csv(self.pairpath,index=False)

            self.cur_pair = pair_dict
            self.showPair()
    
    def restart_video(self):
        """
            Restarts the video
        """
        if not self.DONE:
            for pos in ['right','left']:
                if(self.vid[pos] is not None):
                    self.vid[pos].set(cv2.CAP_PROP_POS_FRAMES, 0)
                
                self.video_progress['value'] = 0

    def update_cur_pair(self):
        """
            Gets current pair according to datanumber
        """
        if not self.DONE:
            # print(f'Will select {self.datamixer[self.datanumber%self.vid_num]} for left side')
            rando_from = self.pairs[self.pairs['left']==self.all_data[self.datamixer[self.datanumber%self.vid_num]]] # group with left equal to specified file
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

    def train_check(self):
        """
            Checks if enough data has been ranked to train a new predictor.
            If yes, starts training in the background.
        """
        if(self.ranked_data_since_last>=20):
            self.ranked_data_since_last=0
            self.start_training()

    def reset_data(self):
        self.make_pair_data()
        self.set_done(False)

        self.pairs=pd.read_csv(self.pairpath)

        self.datanumber=0
        self.ranked_data_since_last=0
        print("Before showpair, done is :",self.DONE)

        self.update_cur_pair()
        # self.make_training_data()
        self.showPair()
    
    def currentPair(self):
        return self.cur_pair
    
    def showPair(self):
        if(self.DONE):
            return
        else:
            vidpaths = self.cur_pair

            score = {}
            score['right'] = f'Loading score...'
            score['left'] = f'Loading score...'

            for pos in ['right','left']: 
                self.vidCaption[pos]['text'] = score[pos]
                self.showVid(os.path.join(self.vidpath,vidpaths[pos]),pos)
                Thread(target=self._show_score).start()

    def load_completed(self):
        return self.loaded['left'].is_set() and self.loaded['right'].is_set()

    def wait_load(self):
        self.loaded['left'].wait()
        self.loaded['right'].wait()

    def _show_score(self):
        """
            Shows the score of the current pair
        """
        if not self.load_completed():
            self.wait_load()
        
        output = self.reward_trainer.estimate_pair(self.tens_vid['left'],self.tens_vid['right']) # (2,)

        score = {}
        score['left'] = f'{output[0].item()*100:.2f}%'
        score['right'] = f'Right : {output[1].item()*100:.2f}%'
        for pos in ['right','left']: 
            self.vidCaption[pos]['text'] = score[pos]

    def set_done(self,value):
        if(value):
            self.DONE=value
            for direc in ['left','right']:
                self.canvas[direc].delete("all")
                self.stopVid(direc)
            self.error_text.set("No more data to label (pairs folder is empty). To restart -> Options -> Delete all data")
            self.error_msg.pack(side=BOTTOM)
        else:
            self.DONE=value
            self.error_text.set("")
            # self.error_msg.pack_forget()
    
    def stopVid(self,position='left'):
        if(self.vid[position] is not None and self.vid[position].isOpened()):
            self.vid[position].release()
        if(self._after_id[position]):
            self.fenetre.after_cancel(self._after_id[position])
            self._after_id[position] = None

    def load_tens_video(self, video_path, position):
        self.tens_vid[position] = (torchvision.io.read_video(video_path,output_format='TCHW',pts_unit='sec')[0]).float()/255.
        # print(f'Loaded {position} video tensor, of shape : ', self.tens_vid[position].shape)
        self.loaded[position].set()


    def showVid(self,vidpath,position='left'):
        self.stopVid(position)
        self.loaded[position].clear()
        tens_thread = Thread(target=self.load_tens_video,args=(vidpath,position))
        tens_thread.start()

        self.vid[position]= cv2.VideoCapture(vidpath)
        if not self.vid[position].isOpened():
            raise Exception(f"Error: Couldn't open video file {vidpath}")
        
        width = int(self.vid[position].get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.vid[position].get(cv2.CAP_PROP_FRAME_HEIGHT))

        ratio = min(self.CANVWIDTH/width,self.CANVHEIGHT/height)
        re_size  = (int(ratio*width),int(ratio*height))

        self.update_vid(re_size,position)
    
    def update_vid(self,size,position):
        ret, frame = self.vid[position].read()
        if ret:
            frame = cv2.resize(frame, size)
            self.photo[position] = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            self.canvas[position].create_image(self.CANVWIDTH/2,self.CANVHEIGHT/2,anchor=CENTER,image=self.photo[position])
            if(position=='right'): # Hacky but to update just once every double update
                self.video_progress['value'] = self.vid['right'].get(cv2.CAP_PROP_POS_FRAMES)/self.vid['right'].get(cv2.CAP_PROP_FRAME_COUNT)*100
                # self.update()
        else:
            # Reset the video to the beginning (both)
            for pos in ['right','left']:
                self.vid[pos].set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.video_progress['value'] = 0
        self._after_id[position] = self.fenetre.after(10, lambda : self.update_vid(size,position))  # ref  resh every 20ms
        

    def __del__(self):
        for vid in self.vid.values():
            if(vid.isOpened()):
                vid.release()
        os._exit(0)
    
    def create_datapoint(self,annotation):
        """
            Creates datapoint given annotation and current images. To be called in a THREAD !
            Args : 
            annotation : float, 0, 0.5 or 1
            return_path : list, list to append the path to the datapoint
        """
        if not self.load_completed():
            self.wait_load()

        return self.reward_trainer.create_datapoint(self.tens_vid['left'],self.tens_vid['right'],annotation)


    def train_predictor(self):
        """
            Trains predictor, and saves it in datapath/predictors
        """
        print('Training predictor !')
        self.reward_trainer.train_model()
        print('Training finished !')

    def start_training(self):
        training_thread = Thread(target=self.train_predictor)
        training_thread.start()