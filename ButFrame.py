from tkinter import *
from tkinter import filedialog

import os, json
import shutil
from ViewFrame import ViewFrame




class BestButFrame(Frame):
    """
        Frame with as two buttons, one for left and one for right.
    """

    def __init__(self, fenetre, ViewFrame : ViewFrame, datafolder, **kwargs):
        Frame.__init__(self,fenetre,**kwargs)
        

        self.ViewFrame=ViewFrame
        
        self.moves =[]
        self.rows=[]
        self.buttons=[]

        self.createButtons()
        
        self.data_so_far=[]
        self.datapath = datafolder
        self.outputfile = os.path.join(self.datapath,"output.json")
        if(not os.path.exists(self.outputfile)):
            with open(self.outputfile,'w') as f:
                json.dump([],f)
    
        with open(self.outputfile,'r') as f:
            self.data_so_far = json.load(f)
        

    def buttonfunc(self,side):
        curPair= self.ViewFrame.currentPair() # 'left' 'right' dictionary with video names.
        
        if(curPair is not None):
            self.moves.append(curPair)
            if len(self.moves)>200:
                self.moves.pop(0)
            self.data_so_far.append({**curPair,"side":side})

            with open(self.outputfile,'w') as f:
                json.dump(self.data_so_far,f)

            self.ViewFrame.next_data()

    def destroyButtons(self):
        for but in self.buttons:
            but.destroy()

        for row in self.rows:
            row.destroy()

        self.buttons=[]
        self.rows=[]
        self.moves=[]

    def keyReleased(self,event):
        if(event.keysym=="Left"):
            self.buttonfunc(0)
        elif(event.keysym=="Right"):
            self.buttonfunc(1)
        elif(event.keysym=="BackSpace"):
            self.undo()

    def createButtons(self):
        nbrows = 3
        
        for i in range(nbrows):
            self.rows.append(Frame(self))


        
        for i,name in enumerate(['L More Interesting','R More Interesting']):
            print("Classname is :{}".format(name))
            self.buttons.append(Button(self.rows[0], command=lambda j=i: self.buttonfunc(j),text=name,
                font=("Unispace", 12, "bold"),activebackground="sky blue",bg="blue",foreground="sky blue",width=-15))

        self.rows.append(Frame(self))
        self.buttons.append(Button(self.rows[1],command=self.undo,text="UNDO",font=("Unispace", 12, "bold"),
            activebackground="red",bg="light coral",foreground="navy",width=-15))
        self.buttons.append(Button(self.rows[2],command=self.upload_data,text="EXPORT RESULTS",font=("Unispace", 12, "bold"),
                                   activebackground="green", bg="light green", foreground="green", width=-15))
        for but in self.buttons:
            but.pack(side=LEFT,fill=X,padx=1,expand=1)
        for row in self.rows:
            row.pack(side=TOP,fill=BOTH)
    
    def upload_data(self):
        savedata_dir_name = filedialog.askdirectory(title="Select a folder to save the data")
        os.makedirs(savedata_dir_name,exist_ok=True)

        with open(os.path.join(savedata_dir_name,"SENDME.json"),'w') as f:
            json.dump(self.data_so_far,f)

    def undo(self):
        if(len(self.moves)>0):
            print(f"undoing {self.moves[-1]}")

            self.ViewFrame.previous_data(self.moves[-1])

            self.moves.pop(-1)
            self.data_so_far.pop(-1)
            with open(self.outputfile,'w') as f:
                json.dump(self.data_so_far,f)

    def reset_data(self):
        self.moves=[]
        self.data_so_far=[]
        
class MultiButFrame(Frame):
    """
        Frame with as many buttons as there are classes, as determined
        by the number of folders in Data/
    """

    def __init__(self, fenetre, ViewFrame : ViewFrame,**kwargs):
        Frame.__init__(self,fenetre,**kwargs)
        

        self.ViewFrame=ViewFrame
        
        self.moves =[]
        self.rows=[]
        self.buttons=[]

        self.createButtons()
        super().bind("1",lambda e: print("jeff"))


    def buttonfunc(self,targetfolder):
        self.moves.append((self.ViewFrame.currentPair(),targetfolder))
        if len(self.moves)>20:
            self.moves.pop(0)
        print("t-folder : ",targetfolder)
        shutil.move(self.ViewFrame.currentPair(), targetfolder)
        self.ViewFrame.next_data()

    def destroyButtons(self):
        for but in self.buttons:
            but.destroy()

        for row in self.rows:
            row.destroy()

        self.buttons=[]
        self.rows=[]
        self.moves=[]

    def keyReleased(self,event):
        if(event.char.isnumeric()):
            if(int(event.char)<len(self.buttons)):
                self.buttons[int(event.char)].invoke()
        
        # if(event.keysym=="Left"):
        #     self.buttons[0].invoke()
        # elif(event.keysym=="Right"):
        #     self.buttons[1].invoke()

    def createButtons(self):
        datapath = "Data"
        classnames = [i for i in os.listdir(datapath) if os.path.isdir(os.path.join(datapath,i))]
        nbclass = len(classnames)
        nbrows = nbclass//4+1
        
        for i in range(nbrows):
            self.rows.append(Frame(self))

        rownbr=-1

        
        for i,name in enumerate(classnames):
            if(i % 4==0):
                rownbr+=1
            print("Classname is :{}".format(name))
            self.buttons.append(Button(self.rows[rownbr], command=lambda a=(os.path.join(datapath,name)): self.buttonfunc(a),text=name,
                font=("Unispace", 12, "bold"),activebackground="sky blue",bg="blue",foreground="sky blue",width=-15))

        self.rows.append(Frame(self))
        self.buttons.append(Button(self.rows[-1],command=self.undo,text="UNDO",font=("Unispace", 12, "bold"),
            activebackground="red",bg="light coral",foreground="navy",width=-15))

        for but in self.buttons:
            but.pack(side=LEFT,fill=X,padx=1,expand=1)
        for row in self.rows:
            row.pack(side=TOP,fill=BOTH)
        
    def undo(self):
        if(len(self.moves)>0):
            shutil.move(os.path.join(self.moves[-1][1],os.path.basename(self.moves[-1][0])), self.ViewFrame.datapath)
            self.ViewFrame.previous_data()
            self.moves.pop(-1)