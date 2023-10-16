from tkinter import *

import os, json
import shutil
from ViewFrame import ViewFrame




class BestButFrame(Frame):
    """
        Frame with as two buttons, one for left and one for right.
    """

    def __init__(self, fenetre, ViewFrame : ViewFrame,**kwargs):
        Frame.__init__(self,fenetre,**kwargs)
        

        self.ViewFrame=ViewFrame
        
        self.moves =[]
        self.rows=[]
        self.buttons=[]

        self.createButtons()
        
        self.data_so_far=[]
        self.outputfile = os.path.join("Data","output.json")
        if(not os.path.exists(self.outputfile)):
            with open(self.outputfile,'w') as f:
                json.dump([],f)
    
        with open(self.outputfile,'r') as f:
            self.data_so_far = json.load(f)
        

    def buttonfunc(self,side):
        curPair, curPairDict = self.ViewFrame.currentPair(),self.ViewFrame.currentPairDict()
        
        if(curPair is not None and curPairDict is not None):
            curPairPure = {k:os.path.splitext(v)[0] for k,v in curPairDict.items()}
            self.moves.append((curPair,curPairDict))
            if len(self.moves)>200:
                self.moves.pop(0)
            self.data_so_far.append({**curPairPure,"side":side})
            # shutil.move(self.ViewFrame.currentPair(), targetfolder)
            with open(self.outputfile,'w') as f:
                json.dump(self.data_so_far,f)

            os.remove(self.ViewFrame.currentPair())
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
        datapath = "Data"
        nbclass = 2
        nbrows = 1
        
        for i in range(nbrows):
            self.rows.append(Frame(self))

        rownbr=-1

        
        for i,name in enumerate(['L More Interesting','R More Interesting']):
            print("Classname is :{}".format(name))
            self.buttons.append(Button(self.rows[rownbr], command=lambda j=i: self.buttonfunc(j),text=name,
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
            print(f"undoing {self.moves[-1]}")
            with open(self.moves[-1][0],'w') as f:
                json.dump(self.moves[-1][1],f)
            self.ViewFrame.previous_data()
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