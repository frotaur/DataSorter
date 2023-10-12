from tkinter import *

import os
import shutil


class MultiButFrame(Frame):
    """
        Frame with as many buttons as there are classes, as determined
        by the number of folders in Data/
    """

    def __init__(self, fenetre, picframe,**kwargs):
        Frame.__init__(self,fenetre,**kwargs)
        

        self.picframe=picframe
        
        self.moves =[]
        self.rows=[]
        self.buttons=[]

        self.createButtons()
        super().bind("1",lambda e: print("caca"))


    def buttonfunc(self,targetfolder):
        self.moves.append((self.picframe.currentImgPath(),targetfolder))
        if len(self.moves)>20:
            self.moves.pop(0)
        print("t-folder : ",targetfolder)
        shutil.move(self.picframe.currentImgPath(), targetfolder)
        self.picframe.next_image()

    def destroyButtons(self):
        for but in self.buttons:
            but.destroy()

        for row in self.rows:
            row.destroy()

        self.buttons=[]
        self.rows=[]
        self.moves=[]

    def keyReleased(self,event):
        print("NYAK")
        if(event.char.isnumeric()):
            if(int(event.char)<len(self.buttons)):
                self.buttons[int(event.char)].invoke()

    def createButtons(self):
        datapath = "Data\\"
        classnames = [i for i in os.listdir(datapath) if os.path.isdir(datapath+i)]
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
            shutil.move(os.path.join(self.moves[-1][1],os.path.basename(self.moves[-1][0])), self.picframe.imgpath)
            self.picframe.previous_image()
            self.moves.pop(-1)