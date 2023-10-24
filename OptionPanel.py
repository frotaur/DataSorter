from tkinter import *

import os
import shutil,requests,zipfile,pathlib


def center_geometry(fenetre,w,h):
    fenetre.update_idletasks()
    # Get the width and height of the parent window
    fenetre_width = fenetre.winfo_width()
    fenetre_height = fenetre.winfo_height()

    # Get the position of the fenetre window on the screen
    fenetre_x = fenetre.winfo_rootx()
    fenetre_y = fenetre.winfo_rooty()

    # Calculate the x and y coordinates to center the popup window
    x = (fenetre_width - w) // 2 + fenetre_x
    y = (fenetre_height - h) // 2 + fenetre_y

    return f'{w}x{h}+{x}+{y}'

class OptionPanel(Toplevel):
    """
        Panel that opens when no raw-image folder is found
        Asks user to enter the name of the folder
    """
    def __init__(self, fenetre, **kwargs):
        Toplevel.__init__(self, fenetre, **kwargs)

        self.attributes("-topmost","true")  # Always on top
        self.imgfolder=None
        self.quittime=False
        self.kenobi = True
        self.msgtimer = 0
        self.confirmColor= False
        self.reset = False
        self.bind('<Button-1>', self.onClick)
        self.msglist = ["Hello there !", "General Kenobi...", "You are a bold one","Kill him!",
        "Back away !", "I will deal with this", "Jedi slime myself","Your move","You fool!",
        "I've been trained", "In the jedi arts","by Count Dooku", "Attack, Kenobi!"]

        # Set geometry
        self.geometry("300x100")
        self.minsize(300, 100)  # set minimum window size value
        self.maxsize(300, 100)  # set maximum window size value

        # Define widgets
        self.buttonFrame=Frame(self)
        self.questext = StringVar()
        self.questext.set("Name of raw-data folder ?:")
        
        self.questlabel = Label(self, textvariable=self.questext)

        self.entrytext=StringVar()
        self.entrytext.set("Folder name (case sensitive)")
        self.entry = Entry(self, width=30, textvariable=self.entrytext)

        self.execMessage = StringVar(value="Hello there !")
        self.execButton = Button(self.buttonFrame, command=self.submit, text="Submit",width=-5,
            font=("Consolas",8,"bold"))
        self.execLabel= Label(self,textvariable=self.execMessage,font=("Unispace",10,"bold"))

        self.delButton = Button(self.buttonFrame, command=self.delAll, text="Delete classes",width=-5,
            background="red",activebackground="red",font=("Consolas",8,"bold"))

        # Bind enter for entry
        self.entry.bind("<Return>",lambda e: self.submit())

        # Pack everything
        self.questlabel.pack(side=TOP)
        self.entry.pack(side=TOP, padx=2, pady=2)
        self.buttonFrame.pack(side=TOP,expand=1,fill=BOTH)
        self.execButton.pack(side=LEFT,padx=2,fill=X,expand=1)
        self.delButton.pack(side=LEFT,padx=2,fill=X,expand=1)
        self.execLabel.pack(fill=X,expand=1)

        # Fast-forward in case folder here
        if("RawData" in os.listdir()):
            self.entrytext.set("RawData")
            self.submit()
            self.execMessage.set("Raw data folder found !")
            self.entrytext.set("Class name")

        # Launch kenobi sequence after 5 seconds
        self.after(5000,self.revertMessage)

    def onClick(self,event):
        """ If there is a click, deactivate Kenobi easter-egg"""
        self.kenobi = False
        self.unbind("<Button-1>")


    def onEnter(self,event):
        print("SELECT PRESENT : {}".format(self.entry.select_present()))

    def submit(self):
        """ Submit the raw-image folder """
        foldername=self.entrytext.get()
        if(foldername not in os.listdir()):
            self.execLabel.configure(foreground="black",background="red",font=("Unispace",10,"bold"))
            self.execMessage.set("Non-existing directory")
        else:
            self.imgfolder=foldername
            self.execLabel.configure(foreground="black",background="green",font=("Unispace",10,"bold"))
            self.execMessage.set("Raw images folder set !")

            self.questext.set("Add a class folder ?")
            self.execButton.configure(text="Add Class",command=self.createClass)
            self.entry.bind("<Return>",lambda e: self.createClass())
            self.entrytext.set("")


    def createClass(self):
        """ Create class folder using name given"""
        classname=self.entry.get()
        if(not os.path.exists(os.path.join("Data",classname))):
            os.mkdir(os.path.join("Data",classname))
            if(self.confirmColor):
                bg="green"
            else:
                bg="lawn green"
            self.confirmColor=not self.confirmColor
            self.execLabel.configure(foreground="black",background=bg,font=("Unispace",10,"bold"))
            self.execMessage.set("Class {} Added !".format(classname))
            self.entrytext.set("")
        else:
            if(self.confirmColor):
                bg="green"
            else:
                bg="lawn green"
            self.confirmColor=not self.confirmColor
            self.execLabel.configure(foreground="black",background=bg,font=("Unispace",10,"bold"))
            self.execMessage.set("Class {} Already Exists !".format(classname))
            self.entrytext.set("")


    def delAll(self):
        """ Delete all class folders and files"""
        top = WarningPanel(self)
        top.wait_window()
        doDelete = top.getAnswer()
        if(doDelete):
            if(len(os.listdir("Data//"))==0):
                self.execLabel.configure(foreground="black",background="orange",font=("Unispace",10,"bold"))
                self.execMessage.set("Class folder already empty")
            else:
                for classfold in os.listdir("Data//"):
                    # First put all data files in rawImages
                    for file in os.listdir(os.path.join("Data//",classfold)):
                        shutil.move(os.path.join("Data",classfold,file),self.getFolder())
                    shutil.rmtree(os.path.join("Data//"+classfold))
                    self.reset=True

                self.execLabel.configure(foreground="black",background="orange",font=("Unispace",10,"bold"))
                self.execMessage.set("Deleted all class folders!")

    def getFolder(self):
        return self.imgfolder

    def isReset(self):
        return self.reset
        
    def revertMessage(self):
        if(self.kenobi):
            self.msgtimer+=1
            self.execMessage.set(self.msglist[self.msgtimer % (len(self.msglist))])
            self.execLabel.configure(foreground="black",background="SystemButtonFace",font=("Unispace",10,"bold"))
            self.after(5000,self.revertMessage)

class WarningPanel(Toplevel):
    def __init__(self,fenetre,message="Are you sure?",**kwargs):
        Toplevel.__init__(self,fenetre,**kwargs)
        self.attributes("-topmost","true")
        self.warningText = StringVar()
        self.warningText.set(message)

        width=220
        height=50

        # Set the position of the popup window
        self.geometry(center_geometry(fenetre,width,height))

        self.minsize(width, height)  # set minimum window size value
        self.maxsize(width, height)  # set maximum window size value
        self.answer = False
        self.butFrame=Frame(self)
        
        self.warninglabel = Label(self, textvariable=self.warningText,font=("Unispace",10,"bold"),fg="Red")
        self.cancelBut = Button(self.butFrame, command=self.cancel, text="Cancel",width=-5,
            font=("Consolas",8,"bold"))
        
        self.oksure = Button(self.butFrame, command=self.ok, text="Yes sure",width=-5,
            background="red",activebackground="red",font=("Consolas",8,"bold"))

        self.cancelBut.pack(side=LEFT,padx=5)
        self.oksure.pack(padx=5)

        self.warninglabel.pack(side=TOP)
        self.butFrame.pack(side=TOP,pady=5,padx=2)

    def cancel(self):
        self.answer=False
        self.destroy()

    def ok(self):
        self.answer = True
        self.destroy()

    def getAnswer(self):
        return self.answer


class ResetOptionPanel(Toplevel):
    """
        Panel used to reset the pairs folder, as well as the output file.
    """
    def __init__(self, fenetre, raw_img_folder=None, **kwargs):
        Toplevel.__init__(self, fenetre, **kwargs)

        self.destroy_early = raw_img_folder is None  

        self.attributes("-topmost","true")  # Always on top
        self.quittime=False

        self.confirmColor= False
        self.reset = False

        width =300
        height=100


        self.geometry(center_geometry(fenetre,width,height))


        # Define widgets
        self.buttonFrame=Frame(self)
        self.questext = StringVar()
        self.questext.set("Name of raw-data folder ?:")
        
        self.questlabel = Label(self, textvariable=self.questext)

        self.entrytext=StringVar()
        self.entrytext.set("Folder name (case sensitive)")
        self.entry = Entry(self, width=30, textvariable=self.entrytext)

        self.execMessage = StringVar(value="Hello there !")
        self.execButton = Button(self.buttonFrame, command=self.submit_img_folder, text="Submit",width=-5,
            font=("Consolas",8,"bold"))
        self.execLabel= Label(self,textvariable=self.execMessage,font=("Unispace",10,"bold"))

        self.delButton = Button(self.buttonFrame, command=self.delAll, text="Delete ALL",width=-5,
            background="red",activebackground="red",font=("Consolas",8,"bold"))

        # Bind enter for entry
        self.entry.bind("<Return>",lambda e: self.submit_img_folder())

        # Pack everything
        self.questlabel.pack(side=TOP)
        self.entry.pack(side=TOP, padx=2, pady=2)
        self.buttonFrame.pack(side=TOP,expand=1,fill=BOTH)
        self.execButton.pack(side=LEFT,padx=2,fill=X,expand=1)
        self.delButton.pack(side=LEFT,padx=2,fill=X,expand=1)
        self.execLabel.pack(fill=X,expand=1)

        self.imgfolder=raw_img_folder
        if self.imgfolder is None and os.path.exists("RawData"):
            self.imgfolder="RawData"
    
        # Fast-forward in case folder here
        if(self.imgfolder is not None):
            # Awful code here, just too lazy to rewrite, so pathcing up
            self.entrytext.set(self.imgfolder)
            self.submit_img_folder()
            self.execMessage.set("Raw data folder found !")
            self.entry.destroy()
            self.execButton.destroy()
        
    def onEnter(self,event):
        print("SELECT PRESENT : {}".format(self.entry.select_present()))

    def submit_img_folder(self):
        """ Submit the raw-image folder """
        foldername=self.entrytext.get()
        if(foldername not in os.listdir()):
            self.execLabel.configure(foreground="black",background="red",font=("Unispace",10,"bold"))
            self.execMessage.set("Non-existing directory")
        else:
            self.imgfolder=foldername
            self.execLabel.configure(foreground="black",background="green",font=("Unispace",10,"bold"))
            self.execMessage.set("Raw images folder set !")

            self.questext.set("Reset data ?")
            
            self.entry.destroy()
            self.execButton.destroy()
        
        if(self.destroy_early):
            self.destroy()

    def delAll(self):
        """ Delete all class folders and files"""
        top = WarningPanel(self)
        top.wait_window()
        doDelete = top.getAnswer()
        if(doDelete):
            shutil.rmtree("Data")
            os.makedirs("Data",exist_ok=True)
            shutil.rmtree("pairs")
            self.execLabel.configure(foreground="black",background="orange",font=("Unispace",10,"bold"))
            self.execMessage.set("Deleted all data!")
            self.reset = True

            self.destroy()
    
    
    def getFolder(self):
        return self.imgfolder

    def isReset(self):
        return self.reset
    
class DownloadPanel(Toplevel):
    """
        Panel used to download data from internet
    """
    def __init__(self, fenetre, raw_img_folder, **kwargs):
        Toplevel.__init__(self, fenetre, **kwargs)

        self.attributes("-topmost","true")  # Always on top

        self.img_folder = raw_img_folder

        width =300
        height=100

        self.geometry(center_geometry(fenetre,width,height))


        # Define widgets
        self.buttonFrame=Frame(self)

        self.questext = StringVar()
        self.questext.set("Download data from repository ? (will empty raw data folder and pairs folder):")
        self.questlabel = Label(self, textvariable=self.questext,    wraplength=280,justify=LEFT )

        self.execButton = Button(self.buttonFrame, command=self.download, text="Download",width=-5,
            font=("Consolas",8,"bold"))

        # Pack everything
        self.questlabel.pack(side=TOP)
        self.buttonFrame.pack(side=TOP,expand=1,fill=BOTH)
        self.execButton.pack(side=LEFT,padx=2,fill=X,expand=1)

    def download(self):
        repo_zip_url = 'https://github.com/frotaur/Lenia_Data/archive/refs/heads/main.zip'
        extract_to = pathlib.Path(__file__).parent.as_posix()
        zip_path = os.path.join(extract_to,'repo.zip')

        shutil.rmtree(self.img_folder) # Remove existing data
        if(os.path.exists(os.path.join(extract_to,'pairs'))):
            shutil.rmtree(os.path.join(extract_to,'pairs'))
        # # Download repo as zip
        response = requests.get(repo_zip_url, stream=True)
        response.raise_for_status()
        with open(zip_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)

        # # Extract zip
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)

        # # Get the path to the main folder inside the extracted files
        main_folder_path = os.path.join(extract_to, 'Lenia_Data-main')

        # # Specify the folder to keep
        keep_folder_path = os.path.join(main_folder_path, 'RawData')

        shutil.move(keep_folder_path,extract_to)
        # # Delete all other files and folders
        shutil.rmtree(main_folder_path)

        # Remove the downloaded zip file
        os.unlink(zip_path)

        self.questext.set("Downloaded data from repository !\n Making pairs... Please wait")
        # Sleep 1 second before auto-destroying
        self.after(1000,self.destroy)