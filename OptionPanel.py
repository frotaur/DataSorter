from tkinter import *

import os
import shutil,requests,zipfile,pathlib, math
from tqdm import tqdm
from tkinter import ttk

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
    def __init__(self, fenetre, datafolder, raw_img_folder=None, **kwargs):
        Toplevel.__init__(self, fenetre, **kwargs)
        
        self.datafolder = datafolder
        self.pairfolder = os.path.join(datafolder,"pairs")

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
        self.questext.set("This should not be seen")
        
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
        if self.imgfolder is None:
            raise ValueError(f"RawData folder not found !")
    
        # Fast-forward in case folder here
        if(self.imgfolder is not None):
            # Awful code here, just too lazy to rewrite, so pathcing up
            self.entrytext.set(self.imgfolder)
            self.submit_img_folder()
            self.execMessage.set("Hello what's up ?")
            self.questext.set("Reset data ?")
            self.entry.destroy()
            self.execButton.destroy()
        
    def onEnter(self,event):
        print("SELECT PRESENT : {}".format(self.entry.select_present()))

    def submit_img_folder(self):
        """ Submit the raw-image folder """
        foldername=self.entrytext.get()
        if(foldername not in os.listdir()):
            self.execLabel.configure(foreground="black",background="green",font=("Unispace",10,"bold"))
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
            if (os.path.isdir(self.datafolder)):
                shutil.rmtree(self.datafolder)
            os.makedirs(self.datafolder,exist_ok=True)
            if (os.path.isdir(self.pairfolder)):
                shutil.rmtree(self.pairfolder)
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
    def __init__(self, fenetre, raw_img_folder, root_folder, **kwargs):
        Toplevel.__init__(self, fenetre, **kwargs)

        self.attributes("-topmost","true")  # Always on top

        self.img_folder = raw_img_folder
        self.root_folder = root_folder

        width =300
        height=100

        self.geometry(center_geometry(fenetre,width,height))


        # Define widgets
        self.buttonFrame=Frame(self)

        self.questext = StringVar()
        self.questext.set("Download data ? (~1GB)")
        self.questlabel = Label(self, textvariable=self.questext,    wraplength=280,justify=LEFT )

        self.execButton = Button(self.buttonFrame, command=self.download, text="Download",width=-5,
            font=("Consolas",8,"bold"))

        self.loading_bar = ttk.Progressbar(self, orient="horizontal", length=150, mode="determinate")
        # Pack everything
        self.questlabel.pack(side=TOP)
        self.buttonFrame.pack(side=TOP,expand=1,fill=BOTH)
        self.execButton.pack(side=LEFT,padx=2,fill=X,expand=1)


    def download(self):
        self.execButton.destroy()
        self.loading_bar.pack(side=TOP,expand=1,fill=BOTH)
        repo_zip_url = 'https://github.com/frotaur/Lenia_Data/archive/refs/heads/main.zip'
    
        zip_path = os.path.join(self.root_folder,'repo.zip')

        shutil.rmtree(self.img_folder) # Remove existing data
        if(os.path.exists(os.path.join(self.root_folder,'pairs'))):
            shutil.rmtree(os.path.join(self.root_folder,'pairs'))
        self.questext.set("Downloading data from repository... Please wait")

        self.update_idletasks()
        # # Download repo as zip
        response = requests.get(repo_zip_url, stream=True)
        response.raise_for_status()
        count = 0
        with open(zip_path, 'wb') as file:
            self.loading_bar.start()
            percent_bytes=0
            for chunk in tqdm(response.iter_content(chunk_size=10240), unit_scale=0.01, unit='MB',total=1e5):
                count+=1
                percent_bytes += 0.001
                if(int(percent_bytes)>=1):
                    self.loading_bar['value'] += int(percent_bytes)
                    percent_bytes = 0
                    self.update()
                file.write(chunk)

        # # Extract zip
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(self.root_folder)

        # # Get the path to the main folder inside the extracted files
        main_folder_path = os.path.join(self.root_folder, 'Lenia_Data-main')

        # # Specify the folder to keep
        keep_folder_path = os.path.join(main_folder_path, 'RawData')

        shutil.move(keep_folder_path,self.root_folder)
        # # Delete all other files and folders
        shutil.rmtree(main_folder_path)

        # Remove the downloaded zip file
        os.unlink(zip_path)

        self.questext.set("Downloaded data from repository !\n Making pairs... Please wait")
        self.loading_bar['value'] = 99
        self.update_idletasks()
        # Sleep 1 second before auto-destroying
        self.after(1000,self.destroy)