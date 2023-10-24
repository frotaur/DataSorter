# DataSorter for Human feedback

## How to use :
#### If you don't want to download data : 
In a folder, put a list of videos that you want to compare. Name the folder 'RawData' otherwise you will be prompted to download
#### If you don't have data locally :
Ensure there is no RawData folder in the root directory, or at least that it is empty. Launching the program will prompt for download.

Launch the program (`python main.py`). (if necessary, run pip -r requirements.txt to install dependencies).

If you are prompted to download, click download to initiate. The program will seem to freeze, but everything is going according to plan. After downloading the videos, the program will prepare the pairs to compare. <span style="color:red">WARNING : right now I make one file per pair. So, the program will generate ALOT of files (~200k if 600 RawData). These are tiny files, but sometimes opening the folder slows down the computer. This will be solve in a further update, but for now limit the size of RawData to 1000.</span>
Follow the prompts (setting the name of the raw image folder if its not detected.). If everything goes according to plan, two videos should be displayed, with buttons underneat 'L is more interesting', 'R is more interesting'. Clicking one of these will categorize the pair and continue to the next one. 

NOTE : you can also use the left and right arrow-keys to choose which is more interesting, which is more ergonomic, I think.

You can UNDO your last choice either by pressing backspace, or clicking the undo button. You can UNDO several choices, up to 100, BUT they must have been made in the current session (in other words, if you closed the program and resumed, it won't work).

Once you categorized all the pairs, you will get a message telling you so. You cannot UNDO anymore at this point. To reset the data and restart the categorization process, click on OPTIONS, then follow the prompts to delete all data. Do note that the data about video pair categorization will be lost.


To get to the data, look in a folder that was created, name 'Data'. It will contain a single JSON file 'output.json'. It contains an array of dictionaries of the form :
` {'left' : 'vidnameleft', 'right' : 'vidnameright', 'side':0 or 1}`
The left and right keys are the name (hashes) of the videos, while the side is 0 if the left is preferred, 1 if the right is preferred.

In python,to read the data :
``` 
    import json
    data=json.load(JSON_PATH)
```
And data will be an array of dicts as described.

Let me know if any bugs arise (which they will).

## NOTE :
On mac, there may be an error saying that '_tkinter' is missing or something. Python must be compiled with tkinter, and on Mac you should run :
` brew install python-tk@3.10`

Replace with the correct version of python.