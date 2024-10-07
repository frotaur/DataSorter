from torchenhanced import ConfigModule, Trainer
import torch, torch.nn as nn, torchvision
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
import json, os
from tqdm import tqdm


class RewardTrainer(Trainer):
    """
        Base class, wrapper for a reward model trainer.
        Contains utility methods to train a reward model, 
        given annotation on a dataset.

        Idea :
        create_datapoint :
        method which will be called
        each time an annotation is made. The argument will be basically
        the two datas that are shown (video, image, whatever), and the
        annotation for which is preferred. This function should take that information, 
        and create a datapoint in the appropriate folder. It will then use this folder
        to create a dataset, which will be used to train a model.
    
        train_model :
        this simply makes a training run of the reward model. We'll see if I can use
        the native 'train_steps' from torchenhanced, if not, I'll just re-implement it
        completely. I'll still inherit from trainer I think, to have access to the saving
        and loading of the model, which might be useful.

        estimate_pair :
        given a pair of datas as in 'create_datapoint', this should return the estimation
        from the model. Depending on the model, this might differ. It will be used by the 
        main program to sort the images to show, to show only ones which are unsure (probably, to be seen).


    """

    def __init__(self, model : nn.Module, data_loc, optimizer, scheduler, no_logging=True, device='cpu'):
        """
            Args:
            model : model to be trained. Probably to be hardcoded in inheriting classes
            
        """
        
        super().__init__(model=model, optim=optimizer, scheduler=scheduler, save_loc='reward_model_train',project_name='reward_train',device=device,
                         no_logging=no_logging)
        
        self.data_fold = data_loc

        os.makedirs(data_loc, exist_ok=True)
    def create_datapoint(self, data1, data2, annotation) -> str:
        """
            Create a datapoint from the two datas given, provided the annotation.
            Returns string for the location of the datapoint.
            TO BE REDEFINED IN INHERITING CLASSES.
        """
        raise NotImplementedError('Method create_datapoint must be implemented in inheriting class.')

    def train_model(self):
        """
            Train the model on the dataset created by create_datapoint.
            TO BE REDEFINED IN INHERITING CLASSES.
        """
        raise NotImplementedError('Method train_model must be implemented in inheriting class.')
    
    def estimate_pair(self, data):
        """
            Estimate the reward for the data given.
            TO BE REDEFINED IN INHERITING CLASSES.
        """
        raise NotImplementedError('Method estimate_pair must be implemented in inheriting class.')
    
    def annotation_to_data(self, annotations_file:str, video_folder:str):
        """
            Given an annotations file, and a data folder, will create the datapoints corresponding to the annotations.
            NOTE : right now, this works only if the video_folder contains mp4 videos. To be chnaged in the future, if needed.
        """
        with open(annotations_file, 'r') as f:
            annotations = json.load(f)
        
        print('Creating dataset...')
        missing =0 
        for annotation in tqdm(annotations):
            dataname1 = f"{video_folder}/{annotation['left']}.mp4"
            dataname2 = f"{video_folder}/{annotation['right']}.mp4"
            score = annotation['side']
            
            try:
                data1 = (torchvision.io.read_video(dataname1, output_format='TCHW', pts_unit='sec')[0]).float() / 255.
                data2 = (torchvision.io.read_video(dataname2, output_format='TCHW', pts_unit='sec')[0]).float() / 255.
            except FileNotFoundError:
                print(f'File not found: {dataname1}.mp4 or {dataname2}.mp4. Skipping this pair.')
                print(f'REMOVING pair from annotation, as it is outdated')
                annotations.remove(annotation)
                missing+=1
                continue

            self.create_datapoint(data1, data2, score)
        
        with open(annotations_file, 'w') as f:
            json.dump(annotations, f)
        
        print(f'Dataset created! Removed {missing} missing pairs from the annotations file.')
    
    @property
    def num_datapoints(self):
        """
            Returns the number of datapoints in the dataset.
        """
        return len(os.listdir(self.data_fold))