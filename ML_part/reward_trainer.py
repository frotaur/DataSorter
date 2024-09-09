from torchenhanced import ConfigModule, Trainer
import torch, torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR


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

    def __init__(self, model : nn.Module, data_loc, optimizer, scheduler, device='cpu'):
        """
            Args:
            model : model to be trained. Probably to be hardcoded in inheriting classes
            
        """
        
        super().__init__(model=model, optim=optimizer, scheduler=scheduler, save_loc='reward_model_train',device=device)