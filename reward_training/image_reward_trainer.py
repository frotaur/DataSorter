from .reward_models import SqueezeReward
from .reward_trainer import RewardTrainer
from .datasets import ImageRewardDataset

import torch, torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
import os

class ImageRewardTrainer(RewardTrainer):
    """
        Reward trainer based on a SqueezeNet model, which
        is used to create a reward model for images.
    """

    def __init__(self, device='cpu'):
        model = SqueezeReward(device)
        optim = AdamW(model.parameters(),lr=1e-3)
        schedu = LinearLR(optim,start_factor=1e-5, end_factor=1, total_iters=300)
        super().__init__(model=model, data_loc='image_data', optimizer=optim, 
                         scheduler=schedu, device=device)

        self.dataset =  ImageRewardDataset(self.data_fold)
        
        print("WARNING : DUMMY LOSS USED FOR NOW, TO BE CHANGED")

    def create_datapoint(self, data1, data2, annotation):
        """ 
        Creates a datapoint for the dataset of the image reward model. 

        Args : 
        data1/2 : tensor, (T,3,H,W) or (3,H,W) tensor representing the video or last frame
        annotation : int, 0, 1 or 0.5, representing the annotation of the user
        """

        # Check if the data is a video or a single frame
        if len(data1.shape) == 4:
            data1 = data1[-1]
            data2 = data2[-1]

        # Save the data in the appropriate folder
        torch.save({'data1':data1, 'data2':data2, 'annotation':annotation}, f'./{self.data_fold}/{len(os.listdir(self.data_fold))}.pt')

    def estimate_pair(self, image):
        """
            Estimates the reward for a pair of (batched) images.

            Args:
            image : (B,3,H,W) tensors representing the images to compare.

            Returns:
            r : (B,1) tensor of floats, representing the reward for each image.
        """
        r = self.model(image) # (B,1)

        return r
    
    def process_batch(self, batch_data):
        data1, data2, annotation = batch_data
        data1 = data1.to(self.device)
        data2 = data2.to(self.device)
        annotation = annotation.to(self.device)

        r1 = self.model(data1)
        r2 = self.model(data2)

        ## NOTE, WARNING : DUMMY LOSS USED FOR NOW, TO BE CHANGED
        # Compute annotation with sigmoid diff :
        loss = F.mse_loss(F.sigmoid(r1-r2), annotation)

        return loss

    def get_loaders(self, batch_size, num_workers=0):
        """
            Returns the dataloader for the dataset.

            TODO : add validation, separating the dataset.
        """
        train_dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=batch_size, num_workers=num_workers,shuffle=True)
        return train_dataloader, None

    def train_model(self):
        """
            Trains the reward model on the dataset created by create_datapoint.
        """
        self.train_steps(steps=1000, batch_size=10, save_every=1e6, pickup=False)
        print('Training done !')