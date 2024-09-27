from .reward_models.squeezenet import SqueezeReward
from .reward_trainer import RewardTrainer
from .datasets import ImageRewardDataset

from torch.utils.data import DataLoader, Subset
import torch, torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
import os

class ImageRewardTrainer(RewardTrainer):
    """
        Reward trainer based on a SqueezeNet model, which
        is used to create a reward model for images.
    """

    def __init__(self, no_logging = True, device='cpu'):
        model = SqueezeReward(device)
        optim = AdamW(model.parameters(),lr=1e-3)
        schedu = LinearLR(optim,start_factor=1e-5, end_factor=1, total_iters=300)
        super().__init__(model=model, data_loc='image_data', optimizer=optim, 
                         scheduler=schedu, no_logging=no_logging, device=device)

        self.dataset =  ImageRewardDataset(self.data_fold)
        
        self.train_dataset = None # Created on the spot
        self.val_dataset = None # Created on the spot

    def create_datapoint(self, data1, data2, annotation) -> str:
        """ 
        Creates a datapoint for the dataset of the image reward model. 

        Args : 
        data1/2 : tensor, (T,3,H,W) or (3,H,W) tensor representing the video or last frame
        annotation : int, 0, 1 or 0.5, representing the annotation of the user

        Returns: 
        str, the path to the saved datapoint
        """

        data1, data2 = self._get_image_from_data(data1), self._get_image_from_data(data2)

        data_path = f'./{self.data_fold}/{len(os.listdir(self.data_fold))}.pt'
        # Save the data in the appropriate folder
        torch.save({'data1':data1, 'data2':data2, 'annotation':torch.tensor([annotation,1-annotation],dtype=torch.float)}, data_path)
        
        return data_path

    def _get_image_from_data(self, data):
        """
            Returns the image from the data, which can be a video or a single frame.

            Args:
            data : (T,3,H,W) or (3,H,W) tensor representing the video or last frame

            Returns:
            (3,H,W) tensor, representing the image.
        """
        # Use clone() to avoid memory issues when saving
        if len(data.shape) == 4:
            return data[-1].clone()
        return data

    @torch.no_grad()
    def estimate_pair(self, data1, data2):
        """
            Estimates the reward for a pair of (batched) images.

            Args:
            data : (3,H,W) or (T,3,H,W) tensors representing the videos or last frame to compare.

            Returns:
            (2,) tensor of floats, representing the probability of winning for each image.
        """

        img1, img2 = self._get_image_from_data(data1)[None].to(self.device), self._get_image_from_data(data2)[None].to(self.device)
        rewards = torch.stack([self.model(img1),self.model(img2)],dim=1).squeeze() # (2,) 
        rewards = F.softmax(rewards, dim=0) # (2,), adjusted probabilities

        return rewards
    
    def process_batch(self, batch_data):
        data1, data2, annotation = batch_data
        data1 = data1.to(self.device) # (B,3,H,W)
        data2 = data2.to(self.device) # (B,3,H,W)
        annotation = annotation.to(self.device) # (B,2)

        rewards = torch.stack([self.model(data1),self.model(data2)],dim=1).squeeze(-1) # (B,2) 

        loss = self.unadjusted_cross_entropy(rewards, annotation)
    
        return loss

    def adjusted_cross_entropy(self, logits, target):
        probas = F.softmax(logits, dim=1)
        probas = 0.9*probas + 0.05 # insert a 'randomness' of 10%
        
        return F.nll_loss(probas.log(), target, reduction='mean') # Doesn't work for soft labels

    def unadjusted_cross_entropy(self, logits, target):
        return F.cross_entropy(logits, target, reduction='mean')

    def get_loaders(self, batch_size, num_workers=0):
        """
            Returns the dataloader for the dataset.
        """
        mixer = torch.randperm(len(self.dataset))

        self.train_dataset = Subset(self.dataset, mixer[:int(0.85*len(mixer))])
        self.val_dataset = Subset(self.dataset, mixer[int(0.85*len(mixer)):])

        train_dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=batch_size, num_workers=num_workers,shuffle=True)
        valid_dataloader = torch.utils.data.DataLoader(self.val_dataset, batch_size=batch_size, num_workers=num_workers,shuffle=True)

        return train_dataloader, valid_dataloader

    def train_model(self):
        """
            Trains the reward model on the dataset created by create_datapoint.
        """
        self.dataset.refresh() # Refreshes the dataset before launching training.
        self.train_steps(steps=50, batch_size=10, save_every=1e6, pickup=False)
        print('Training done !')