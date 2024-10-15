from .reward_models.vjepa_reward import VJEPAReward
from .reward_trainer import RewardTrainer
from .datasets import DiskRewardDataset

from torch.utils.data import DataLoader, Subset
import torch, torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
import os

class VJepaRewardTrainer(RewardTrainer):
    """
        DEPRECATED :
        Reward trainer based on a SqueezeNet model, which
        is used to create a reward model for images.
    """

    def __init__(self, vjepa_size='large', vjepa_weights_file=None, head_skip=1,
                  lr_jepa=1e-6, no_logging = True, device='cpu'):
        """
            vjepa_size : str, 'large' or 'tiny', size of the VJEPA model
            vjepa_weights_file : str, path to the VJEPA weights
            head_skip : int, size of jumps in the kept head tokens. 1 means all tokens are kept.
            lr_jepa : float, learning rate for the VJEPA model. Pass 0. to freeze the model.
            no_logging : bool, whether to log the training or not
            device : str, device to train the model on
        """
        assert vjepa_size in ['large','tiny'], 'vjepa_size must be either large or tiny'
        
        model = VJEPAReward(vjepa_size=vjepa_size,vjepa_weights=vjepa_weights_file, head_skip=head_skip,device=device)
        vjepa_params = model.vjepa_params()
        if(lr_jepa<=1e-8):
            for param in vjepa_params:
                param.requires_grad = False
            optim = AdamW(model.head_params(),lr=1e-3)
        else :
            optim = AdamW([
                {'params': vjepa_params , 'lr': lr_jepa},
                {'params': model.head_params(), 'lr': 1e-4}
            ])
        schedu = LinearLR(optim,start_factor=1e-5, end_factor=1, total_iters=100)

        super().__init__(model=model, data_loc='vjepa_data', optimizer=optim, 
                         scheduler=schedu, no_logging=no_logging, device=device)

        self.dataset =  DiskRewardDataset(self.data_fold)
        
        self.train_dataset = None # Created on the spot
        self.val_dataset = None # Created on the spot
        
        self.input_shape = model.input_shape

    def create_datapoint(self, data1, data2, annotation) -> str:
        """ 
        Creates a datapoint for the dataset of the image reward model. 

        Args : 
        data1/2 : tensor, (T,3,H,W) representing the video
        annotation : int, 0, 1 or 0.5, representing the annotation of the user

        Returns: 
        str, the path to the saved datapoint
        """

        data1, data2 = self._process_video(data1), self._process_video(data2)

        data_path = f'./{self.data_fold}/{len(os.listdir(self.data_fold))}.pt'
        # Save the data in the appropriate folder
        torch.save({'data1':data1, 'data2':data2, 'annotation':torch.tensor([annotation,1-annotation],dtype=torch.float)}, data_path)
        
        return data_path

    def _process_video(self, video):
        """
            Given a video tensor, returns the processed tensor.

            Args:
            data : (T,3,H,W) representing the video

            Returns:
            (T',3,H',W') tensor, processed video
        """
        tar_T, _, tar_H, tar_W = self.input_shape
        T = video.shape[0]
        assert tar_T <= video.shape[0], f'Video {video.shape[0]} frames, need at least {tar_T} frames'

        # Take tar_T equally spaced frames
        video = video[torch.linspace(0,T-1,tar_T).long()]

        video = torch.einsum('tchw->cthw', video) # interpolate expects channels first
        # Resize the frames
        video = F.interpolate(video, size=(tar_H,tar_W), mode='bilinear')
        video = torch.einsum('cthw->tchw', video) # back to normal
        
        assert video.shape == self.input_shape, f'Video shape {video.shape} not equal to {self.input_shape}'

        return video

    @torch.no_grad()
    def estimate_pair(self, data1, data2, preprocessed=False):
        """
            Estimates the reward for a pair of (un-batched) videos.

            Args:
            data1/2 : (T,3,H,W) tensors representing the videos to compare.	
            preprocessed : bool, whether the data is preprocessed or not

            Returns:
            (2,) tensor of floats, representing the probability of winning for each image.
        """
        if(not preprocessed):
            data1, data2 = self._process_video(data1), self._process_video(data2)

        data1, data2 = data1[None].to(self.device), data2[None].to(self.device)

        probas = torch.stack([self.model(data1),self.model(data2)],dim=1).squeeze() # (2,) 
        probas = F.softmax(probas, dim=0) # (2,), adjusted probabilities

        return probas
    
    def process_batch(self, batch_data):
        data1, data2, annotation = batch_data
        data1 = data1.to(self.device) # (B,T,3,H,W)
        data2 = data2.to(self.device) # (B,T,3,H,W)
        annotation = annotation.to(self.device) # (B,2)

        rewards = torch.stack([self.model(data1),self.model(data2)],dim=1).squeeze(-1) # (B,2) 

        loss = self.unadjusted_cross_entropy(rewards, annotation)
    
        return loss

    def process_batch_valid(self, batch_data):
        data1, data2, annotation = batch_data
        data1 = data1.to(self.device) # (B,T,3,H,W)
        data2 = data2.to(self.device) # (B,T,3,H,W)
        annotation = annotation.to(self.device) # (B,2)

        rewards = torch.stack([self.model(data1),self.model(data2)],dim=1).squeeze(-1) # (B,2) 

        loss = self.unadjusted_cross_entropy(rewards, annotation)
    
        return loss

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
        self.train_steps(steps=400, batch_size=10, valid_every=20, step_log=2, save_every=1e6, pickup=False,
                         num_workers=4)
        print('Training done !')
