# functions for ranking sets of videos or preprocessed tensors.
# What should it output ? Probably a json with the video scores,
# the keys being the video names.
from reward_training import RewardTrainer
from torchenhanced import ConfigModule
import torch, os, json
from tqdm import tqdm
import cv2

class VideoRanker:
    """
        Class with utility to rank videos using a reward model
    """

    def __init__(self, reward_trainer:RewardTrainer, model:ConfigModule = None, device='cpu'):
        """
            Args:
            reward_trainer : used to process the videos, and get the reward model
            model : if not None, will use this model to rank videos. Otherwise, use the reward_trainer's model
            device : str, device to use
        """
        self.trainer =reward_trainer
        if(model is None):
            self.model = reward_trainer.model
        else :
            self.model = model

        self.device = device

        self.model.to(device)

        self.model.eval()

    @torch.no_grad()
    def _rank_tensor(self, input_tensor):
        """
            Ranks a preprocessed, batched video tensor

            Args :
            input_tensor :(B,*) tensor, should match shape expected by the model
        """
        input_tensor = input_tensor.to(self.device)

        scores = self.model(input_tensor) # (B,)

        return scores

    def brand_videos(video_folder:str, scores:dict, output_folder:str):
        """
            Adds to the video the score from the score file.

            Args:
            video_folder: path to the folder containing the videos
            scores: keys are the video names, values are the scores
            output_folder: path to the folder where the branded videos will be saved
        """
        os.makedirs(output_folder, exist_ok=True)

        # Font settings
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_color = (255, 255, 255)  # White color
        thickness = 2

        # Process each video
        for video_name, score in tqdm(scores.items()):
            video_path = os.path.join(video_folder, video_name)
            if not os.path.exists(video_path):
                print(f"Video file not found: {video_path}")
                continue
            
            # Open the video
            cap = cv2.VideoCapture(video_path)
            
            # Get video properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            
            # Create VideoWriter object
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(os.path.join(output_folder,f"{score}_{video_name}"), fourcc, fps, (width, height))
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Add score to the frame
                cv2.putText(frame, f"Score: {score:.2f}", (10, 30), font, font_scale, font_color, thickness)
                
                # Write the frame
                out.write(frame)
            
            # Release everything
            cap.release()
            out.release()

        print("All videos processed.")

    def score_videos(self, video_folder):
        """
            Ranks the videos in a folder. First processes them to tensor,
            then proceed to the ranking

            Args:
            video_folder : str, path to the folder containing the videos
        """
        print('Ranking (conver video to tensor takes a while)')

        scores = {}

        video_names = os.listdir(video_folder)
        for i, video in enumerate(tqdm(video_names)):
            video_tensor = self.trainer.video_to_tensor(os.path.join(video_folder,video))
            scores[video] = self._rank_tensor(video_tensor[None]).item()
        
        return scores

    def score_and_brand_videos(self, video_folder, out_folder, scores_name=None):
        """
            Scores all videos in folder, and copies them to the output folder,
            branded with their score.
        """
        scores = self.score_videos(video_folder)

        VideoRanker.brand_videos(video_folder, scores, out_folder)

        os.makedirs('./scorings',exist_ok=True)
        if scores_name is None:
            scores_name = f'scores_{self.model.__class__.__name__}.json'
        
        with open(os.path.join('scorings',scores_name+'.json'),'w') as f:
            json.dump(scores,f)

if __name__=='__main__':
    from reward_training import CLIPVIPReward, VideoRewardTrainer

    trainer = VideoRewardTrainer(model=CLIPVIPReward(32, minihead=True), lr_body=0, device='cuda')

    trainer.load_model_from_state('./lenia_rlhf/reward_train/state/clipvip_lr0.0.state')

    ranker = VideoRanker(trainer, device='cuda')

    scores = ranker.rank_videos('./OldVideos')
    
    os.makedirs('./scorings',exist_ok=True)
    with open(os.path.join('scorings','scores_clipvip_oldvideo.json'),'w') as f:
        json.dump(scores,f)