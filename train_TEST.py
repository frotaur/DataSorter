from reward_training import VideoRewardTrainer, ImageRewardTrainer
from reward_training import VJEPAReward, CLIPVIPReward

# model = VJEPAReward(vjepa_size='tiny',head_skip=1,vjepa_weights=None)
# model = CLIPVIPReward(clipvip_weights='checkpoints/clipvip_32.pt', minihead=True)
# v_trainer = VideoRewardTrainer(model, lr_body=0., no_logging=False, data_loc='clipvip_data', device='cuda:1')
i_trainer = ImageRewardTrainer(lr_body=0., no_logging=False, device='cuda:0')
# v_trainer.annotation_to_data('Data/output.json', 'RawData/Videos')
# trainer.train_model()

# i_trainer.annotation_to_data('Data/output.json', 'RawData/Videos')
# trainer = ImageRewardTrainer(no_logging=False, device='cuda:3')
i_trainer.train_model(steps=400,batch_size=10)
# v_trainer.train_model(steps=200,batch_size=50)