from reward_training import VJepaRewardTrainer, VideoRewardTrainer
from reward_training import VJEPAReward, CLIPVIPReward

# model = VJEPAReward(vjepa_size='tiny',head_skip=1,vjepa_weights=None)
model = CLIPVIPReward(clipvip_weights='checkpoints/clipvip_32.pt', minihead=True)
v_trainer = VideoRewardTrainer(model, lr_body=1e-6, no_logging=False, data_loc='clipvip_data', device='cuda:2')

# v_trainer.annotation_to_data('Data/output.json', 'RawData/Videos')
# trainer = VJepaRewardTrainer(vjepa_size='large',head_skip=4,vjepa_weights_file='checkpoints/vitl16.pth', lr_jepa=1e-7, device='cuda:1',no_logging=False)

# trainer.train_model()

# trainer = ImageRewardTrainer(no_logging=False, device='cuda:3')

v_trainer.train_model(steps=400,batch_size=10)