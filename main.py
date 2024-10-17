from MainWindow import MainWindow
from reward_training import ImageRewardTrainer, VideoRewardTrainer, CLIPVIPReward

def main():
    # reward_train = ImageRewardTrainer('cuda')
    model = CLIPVIPReward(patch_size=32, clipvip_weights='checkpoints/clipvip_32.pt',minihead=True,device='cuda')
    reward_train = VideoRewardTrainer(model=model,lr_body=0.,no_logging=True, device='cuda')
    fenetre = MainWindow(reward_trainer=reward_train,live_training=False)

    fenetre.mainloop()


if __name__ == "__main__":
    main()
