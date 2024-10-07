from MainWindow import MainWindow
from reward_training import ImageRewardTrainer, VJepaRewardTrainer

def main():
    # reward_train = ImageRewardTrainer('cuda')
    reward_train = VJepaRewardTrainer('checkpoints/vitl16.pth', lr_jepa=0., no_logging=False,device='cuda')
    fenetre = MainWindow(reward_trainer=reward_train)

    fenetre.mainloop()


if __name__ == "__main__":
    main()
