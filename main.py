from MainWindow import MainWindow
from reward_training import ImageRewardTrainer

def main():
    reward_train = ImageRewardTrainer('cuda')
    fenetre = MainWindow(reward_trainer=reward_train)

    fenetre.mainloop()


if __name__ == "__main__":
    main()
