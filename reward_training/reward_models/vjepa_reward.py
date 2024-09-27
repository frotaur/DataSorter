from v_jepa import get_vit_large
from torchenhanced import ConfigModule

class VJEPAReward(ConfigModule):
    pass
    # TODO : Make the VJEPA reward. Essentially, ask for a path to the vjepa weights.
    # Then, load the model, and treat the features of VJEPA to output a number
    # Probably nice to have an helper function, that outputs the head parameters
    # vs the VJEPA parameters, so that when I have the reward trainer, I can assign
    # a different LR to the head and the VJEPA parameters.