from .v_jepa import get_vit_large
from torchenhanced import ConfigModule, DevModule
import torch, torch.nn as nn


class VJEPAReward(ConfigModule):
    """
        Given VJEPA weights (only one option for now),
        creates the reward model for videos.

        Expected video shape : (B, 3, T, H, W)
    """

    def __init__(self, vjepa_weights=None, num_frames=16, device='cpu'):
        """
            vjepa_weights : path to the VJEPA weights, if None, random weights are used
            num_frames : number of frames in the video, should be
            16 as it is how it's trained.

        """
        configo = dict(vjepa_weights=vjepa_weights)
        super().__init__(configo, device=device)
        
        self.vjepa = get_vit_large(pre_weights_file=vjepa_weights)
        self.vjepa.to(device)
        self.vjepa.eval()

        # Freeze the weights of vjepa
        for param in self.vjepa.parameters():
            param.requires_grad = False
        
        out_tokens, out_dim = self._get_vjepa_output_shape(num_frames)
        # Replace the last two layers (final_conv and classifier)
        minihead = MiniBlock(embed_dim=out_dim, num_tokens=out_tokens, device=device)

        last_linear = nn.Linear(out_dim, 1)

        self.score_head = nn.ModuleDict(dict(minihead=minihead, last_linear=last_linear))
        self.to(device)

        self.input_shape = (3,num_frames,224,224)

    def vjepa_params(self):
        """
            Returns the parameters of the VJEPA model
        """
        return self.vjepa.parameters()

    def head_params(self):
        """
            Returns the parameters of the head of the model
        """
        return self.score_head.parameters()

    def _get_vjepa_output_shape(self, num_frames):
        """
            Returns the shape of the output of the VJEPA model
        """
        input_image = torch.randn(1,3,num_frames,224,224, device=self.device)
        output = self.vjepa(input_image) # (1, patch_num, patch_dim)

        return output.shape[1:]

    def forward(self, x):
        """
            x : (B, 3, T, H, W) tensor, correct dimensions for the model

            Returns : (B,) tensor of scores
        """
        B, C, T, H, W = x.shape
        assert C == 3, 'Input must have 3 channels'

        # x = torch.nn.functional.interpolate(x, size=(224, 224), mode='bilinear')
        x = self.vjepa(x) # (B,T,D)
        x = self.score_head['minihead'](x) # (B,T,D)
       
        x = self.score_head['last_linear'](x[:,-1,:]) # (B,1) Linear head on last token

        return x.squeeze(1)

 
class MiniAttentionHead(ConfigModule):
    """
        Mini causal transformer head to be used on top of VJEPA representations.
    """

    def __init__(self, in_dim, num_tokens):
        """
            in_dim : int, dimension of the input
            num_tokens : int, number of tokens to output
        """
        configo = dict(in_dim=in_dim, num_tokens=num_tokens)
        super().__init__(configo)
        
        self.attention = torch.nn.MultiheadAttention(in_dim, num_heads=4, dropout=0.1, batch_first=True)
        self.score_head = torch.nn.Linear(in_dim, 1)
        self.pos_embedder = torch.nn.Embedding(num_tokens, in_dim)

        self.register_buffer('cant_attend',torch.tril(torch.ones(num_tokens, num_tokens), diagonal=0))
        self.cant_attend = self.cant_attend==0

    def forward(self, x):
        """
            x : (B, T, in_dim) tensor
        """
        B, T, in_dim = x.shape
        x = x + self.pos_embedder(torch.arange(T).to(x.device)) # Pos embed

        out, _ = self.attention(x,x,x, is_causal=True, attn_mask=self.cant_attend[:T,:T])

        return out

class MiniBlock(DevModule):
    """
    One transformer block/layer, fast causal attention followed by a MLP.

    Args:
        embed_dim: number of embedding dimensions
        n_heads: number of attention heads
        attn_length: length of the attention window
        mlp_ratio: ratio of mlp hidden dim to embedding dim
        dropout: (optional) dropout probability
    """

    def __init__(
        self,
        embed_dim: int,
        num_tokens: int,
        device='cpu'
        ):
        super().__init__(device=device)

        self.ln_1 = nn.LayerNorm(embed_dim)
        self.attn = MiniAttentionHead(
            in_dim=embed_dim,
            num_tokens=num_tokens,
        )

        self.ln_2 = nn.LayerNorm(embed_dim)

        self.mlp = nn.ModuleDict(
            dict(
                c_fc=nn.Linear(embed_dim, int(2 * embed_dim)),
                act=nn.GELU(),
                c_proj=nn.Linear(int(2 * embed_dim), embed_dim),
                dropout=nn.Dropout(0.1),
            )
        )

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp["dropout"](
            self.mlp["c_proj"](self.mlp["act"](self.mlp["c_fc"](self.ln_2(x))))
        )

        return x
