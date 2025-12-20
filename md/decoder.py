
import torch.nn as nn 
from .blocks import ConcatCondishBlock, JustCondishBlock

class Decoder(nn.Module):
    def __init__(self, 
            intermediate_sizes, 
            forcing_dim, 
            activation = nn.GELU,
            sfno_embed_dim=0,
            n_statics=0,
            statics = None,
            rank = 32,
            conditioning_rank = 4,
            conditioning_hidden_channels = 2,
            conditioning_operator_type = "spectral_fc"
        ):
        super(Decoder, self).__init__()
        in_shape, out_shape = intermediate_sizes[0], intermediate_sizes[1]
        self.statics = statics 
        
        self.first_layer = ConcatCondishBlock(
            in_shape,
            out_shape,
            n_conditioning_channels=forcing_dim,
            n_static_channels=n_statics,
            activation=activation,
            is_sfno_block = False if sfno_embed_dim == 0 else True, 
            sfno_hidden_layer= sfno_embed_dim,
            has_resampled_condition = True,
            rank = rank,
            conditioning_operator_type = conditioning_operator_type, 
            conditioning_hidden_channels = conditioning_hidden_channels,
            conditioning_rank = conditioning_rank
        )

        intermediate_sizes = intermediate_sizes[1:]
        self.latent_channels = intermediate_sizes[-1][0]

        layers = []
        for in_shape, out_shape in zip(intermediate_sizes[:-1], intermediate_sizes[1:]):
            layers.append(
                JustCondishBlock(
                    in_shape,
                    out_shape,
                    n_conditioning_channels=forcing_dim,
                    activation=activation,
                    is_sfno_block = False if sfno_embed_dim == 0 else True, 
                    sfno_hidden_layer= sfno_embed_dim,
                    has_resampled_condition=True,
                    rank = rank,
                    conditioning_operator_type = conditioning_operator_type,
                    conditioning_hidden_channels = conditioning_hidden_channels,
                    conditioning_rank = conditioning_rank
                )
            )

        self.layers = nn.ModuleList(layers)

    
    def forward(self, x, cs):
        cs_r = cs[::-1]
        ss = self.statics[::-1]
        x, c_r = self.first_layer(x, cs_r[0], c_resampled=cs_r[1], statics = ss[0] )
        for _, layer in enumerate(self.layers): 
            x, c_r = layer(x, cs_r[_+2], c_is_resampled=True )

        return  x
