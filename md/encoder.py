
import torch.nn as nn 
from .blocks import ConcatCondishBlock, JustCondishBlock
import torch 

class VariationalEncoder(nn.Module):
    def __init__(self, 
            intermediate_sizes, 
            forcing_dim, 
            n_statics,
            activation = nn.GELU,
            sfno_embed_dim=0,
            varnames = None,
            statics = None,
            device=torch.device('cuda'),
            rank = 32,
            conditioning_rank = 4,
            conditioning_operator_type = 'spectral_fc'
        ):
        super(VariationalEncoder, self).__init__()
        self.device = device
        
        # intermediate sizes is initially small-to-big 
        intermediate_sizes_r = intermediate_sizes[::-1]

        if len(intermediate_sizes_r) == 1: 
            intermediate_sizes_r = [intermediate_sizes_r[0] for _ in range(2)]

        # number of channels in Z 
        self.latent_channels = intermediate_sizes_r[-1][0]
        
        # double output channels of final layer for mu and log var 
        intermediate_sizes_r[-1] = (intermediate_sizes_r[-1][0]*2, intermediate_sizes_r[-1][1], intermediate_sizes_r[-1][2])
        
        self.statics = statics 
        in_shape, out_shape = intermediate_sizes_r[0], intermediate_sizes_r[1]

        self.first_layer = ConcatCondishBlock(
            in_shape,
            out_shape,
            n_conditioning_channels=forcing_dim,
            n_static_channels=n_statics,
            activation=activation,
            is_sfno_block = False if sfno_embed_dim == 0 else True, 
            sfno_hidden_layer= sfno_embed_dim,
            varnames = varnames,
            rank = rank,
            conditioning_rank = conditioning_rank,
            conditioning_operator_type = conditioning_operator_type
        )

        intermediate_sizes_r = intermediate_sizes_r[1:]

        layers = []
        for in_shape, out_shape in zip(intermediate_sizes_r[:-1], intermediate_sizes_r[1:]):
            layers.append(
                JustCondishBlock(
                    in_shape,
                    out_shape,
                    n_conditioning_channels=forcing_dim,
                    activation=activation,
                    is_sfno_block = False if sfno_embed_dim == 0 else True, 
                    sfno_hidden_layer= sfno_embed_dim,
                    rank = rank,
                    conditioning_rank = conditioning_rank,
                    conditioning_operator_type = conditioning_operator_type
                )
            )

        self.layers = nn.ModuleList(layers)

        if self.statics is None:
            self.statics = [None for _ in range(len(self.layers)+1) ] 
        else:
            self.statics = self.embed_conditions(self.statics)

    def forward(self, x, c):
        cs = [c]
        x, c_r = self.first_layer(x, c, self.statics[0] )
        cs.append(c_r)
        for _, layer in enumerate(self.layers): 
            x, c_r = layer(x, c_r)#, self.statics[_+1])
            cs.append(c_r)

        return  x[:, :self.latent_channels, :, :], x[:, self.latent_channels:, :, :], cs 

    def embed_conditions(self, c):
        cs = [c]
        c_resampled = self.first_layer.condition_resample(c)
        cs.append(c_resampled)
        for layer in self.layers:
            c_resampled = layer.condition_resample(c_resampled)
            cs.append(c_resampled)
        return cs 

