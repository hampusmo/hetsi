# General set of networks and layers for the model

import torch
from torch import nn as nn

class lRFFLayer(nn.Module):
    def __init__(self, input_dim, embed_dim, sigma = 3e1):
        super(lRFFLayer, self).__init__()

        w1 = torch.randn((embed_dim // 2, input_dim)) * sigma
        w2 = torch.randn((embed_dim // 2, input_dim)) * sigma

        self.w1 = nn.Parameter(w1)
        self.w2 = nn.Parameter(w2)

    def forward(self, x):
        c = torch.cos(nn.functional.linear(x, self.w1))
        s = torch.sin(nn.functional.linear(x, self.w2))
        return torch.cat([c,s], dim = -1)


class RFFMLP(nn.Module):
    def __init__(self, spec: list[int], activation = None, scale = 3e1, dtype = torch.float32):
        super(RFFMLP, self).__init__()

        if torch.is_complex(torch.tensor(1., dtype = dtype)):
            self.complex = True
            spec[-1] = spec[-1] * 2
        
        else:
            self.complex = False

        layers = []
        layers.append(lRFFLayer(spec[0], spec[1], sigma = scale))

        for i in range(2, len(spec)):
            layers.append(nn.Linear(spec[i-1], spec[i]))

            if activation is not None:
                layers.append(activation)
        
        _ = layers.pop(-1) # remove final activation

        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        if self.complex:
            p = self.net(x)
            return torch.complex(*torch.split(p, p.shape[-1] // 2, dim = -1))
        
        else:
            return self.net(x)

    

        