import torch.nn as nn
from onpolicy.algorithms.utils.mlp import MLPLayer

class Encoder(MLPLayer):
    def __init__(self, input_dim, hidden_size, layer_N, use_orthogonal, use_ReLU, use_feature_normalization):
        super(Encoder, self).__init__(input_dim, hidden_size, layer_N, use_orthogonal, use_ReLU)
        self._use_feature_normalization=use_feature_normalization

        if self._use_feature_normalization:
            self.feature_norm = nn.LayerNorm(input_dim)
    
    def forward(self, x):
        if self._use_feature_normalization:
            x = self.feature_norm(x)
        x=super().forward(x)
        return x