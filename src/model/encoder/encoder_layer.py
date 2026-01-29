import torch.nn as nn
from src.model.common.multi_head_layer import MultiHeadAttentionLayer
from src.model.common.pos_wise_feed_layer import PositionWiseFeedforwardLayer

class EncoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, dropout, device):
        super().__init__()

        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.position_wise_feedforward = PositionWiseFeedforwardLayer(hid_dim, pf_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask):
        # Self Attention
        _src, _ = self.self_attention(src, src, src, src_mask)

        # Residual + Norm
        src = self.self_attn_layer_norm(src + self.dropout(_src))

        # Feedforward
        _src = self.position_wise_feedforward(src)

        # Residual + Norm
        src = self.ff_layer_norm(src + self.dropout(_src))

        return src