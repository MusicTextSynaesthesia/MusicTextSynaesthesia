import torch
import torch.nn as nn
from transformers import BertModel, BertConfig


class LSTMMapping(nn.Module):
    def __init__(self, in_dim, h_dim, layer_num=1):
        super(LSTMMapping, self).__init__()
        self.lstm = nn.LSTM(in_dim, h_dim//2, num_layers=layer_num, batch_first=True, bidirectional=True)

    def forward(self, x):
        x, _ = self.lstm(x)
        return x


class LSTMAttnMapping(nn.Module):
    def __init__(self, in_dim, h_dim, layer_num=1, num_heads=8):
        super(LSTMAttnMapping, self).__init__()
        self.layers = nn.ModuleList(nn.ModuleList([nn.LSTM(in_dim, h_dim//2, bidirectional=True),
                                                   nn.MultiheadAttention(in_dim, num_heads=num_heads),
                                                   nn.LayerNorm(in_dim)]) for _ in range(layer_num))

    def forward(self, x, k=None):
        x = x.transpose(0, 1)
        k = x if k is None else k.transpose(0, 1)
        for lstm, attn, ln in self.layers:
            residual = x
            x, _ = lstm(x)
            x, _ = attn(x, k, k)
            x, k = residual + ln(x), x
        return x.transpose(0, 1)


class ResLSTMMapping(nn.Module):
    def __init__(self, h_dim, layer_num=1):
        super(ResLSTMMapping, self).__init__()
        self.layer_num = layer_num
        self.layers = nn.ModuleList(nn.Sequential(LSTMMapping(h_dim, h_dim), nn.LayerNorm(h_dim)) for _ in range(layer_num))

    def forward(self, x):
        for i in range(self.layer_num):
            x = x + self.layers[i](x)
        return x


class TransformerMapping(nn.Module):
    def __init__(self, in_dim, h_dim, layer_num=1, max_pos=2048):
        super(TransformerMapping, self).__init__()
        self.linear = nn.Linear(in_dim, h_dim)
        self.max_pos = max_pos
        self.transformer = BertModel(
            BertConfig.from_pretrained("bert-base-uncased", cache_dir="cache", vocab_size=4,
                                       hidden_size=h_dim, num_hidden_layers=layer_num, max_position_embeddings=max_pos))

    def forward(self, x):
        x = self.linear(x[:, :self.max_pos])
        x = self.transformer(inputs_embeds=x).last_hidden_state
        return x
