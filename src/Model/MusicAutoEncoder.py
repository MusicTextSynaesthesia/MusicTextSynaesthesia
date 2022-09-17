from torch import nn
from Model.modules.CNNEncoder import CNNEncoder
from Model.modules.CNNDecoder import CNNDecoder


class MusicAutoEncoder(nn.Module):
    def __init__(self, args):
        super(MusicAutoEncoder, self).__init__()
        self.wave_encoder = CNNEncoder(args.music_dim, args.latent_dim)
        self.wave_decoder = CNNDecoder(args.music_dim)
        self.max_layer_num = len(self.wave_encoder.layers) + 1
        self.layer_num = args.init_layer_num

    def add_layer_num(self):
        if self.layer_num < self.max_layer_num:
            self.layer_num += 1
            print(f"adding layer to {self.layer_num}/{self.max_layer_num}")
            return True
        return False

    def forward(self, waveform):
        spectrogram = self.wave_encoder.SpecTrans(waveform).unsqueeze(1)
        encoded = self.wave_encoder.layers[:self.layer_num](spectrogram)
        if self.layer_num == self.max_layer_num:
            encoded = self.wave_encoder.mapping(encoded.squeeze(2).transpose(1, 2))
            encoded = encoded.unsqueeze(2).transpose(1, 3)
        decoded = self.wave_decoder.layers[-self.layer_num:](encoded)
        return decoded
