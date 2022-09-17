import torchaudio
from .CNNModules import *
from .TimeSeriesModules import *


class CNNEncoder(nn.Module):
    def __init__(self, dim=64, output_dim=768):
        super(CNNEncoder, self).__init__()
        self.SpecTrans = torchaudio.transforms.Spectrogram(n_fft=510, normalized=True).cuda()
        self.layers = [object()]*8
        self.layers[0] = nn.Sequential(nn.Conv2d(1, dim, kernel_size=3, stride=2, padding=1), nn.LeakyReLU())
        self.layers[1] = nn.Sequential(nn.Conv2d(dim, dim*2, kernel_size=3, stride=2, padding=1), nn.LeakyReLU())
        self.layers[2] = nn.Sequential(nn.Conv2d(dim*2, dim*2, kernel_size=3, stride=2, padding=1), nn.LeakyReLU())
        self.layers[3] = nn.Sequential(nn.Conv2d(dim*2, dim*2, kernel_size=3, stride=2, padding=1), nn.LeakyReLU())
        self.layers[4] = nn.Sequential(nn.Conv2d(dim*2, dim*4, kernel_size=3, stride=2, padding=1), nn.LeakyReLU())
        self.layers[5] = nn.Sequential(nn.Conv2d(dim*4, dim*8, kernel_size=3, stride=(2, 1), padding=1), nn.LeakyReLU())
        self.layers[6] = nn.Sequential(nn.Conv2d(dim*8, dim*12, kernel_size=3, stride=2, padding=1), nn.LeakyReLU())
        self.layers[7] = nn.Sequential(nn.Conv2d(dim*12, dim*12, kernel_size=3, stride=(2, 1), padding=1), nn.LeakyReLU())
        self.layers = nn.Sequential(*self.layers)
        self.mapping = TransformerMapping(dim*12, output_dim, layer_num=2)

    def forward(self, waveform):
        spectrogram = self.SpecTrans(waveform).unsqueeze(1)
        out = self.layers(spectrogram)
        out = self.mapping(out.squeeze(2).transpose(1, 2))
        return out


class ShortChunkCNN(nn.Module):
    def __init__(self, dim=64, output_dim=768):
        super(ShortChunkCNN, self).__init__()
        self.SpecTrans = torchaudio.transforms.Spectrogram(n_fft=510, normalized=True).cuda()
        # Short-chunk CNN
        self.spec_bn = nn.BatchNorm2d(1)
        self.layers = [object()]*8
        self.layers[0] = Conv_2d(1, dim, pooling=2)
        self.layers[1] = Res_2d_mp(dim, dim, pooling=2)
        self.layers[2] = Conv_2d(dim, dim*2, pooling=2)
        self.layers[3] = Res_2d_mp(dim*2, dim*2, pooling=2)
        self.layers[4] = Res_2d_mp(dim*2, dim*2, pooling=2)
        self.layers[5] = Res_2d_mp(dim*2, dim*2, pooling=(2,3))
        self.layers[6] = Conv_2d(dim*2, dim*4, pooling=(2,3))
        self.layers[7] = Conv_emb(dim*4, dim*4)
        self.layers = nn.Sequential(*self.layers)
        self.mapping = TransformerMapping(dim*4, output_dim, layer_num=2)

    def forward(self, waveform):
        spectrogram = self.SpecTrans(waveform).unsqueeze(1).unsqueeze(1)
        out = self.spec_bn(spectrogram)
        out = self.layers(out)
        out = self.mapping(out.squeeze(2).transpose(1, 2))
        return out
