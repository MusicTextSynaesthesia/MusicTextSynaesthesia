import torchaudio
from .CNNModules import *


class CNNDecoder(nn.Module):
    def __init__(self, dim=64):
        super(CNNDecoder, self).__init__()
        self.layers = [object()]*8
        self.layers[0] = nn.Sequential(nn.ConvTranspose2d(dim*12, dim*12, kernel_size=3, stride=(2, 1), padding=1, output_padding=(1, 0)), nn.LeakyReLU())
        self.layers[1] = nn.Sequential(nn.ConvTranspose2d(dim*12, dim*8, kernel_size=3, stride=2, padding=1, output_padding=1), nn.LeakyReLU())
        self.layers[2] = nn.Sequential(nn.ConvTranspose2d(dim*8, dim*4, kernel_size=3, stride=(2, 1), padding=1, output_padding=(1, 0)), nn.LeakyReLU())
        self.layers[3] = nn.Sequential(nn.ConvTranspose2d(dim*4, dim*2, kernel_size=3, stride=2, padding=1, output_padding=1), nn.LeakyReLU())
        self.layers[4] = nn.Sequential(nn.ConvTranspose2d(dim*2, dim*2, kernel_size=3, stride=2, padding=1, output_padding=1), nn.LeakyReLU())
        self.layers[5] = nn.Sequential(nn.ConvTranspose2d(dim*2, dim*2, kernel_size=3, stride=2, padding=1, output_padding=1), nn.LeakyReLU())
        self.layers[6] = nn.Sequential(nn.ConvTranspose2d(dim*2, dim, kernel_size=3, stride=2, padding=1, output_padding=1), nn.LeakyReLU())
        self.layers[7] = nn.Sequential(nn.ConvTranspose2d(dim, 1, kernel_size=3, stride=2, padding=1, output_padding=1), nn.LeakyReLU())
        self.layers = nn.Sequential(*self.layers)

    def forward(self, encoded):
        encoded = encoded.unsqueeze(2).transpose(1, 3)
        out = self.layers(encoded)
        return out
