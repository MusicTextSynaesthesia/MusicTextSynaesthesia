import torch
from .BaseConfig import BaseConfig
from Model import MusicAutoEncoder, TextAutoEncoder
from Trainer import MusicAutoEncoderTrainer, TextAutoEncoderTrainer
from Model.modules.Pooler import AttentionPooler, MeanPooler


class MusicAutoEncoderConfig(BaseConfig):
    Model = MusicAutoEncoder
    Trainer = MusicAutoEncoderTrainer

    model_name = "MusicAutoEncoder"
    gradient_checkpoint = False
    load_pretrained = False
    epoch = 1000
    batch_size = 34
    optimizer = torch.optim.Adam
    lr = 5e-5
    weight_decay = 0
    init_layer_num = 2
    valid_in_train = False


class TextAutoEncoderConfig(BaseConfig):
    Model = TextAutoEncoder
    Trainer = TextAutoEncoderTrainer

    model_name = "TextAutoEncoder"
    gradient_checkpoint = False
    load_pretrained = False
    epoch = 30
    batch_size = 8
    optimizer = torch.optim.Adam
    lr = 5e-5
    weight_decay = 0
    pooler = AttentionPooler

    def __init__(self, *args, **kwargs):
        super(TextAutoEncoderConfig, self).__init__(*args, **kwargs)
        self.generate_config.max_len = 100
        self.generate_config.do_sample = False
        self.generate_config.num_beams = 1
        self.generate_config.top_k = 50
        self.generate_config.num_return_sequences = 1
