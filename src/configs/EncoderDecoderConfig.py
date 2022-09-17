from .BaseConfig import BaseConfig
from Model import EncoderDecoder
from Trainer import EncoderDecoderTrainer


class BaseEncoderDecoderConfig(BaseConfig):
    Model = EncoderDecoder
    Trainer = EncoderDecoderTrainer

    model_name = "EncoderDecoder"
    tokenizer = "bert-base-uncased"
