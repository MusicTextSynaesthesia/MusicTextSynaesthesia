from .BaseConfig import BaseConfig
from Model import TwoStreamModel
from Trainer import JointClassifyTrainer, JointConfusionTrainer


class JointClassifyConfig(BaseConfig):
    Model = TwoStreamModel
    Trainer = JointClassifyTrainer

    model_name = "JointClassifyModel"
    load_pretrained = True
    load_frozen_pretrained = False


class JointConfusionConfig(BaseConfig):
    Model = TwoStreamModel
    Trainer = JointConfusionTrainer

    model_name = "JointConfusionModel"
    load_pretrained = True
    load_frozen_pretrained = False
