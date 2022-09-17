from .BaseConfig import BaseConfig
from Model import TwoStreamModel
from Trainer import CoordinateReconTrainer, CoordinateCrossContrastiveTrainer, CoordinateGTPContrastiveTrainer
from Trainer import CoordinatePairwiseContrastiveTrainer, CoordinateTripletContrastiveTrainer


class CoordinateReconConfig(BaseConfig):
    Model = TwoStreamModel
    Trainer = CoordinateReconTrainer

    epoch = 5
    model_name = "CoordinateReconModel"
    load_pretrained = True
    load_frozen_pretrained = False


class CoordinateCrossContrastiveConfig(BaseConfig):
    Model = TwoStreamModel
    Trainer = CoordinateCrossContrastiveTrainer

    model_name = "CoordinateCrossContrastive"
    similarity_metric = "BLEU"
    divergence_metric = "mse"
    epoch = 5
    momentum = 0.999
    n_samples = 32
    temperature = 32
    cross_weight = 500


class CoordinatePairwiseContrastiveConfig(BaseConfig):
    Model = TwoStreamModel
    Trainer = CoordinatePairwiseContrastiveTrainer

    model_name = "CoordinatePairwiseContrastive"
    similarity_metric = "BLEU"
    divergence_metric = "mse"
    epoch = 5
    temperature = 32
    cross_weight = 100


class CoordinateTripletContrastiveConfig(BaseConfig):
    Model = TwoStreamModel
    Trainer = CoordinateTripletContrastiveTrainer

    model_name = "CoordinateTripletContrastive"
    similarity_metric = "BLEU"
    divergence_metric = "mse"
    epoch = 5
    margin = 0.1
    cross_weight = 0.1


class CoordinateGTPContrastiveConfig(BaseConfig):
    Model = TwoStreamModel
    Trainer = CoordinateGTPContrastiveTrainer

    model_name = "CoordinateGTPContrastive"
    similarity_metric = "BLEU"
    divergence_metric = "mse"
    epoch = 5
    momentum = 0.999
    n_samples = 32
    temperature = 32
    cross_weight = 500
