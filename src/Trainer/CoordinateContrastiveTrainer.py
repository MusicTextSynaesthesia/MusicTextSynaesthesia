import torch
from torch.utils.checkpoint import checkpoint
from .CoordinateReconTrainer import CoordinateReconTrainer, wave_sample


class CoordinateContrastiveTrainer(CoordinateReconTrainer):
    def __init__(self, args, model, tokenizer, optimizer):
        super(CoordinateContrastiveTrainer, self).__init__(args, model, tokenizer, optimizer)
        self.mse = torch.nn.MSELoss()

    def model_forward(self, sent_tokens, waveform, pair):
        in_tokens, out_tokens, tags = sent_tokens[:, :-1], sent_tokens[:, 1:], pair["tags"]
        length = torch.randint(20, min(waveform.shape[1] // 64 // 255, 512), (1,)).item() * 64
        sample_wave = wave_sample(waveform, 255 * length - 1)
        spectrogram = self.SpecTrans(sample_wave).unsqueeze(1)
        encoded = self.model.wave_encoder(sample_wave)
        decoded = self.model.decode_wave(encoded)
        loss = self.compute_loss(decoded, spectrogram)

        encoded_t = self.model.encode_text(sent_tokens)
        encoded_t = encoded_t * 0.9 + torch.rand_like(encoded_t) * 0.1
        logits = self.model.decode_text(in_tokens, encoded_t)
        loss += self.criterion(logits.transpose(1, 2), out_tokens)

        encoded_w = self.model.encode_wave(waveform)
        aug_waves = [af(self.args, waveform) for af in self.args.aug_func]
        aug_enc = [checkpoint(self.model.encode_wave, x) for x in aug_waves]
        loss += self.args.contrastive_weight * sum([self.mse(x, encoded_w) for x in aug_enc])/len(aug_enc)
        logits = self.model.decode_text(in_tokens, encoded_w)
        loss += self.criterion(logits.transpose(1, 2), out_tokens)
        return loss, encoded_w, encoded_t
