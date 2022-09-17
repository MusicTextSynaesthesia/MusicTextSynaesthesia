import torch
import torchaudio
from .BaseTrainer import Trainer


def wave_sample(waveform, length):
    start = torch.randint(0, waveform.shape[1]-length, (1,)).item()
    return waveform[:, start: start+length]


class CoordinateReconTrainer(Trainer):
    def __init__(self, args, model, tokenizer, optimizer):
        super(CoordinateReconTrainer, self).__init__(args, model, tokenizer, optimizer)
        self.best_bleu = 0
        self.SpecTrans = torchaudio.transforms.Spectrogram(n_fft=510, normalized=True).cuda()
        self.mse = torch.nn.MSELoss().cuda()
        self.criterion = torch.nn.CrossEntropyLoss().cuda()

    def compute_loss(self, recon, refer):
        loss = self.mse(recon, refer)
        loss += self.mse(recon[:, :, 1:, :] - recon[:, :, :-1, :], refer[:, :, 1:, :] - refer[:, :, :-1, :])
        loss += self.mse(recon[:, :, :, 1:] - recon[:, :, :, :-1], refer[:, :, :, 1:] - refer[:, :, :, :-1])
        return loss

    def save_model(self, epoch_metrics):
        if self.args.save_last_checkpoint:
            pass
            # torch.save(self.model.state_dict(), f"{self.args.model_save_dir}/LastModel.pkl")
        if epoch_metrics['BLEU'] > self.best_bleu:
            self.best_bleu = epoch_metrics['BLEU']
            torch.save(self.model.state_dict(), f"{self.args.model_save_dir}/BestModel.pkl")
        print(f"Epoch{self.epoch} Loss: {epoch_metrics['Loss']} (best: {self.best_bleu})")

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
        logits = self.model.decode_text(in_tokens, encoded_w)
        loss += self.criterion(logits.transpose(1, 2), out_tokens)
        return loss, encoded_w, encoded_t

    def train_step(self, sent_tokens, waveform, pair):
        try:
            loss, encoded_w, encoded_t = self.model_forward(sent_tokens, waveform, pair)
            loss.backward()
        except RuntimeError as e:
            print(e)
            return torch.tensor(0)
        return loss

    def valid_step(self, sent_tokens, waveform, pair):
        n = self.args.generate_config.num_return_sequences
        in_tokens, out_tokens = sent_tokens[:, :-1], sent_tokens[:, 1:]
        try:
            encoded = self.model.encode_wave(waveform)
            logits = self.model.decode_text(in_tokens, encoded)
            loss = self.criterion(logits.transpose(1, 2), out_tokens)
            generated = self.model.generate(waveform, self.tokenizer.cls_token_id, self.tokenizer.sep_token_id, self.args.generate_config)
        except RuntimeError as e:
            print(e)
            return 0, [[self.tokenizer.pad_token_id]]*n
        return loss.item(), generated
