import torch
import sacrebleu
from .BaseTrainer import Trainer


class EncoderDecoderTrainer(Trainer):
    def __init__(self, args, model, tokenizer, optimizer):
        super(EncoderDecoderTrainer, self).__init__(args, model, tokenizer, optimizer)
        self.best_bleu = 0
        self.criterion = torch.nn.CrossEntropyLoss().cuda()

    def save_model(self, epoch_metrics):
        if self.args.save_last_checkpoint:
            torch.save(self.model.state_dict(), f"{self.args.model_save_dir}/LastModel.pkl")
        if epoch_metrics['BLEU'] > self.best_bleu:
            self.best_bleu = epoch_metrics['BLEU']
            torch.save(self.model.state_dict(), f"{self.args.model_save_dir}/BestModel.pkl")
        print(f"Epoch{self.epoch} Loss: {epoch_metrics['Loss']} (best: {self.best_bleu})")

    def train_step(self, sent_tokens, waveform, pair):
        in_tokens, out_tokens = sent_tokens[:, :-1], sent_tokens[:, 1:]
        try:
            logits = self.model(waveform, in_tokens)
            loss = self.criterion(logits.transpose(1, 2), out_tokens)
            loss.backward()
        except RuntimeError as e:
            print(e)
            return torch.tensor(0)
        return loss

    def valid_step(self, sent_tokens, waveform, pair):
        n = self.args.generate_config.num_return_sequences
        in_tokens, out_tokens = sent_tokens[:, :-1], sent_tokens[:, 1:]
        try:
            logits = self.model(waveform, in_tokens).transpose(1, 2)
            loss = self.criterion(logits, out_tokens)
            generated = self.model.generate(waveform, self.tokenizer.cls_token_id, self.tokenizer.sep_token_id, self.args.generate_config)
        except RuntimeError as e:
            print(e)
            return 0, [[self.tokenizer.pad_token_id]] * n
        return loss.item(), generated
