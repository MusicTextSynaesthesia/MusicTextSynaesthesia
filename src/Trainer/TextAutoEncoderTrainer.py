import torch
import tqdm
import sacrebleu
from .BaseTrainer import Trainer


class TextAutoEncoderTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super(TextAutoEncoderTrainer, self).__init__(*args, **kwargs)
        self.best_bleu = 0
        self.criterion = torch.nn.CrossEntropyLoss().cuda()

    def train(self):
        with open("train.en") as f:
            nist_dataset = [s for s in f.readlines() if "@-@" not in s and 10 < s.count(" ")]
            nist_dataset = sorted(nist_dataset, key=len)
        classic_dataset = [pair["sent_text"] for _, _, pair in self.train_dataset]
        for self.epoch in range(self.args.epoch):
            all_dataset = nist_dataset[self.epoch::50] + classic_dataset
            self.p_bar = tqdm.tqdm(all_dataset, desc=f"train epoch{self.epoch}")
            for self.i, sent_text in enumerate(self.p_bar):
                sent_tokens = self.tokenizer(sent_text, return_tensors='pt')
                loss = self.train_step(sent_tokens["input_ids"].cuda(), None, None)
                if self.i % self.args.batch_size == self.args.batch_size-1 or self.i == len(all_dataset)-1:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                self.p_bar.set_postfix(loss=loss.item())
            epoch_metrics = self.valid(self.valid_dataset)
            self.save_model(epoch_metrics)

    def train_step(self, sent_tokens, waveform: None, pair: None):
        in_tokens, out_tokens = sent_tokens[:, :-1], sent_tokens[:, 1:]
        try:
            logits = self.model(in_tokens)
            loss = self.criterion(logits.transpose(1, 2), out_tokens)
            loss.backward()
        except RuntimeError as e:
            print(e)
            return torch.tensor(0)
        return loss

    def valid(self, dataset, desc="Valid"):
        self.model.eval()
        p_bar = tqdm.tqdm(self.valid_dataset, desc=f"validation")
        sys, ref = [], []
        with torch.no_grad():
            for c, (sent_tokens, waveform, pair_info) in enumerate(p_bar):
                sent_tokens = sent_tokens["input_ids"].cuda()[:, :512]
                gen_text = self.model.generate_text(sent_tokens, self.tokenizer.cls_token_id, self.tokenizer.sep_token_id, self.args.generate_config)
                sys.append(self.tokenizer.batch_decode(gen_text, skip_special_tokens=True))
                ref.append(self.tokenizer.decode(sent_tokens[0], skip_special_tokens=True))
            bleu = sacrebleu.corpus_bleu(ref, [[s if s else " " for s in l] for l in zip(*sys)])
        self.model.train()
        return {"epoch": self.epoch, "BLEU": bleu.score}

    def save_model(self, epoch_metrics):
        if self.args.save_last_checkpoint:
            torch.save(self.model.state_dict(), f"{self.args.model_save_dir}/LastModel.pkl")
        if epoch_metrics["BLEU"] > self.best_bleu:
            self.best_bleu = epoch_metrics["BLEU"]
            torch.save(self.model.state_dict(), f"{self.args.model_save_dir}/BestModel.pkl")
        print(f"Epoch{self.epoch} BLEU: {epoch_metrics['BLEU']}(best: {self.best_bleu})")
