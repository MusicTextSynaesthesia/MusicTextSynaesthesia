import json
import tqdm
import torch
from typing import List, Tuple, Dict
from dataset import ClassicDataset
from Metrics import all_metrics


class Trainer:
    def __init__(self, args, model, tokenizer, optimizer):
        self.args = args
        self.train_dataset = ClassicDataset(tokenizer, args.train_split, args)
        self.valid_dataset = ClassicDataset(tokenizer, args.valid_split, args)
        self.model = model
        self.model.trainer = self
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.p_bar = None
        self.epoch = 0
        self.i = 0

    def train(self):
        for self.epoch in range(self.args.epoch):
            self.p_bar = tqdm.tqdm(self.train_dataset, desc=f"train epoch{self.epoch}")
            for self.i, (sent_tokens, waveform, pair) in enumerate(self.p_bar):
                loss = self.train_step(sent_tokens["input_ids"].cuda(), waveform.cuda(), pair)
                if self.i % self.args.batch_size == self.args.batch_size-1 or self.i == len(self.train_dataset)-1:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                self.p_bar.set_postfix(loss=loss.item())
            if self.args.valid_in_train:
                self.valid(self.train_dataset[::16], "Valid in Train")
            epoch_metrics = self.valid(self.valid_dataset, "Valid")
            self.save_model(epoch_metrics)

    def valid(self, dataset, desc="Valid"):
        with torch.no_grad():
            self.model.eval()
            self.p_bar = tqdm.tqdm(dataset, desc=f"validation")
            sys, ref, losses = [], [], []
            for sent_tokens, waveform, pair in self.p_bar:
                sent_tokens = sent_tokens["input_ids"].cuda()
                loss, generated_text = self.valid_step(sent_tokens, waveform.cuda(), pair)
                losses.append(loss)
                sys.append(self.tokenizer.batch_decode(generated_text, skip_special_tokens=True))
                ref.append(self.tokenizer.decode(sent_tokens[0], skip_special_tokens=True))
                self.p_bar.set_postfix(loss=loss)
            epoch_metrics = all_metrics(sys, ref, desc, epoch=self.epoch, Loss=sum(losses)/len(losses))
            self.model.train()
        return epoch_metrics

    def save_model(self, epoch_metrics):
        pass

    def train_step(self, sent_tokens: torch.Tensor, waveform: torch.Tensor, pair: Dict) -> torch.Tensor:
        pass

    def valid_step(self, sent_tokens: torch.Tensor, waveform: torch.Tensor, pair: Dict) -> Tuple[float, List]:
        pass

    def generate(self):
        n = self.args.generate_config.num_return_sequences
        self.model.eval()
        with torch.no_grad():
            sys, ref = [], []
            for sent_tokens, waveform, pair in self.valid_dataset:
                sent_tokens = sent_tokens["input_ids"].cuda()
                _, generated_text = self.valid_step(sent_tokens, waveform.cuda(), pair)
                sys.append(self.tokenizer.batch_decode(generated_text[0:n], skip_special_tokens=True))
                ref.append(self.tokenizer.decode(sent_tokens[0], skip_special_tokens=True))
                print("R: ", ref[-1])
                print("S: ", sys[-1])
        with open(f"{self.args.output_dir}/{self.args.model_name}_generated.txt", "w") as f:
            json.dump({"reference": ref, "generate": sys}, f)
        self.model.train()
