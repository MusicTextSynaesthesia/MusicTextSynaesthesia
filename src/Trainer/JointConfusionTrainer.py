import torch
import tqdm
from sklearn.metrics import f1_score
from .BaseTrainer import Trainer
from Metrics import all_metrics


class JointConfusionTrainer(Trainer):
    def __init__(self, args, model, tokenizer, optimizer):
        super(JointConfusionTrainer, self).__init__(args, model, tokenizer, optimizer)
        self.best_bleu = 0
        self.discriminator = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(args.pool_size*args.latent_dim, 2)).cuda()
        self.d_optim = args.optimizer(self.discriminator.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.criterion = torch.nn.CrossEntropyLoss().cuda()

    def save_model(self, epoch_metrics):
        if self.args.save_last_checkpoint:
            torch.save(self.model.state_dict(), f"{self.args.model_save_dir}/LastModel.pkl")
        if epoch_metrics['BLEU'] > self.best_bleu:
            self.best_bleu = epoch_metrics['BLEU']
            torch.save(self.model.state_dict(), f"{self.args.model_save_dir}/BestModel.pkl")
        print(f"Epoch{self.epoch} Loss: {epoch_metrics['Loss']} (best: {self.best_bleu})")

    def train_step(self, sent_tokens, waveform, pair):
        in_tokens, out_tokens, tags = sent_tokens[:, :-1], sent_tokens[:, 1:], pair["tags"]
        try:
            self.discriminator.requires_grad_(False)
            encoded_w = self.model.encode_wave(waveform)
            logits = self.model.decode_text(in_tokens, encoded_w)
            loss = self.criterion(logits.transpose(1, 2), out_tokens)
            loss -= 10*self.criterion(self.discriminator(encoded_w), torch.LongTensor([0]).cuda())

            encoded_t = self.model.encode_text(sent_tokens)
            logits = self.model.decode_text(in_tokens, encoded_t)
            loss += self.criterion(logits.transpose(1, 2), out_tokens)
            loss -= 10*self.criterion(self.discriminator(encoded_t), torch.LongTensor([1]).cuda())
            loss.backward()
            self.discriminator.requires_grad_(True)

            loss_d = self.criterion(self.discriminator(encoded_w.detach()), torch.LongTensor([0]).cuda())
            loss_d += self.criterion(self.discriminator(encoded_t.detach()), torch.LongTensor([1]).cuda())
            loss_d.backward()
            self.d_optim.step()
        except RuntimeError as e:
            print(e)
            return torch.tensor(0)
        return loss

    def valid(self, dataset, desc="Valid"):
        with torch.no_grad():
            self.model.eval()
            self.p_bar = tqdm.tqdm(dataset, desc=f"validation")
            sys, ref, losses = [], [], []
            pred, refer = [], []
            for sent_tokens, waveform, pair in self.p_bar:
                sent_tokens = sent_tokens["input_ids"].cuda()
                loss, (generated_text, classes) = self.valid_step(sent_tokens, waveform.cuda(), pair)
                losses.append(loss)
                sys.append(self.tokenizer.batch_decode(generated_text, skip_special_tokens=True))
                ref.append(self.tokenizer.decode(sent_tokens[0], skip_special_tokens=True))
                pred += classes
                refer += [0, 1]
                self.p_bar.set_postfix(loss=loss)
            f1 = f1_score(refer, pred)
            epoch_metrics = all_metrics(sys, ref, desc, epoch=self.epoch, Loss=sum(losses)/len(losses), F1=f1)
            self.model.train()
        return epoch_metrics

    def valid_step(self, sent_tokens, waveform, pair):
        n = self.args.generate_config.num_return_sequences
        in_tokens, out_tokens = sent_tokens[:, :-1], sent_tokens[:, 1:]
        try:
            encoded = self.model.encode_wave(waveform)
            logits = self.model.decode_text(in_tokens, encoded)
            loss = self.criterion(logits.transpose(1, 2), out_tokens)
            classes = [self.discriminator(encoded).argmax(dim=-1).item(), self.discriminator(self.model.encode_text(sent_tokens)).argmax(dim=-1).item()]
            generated = self.model.generate(waveform, self.tokenizer.cls_token_id, self.tokenizer.sep_token_id, self.args.generate_config)
        except RuntimeError as e:
            print(e)
            classes = [0, 1]
            return 0, ([[self.tokenizer.pad_token_id]]*n, classes)
        return loss.item(), (generated, classes)

    def generate(self):
        pass
