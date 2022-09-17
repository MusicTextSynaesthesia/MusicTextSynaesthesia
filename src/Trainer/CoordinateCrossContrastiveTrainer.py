import torch
import tqdm
import sacrebleu
import multiprocessing
from copy import deepcopy
from .CoordinateReconTrainer import CoordinateReconTrainer, wave_sample


def bleu_scores(args):
    texts, idx = args
    return [sacrebleu.corpus_bleu([texts[idx]], [[t]]).score for t in texts]


class CoordinateCrossContrastiveTrainer(CoordinateReconTrainer):
    def __init__(self, args, model, tokenizer, optimizer):
        super(CoordinateCrossContrastiveTrainer, self).__init__(args, model, tokenizer, optimizer)
        with torch.no_grad():
            assert args.similarity_metric.lower() in ("bert", "bleu")
            assert args.divergence_metric.lower() in ("mse", "r")
            if args.similarity_metric.lower() == "bert":
                model.load_pretrained(args)
                sentences = [sent_tokens.input_ids[:, :512].cuda() for sent_tokens, _, _ in self.train_dataset]
                self.reps = torch.cat([model.encode_text(s) for s in sentences]).view(-1, 20*768)
                self.reps = self.reps / torch.norm(self.reps, dim=1, keepdim=True)
                self.sims = self.reps @ self.reps.T
                self.indices = torch.argmax(self.sims, dim=1)
            elif args.similarity_metric.lower() == "bleu":
                pool = multiprocessing.Pool(8)
                sentences = [p["sent_text"] for _, _, p in self.train_dataset]
                self.sims = [*pool.imap(bleu_scores, tqdm.tqdm([(sentences, i) for i in range(len(sentences))], desc="Building BLEU Similarity"))]
                self.sims = torch.tensor(self.sims).cuda() / 100

    def cross_loss(self, p, q):
        if self.args.divergence_metric.lower() == "mse":
            p, q = p.mul(self.args.temperature).softmax(0), q.mul(self.args.temperature).softmax(0)
            loss = self.mse(p, q)
        else:
            loss = -((p*q).mean() - p.mean()*q.mean()) / (p.std(False) * q.std(False) + 1e-8)
        return loss

    def train_step(self, sent_tokens, waveform, pair):
        try:
            loss, encoded, _ = self.model_forward(sent_tokens, waveform, pair)
            # indices = torch.randint(len(self.train_dataset), (self.args.n_samples,)).cuda()
            indices = torch.tensor([i % len(self.sims) for i in range(self.i, self.i-self.args.n_samples, -1)]).cuda()
            q = self.sims[self.i].index_select(0, indices)
            waves = [self.train_dataset[i][1] for i in indices]
            with torch.no_grad():
                waves_enc = torch.cat([self.model.pool_wave(self.model.wave_encoder(w)) for w in waves])
            waves_enc = waves_enc.view(self.args.n_samples, 20 * 768)
            encoded = encoded.view(1, -1)
            p = (encoded * waves_enc / torch.norm(encoded) / torch.norm(waves_enc, dim=1, keepdim=True)).sum(dim=-1)
            loss += self.args.cross_weight * self.cross_loss(p, q)

            loss.backward()
        except RuntimeError as e:
            print(e)
            return torch.tensor(0)
        return loss


class CoordinatePairwiseContrastiveTrainer(CoordinateCrossContrastiveTrainer):
    def train_step(self, sent_tokens, waveform, pair):
        try:
            loss, encoded, _ = self.model_forward(sent_tokens, waveform, pair)
            index = self.sims[self.i].argmax()
            wave = self.train_dataset[index][1]
            waves_enc = self.model.pool_wave(self.model.wave_encoder(wave)).view(1, -1)
            encoded = encoded.view(1, -1)
            p = (encoded * waves_enc / torch.norm(encoded) / torch.norm(waves_enc, dim=1, keepdim=True)).sum()
            loss += self.args.cross_weight * self.cross_loss(p, self.sims[self.i][index])
            loss.backward()
        except RuntimeError as e:
            print(e)
            return torch.tensor(0)
        return loss


class CoordinateTripletContrastiveTrainer(CoordinateCrossContrastiveTrainer):
    def train_step(self, sent_tokens, waveform, pair):
        try:
            loss, encoded, _ = self.model_forward(sent_tokens, waveform, pair)
            encoded = encoded.view(1, -1)
            p = self.train_dataset[self.sims[self.i].argmax()][1]
            p = self.model.pool_wave(self.model.wave_encoder(p)).view(1, -1)
            p = (encoded * p / torch.norm(encoded) / torch.norm(p)).sum()
            n = self.train_dataset[torch.randint(len(self.train_dataset), (1,))][1]
            n = self.model.pool_wave(self.model.wave_encoder(n)).view(1, -1)
            n = (encoded * n / torch.norm(encoded) / torch.norm(n)).sum()
            loss += self.args.cross_weight * max(n - p + self.args.margin, 0)
            # loss += self.args.cross_weight * max(p - n + self.args.margin, 0)
            loss.backward()
        except RuntimeError as e:
            print(e)
            return torch.tensor(0)
        return loss


class CoordinateGTPContrastiveTrainer(CoordinateCrossContrastiveTrainer):
    def __init__(self, args, model, tokenizer, optimizer):
        super(CoordinateGTPContrastiveTrainer, self).__init__(args, model, tokenizer, optimizer)
        self.memory_bank, self.indices = [], []
        self.momentum_pooler = deepcopy(model.pool_wave).cuda()
        self.momentum_encoder = deepcopy(model.wave_encoder).cuda()

    def momentum_step(self):
        for e, m in [(self.model.pool_wave, self.momentum_pooler), (self.model.wave_encoder, self.momentum_encoder)]:
            for p, pm in zip(e.parameters(), m.parameters()):
                pm.data = pm.data * self.args.momentum + p.data * (1 - self.args.momentum)

    def train_step(self, sent_tokens, waveform, pair):
        try:
            loss, encoded, _ = self.model_forward(sent_tokens, waveform, pair)
            with torch.no_grad():
                self.memory_bank.append(self.momentum_pooler(self.momentum_encoder(waveform)).view(1, -1))
                self.indices.append(self.i)
                self.memory_bank = self.memory_bank[-self.args.n_samples:]
                q = self.sims[self.i].index_select(0, torch.LongTensor(self.indices).cuda()[-self.args.n_samples:])
                waves_enc = torch.cat(self.memory_bank, dim=0)
            encoded = encoded.view(1, -1)
            if len(self.memory_bank) > 1:
                p = (encoded * waves_enc / torch.norm(encoded) / torch.norm(waves_enc, dim=1, keepdim=True)).sum(dim=-1)
                loss += self.args.cross_weight * self.cross_loss(p, q)

            self.momentum_step()
            loss.backward()
        except RuntimeError as e:
            print(e)
            return torch.tensor(0)
        return loss
