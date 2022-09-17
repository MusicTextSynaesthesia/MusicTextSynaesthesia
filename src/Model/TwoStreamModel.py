import torch
from torch import nn
from Model.modules.CNNEncoder import CNNEncoder
from Model.modules.CNNDecoder import CNNDecoder
from Model.modules.BertModel import BertEncoder, BertDecoder
from Model.modules.Pooler import MeanPooler, AttentionPooler
from dataset import labels
from Model.utils import prefix_state_dict


class TwoStreamModel(nn.Module):
    def __init__(self, args):
        super(TwoStreamModel, self).__init__()
        self.args = args
        self.wave_encoder = CNNEncoder(self.args.music_dim, self.args.latent_dim)
        self.wave_decoder = CNNDecoder(args.music_dim)
        self.text_encoder = BertEncoder(layer=5)
        self.text_decoder = BertDecoder(layer=5)
        self.pool_wave = MeanPooler(args.latent_dim, args.pool_size)
        self.pool_text = AttentionPooler(args.latent_dim, args.pool_size)
        self.classifiers = nn.ModuleDict({k: nn.Linear(args.pool_size*args.latent_dim, len(labels[k])) for k in args.tags})

    def load_pretrained(self, args):
        wave_model = torch.load(f"checkpoints/MusicAutoEncoder_{args.model_name[-1]}/BestModel.pkl", "cpu")
        text_model = torch.load(f"checkpoints/TextAutoEncoder_{args.model_name[-1]}/BestModel.pkl", "cpu")
        self.wave_encoder.load_state_dict(prefix_state_dict(wave_model, "wave_encoder"))
        self.wave_decoder.load_state_dict(prefix_state_dict(wave_model, "wave_decoder"))
        self.text_encoder.load_state_dict(prefix_state_dict(text_model, "text_encoder"))
        self.text_decoder.load_state_dict(prefix_state_dict(text_model, "text_decoder"))
        self.pool_text.load_state_dict(prefix_state_dict(text_model, "pool_text"))

    def encode_text(self, tokens):
        encoded = self.text_encoder(tokens)
        encoded = self.pool_text(encoded)
        return encoded

    def decode_text(self, tokens, encoded):
        dec_out = self.text_decoder(tokens, encoder_hidden_states=encoded).logits
        return dec_out

    def encode_wave(self, waveform):
        encoded = self.wave_encoder(waveform)
        encoded = self.pool_wave(encoded)
        return encoded

    def decode_wave(self, encoded):
        decoded = self.wave_decoder(encoded)
        return decoded

    def classify(self, encoded):
        h = self.pool_wave(encoded).view(-1, self.args.pool_size*self.args.latent_dim)
        probs = {k: self.classifiers[k](h) for k in self.args.tags}
        return probs

    def generate(self, waveform, bos_token_id, eos_token_id, args, **kwargs):
        enc_output = self.wave_encoder(waveform)
        enc_output = self.pool_wave(enc_output)
        return self.text_decoder.generate(encoder_hidden_states=enc_output, max_length=args.max_len,
                                     bos_token_id=bos_token_id, eos_token_id=eos_token_id,
                                     do_sample=args.do_sample, num_beams=args.num_beams, top_k=args.top_k,
                                     num_return_sequences=args.num_return_sequences, **kwargs)
