import torch
from torch import nn
from Model.modules.CNNEncoder import CNNEncoder
from Model.modules.BertModel import BertDecoder
from Model.modules.Pooler import MeanPooler
from Model.utils import prefix_state_dict


class EncoderDecoder(nn.Module):
    def __init__(self, args):
        super(EncoderDecoder, self).__init__()
        self.args = args
        self.wave_encoder = CNNEncoder(self.args.music_dim, self.args.latent_dim)
        self.pool_wave = MeanPooler(args.latent_dim, args.pool_size)
        self.text_decoder = BertDecoder(layer=5)

    def load_pretrained(self, args):
        wave_model = torch.load(f"checkpoints/MusicAutoEncoder_{args.model_name[-1]}/BestModel.pkl", "cpu")
        text_model = torch.load(f"checkpoints/TextAutoEncoder_{args.model_name[-1]}/BestModel.pkl", "cpu")
        self.wave_encoder.load_state_dict(prefix_state_dict(wave_model, "wave_encoder"))
        self.pool_wave.load_state_dict(prefix_state_dict(wave_model, "pool_wave"))
        self.text_decoder.load_state_dict(prefix_state_dict(text_model, "text_decoder"))

    def forward(self, waveform, input_ids):
        enc_output = self.wave_encoder(waveform)
        enc_output = self.pool_wave(enc_output)
        output = self.text_decoder(input_ids=input_ids, encoder_hidden_states=enc_output).logits
        return output

    def generate(self, waveform, bos_token_id, eos_token_id, args, **kwargs):
        enc_output = self.wave_encoder(waveform)
        enc_output = self.pool_wave(enc_output)
        return self.text_decoder.generate(encoder_hidden_states=enc_output, max_length=args.max_len,
                                     bos_token_id=bos_token_id, eos_token_id=eos_token_id,
                                     do_sample=args.do_sample, num_beams=args.num_beams, top_k=args.top_k,
                                     num_return_sequences=args.num_return_sequences, **kwargs)
