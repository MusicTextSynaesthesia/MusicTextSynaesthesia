import torch
from torch import nn
from Model.modules.BertModel import BertEncoder, BertDecoder


class TextAutoEncoder(nn.Module):
    def __init__(self, args):
        super(TextAutoEncoder, self).__init__()
        self.text_encoder = BertEncoder(layer=5)
        self.pool_text = args.pooler(args.latent_dim, args.pool_size)

        self.text_decoder = BertDecoder(layer=5)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        encoded = self.text_encoder(x)
        encoded = self.pool_text(encoded)
        encoded = encoded*0.9 + torch.rand_like(encoded)*0.1
        dec_out = self.text_decoder(x, encoder_hidden_states=encoded).logits
        return dec_out

    def generate_text(self, x, bos_token_id, eos_token_id, args, **kwargs):
        encoded = self.text_encoder(x)
        encoded = self.pool_text(encoded)
        encoded = encoded*0.9 + torch.rand_like(encoded)*0.1
        return self.text_decoder.generate(encoder_hidden_states=encoded, max_length=args.max_len,
                                     bos_token_id=bos_token_id, eos_token_id=eos_token_id,
                                     do_sample=args.do_sample, num_beams=args.num_beams, top_k=args.top_k,
                                     num_return_sequences=args.num_return_sequences, **kwargs)
