from transformers import BertLMHeadModel, BertModel, BertConfig
import torch


class BertEncoder(BertModel):
    def __init__(self, layer=5):
        config = BertConfig.from_pretrained("bert-base-uncased", cache_dir="cache", num_hidden_layers=layer)
        super(BertEncoder, self).__init__(config)

    def forward(self, input_ids):
        return super(BertEncoder, self).forward(input_ids).last_hidden_state


class BertDecoder(BertLMHeadModel):
    def __init__(self, layer=5):
        config = BertConfig.from_pretrained("bert-base-uncased", cache_dir="cache")
        config.is_decoder, config.add_cross_attention, config.num_hidden_layers = True, True, layer
        super(BertDecoder, self).__init__(config)

    def prepare_inputs_for_generation(self, input_ids, encoder_hidden_states=None, **kwargs):
        if encoder_hidden_states is not None:
            encoder_hidden_states = encoder_hidden_states.expand(input_ids.shape[0], -1, -1)
        return {"input_ids": input_ids, "encoder_hidden_states": encoder_hidden_states}

    def load_pretrained_embedding(self, device):
        bert = BertModel.from_pretrained("bert-base-uncased", cache_dir="cache").to(device)
        bert.requires_grad_(False)
        def hook_forward(*args, **kwargs):
            with torch.no_grad():
                embedding = bert(kwargs["input_ids"]).last_hidden_state
            return embedding
        self.bert.embeddings.forward = hook_forward
