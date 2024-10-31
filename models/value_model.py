import torch
from torch import nn
from torch.nn.functional import softmax
from transformers import LlamaForCausalLM
import gc
import torch.nn.functional as F 

class ValueModel(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)

        self.num_padding_at_beginning = 0
        if hasattr(self.config, "word_embed_proj_dim"):
            self.v_head = nn.Linear(self.config.word_embed_proj_dim,
                                    1,
                                    bias=False)
        else:
            self.config.n_embd = self.config.hidden_size if hasattr(
                self.config, "hidden_size") else self.config.n_embd
            self.v_head = nn.Linear(self.config.n_embd, 1, bias=False)
        self.PAD_ID = 2


    def forward(self,
                input_ids=None,
                past_key_values=None,
                attention_mask=None,
                label=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                use_cache=False,
                rank=None):
        if self.config.model_type == "llama":
            kwargs = dict()
        else:
            kwargs = dict(head_mask=head_mask)

        transformer_outputs = self.model(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs)

        hidden_states = transformer_outputs[0]
        rewards = self.v_head(hidden_states).squeeze(-1)

        bs = input_ids.shape[0]
        values = torch.zeros(bs, device=input_ids.device)

        seq_len = input_ids.shape[1]

        for i in range(bs):
            input_id = input_ids[i]
            reward = rewards[i]
            c_inds = (input_id == self.PAD_ID).nonzero()
            # assert self.PAD_ID == 0
            c_ind = c_inds[self.num_padding_at_beginning].item() if len(
                c_inds
            ) > self.num_padding_at_beginning else seq_len
            # Fill the values tensor with the end scores
            values[i] = reward[c_ind - 1]

        return {
            "values": values,
        }

