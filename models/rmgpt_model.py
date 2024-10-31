import torch
from torch import nn

from transformers import LlamaForCausalLM


def linear_activation(x):
    return x


ACT2FN = {
    "linear": linear_activation,
    "sigmoid": torch.nn.Sigmoid(),
}


class RMGPTModel(LlamaForCausalLM):

    def __init__(self, config):
        super().__init__(config)

        rm_head_intermediate_size = 100
        self.rm_layer1 = nn.Linear(config.hidden_size, rm_head_intermediate_size, bias=True)
        self.rm_intermediate_activation = torch.nn.functional.gelu
        self.rm_layer2 = nn.Linear(rm_head_intermediate_size, 1, bias=True)

        self.activation_func = ACT2FN["sigmoid"]

    def forward(self,
                input_ids=None,
                past_key_values=None,
                attention_mask=None,
                label=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                use_cache=False,
                pad_id=32004,
                loss_mask=None,
                rank=None):
        if self.config.model_type == "llama":
            kwargs = dict()
        else:
            kwargs = dict(head_mask=head_mask)

        transformer_outputs = self.model(input_ids,
                                         past_key_values=past_key_values,
                                         attention_mask=attention_mask,
                                         inputs_embeds=inputs_embeds,
                                         use_cache=use_cache,
                                         **kwargs)

        hidden_states = transformer_outputs[0]

        rm_layer1_output = self.rm_layer1(hidden_states)
        rm_layer1_output = self.rm_intermediate_activation(rm_layer1_output)
        rm_layer_outputs = self.rm_layer2(rm_layer1_output)  # check shape

        rm_layer_outputs = self.activation_func(rm_layer_outputs)

        row_index = torch.arange(rm_layer_outputs.shape[0], dtype=torch.long, device=rm_layer_outputs.device)
        col_index = loss_mask.sum(dim=1).long() - 1
        # move col_index and row_index to the same device
        col_index = col_index.to(rm_layer_outputs.device)
        rm_outputs = rm_layer_outputs[row_index, col_index].contiguous()  # [bs]

        rm_outputs = rm_outputs.view(-1)
        return {
            "values": rm_outputs,
        }