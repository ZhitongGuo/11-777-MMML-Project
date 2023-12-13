import torch
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_ as __call_trunc_normal_
from torchscale.model.BEiT3 import BEiT3
from torchscale.architecture.config import EncoderConfig
from transformers.modeling_outputs import SequenceClassifierOutput

def trunc_normal_(tensor, mean=0., std=1.):
    __call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)
    
def _get_large_config(
        img_size=224, patch_size=16, drop_path_rate=0, 
        checkpoint_activations=None, mlp_ratio=4, vocab_size=64010, **kwargs
):
    return EncoderConfig(
        img_size=img_size, patch_size=patch_size, vocab_size=vocab_size, multiway=True, 
        layernorm_embedding=False, normalize_output=True, no_output_layer=True, 
        drop_path_rate=drop_path_rate, encoder_embed_dim=1024, encoder_attention_heads=16, 
        encoder_ffn_embed_dim=int(1024 * mlp_ratio), encoder_layers=24, 
        checkpoint_activations=checkpoint_activations, 
    )

def _get_base_config(
        img_size=224, patch_size=16, drop_path_rate=0, 
        checkpoint_activations=None, mlp_ratio=4, vocab_size=64010, **kwargs
):
    return EncoderConfig(
        img_size=img_size, patch_size=patch_size, vocab_size=vocab_size, multiway=True, 
        layernorm_embedding=False, normalize_output=True, no_output_layer=True, 
        drop_path_rate=drop_path_rate, encoder_embed_dim=768, encoder_attention_heads=12, 
        encoder_ffn_embed_dim=int(768 * mlp_ratio), encoder_layers=12, 
        checkpoint_activations=checkpoint_activations, 
    )
    
def get_aggregated(output, lens, method):
    """
    Get the aggregated hidden state of the encoder.
    B x D
    """
    if method == 'mean':
        return torch.stack([output[i, :j, :].mean(0) for i, j in enumerate(lens)], dim=0)
    elif method == 'last':
        return torch.stack([output[i, j-1, :] for i, j in enumerate(lens)], dim=0)
    elif method == 'first':
        return output[:, 0, :]
    
class BEiT3Wrapper(nn.Module):
    def __init__(self, args, **kwargs):
        super().__init__()
        self.args = args
        self.beit3 = BEiT3(args)
        self.apply(self._init_weights)
        #for _, parameters in self.beit3.named_parameters():
            #parameters.requires_grad = False

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def get_num_layers(self):
        return self.beit3.encoder.num_layers

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'beit3.encoder.embed_positions.A.weight', 'beit3.vision_embed.cls_token', 'logit_scale'}

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

class BiAttention(nn.Module):
    def __init__(self, input_size, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.input_linear = nn.Linear(input_size, 1, bias=False)
        self.memory_linear = nn.Linear(input_size, 1, bias=False)
        self.dot_scale = nn.Parameter(
            torch.zeros(size=(input_size,)).uniform_(1. / (input_size ** 0.5)),
            requires_grad=True)
        self.init_parameters()

    def init_parameters(self):
        return

    def forward(self, context, memory, mask):
        bsz, input_len = context.size(0), context.size(1)
        memory_len = memory.size(1)
        context = self.dropout(context)
        memory = self.dropout(memory)

        input_dot = self.input_linear(context)
        memory_dot = self.memory_linear(memory).view(bsz, 1, memory_len)
        cross_dot = torch.bmm(
            context * self.dot_scale,
            memory.permute(0, 2, 1).contiguous())
        att = input_dot + memory_dot + cross_dot
        if mask is not None:
            att = att - 1e30 * (1 - mask[:, None])

        weight_one = F.softmax(att, dim=-1)
        output_one = torch.bmm(weight_one, memory)
        weight_two = (F.softmax(att.max(dim=-1)[0], dim=-1)
                      .view(bsz, 1, input_len))
        output_two = torch.bmm(weight_two, context)
        return torch.cat(
            [context, output_one, context * output_one,
             output_two * output_one],
            dim=-1)


class Pooler(nn.Module):
    def __init__(self, input_features, output_features, norm_layer):
        super().__init__()
        self.norm = norm_layer(input_features)
        self.dense = nn.Linear(input_features, output_features)
        self.activation = nn.Tanh()

    def forward(self, x):
        cls_rep = x
        cls_rep = self.norm(cls_rep)
        pooled_output = self.dense(cls_rep)
        pooled_output = self.activation(pooled_output)
        return pooled_output
    
class BEiT3ForVisualQuestionAnswering(BEiT3Wrapper):
    def __init__(
            self, 
            args, 
            num_classes, 
            norm_layer=nn.LayerNorm, 
            **kwargs
    ):
        super(BEiT3ForVisualQuestionAnswering, self).__init__(args=args)
        embed_dim = args.encoder_embed_dim
        self.pooler = Pooler(
            input_features=embed_dim, 
            output_features=embed_dim, 
            norm_layer=norm_layer, 
        )
        self.pooler2 = Pooler(
            input_features=embed_dim, 
            output_features=embed_dim, 
            norm_layer=norm_layer, 
        )
        self.attn = BiAttention(embed_dim, 0.0)
        self.head = nn.Sequential(
            nn.Linear(embed_dim * 4, embed_dim * 2),
            norm_layer(embed_dim * 2),
            nn.GELU()
        )
        self.head.apply(self._init_weights)
        self.linear = nn.Linear(embed_dim * 2, num_classes)

    def forward(self, state_input_ids, state_attention_mask, action_input_ids, action_attention_mask, sizes, images=None, labels=None):
        sizes = sizes.tolist()
        action_input_ids = F.pad(action_input_ids, (0, 128 - action_input_ids.shape[-1]), value = 1)
        action_attention_mask = F.pad(action_attention_mask, (0, 128 - action_attention_mask.shape[-1]), value = 0)
        state_outputs = self.beit3(
            textual_tokens=state_input_ids, 
            visual_tokens=images, 
            text_padding_position=state_attention_mask, 
        )
        action_outputs = self.beit3(
            textual_tokens=action_input_ids, 
            visual_tokens=None, 
            text_padding_position=action_attention_mask, 
        )
        state_rep = state_outputs["encoder_embedding"]
        state_rep = self.pooler(state_rep)
        state_mask = state_outputs['encoder_padding_mask']
        state_rep = torch.cat([state_rep[i:i+1].repeat(j, 1, 1) for i, j in enumerate(sizes)], dim=0)
        state_mask = torch.cat([state_mask[i:i+1].repeat(j, 1) for i, j in enumerate(sizes)], dim=0)
        action_rep = self.pooler2(action_outputs['encoder_embedding'])
        cls_rep = self.attn(action_rep, state_rep, state_mask)
        ln = self.head(cls_rep)
        act_values = get_aggregated(ln, action_attention_mask.sum(1).tolist(), 'mean')
        act_values = self.linear(act_values).squeeze(1)
        logits = [F.log_softmax(_, dim=0) for _ in act_values.split(sizes)]
        loss = None
        if labels is not None:
            loss = -sum(
                logit[label] for logit, label in zip(logits, labels)
            ) / len(logits)
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
        )