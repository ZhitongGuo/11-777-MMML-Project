import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    BertModel,
    BertConfig,
    PretrainedConfig,
    PreTrainedModel,
    Blip2Processor,
    Blip2Model
)
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from .modules import EncoderRNN, BiAttention, get_aggregated
from .bert import BertConfigForWebshop
import requests # pull images from url


class BlipForWebshop(PreTrainedModel):

    def __init__(self, config, token_embed_size=32128):
        super().__init__(config)

        self.processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        self.blip = Blip2Model.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16)
        self.blip.language_model.resize_token_embeddings(token_embed_size)

        embedding_size = 1024 # hardcode it to BLIP-2's actual size
        self.attn = BiAttention(embedding_size, 0.0)
        self.final_layers = torch.nn.Sequential(
            nn.Linear(embedding_size * 4, embedding_size),
            nn.ReLU(),
            nn.Linear(embedding_size, 1)
        )

        # # Usage:
        # inputs = processor(images=image, text=prompt, return_tensors="pt").to(device, torch.float16)
        # outputs = model(**inputs)
        
        # self.attn = BiAttention(embedding_size, 0.0)
        # self.linear_1 = nn.Linear(embedding_size * 4, embedding_size)
        # self.relu = nn.ReLU()
        # self.linear_2 = nn.Linear(embedding_size, 1)
        # if config.image:
        #     self.image_linear = nn.Linear(512, embedding_size)
        # else:
        #     self.image_linear = None

        # # for state value prediction, used in RL
        # self.linear_3 = nn.Sequential(
        #         nn.Linear(embedding_size, 128),
        #         nn.LeakyReLU(),
        #         nn.Linear(128, 1),
        #     )
    
    def forward(self, state_encodings, action_encodings, sizes, images, labels):
        state_rep = self.blip(**state_encodings)
        action_rep = self.blip(**action_encodings)
        print("state_rep.shape={}".format(state_rep.shape))
        print("action_rep.shape={}".format(action_rep.shape))
        return state_rep
    

    # def forward(self, state_input_ids, state_attention_mask, action_input_ids, action_attention_mask, sizes, images=None, labels=None):
    #     sizes = sizes.tolist()
    #     # print(state_input_ids.shape, action_input_ids.shape)
    #     state_rep = self.t5.encoder(state_input_ids, attention_mask=state_attention_mask)[0]
    #     if images is not None and self.image_linear is not None:
    #         images = self.image_linear(images)
    #         state_rep = torch.cat([images.unsqueeze(1), state_rep], dim=1)
    #         state_attention_mask = torch.cat([state_attention_mask[:, :1], state_attention_mask], dim=1)
    #     action_rep = self.t5.encoder(action_input_ids, attention_mask=action_attention_mask)[0]
    #     state_rep = torch.cat([state_rep[i:i+1].repeat(j, 1, 1) for i, j in enumerate(sizes)], dim=0)
    #     state_attention_mask = torch.cat([state_attention_mask[i:i+1].repeat(j, 1) for i, j in enumerate(sizes)], dim=0)
    #     act_lens = action_attention_mask.sum(1).tolist()
    #     state_action_rep = self.attn(action_rep, state_rep, state_attention_mask)
    #     state_action_rep = self.relu(self.linear_1(state_action_rep))
    #     act_values = get_aggregated(state_action_rep, act_lens, 'mean')
    #     act_values = self.linear_2(act_values).squeeze(1)

    #     logits = [F.log_softmax(_, dim=0) for _ in act_values.split(sizes)]

    #     loss = None
    #     if labels is not None:
    #         loss = - sum([logit[label] for logit, label in zip(logits, labels)]) / len(logits)
        
    #     return SequenceClassifierOutput(
    #         loss=loss,
    #         logits=logits,
    #     )

    # def rl_forward(self, state_batch, act_batch, value=False, q=False, act=False):
    #     act_values = []
    #     act_sizes = []
    #     values = []
    #     for state, valid_acts in zip(state_batch, act_batch):
    #         with torch.set_grad_enabled(not act):
    #             state_ids = torch.tensor([state.obs]).cuda()
    #             state_mask = (state_ids > 0).int()
    #             act_lens = [len(_) for _ in valid_acts]
    #             act_ids = [torch.tensor(_) for _ in valid_acts]
    #             act_ids = nn.utils.rnn.pad_sequence(act_ids, batch_first=True).cuda()
    #             act_mask = (act_ids > 0).int()
    #             act_size = torch.tensor([len(valid_acts)]).cuda()
    #             if self.image_linear is not None:
    #                 images = [state.image_feat]
    #                 images = [torch.zeros(512) if _ is None else _ for _ in images] 
    #                 images = torch.stack(images).cuda()  # BS x 512
    #             else:
    #                 images = None
    #             logits = self.forward(state_ids, state_mask, act_ids, act_mask, act_size, images=images).logits[0]
    #             act_values.append(logits)
    #             act_sizes.append(len(valid_acts))
    #         if value:
    #             v = self.bert(state_ids, state_mask)[0]
    #             values.append(self.linear_3(v[0][0]))
    #     act_values = torch.cat(act_values, dim=0)
    #     act_values = torch.cat([F.log_softmax(_, dim=0) for _ in act_values.split(act_sizes)], dim=0)
    #     # Optionally, output state value prediction
    #     if value:
    #         values = torch.cat(values, dim=0)
    #         return act_values, act_sizes, values
    #     else:
    #         return act_values, act_sizes