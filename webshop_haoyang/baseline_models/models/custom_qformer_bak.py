import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    BertModel,
    BertConfig,
    PretrainedConfig,
    PreTrainedModel,
)
from transformers.modeling_outputs import SequenceClassifierOutput
from .modules import EncoderRNN, BiAttention, get_aggregated
from transformers import Blip2ForConditionalGeneration, AutoProcessor, AutoTokenizer, Blip2Model, BlipModel, BlipTextModel
from PIL import Image


class QFormerConfigForWebshop(PretrainedConfig):
    model_type = "blip2"
    def __init__(
        self,
        pretrained_blip=True,
        image=False,
        **kwargs
    ):
        self.pretrained_blip = pretrained_blip
        self.image = image
        super().__init__(**kwargs)


class QFormerModelForWebshop(PreTrainedModel):

    config_class = QFormerConfigForWebshop

    def __init__(self, config, token_embed_size=30526, embedding_dimension = 2560, blip1 = False):
        super().__init__(config)
        self.blip = Blip2Model.from_pretrained("Salesforce/blip2-opt-2.7b")
        self.blip.language_model = None # save GPU memory
        for param in self.blip.vision_model.parameters(): # freeze ViT in BLIP-2
            param.requires_grad = False
        self.proj_layer = nn.Linear(768, 768) # hidden_dims of both qformer and bert are 768
        
        bert_config = BertConfig.from_pretrained('bert-base-uncased')
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.bert.resize_token_embeddings(30526)
        self.bert_dimension = 768
        self.image_emb_seqlen = 32
        # self.visual2bert = nn.Linear(self.visual_dimension, self.bert_dimension)

        self.attn = BiAttention(self.bert_dimension, 0.0)
        self.linear_1 = nn.Linear(self.bert_dimension * 4, self.bert_dimension)
        self.relu = nn.ReLU()
        self.linear_2 = nn.Linear(self.bert_dimension, 1)

        # for state value prediction, used in RL
        self.linear_3 = nn.Sequential(
            nn.Linear(self.bert_dimension, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, state_input_ids, state_attention_mask, action_input_ids, action_attention_mask, sizes, raw_images, labels=None):
        sizes = sizes.tolist()
        state_rep = self.bert(state_input_ids, attention_mask=state_attention_mask)[0]
        image_emb = self.blip.get_qformer_features(pixel_values=raw_images).last_hidden_state
        image_emb = self.proj_layer(image_emb)
        # print(state_rep.shape)
        # print(image_emb.shape)

        state_rep = torch.cat([image_emb, state_rep], dim=1)
        image_emb_mask = torch.ones(state_attention_mask.shape[0], self.image_emb_seqlen).cuda()
        state_attention_mask = torch.cat([image_emb_mask, state_attention_mask], dim=1)

        assert state_attention_mask.shape[1] == state_rep.shape[1]

        action_rep = self.bert(action_input_ids, attention_mask=action_attention_mask)[0]
        state_rep = torch.cat([state_rep[i:i+1].repeat(j, 1, 1) for i, j in enumerate(sizes)], dim=0)
        state_attention_mask = torch.cat([state_attention_mask[i:i+1].repeat(j, 1) for i, j in enumerate(sizes)], dim=0)
        act_lens = action_attention_mask.sum(1).tolist()
        state_action_rep = self.attn(action_rep, state_rep, state_attention_mask)
        state_action_rep = self.relu(self.linear_1(state_action_rep))
        act_values = get_aggregated(state_action_rep, act_lens, 'mean')
        act_values = self.linear_2(act_values).squeeze(1)

        logits = [F.log_softmax(_, dim=0) for _ in act_values.split(sizes)]

        loss = None
        if labels is not None:
            loss = - sum([logit[label] for logit, label in zip(logits, labels)]) / len(logits)
        
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
        )