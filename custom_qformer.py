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
    
    # Expected shapes of forward():
    # Shape of state_input_ids: torch.Size([1, 512])
    # Shape of state_attention_mask: torch.Size([1, 512])
    # Shape of action_input_ids: torch.Size([12, 52])
    # Shape of action_attention_mask: torch.Size([12, 52])
    # Shape of raw_images: torch.Size([1, 3, 224, 224])
    
    # Adapted from BERT RL
    def rl_forward(self, state_batch, act_batch, value=False, q=False, act=False):
        act_values = []
        act_sizes = []
        values = []
        for state, valid_acts in zip(state_batch, act_batch):
            with torch.set_grad_enabled(not act):
                state_ids = torch.tensor([state.obs]).cuda()
                state_mask = (state_ids > 0).int()
                act_lens = [len(_) for _ in valid_acts]
                act_ids = [torch.tensor(_) for _ in valid_acts]
                act_ids = nn.utils.rnn.pad_sequence(act_ids, batch_first=True).cuda()
                act_mask = (act_ids > 0).int()
                act_size = torch.tensor([len(valid_acts)]).cuda()
                raw_images = state.raw_image.cuda()

                logits = self.forward(state_ids, state_mask, act_ids, act_mask, act_size, raw_images).logits[0]
                act_values.append(logits)
                act_sizes.append(len(valid_acts))
            if value:
                v = self.bert(state_ids, state_mask)[0]
                values.append(self.linear_3(v[0][0]))
        act_values = torch.cat(act_values, dim=0)
        act_values = torch.cat([F.log_softmax(_, dim=0) for _ in act_values.split(act_sizes)], dim=0)
        # Optionally, output state value prediction
        if value:
            values = torch.cat(values, dim=0)
            return act_values, act_sizes, values
        else:
            return act_values, act_sizes