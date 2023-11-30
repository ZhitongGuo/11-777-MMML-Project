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
import requests

device = "cuda" if torch.cuda.is_available() else "cpu"

# class DimensionReducer(nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super(DimensionReducer, self).__init__()
#         self.linear = nn.Linear(input_dim, output_dim)
    
#     def forward(self, x):
#         # Assume x has shape (batch_size, seq_len, input_dim)
#         batch_size, seq_len, _ = x.size()
#         # Reshape x to combine the batch and sequence dimensions
#         x = x.view(-1, self.linear.in_features)
#         # Apply the linear layer
#         x = self.linear(x)
#         # Reshape x to split the batch and sequence dimensions again
#         x = x.view(batch_size, seq_len, self.linear.out_features)
#         return x


class BlipConfigForWebshop(PretrainedConfig):
    # model_type = "bert"

    def __init__(
        self,
        pretrained_blip=True,
        image=False,
        **kwargs
    ):
        self.pretrained_blip = pretrained_blip
        self.image = image
        super().__init__(**kwargs)


# self.blipprocessor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
# url = https://upload.wikimedia.org/wikipedia/commons/4/49/A_black_image.jpg
# self.placement = Image.open(requests.get(url, stream=True).raw)
# processor(images=self.placement, text=prompt, return_tensors="pt").to(device, torch.float16)

# self.blip.resize_token_embeddings(token_embed_size)
class BlipModelForWebshop(PreTrainedModel):

    config_class = BlipConfigForWebshop

    def __init__(self, config, token_embed_size=50269, embedding_dimension = 2560, blip1 = False):
        super().__init__(config)
        self.blip = Blip2Model.from_pretrained("Salesforce/blip2-opt-2.7b")
        for param in self.blip.parameters():
            param.requires_grad = False
        # self.blip = Blip2Model.from_pretrained("Salesforce/blip2-flan-t5-xxl")
        # assert False
        self.blip.language_model.resize_token_embeddings(token_embed_size)
        self.embedding_dimension = 2560

        
        #self.embedding_dimension = embedding_dimension #blip1 is to be 768, blip2 is 2560
        # self.state_reducer = DimensionReducer(token_embed_size, embedding_dimension) #768 is the embedding dimension
        # self.action_reducer = nn.AvgPool1d(kernel_size=66, stride=66)

        self.attn = BiAttention(embedding_dimension, 0.0)
        self.linear_1 = nn.Linear(embedding_dimension * 4, embedding_dimension)
        self.relu = nn.ReLU()
        self.linear_2 = nn.Linear(embedding_dimension, 1)
        if config.image:
            self.image_linear = nn.Linear(512, embedding_dimension)
        else:
            self.image_linear = None

        # for state value prediction, used in RL
        self.linear_3 = nn.Sequential(
            nn.Linear(embedding_dimension, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, state_input_ids, state_attention_mask, action_input_ids, action_attention_mask, sizes, raw_images=None, labels=None):
        sizes = sizes.tolist()

        # if action_input_ids.shape[0] > 6:
        #     action_input_ids = action_input_ids[:6]
        #     action_attention_mask = action_attention_mask[:6]
        #     sizes = [6]
        state_rep = self.blip.language_model(state_input_ids, attention_mask=state_attention_mask, output_hidden_states= True, return_dict= True)["hidden_states"][-1] #last hidden states
        # print(state_rep.shape) #last hidden state, CausalLMOutputWithPast

        if images is not None and self.image_linear is not None:
            images = self.image_linear(images)
            state_rep = torch.cat([images.unsqueeze(1), state_rep], dim=1)
            state_attention_mask = torch.cat([state_attention_mask[:, :1], state_attention_mask], dim=1)

        action_rep = []
        for i in range(action_input_ids.shape[0]):
            action_rep.append(self.blip.language_model(action_input_ids[i].unsqueeze(dim=0), attention_mask=action_attention_mask[i].unsqueeze(dim=0), output_hidden_states= True, return_dict= True)["hidden_states"][-1])
        action_rep = torch.cat(action_rep, dim=0)
        # print(action_rep.shape)

        state_rep = torch.cat([state_rep[i:i+1].expand(j, -1, -1) for i, j in enumerate(sizes)], dim=0)
        state_attention_mask = torch.cat([state_attention_mask[i:i+1].expand(j, -1) for i, j in enumerate(sizes)], dim=0)
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