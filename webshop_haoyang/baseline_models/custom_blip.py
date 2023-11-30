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
from .modules import EncoderRNN, BiAttention, get_aggregated, SelfAttention


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

    def __init__(self, config, token_embed_size=50269, embedding_dimension = 768, blip1 = False):
        super().__init__(config)
        self.blip1 = blip1
        if self.blip1 == 2:
            self.blip = Blip2Model.from_pretrained("Salesforce/blip2-opt-2.7b")
            self.blip.language_model.resize_token_embeddings(token_embed_size)
            self.embedding_dimension = 2560
        else:
            self.blip = BlipTextModel.from_pretrained("Salesforce/blip-image-captioning-base") #, torch_dtype=torch.float16
            self.blip.resize_token_embeddings(token_embed_size)  #because some special tokens are added for the action size
            self.embedding_dimension = 768
        
        self.embedding_dimension = embedding_dimension #blip1 is to be 768, blip2 is 2560
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

    def forward(self, state_input_ids, state_attention_mask, action_input_ids, action_attention_mask, sizes, images=None, labels=None):
        sizes = sizes.tolist()
        # print("state_input_ids.shape, action_input_ids.shape", state_input_ids.shape, action_input_ids.shape)
        #torch.Size([1, 512, 768]) torch.Size([12, 50])

        if self.blip1 == True:
            print(self.blip1)
            state_rep = self.blip.language_model(state_input_ids, attention_mask=state_attention_mask, output_hidden_states= True, return_dict= True)["hidden_states"][-1] #last hidden states
            print(state_rep.shape) #last hidden state, CausalLMOutputWithPast
        else:
            print(self.blip1)
            state_rep = self.blip.encoder(state_input_ids.float(), attention_mask=state_attention_mask, output_hidden_states= True, return_dict= True)["hidden_states"][-1] #last hidden states
            print(state_rep.shape) #last hidden state, CausalLMOutputWithPast
        #encoding

        #get_text_features
        #torch.Size([1, 512, 50272]) torch.Size([1, 1, 512])

        # state_rep = state_rep.reshape(512, 245, 98)
        #torch.Size([1, 245, 50272]), (512, 768)

        # state_reducer = DimensionReducer(state_rep.shape[-1], 768)
        # state_rep = self.state_reducer(state_rep)
        # state_rep = state_rep[:, :, :768]
        # print("state_rep.shape, images.unsqueeze(1).shape", state_rep.shape, images.unsqueeze(1).shape)

        if images is not None and self.image_linear is not None:
            images = self.image_linear(images)
            state_rep = torch.cat([images.unsqueeze(1), state_rep], dim=1)
            state_attention_mask = torch.cat([state_attention_mask[:, :1], state_attention_mask], dim=1)
        
        if self.blip == 2:
            action_rep = self.blip.language_model(action_input_ids, attention_mask=action_attention_mask, output_hidden_states= True, return_dict= True)["hidden_states"][-1] #last hidden states
            print(action_rep.shape)
        else:
            action_rep = self.blip.encoder(action_input_ids, attention_mask=action_attention_mask, output_hidden_states= True, return_dict= True)["hidden_states"][-1] #last hidden states
            print(action_rep.shape)
        
        # print(action_rep.shape, "this is action_rep shape")
        # action_rep = action_rep[:, :, :768]
        # act_reducer = DimensionReducer(action_rep.shape[-1], 768)
        # print(action_rep.shape, "this is action_rep shape")
        # action_rep = self.action_reducer(action_rep)

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
