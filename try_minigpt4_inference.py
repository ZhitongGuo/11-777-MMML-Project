from PIL import Image
import torch

from minigpt4.models.minigpt4 import MiniGPT4
from minigpt4.processors.blip_processors import Blip2ImageTrainProcessor, Blip2ImageEvalProcessor
from transformers import StoppingCriteriaList
from minigpt4.conversation.conversation import StoppingCriteriaSub

# hardcoded configs
model_config = {
    'arch': 'minigpt4',
    'image_size': 224,
    'drop_path_rate': 0,
    'use_grad_checkpoint': False,
    'vit_precision': 'fp16',
    'freeze_vit': True,
    'freeze_qformer': True,
    'num_query_token': 32,
    'prompt': '',
    'llama_model': 'vicuna_7b',
    'model_type': 'pretrain_vicuna0',
    'max_txt_len': 160,
    'end_sym': '###',
    'low_resource': True,
    'prompt_template': '###Human: {} ###Assistant: ',
    'ckpt': 'prerained_minigpt4_7b.pth',
    'device_8bit': 0
}
gpu_id = 0
device = torch.device("cuda:{}".format(gpu_id))

# LLM hyper-params
max_new_tokens=300
num_beams=1
min_length=1
top_p=0.9
repetition_penalty=1.05
length_penalty=1
temperature=1.0
max_length=2000

model = MiniGPT4.from_config(model_config).to('cuda:{}'.format(gpu_id))
train_processor = Blip2ImageTrainProcessor.from_config({'name': 'blip2_image_train', 'image_size': 224})
eval_processor = Blip2ImageEvalProcessor.from_config({'name': 'blip2_image_eval', 'image_size': 224})
# print(model)
# print(train_processor)
# print(eval_processor)

# Adapted from minigpt4/conversation/conversation.py
# image is a path
def preprocess_img(image, processor):
    raw_image = Image.open(image).convert('RGB')
    return processor(raw_image).unsqueeze(0).to(device)

image = preprocess_img("downloaded_image.jpg", eval_processor)
image_emb, _ = model.encode_img(image) # image --> ViT --> Q-Former --> image_emb
# print(image)
print(image.shape)
# print(image_emb)
print(image_emb.shape)

prompt = """Give the following image: <Img>ImageContent</Img>. 
    You will be able to see the image once I provide it to you. 
    Please answer my questions. ###Human: <Img><ImageHere></Img> """
question = "Does this product describe a black women's coat? "
suffix = "###Assistant: "
text = prompt + question + suffix

embs = model.get_context_emb(text, [image_emb]) # batch_size, seq_len, hidden_dim
print(embs.shape)
current_max_len = embs.shape[1] + max_new_tokens
begin_idx = max(0, current_max_len - max_length)
embs = embs[:, begin_idx:]
print(embs.shape)

# copied from demo.py
stop_words_ids = [[835], [2277, 29937]]
stop_words_ids = [torch.tensor(ids).to(device='cuda:{}'.format(gpu_id)) for ids in stop_words_ids]
stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

generation_kwargs = dict(
    inputs_embeds=embs,
    max_new_tokens=max_new_tokens,
    stopping_criteria=stopping_criteria,
    num_beams=num_beams,
    do_sample=True,
    min_length=min_length,
    top_p=top_p,
    repetition_penalty=repetition_penalty,
    length_penalty=length_penalty,
    temperature=float(temperature),
)
print(generation_kwargs)

def model_generate(*args, **kwargs):
    # for 8 bit and 16 bit compatibility
    with model.maybe_autocast():
        output = model.llama_model.generate(*args, **kwargs)
    return output

output_token = model_generate(**generation_kwargs)[0]
output_text = model.llama_tokenizer.decode(output_token, skip_special_tokens=True)
output_text = output_text.split('###')[0]  # remove the stop sign '###'
output_text = output_text.split('Assistant:')[-1].strip()
print("Output Text: {}".format(output_text))
print("Output Text: {}".format(output_text))