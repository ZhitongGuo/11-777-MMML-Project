# load WebEnV
import os
import sys
import json
from train_rl import parse_args as webenv_args
from env import WebEnv  # TODO: just use webshopEnv?
import torch
from models.custom_qformer import QFormerConfigForWebshop, QFormerModelForWebshop
from PIL import Image
import torch
from train_choice_il_qformer import *
from transformers import BartForConditionalGeneration, BartTokenizer
from tqdm import tqdm
from functools import partial
import random
from minigpt4.models.minigpt4 import MiniGPT4
from minigpt4.processors.blip_processors import Blip2ImageTrainProcessor, Blip2ImageEvalProcessor
from transformers import StoppingCriteriaList
from minigpt4.conversation.conversation import StoppingCriteriaSub

IMAGE_PATH = "../all_images"

''' for debug'''
import os
print(os.getcwd())
os.chdir('baseline_models/')
print(os.getcwd())

# TODO: give an exmaple with images. one-shot example prompt
init_prompt = """Instruction:
i would like a 3 ounce bottle of bright citrus deodorant for sensitive skin, and price lower than 50.00 dollars
[Search]

Action: search[3 ounce bright citrus deodorant sensitive skin]
Observation:
[Back to Search]
Page 1 (Total results: 50)
[Next >]
[B078GWRC1J]
Bright Citrus Deodorant by Earth Mama | Natural and Safe for Sensitive Skin, Pregnancy and Breastfeeding, Contains Organic Calendula 3-Ounce
$10.99
[B078GTKVXY]
Ginger Fresh Deodorant by Earth Mama | Natural and Safe for Sensitive Skin, Pregnancy and Breastfeeding, Contains Organic Calendula 3-Ounce
$10.99
[B08KBVJ4XN]
Barrel and Oak - Aluminum-Free Deodorant, Deodorant for Men, Essential Oil-Based Scent, 24-Hour Odor Protection, Cedar & Patchouli Blend, Gentle on Sensitive Skin (Mountain Sage, 2.7 oz, 2-Pack)
$15.95

Action: think[B078GWRC1J and B078GTKVXY are bright citrus deodorant less then 50 dollars. I can check B078GWRC1J first.]
Observation: OK.

Action: click[B078GWRC1J]
Observation:
[Back to Search]
[< Prev]
scent [assorted scents][bright citrus][calming lavender][ginger fresh][simply non-scents]
size [travel set (4-pack)][3 ounce (pack of 1)][3-ounce (2-pack)]
Bright Citrus Deodorant by Earth Mama | Natural and Safe for Sensitive Skin, Pregnancy and Breastfeeding, Contains Organic Calendula 3-Ounce
Price: $10.99
Rating: N.A.
[Description]
[Features]
[Reviews]
[Buy Now]

Action: think[For 3 ounce bottle of bright citrus deodorant for sensitive skin, the item has options 'bright citrus' and '3 ounce (pack of 1)' and seems good to buy.]
Observation: OK.

Action: click[bright citrus]
Observation: You have clicked bright citrus.

Action: click[3 ounce (pack of 1)]
Observation: You have clicked 3 ounce (pack of 1).

Action: click[Buy Now]
"""

FEAT_CONV = '/home/haoyang/webshop/data/feat_conv.pt'
feat_conv = torch.load(FEAT_CONV)
cache = {"asin2name": None, "name2asin": None}
DEFAULT_FILE_PATH = "../data/items_shuffle.json"
# We want to map image url to its product ASIN because in WebEnv we only have the URL
# and we need ASIN to access downloaded images
url2asin = {}
f = open(DEFAULT_FILE_PATH, 'r')
data = json.load(f)
for d in data:
    if ('product_information' not in d) or ('images' not in d) or ('asin' not in d):
        continue
    url = d['images'][0]
    if len(url) == 0 and len(d['images']) > 1:
        url = d['images'][1]
    url2asin[url] = d['asin'].upper()
data = None # save RAM
f.close()
args = webenv_args()[0]
env = WebEnv(args, split='test', feat_conv=feat_conv, cache=cache, url2asin=url2asin)
print('env loaded')

# load Model
bart_tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
image_processor = Blip2ImageEvalProcessor.from_config({'name': 'blip2_image_eval', 'image_size': 224})
my_data_collator = partial(data_collator, image_processor=image_processor)


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")
    parser.add_argument("--model_path", type=str, default="./ckpts/web_click/epoch_9/model.pth", help="Where to store the final model.")
    parser.add_argument("--mem", type=int, default=0, help="State with memory")
    parser.add_argument("--bart_path", type=str, default='./ckpts/web_search/checkpoint-800', help="BART model path if using it")
    parser.add_argument("--bart", type=bool, default=True, help="Flag to specify whether to use bart or not (default: True)")
    parser.add_argument("--image", type=bool, default=True, help="Flag to specify whether to use image or not (default: True)")
    parser.add_argument("--softmax", type=bool, default=True, help="Flag to specify whether to use softmax sampling or not (default: True)")
    parser.add_argument("--model_name", type=str, default="qformer")
    args = parser.parse_args()

    return args


def bart_predict(input, model, skip_special_tokens=True, **kwargs):
    input_ids = bart_tokenizer(input)['input_ids']
    input_ids = torch.tensor(input_ids).unsqueeze(0)
    output = model.generate(input_ids, max_length=512, **kwargs)
    return bart_tokenizer.batch_decode(output.tolist(), skip_special_tokens=skip_special_tokens)


def generate_prompt(exprompt, i, observation, action=None):
    if i:
      exprompt += f' {action}\nObservation: {observation}\n\nAction:'
      prompt = exprompt + f' {action}\nObservation: {observation}\n\n<Img><ImageHere></Img>\n\n###Action: '
    else:
      exprompt += f'{observation}\n\nAction:'
      prompt = exprompt + f'{observation}\n\n<Img><ImageHere></Img>\n\n###Action: '
    
    return prompt, exprompt


def predict_v(obs, info, model, prompt, idx, softmax=False, rule=False, bart_model=None):
    valid_acts = info['valid']
    if valid_acts[0].startswith('search['):
        if bart_model is None:
            return valid_acts[-1]
        else:
            goal = process_goal(obs)
            query = bart_predict(goal, bart_model, num_return_sequences=5, num_beams=5)
            # query = random.choice(query)  # in the paper, we sample from the top-5 generated results.
            query = query[0]  #... but use the top-1 generated search will lead to better results than the paper results.
            return f'search[{query}]'
            
    if rule:
        item_acts = [act for act in valid_acts if act.startswith('click[item - ')]
        if item_acts:
            return item_acts[0]
        else:
            assert 'click[buy now]' in valid_acts
            return 'click[buy now]'
                
    states = process(obs)
    actions = list(map(process, valid_acts))
    raw_image_asin = info.get('raw_image').upper()

    eval_processor = Blip2ImageEvalProcessor.from_config({'name': 'blip2_image_eval', 'image_size': 224})
    if raw_image_asin == "NONE":
        print("No image")
        image = eval_processor(Image.new('RGB', (224, 224), (255, 255, 255))).unsqueeze(0).cuda()
    else:
        try:
            pil_image = Image.open(os.path.join(IMAGE_PATH, raw_image_asin + ".jpg"))
            image = eval_processor(pil_image).unsqueeze(0).cuda()
            pil_image.save(f"{idx}.jpg")
            print("Image saved to ")
        except:
            image = eval_processor(Image.new('RGB', (224, 224), (255, 255, 255))).unsqueeze(0).cuda()
            print("This is rare")
    image_emb, _ = model.encode_img(image)
    text_length = 5600
    text = init_prompt + prompt[-(text_length-len(init_prompt)):]
    print("*" * 10)
    print(text)

    # LLM hyper-params
    max_new_tokens=300
    num_beams=1
    min_length=1
    top_p=0.9
    repetition_penalty=1.05
    length_penalty=1
    temperature=1.0
    max_length=2000

    embs = model.get_context_emb(text, [image_emb]) # batch_size, seq_len, hidden_dim
    current_max_len = embs.shape[1] + max_new_tokens
    begin_idx = max(0, current_max_len - max_length)
    embs = embs[:, begin_idx:]
    print("embs.shape={}".format(embs.shape))
    
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

    def model_generate(*args, **kwargs):
        # for 8 bit and 16 bit compatibility
        with model.maybe_autocast():
            output = model.llama_model.generate(*args, **kwargs)
        return output
    
    output_token = model_generate(**generation_kwargs)[0]
    output_text = model.llama_tokenizer.decode(output_token, skip_special_tokens=True) # skip set to false?
    # output_text = output_text.split('###')[0]  # remove the stop sign '###'
    # output_text = output_text.split('Assistant:')[-1].strip()
    print("{}'s Output Text: {}".format(idx, output_text))
    
    action = output_text.lstrip(' ')
    print("**action: ", action)
    print("*" * 10)
    return action


def episode(model, idx=None, verbose=False, softmax=False, rule=False, bart_model=None):
    obs, info = env.reset(idx)
    if verbose:
        print(info['goal'])
    exprompt = ''
    action = None
    for i in range(100):
        prompt, exprompt = generate_prompt(exprompt, i, obs, action)
        print("new prompt: ", exprompt)
        if action and action.startswith('think'):
            obs = 'OK.'
        action = predict_v(obs, info, model, prompt, idx, softmax=softmax, rule=rule, bart_model=bart_model)
        if verbose:
            print(action)
        obs, reward, done, info = env.step(action)
        if done:
            return reward
    return 0


if __name__ == "__main__":
    args = parse_args()
    if args.mem:
        env.env.num_prev_obs = 1
        env.env.num_prev_actions = 5
        print('memory')
    else:
        env.env.num_prev_obs = 0
        env.env.num_prev_actions = 0
        print('no memory')
    
    if args.bart:
        bart_model = BartForConditionalGeneration.from_pretrained(args.bart_path)
        print('bart model loaded', args.bart_path)
    else:
        bart_model = None

    print("Load minigpt4")
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
        'prompt_template': '###Webshop Task: {} ###Assistant: ',
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
    print("Model loaded")

    scores_softmax, scores_rule = [], []
    bar = tqdm(total=500)
    for i in range(500):
        print(i)
        score_softmax = episode(model, idx=i, softmax=args.softmax, bart_model=bart_model)
        score_rule = episode(model, idx=i, rule=True)
        # print(i, '|', score_softmax * 10, score_rule * 10)  # env score is 0-10, paper is 0-100
        scores_softmax.append(score_softmax)
        scores_rule.append(score_rule)
        bar.update(1)
    bar.close()
    score_softmax = sum(scores_softmax) / len(scores_softmax)
    score_rule = sum(scores_rule) / len(scores_rule)
    harsh_softmax = len([s for s in scores_softmax if s == 10.0])
    harsh_rule = len([s for s in scores_rule if s == 10.0])
    print('------')
    print('avg test score (model, rule):', score_softmax * 10, score_rule * 10)
    print('avg test success rate % (model, rule):', harsh_softmax / 500 * 100, harsh_rule / 500 * 100)
