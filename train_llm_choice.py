# Standard library imports
import argparse
import json
import logging
import math
import os
import random
import re
from functools import partial
from pathlib import Path

# Third-party imports
from PIL import Image
import datasets
from datasets import Dataset, load_metric
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm
from tqdm.auto import tqdm
import transformers
from transformers import (
    AdamW,
    AutoConfig,
    AutoImageProcessor,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BertModel,
    BertConfig,
    Blip2Processor,
    DataCollatorWithPadding,
    PretrainedConfig,
    PreTrainedModel,
    SchedulerType,
    default_data_collator,
    get_scheduler,
    T5Tokenizer,
    T5ForConditionalGeneration,
    StoppingCriteriaList
)
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.utils.versions import require_version
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from huggingface_hub import Repository
import wandb
from torch.nn import CrossEntropyLoss
import GPUtil
import gc

# Local application/library specific imports
from models.custom_codellama import CodeLlamaForWebshop
from minigpt4.processors.blip_processors import Blip2ImageTrainProcessor, Blip2ImageEvalProcessor


JSON_PATH = "../data/items_shuffle.json"
TRAJ_PATH = "data/il_trajs_finalized_images.jsonl"
GOAL_PATH = "data/human_goals.json"
IMAGE_PATH = "../all_images"
IMAGE_SIZE = 224
CKPT_PATH = "ckpts"

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}
logger = get_logger(__name__)

def process(s):
    s = s.lower().replace('"', '').replace("'", "").strip()
    s = s.replace('[sep]', '[SEP]')
    return s

def process_goal(state):
    state = state.lower().replace('"', '').replace("'", "")
    state = state.replace('amazon shopping game\ninstruction:', '').replace('webshop\ninstruction:', '')
    state = state.replace('\n[button] search [button_]', '').strip()
    if ', and price lower than' in state:
        state = state.split(', and price lower than')[0]
    return state

# find the last item the agent clicked
def find_image_asin(actions, state_idx):
    for i in range(state_idx - 1, -1, -1):
        match = re.match(r'^click\[(?P<id>[a-z0-9]{10})\]', actions[i])
        if match:
            asin = match.group('id')
            if asin.isalpha(): # ASIN always contain digits
                continue
            return asin
    return "none"

def get_data(split, filter_search=True):
    print('Loading data from {}'.format(TRAJ_PATH))
    with open(TRAJ_PATH, 'r') as json_file:
        json_list = list(json_file)
    human_goals = json.load(open(GOAL_PATH, 'r'))
    
    random.seed(233)
    random.shuffle(json_list)
    
    # split by human goal index
    goal_range = range(len(human_goals))
    if split == 'train':
        goal_range = range(1500, len(human_goals))
    elif split == 'eval':
        goal_range = range(500, 1500)
    elif split == 'test':
        goal_range = range(0, 500)

    bad = cnt = 0
    state_list, action_list, idx_list, size_list = [], [], [], []
    image_list = []
    raw_image_list = []
    num_trajs = 0
    for json_str in json_list:
        result = json.loads(json_str)
        s = process_goal(result['states'][0])
        assert s in human_goals, s
        goal_idx = human_goals.index(s)
        if goal_idx not in goal_range:
            continue
        num_trajs += 1
        if 'images' not in result:
            result['images'] = [0] * len(result['states'])
        for i, (state, valid_acts, idx, image) in enumerate(zip(result['states'], result['available_actions'], result['action_idxs'], result['images'])):
            cnt += 1
            if filter_search and idx == -1:
                continue
            state_list.append(state)

            if image == 0:
                image_list.append([0.] * 512)
                raw_image_list.append("none")
            else:
                image_list.append(image)
                asin = find_image_asin(result['actions'], i) # asin = "none" if not found
                raw_image_list.append(asin)

            if len(valid_acts) > 4:  # do some action space reduction...
                bad += 1
                new_idxs = list(range(2)) + \
                    random.sample(range(2, len(valid_acts)), 2)
                if idx not in new_idxs:
                    new_idxs = new_idxs[:-1] + [idx]
                new_idxs = sorted(new_idxs)
                valid_acts = [valid_acts[i] for i in new_idxs]
                idx = new_idxs.index(idx)
                # print(valid_acts)
            action_list.extend(valid_acts)
            idx_list.append(idx)
            size_list.append(len(valid_acts))
    print('num of {} trajs: {}'.format(split, num_trajs))
    print('total transitions and bad transitions: {} {}'.format(cnt, bad))
    state_list, action_list = list(map(process, state_list)), list(map(process, action_list))
    return state_list, action_list, idx_list, size_list, image_list, raw_image_list

def split_list(X, Y):
    result = []
    start_index = 0
    for y in Y:
        end_index = start_index + y
        sublist = X[start_index:end_index]
        result.append(sublist)
        start_index = end_index
    return result

def get_dataset(split):
    states, actions, idxs, sizes, images, raw_images = get_data(split)
    actions = split_list(actions, sizes)
    dataset = {
        'states': states,
        'actions': actions,
        'sizes': sizes,
        'raw_images': raw_images,
        'labels': idxs
    }
    return Dataset.from_dict(dataset)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Finetune a transformers model on a text classification task")
    parser.add_argument(
        "--task_name",
        type=str,
        default="mprc",
        help="The name of the glue task to train on.",
        choices=list(task_to_keys.keys()),
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
        ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        default="bert-base-uncased",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=1,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float,
                        default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=10,
                        help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=32,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts",
                 "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default="./ckpts/web_click",
                        help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None,
                        help="A seed for reproducible training.")
    parser.add_argument("--push_to_hub", action="store_true",
                        help="Whether or not to push the model to the Hub.")
    parser.add_argument(
        "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument("--hub_token", type=str,
                        help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default="epoch",
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--with_tracking",
        type=int,
        default=1,
        help="Whether to load in all available experiment trackers from the environment and use them for logging.",
    )

    parser.add_argument("--mem", type=int, default=0, help="State with memory")
    parser.add_argument("--image", type=int, default=1,
                        help="State with image")
    parser.add_argument("--pretrain", type=int, default=1,
                        help="Pretrained BERT or not")

    parser.add_argument("--logging_steps", type=int,
                        default=10, help="Logging in training")

    parser.add_argument("--model_name", type=str, default="minigpt4", help="Name of the text encoder model (e.g. bert-base, t5-small, ...)")

    args = parser.parse_args()

    # Sanity checks
    if args.task_name is None and args.train_file is None and args.validation_file is None:
        raise ValueError(
            "Need either a task name or a training/validation file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in [
                "csv", "json"], "`train_file` should be a csv or a json file."
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            assert extension in [
                "csv", "json"], "`validation_file` should be a csv or a json file."

    if args.push_to_hub:
        assert args.output_dir is not None, "Need an `output_dir` to create a repo when `--push_to_hub` is passed."

    return args

# examples = """Instruction:
# i am looking for an easy to install white antler chandelier with 18 antlers and 9 lights, and price lower than 410.00 dollars
# Observation: 
# page 1 (total results: 50)
# durahonn white antler chandelier, retro resin deer horn pendant light, e12 lamp holder, commercial and home lighting for cafes, bars, restaurants, living rooms (15 antlers + 9 lights)
# $379.99
# large antler chandelier 12 lights, bigmaii cabin retro faux antler light fixture rustic resin pendant light farmhouse candle style for living room, brown
# $100.0
# hubrin rustic antler chandelier, resin deer horn pendant light , antler light fixtures 9 light brown e12 candle style for home store (9 lamp arms + 6arms)
# $378.99
# Product image: 
# Available actions: click[back to search]    click[next >]    click[item - durahonn white antler chandelier, retro resin deer horn pendant light, e12 lamp holder, commercial and home lighting for cafes, bars, restaurants, living rooms (15 antlers + 9 lights)]    click[item - durahonn antler chandelier, 9 lights e12 bulbs, brown resin deer horn chandelier, retro antler pendant light for kitchen, bar, living room, dining room (15 antlers + 9 lights)]    click[item - large antler chandelier 12 lights, bigmaii cabin retro faux antler light fixture rustic resin pendant light farmhouse candle style for living room, brown]    

# Think:durahonn white antler chandelier is white antler chandelier less than 410 dollars. I can check durahonn first.
# Action:click[item - durahonn white antler chandelier, retro resin deer horn pendant light, e12 lamp holder, commercial and home lighting for cafes, bars, restaurants, living rooms (15 antlers + 9 lights)]

# Observation: 
# size
# durahonn white antler chandelier, retro resin deer horn pendant light, e12 lamp holder, commercial and home lighting for cafes, bars, restaurants, living rooms (15 antlers + 9 lights)
# price: $379.99
# rating: n.a.
# Product image:
# Available actions: click[back to search]    click[< prev]    click[description]    click[reviews]    click[18 antlers + 9 lights]    

# Think:For durahonn white antler chandelier, the item has options '18 antlers + 9 lights' and seems good to buy.
# Action:click[18 antlers + 9 lights]

# Observation: You have clicked 3 ounce (pack of 1).
# Available actions: click[back to search]    click[< prev]    click[description]    click[buy now]    click[15 antlers + 9 lights]

# Action: click[Buy Now]

# """

# examples = """Instruction:
# i am looking for a white antler chandelier with 18 antlers, and price lower than 410.00 dollars
# Observation: 
# page 1 (total results: 50)
# durahonn white antler chandelier (15 antlers + 9 lights)
# $379.99
# large antler chandelier 12 lights
# $100.0
# Product image: 
# Available actions: click[back to search]    click[next >]    click[item - durahonn white antler chandelier (15 antlers + 9 lights)]    click[item - large antler chandelier 12 lights]    

# Think: durahonn white antler chandelier is white antler chandelier less than 410 dollars. I can check durahonn first.
# Action:click[item - durahonn white antler chandelier (15 antlers + 9 lights)]

# Observation: 
# size
# durahonn white antler chandelier (15 antlers + 9 lights)
# price: $379.99
# rating: n.a.
# Product image:
# Available actions: click[back to search]    click[< prev]    click[description]    click[reviews]    click[18 antlers + 9 lights]    

# Think: For durahonn white antler chandelier, the item has options '18 antlers + 9 lights' and seems good to buy.
# Action: click[18 antlers + 9 lights]

# Observation: You have clicked 3 ounce (pack of 1).
# Available actions: click[back to search]    click[< prev]    click[description]    click[buy now]    click[15 antlers + 9 lights]

# Action: click[Buy Now]

# """

examples = """Instruction:
i am looking for a white antler chandelier with 18 antlers, and price lower than 410.00 dollars
Observation: 
page 1 (total results: 2)
durahonn white antler chandelier (15 antlers + 9 lights)
$379.99
large antler chandelier 12 lights
$100.0
Product image: 
Available actions: click[back to search]    click[next >]    click[item - durahonn white antler chandelier (15 antlers + 9 lights)]    click[item - large antler chandelier 12 lights]    

Think: durahonn white antler chandelier is white antler chandelier less than 410 dollars. I can check durahonn first.
Action:click[item - durahonn white antler chandelier (15 antlers + 9 lights)]

Observation: 
size
durahonn white antler chandelier (15 antlers + 9 lights)
price: $379.99
rating: n.a.
Product image:
Available actions: click[back to search]    click[< prev]    click[description]    click[reviews]    click[18 antlers + 9 lights]    

Think: The item has options '18 antlers + 9 lights' and seems good to buy.
Action: click[18 antlers + 9 lights]

Observation: You have clicked 18 antlers + 9 lights.
Available actions: click[back to search]    click[< prev]    click[description]    click[buy now]    click[15 antlers + 9 lights]

Action: click[Buy Now]

"""


def generate_prompt(observation, action=None):
    obslist = observation.split('\n')
    observation = ''
    i = 0
    for obs in obslist:
        if i == 1:
            i = 2

        if "instruction" in obs:
            i = 1
        
        if obs != '':
            if "[button]" not in obs and "[clicked button]" not in obs:
                observation += obs+'\n'
        
        if i == 2:
            observation += "Observation: "+'\n'
            i = 3

    # observation = observation.replace('[button] ', '[')
    # observation = observation.replace(' [button_]', ']')
    observation = observation.replace("instruction", "Instruction")
    # print("新的开始"+"="*20)
    # print(observation)

    observation += "Product image: <ImageHere>\n"
    observation += "Available actions: "
    for i in action:
        observation += i
        observation += "    "
    
    # print("分割线"+"="*20)
    # print(observation)
    prompt = examples + f'{observation}\n\nFollowing the above format, output the Action you will take to complete the instructed task. Action: '
    return prompt


def truncate_actions(actions, thresh=10):
    result = []
    for act in actions:
        words = act.split(" ")
        if len(words) > thresh:
            result.append(" ".join(words[:thresh]) + "]")
        else:
            result.append(act)
    return result


def process_actions(observation, actions):
    # Extract product names and prices from the observation
    lines = observation.split('\n')
    product_prices = {}
    for line in lines:
        if line.startswith('$'):
            price = line
            product_name = lines[lines.index(line) - 1]
            product_prices[product_name] = price
    
    # Process each action
    new_actions = []
    for action in actions:
        price = "-1.00"
        if action.startswith('click[item - '):
            # Extract the product name from the action
            product_name = action[len('click[item - '):].rstrip(']')
            # Append the price to the action if the product is in the product_prices dictionary
            if product_name in product_prices:
                price = product_prices[product_name]
        new_actions.append((action, price))
    
    return new_actions


def truncate_line(line, thresh=10):
    if len(line.split(" ")) > thresh:
        return " ".join(line.split(" ")[:10])
    return line


def make_concise_states(observation, actions):
    actions_with_prices = dict(process_actions(observation, actions))
    if observation.count('$') == 1: # do not filter on product page
        return observation
    lines = observation.split('\n')
    new_state = []
    num_products = 0
    for i in range(len(lines) - 1):
        line = lines[i]
        if lines[i + 1].startswith('$') or lines[i + 1].startswith('price: '):
            line_as_button = 'click[item - ' + line + ']'
            if line_as_button in actions_with_prices:
                new_state.append(truncate_line(line))
                new_state.append(actions_with_prices[line_as_button])
                num_products += 1
        elif line.startswith('$'):
            continue
        else:
            new_state.append(line)
    return '\n'.join(new_state).replace('total results: 50', f'total_results: {num_products}')


def main():
    args = parse_args()
    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will pick up all supported trackers in the environment
    # accelerator = Accelerator(log_with="wandb", logging_dir=args.output_dir) if args.with_tracking else Accelerator()
    # accelerator = Accelerator()
    # Make one log on every process with the configuration for debugging.

    output_folder = os.path.join(args.output_dir, args.model_name)
    # wandb.init(project="bert_il", config=args, name=output_folder)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    # print(accelerator.state, main_process_only=False)
    # if accelerator.is_local_main_process:
    #     datasets.utils.logging.set_verbosity_warning()
    #     transformers.utils.logging.set_verbosity_info()
    # else:
    #     datasets.utils.logging.set_verbosity_error()
    #     transformers.utils.logging.set_verbosity_ error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)
    
    train_dataset = get_dataset("train")
    eval_dataset = get_dataset("eval")
    train_idx = list(range(len(train_dataset)))
    eval_idx = list(range(len(eval_dataset)))
    
    llama_tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-hf")
    model = CodeLlamaForWebshop(llama_tokenizer).to('cuda:{}'.format(0))
    GPUtil.showUtilization()
    print("Model loaded to GPU")

    print("BACKBONE TYPE: ", model.code_llama)
    
    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataset) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(
            args.max_train_steps / num_update_steps_per_epoch)

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # # Prepare everything with our `accelerator`.
    # model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
    #     model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    # )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataset) / args.gradient_accumulation_steps)
    args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

    # Figure out how many steps we should save the Accelerator states
    if hasattr(args.checkpointing_steps, "isdigit"):
        checkpointing_steps = args.checkpointing_steps
        if args.checkpointing_steps.isdigit():
            checkpointing_steps = int(args.checkpointing_steps)
    else:
        checkpointing_steps = None

    # # We need to initialize the trackers we use, and also store our configuration
    # if args.with_tracking:
    #     experiment_config = vars(args)
    #     # TensorBoard cannot log Enums, need the raw value
    #     experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
    #     accelerator.init_trackers("glue_no_trainer", experiment_config)

    # Get the metric function
    metric = load_metric("accuracy")

    # Train!
    # total_batch_size = args.per_device_train_batch_size * \
    #     accelerator.num_processes * args.gradient_accumulation_steps

    print("***** Running training *****")
    print(f"  Num examples = {len(train_dataset)}")
    print(f"  Num Epochs = {args.num_train_epochs}")
    print(
        f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    # print(
    #     f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    print(
        f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    print(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    # progress_bar = tqdm(range(args.max_train_steps))
    # completed_steps = 0
    # starting_epoch = 0

    train_processor = Blip2ImageTrainProcessor.from_config({'name': 'blip2_image_train', 'image_size': 224})
    eval_processor = Blip2ImageEvalProcessor.from_config({'name': 'blip2_image_eval', 'image_size': 224})

    for epoch in range(args.num_train_epochs):
        model.train()
        if args.with_tracking:
            total_loss = total_step = 0

        random.shuffle(train_idx)
        bar = tqdm(total=len(train_idx))
        step = 0
        avg_train_loss = 0.0

        for i in train_idx[:100]:
            # print(i)
            data = train_dataset[i]
            states = data['states']
            actions = data['actions']
            sizes = data['sizes']
            raw_images = data['raw_images'].upper()
            labels = data['labels']
            if raw_images == "NONE":
                image = Image.new('RGB', (224, 224), (255, 255, 255))
            else:
                try:
                    image = Image.open(os.path.join(IMAGE_PATH, raw_images + ".jpg")).convert('RGB')
                except:
                    image = Image.new('RGB', (224, 224), (255, 255, 255)) # this is rare
            image = train_processor(image).unsqueeze(0).to('cuda:{}'.format(0))

            states = make_concise_states(states, actions)

            # print("---------+++++++++++++PROCESSED ACTIONS")
            # print(process_actions(states, actions))

            prompt = generate_prompt(states, actions)
            num_tokens = llama_tokenizer(
                prompt,
                truncation=False,
                padding=False,
                return_tensors='pt',
            )['input_ids'].shape[-1]

            if num_tokens > 680:
                bar.update(1)
                continue

            # print("=================---------------===============")
            # print(prompt)
            tokenized_labels = llama_tokenizer(
                actions[labels],
                truncation=True,
                max_length=32,
                padding=False,
                return_tensors='pt',
            )['input_ids']

            # import gc
            gc.collect()
            # print("Before forward")
            # GPUtil.showUtilization()
            # print("PROMPT LENGTH=", len(prompt.split(' ')))

            loss = model(prompt, [image], labels=tokenized_labels) / args.gradient_accumulation_steps
            # print("zha le ma???????????????????????????????????????????????????????")
            
            # import gc
            gc.collect()
            # print("After forward")
            # GPUtil.showUtilization()

            loss.backward()
            gc.collect()
            # print("After backward")
            # GPUtil.showUtilization()

            if step % args.gradient_accumulation_steps == 0 or step == len(train_idx) - 1:
                optimizer.step()
                optimizer.zero_grad()
            avg_train_loss += loss.item()
            bar.update(1)
        bar.close()

        random.shuffle(eval_idx)
        bar = tqdm(total=len(eval_idx))
        step = 0
        avg_train_loss = 0.0
        model.eval()
        num_correct = 0
        with torch.no_grad():

            for i in eval_idx:
                torch.cuda.empty_cache()
                data = eval_dataset[i]
                states = data['states']
                # print("新的开始"+"="*20)
                # print(states)
                # print("分割线"+"="*20)
                actions = data['actions']
                # print(actions)
                sizes = data['sizes']
                raw_images = data['raw_images'].upper()
                labels = data['labels']
                if raw_images == "NONE":
                    image = Image.new('RGB', (224, 224), (255, 255, 255))
                else:
                    try:
                        image = Image.open(os.path.join(IMAGE_PATH, raw_images + ".jpg")).convert('RGB')
                    except:
                        image = Image.new('RGB', (224, 224), (255, 255, 255)) # this is rare
                image = eval_processor(image).unsqueeze(0).to('cuda:{}'.format(0))

                states = make_concise_states(states, actions)

                prompt = generate_prompt(states, actions)
                # print("=============================================================================================================")
                
                tokenized_labels = llama_tokenizer(
                    actions[labels],
                    truncation=True,
                    max_length=32,
                    padding=False,
                    return_tensors='pt',
                )['input_ids']
                answer = model.generate(prompt, [image]).strip()
                print("ANSWER: ", answer)
                isPrint = True
                if answer.startswith('click[') and len(answer) > 6 + 4: # the shortest thing to click on is "prev"
                    # print("Good answer")

                    answer_content = answer[6:]
                    if actions[labels][6:].startswith(answer_content) or answer_content.startswith(actions[labels][6:]):
                        # print("CORRECT")
                        num_correct += 1
                        isPrint = False
                
                if isPrint:
                    print("PROMPT: ", prompt)
                    print("GROUND TRUTH:", actions[labels])
                    print("ANSWER: ", answer)

                bar.update(1)
            print("Accuracy: ", num_correct / len(eval_idx))
        bar.close()

        # model.to('cpu')
        # torch.cuda.empty_cache()
        
        # # torch.save(model.state_dict(), f'model_state_dict_epoch_{epoch}.pth')
        # model.to('cuda:0')

    # if output_folder is not None:
    #     with open(os.path.join(output_folder, "all_results.json"), "w") as f:
    #         json.dump({"eval_accuracy": eval_metric["accuracy"]}, f)


if __name__ == "__main__":
    main()
