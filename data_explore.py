import warnings
warnings.filterwarnings("ignore")

import gym
from web_agent_site.envs import WebAgentTextEnv

env = gym.make('WebAgentTextEnv-v0', observation_mode='text', num_products=None, get_image=True)
obs = env.reset()

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")
    parser.add_argument("--model_path", type=str, default="./ckpts/web_click/epoch_9/model.pth", help="Where to store the final model.")
    parser.add_argument("--mem", type=int, default=0, help="State with memory")
    parser.add_argument("--bart_path", type=str, default='./ckpts/web_search/checkpoint-800', help="BART model path if using it")
    parser.add_argument("--bart", type=bool, default=True, help="Flag to specify whether to use bart or not (default: True)")
    parser.add_argument("--image", type=bool, default=True, help="Flag to specify whether to use image or not (default: True)")
    parser.add_argument("--softmax", type=bool, default=True, help="Flag to specify whether to use softmax sampling or not (default: True)")
    args = parser.parse_args()
    return args

args = parse_args()
print(args)

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

config = BertConfigForWebshop(image=args.image)
model = BertModelForWebshop(config)
model.cuda()
model.load_state_dict(torch.load(args.model_path), strict=False)
print('bert il model loaded', args.model_path)

def episode(model, idx=None, verbose=False, softmax=False, rule=False, bart_model=None):
    assert (bart_model is not None)
    obs, info = env.reset(idx)
    if verbose:
        print(info['goal'])
    valid_acts = info['valid']
    goal = process_goal(obs)
    query = bart_predict(goal, bart_model, num_return_sequences=5, num_beams=5)
    # query = random.choice(query)  # in the paper, we sample from the top-5 generated results.
    query = query[0]  #... but use the top-1 generated search will lead to better results than the paper results.
    action = f'search[{query}]'
    for i in range(100):
        action = predict(obs, info, model, softmax=softmax, rule=rule, bart_model=bart_model)
        if verbose:
            print(action)
        obs, reward, done, info = env.step(action)
        obs, reward, done, info = env.step(action)
    
    return 0

for i in range(500):
    score_softmax, score_rule = episode(model, idx=i, softmax=args.softmax, bart_model=bart_model), episode(model, idx=i, rule=True)