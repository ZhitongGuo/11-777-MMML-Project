from web_agent_site import engine
import torch
import clip
from PIL import Image
import requests
from bert_score import score

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

def ranking(top_n_products, query, goal, price, options):
    bert = []
    clip_score = []
    reward_score = []
    for item in top_n_products:
        r_score = engine.get_reward(item, goal, price, options)
        reward_score.append(r_score)
        image = preprocess(Image.open(requests.get(item['images']))).unsqueeze(0).to(device)
        text = clip.tokenize([query]).to(device)
        with torch.no_grad():
            logits_per_image, _ = model(image, text)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()
            clip_score.append(probs[0])
        bert.append(score(text, item['full_description']).numpy()[0])
    return bert, clip_score, reward_score