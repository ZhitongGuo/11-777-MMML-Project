from transformers import ViltProcessor, ViltForQuestionAnswering
import requests
from PIL import Image


# prepare image + question
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
text = "How many cats are there?"

processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

# prepare inputs
encoding = processor(image, text, return_tensors="pt")

# forward pass
outputs = model(**encoding)
logits = outputs.logits
idx = logits.argmax(-1).item()
print("Predicted answer:", model.config.id2label[idx])


with open('items_ins_v2_1000.json', 'r') as f:
  data_ins = json.load(f)

scores = []
for i in tqdm(range(len(data))):
  txt = data[i]['contents']
  url = data[i]['product']['images'][0]
  if url == '':
    continue
  response = requests.get(url)
  img = Image.open(BytesIO(response.content))
  image = preprocess(img).unsqueeze(0)
  text = tokenizer([txt])
  with torch.no_grad(), torch.cuda.amp.autocast():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    text_probs = (100.0 * image_features @ text_features.T)
    scores.append(text_probs[0][0].item())