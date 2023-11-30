import json
import numpy as np
import requests
from PIL import Image
from io import BytesIO
from tqdm import tqdm

JSON_PATH = "data/items_shuffle.json"
DUMP_PATH = "all_images.npz"

# output format: H*W*3
def url_to_image(url):
    response = requests.get(url)
    if response.status_code == 200:
        img = Image.open(BytesIO(response.content))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img_array = np.array(img)
        return img_array
    else:
        response.raise_for_status()

images = {}
f = open(JSON_PATH, 'r')
data = json.load(f)

missing_keys = 0
missing_asin = 0
network_failure = 0
total = 0
max_samples = 1000

bar = tqdm(total=max_samples)
for i, d in enumerate(data):
    total += 1
    if ('product_information' not in d) or ('images' not in d):
        bar.update(1)
        missing_keys += 1
        continue
    if 'asin' not in d:
        bar.update(1)
        missing_asin += 1
        continue
    url = d['images'][0]
    try:
        im = url_to_image(url)
    except:
        network_failure += 1
    images[d['asin']] = im
    bar.update(1)
    if total >= max_samples:
        break
bar.close()

print("missing_keys: {}".format(missing_keys))
print("missing_asin: {}".format(missing_asin))
print("network_failure: {}".format(network_failure))

np.savez(DUMP_PATH, **images)