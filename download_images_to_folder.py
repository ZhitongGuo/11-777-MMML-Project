import json
import requests
from PIL import Image
from io import BytesIO
from tqdm import tqdm
import os

JSON_PATH = "data/items_shuffle.json"
DUMP_DIR = "all_images"  # Set to your desired folder path

if not os.path.exists(DUMP_DIR):
    os.makedirs(DUMP_DIR)

# output format: H*W*3
def url_to_image(url, asin):
    response = requests.get(url)
    if response.status_code == 200:
        img = Image.open(BytesIO(response.content))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img.save(os.path.join(DUMP_DIR, f'{asin}.jpg'), 'JPEG')
    else:
        response.raise_for_status()

f = open(JSON_PATH, 'r')
data = json.load(f)

missing_keys = 0
missing_asin = 0
network_failure = 0
total = 0

bar = tqdm(total=len(data))
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
    if len(url) == 0 and len(d['images']) > 1:
        url = d['images'][1]
    try:
        url_to_image(url, d['asin'])
    except Exception as e:
        print(f"Failed to download or save image for ASIN {d['asin']}: {e}")
        network_failure += 1
    bar.update(1)
    # if total >= max_samples:
    #     break
bar.close()

f.close()  # Don't forget to close the file

print("missing_keys: {}".format(missing_keys))
print("missing_asin: {}".format(missing_asin))
print("network_failure: {}".format(network_failure))
