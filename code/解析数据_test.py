import json
import os
import numpy as np

save_dir = "./dataset/data/"
os.makedirs(save_dir, exist_ok=True)
image_text_dir = "./data"
test_list_fp = "test_without_label.txt"

test_list = np.loadtxt(test_list_fp, delimiter=',', dtype=str)[1:]

emotion_labels = {
    "positive": 0,
    "neutral": 1,
    "negative": 2,
}

translations = []
datas = []

for imgId, label in test_list:
    text_fp = os.path.join(image_text_dir, imgId + '.txt')

    with open(text_fp, 'r', encoding='ANSI') as f:
        text = f.readlines()[0]

    translation = {
        "id": imgId,
        "text_translation": text,
    }

    data = {
        "id": imgId,
        "text": text,
        "emotion_label": 3
    }

    translations.append(translation)
    datas.append(data)

with open(save_dir + "translations_test.json", mode='w', encoding='utf-8') as f:
    json.dump(translations, f, indent=4)

with open(save_dir + "test.json", mode='w', encoding='utf-8') as f:
    json.dump(datas, f, indent=4)