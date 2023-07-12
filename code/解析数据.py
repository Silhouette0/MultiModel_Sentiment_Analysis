import json
import os
import numpy as np


save_dir = "./dataset/data/"
os.makedirs(save_dir, exist_ok=True)
image_text_dir = "./data"
train_list_fp = "train.txt"

train_list = np.loadtxt(train_list_fp, delimiter=',', dtype=str)[1:]

emotion_labels = {
    "positive": 0,
    "neutral": 1,
    "negative": 2,
}

translations = []
datas = []

for imgId, label in train_list:
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
        "emotion_label": emotion_labels[label]
    }

    translations.append(translation)
    datas.append(data)

with open(save_dir + "translations.json", mode='w', encoding='utf-8') as f:
    json.dump(translations, f, indent=4)

# 划分训练集和验证集
l = int(len(datas) * 0.9)

train_set = datas[:l]
val_set = datas[l:]

with open(save_dir + "train.json", mode='w', encoding='utf-8') as f:
    json.dump(train_set, f, indent=4)

with open(save_dir + "val.json", mode='w', encoding='utf-8') as f:
    json.dump(val_set, f, indent=4)

