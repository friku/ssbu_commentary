
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import GPT2Tokenizer
from tqdm import tqdm
import os
import pickle
import sys
import json
import cv2
import torchvision
from PIL import Image
from typing import Tuple
from transformers import T5Tokenizer
from pathlib import Path

DEBUG = True


class SSBUCaptionDataset(Dataset):
    def __init__(self, tokenizer, clip_preprocess, split, prefix_length, transform=None):
        self.clip_preprocess = clip_preprocess
        print(f"Preprocess for {split} ... ")

        textDatasetPaths = Path("C:\\Users\\81804\\whisper\\text_datasets\\mumei").glob("*.json")

        self.labels = []
        for jsonFile in textDatasetPaths:
            with open(jsonFile, 'r',encoding="utf-8") as f:
                corpus = json.load(f)

                for segment in corpus["segments"]:
                    tokens = tokenizer.encode(segment['text'], return_tensors="pt").squeeze(0)
                    segment['tokens'] = tokens
                    segment['path'] = jsonFile
                    self.labels.append(segment)


    def get_img(self, path):
        img = Image.open(path)
        return img

    def __getitem__(self, i):
        segment = self.labels[i]

        img = get_frame_at_time(segment['path'], segment['start'])
        assert img.shape[0] == 3, f"img.shape == {img.shape}"

        img_size = (256,256)
        img = img_preprocess(img, img_size, self.clip_preprocess)

        if self.transform is not None:
            img = self.transform(img)

        tokens, mask = self.pad_tokens(segment['tokens'])
        tokens, mask = tokens.cuda(), mask.cuda()

        return img, tokens, mask, segment['text']


    def __len__(self):
        return len(self.labels)

    def pad_tokens(self, tokens):
        padding = self.max_seq_len - tokens.shape[0]
        if padding > 0:
            tokens = torch.cat((tokens, torch.zeros(padding, dtype=torch.int64) - 1))
        elif padding < 0:
            tokens = tokens[:self.max_seq_len]
        mask = tokens.ge(0)  # mask is zero where we out of sequence
        tokens[~mask] = 0
        mask = mask.float()
        mask = torch.cat((torch.ones(self.prefix_length), mask), dim=0)  # adding prefix mask
        return tokens, mask


def get_frame_at_time(video_path, time_in_seconds):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_MSEC, time_in_seconds * 1000)
    success, image = cap.read()
    if not success:
        return None
    return image


def img_preprocess(img, size:Tuple, clip_preprocess=None) -> torch.Tensor:
    img = cv2.resize(img, size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if clip_preprocess is not None:
        img_tensor = clip_preprocess(img).squeeze(0).to("cuda")

    return img_tensor
