# coding: UTF-8
import os
import torch
import numpy as np
import pickle as pkl
from tqdm import tqdm
import time
from datetime import timedelta

VOCAB_MAX_SIZE = 10000
UNK, PAD = '<UNK>', '<PAD>'


def generate_dataset(config):
    """
    :param config: config.train_path,config.dev_path,config.test_path
    :return: dataset size  [([...], 0), ([...], 1), ...]
    """
    tokenizer = lambda x: [y for y in x]
    # 加载词表
    vocab = pkl.load(open(config.vocab_path, 'rb'))
    print(f"Vocab size: {len(vocab)}")

    def load_dataset(path,pad_size):
        # 文本向量化表示
        texts = []
        with open(path, 'r', encoding='UTF-8') as f:
            for lines in tqdm(f):
                line = lines.strip()
                if not line:
                    continue
                text, label = line.split('\t')
                sequence = []
                token = tokenizer(text)
                # 将句子向量处理成固定长度 pad_size
                seq_len = len(token)
                if seq_len < pad_size:
                    # 句子不够长，<PAD>填充
                    token.extend([vocab.get(PAD)] * (pad_size - seq_len))
                else:
                    # 句子够长，截断
                    token = token[:pad_size]
                    seq_len = pad_size
                # word → id
                for word in token:
                    sequence.append(vocab.get(word, vocab.get(UNK)))
                texts.append((sequence, int(label), seq_len))
        return texts

    train = load_dataset(config.train_path,config.pad_size)
    dev = load_dataset(config.dev_path,config.pad_size)
    test = load_dataset(config.test_path,config.pad_size)

    return vocab, train, dev, test


class DatasetIter(object):
    def __init__(self, batches, batch_size, device):
        self.batch_size = batch_size
        self.batches = batches
        self.num_batches = len(batches) // batch_size
        self.residue = False
        if len(batches) % self.num_batches != 0:
            self.residue = True
        self.idx = 0
        self.device = device

    def _to_tensor(self, datas):
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        y = torch.LongTensor([_[1] for _ in datas]).to(self.device)

        # 句子长度，超过pad_size的设为pad_size
        seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        return (x, seq_len), y

    def __next__(self):
        if self.residue and self.idx == self.num_batches:
            batches = self.batches[self.idx * self.batch_size: len(self.batches)]
            self.idx += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.idx > self.num_batches:
            self.idx = 0
            raise StopIteration
        else:
            batches = self.batches[self.idx * self.batch_size: (self.idx + 1) * self.batch_size]
            self.idx += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.num_batches + 1
        else:
            return self.num_batches


def build_iterator(dataset, config):
    iter = DatasetIter(dataset, config.batch_size, config.device)
    return iter


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


