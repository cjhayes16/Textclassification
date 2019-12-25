# coding: UTF-8
import os
import torch
import numpy as np
import pickle as pkl
from tqdm import tqdm
import time
from datetime import timedelta


MAX_VOCAB_SIZE = 10000
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


    def biGramHash(sequence, t, buckets):
        t1 = sequence[t - 1] if t - 1 >= 0 else 0
        return (t1 * 14918087) % buckets

    def triGramHash(sequence, t, buckets):
        t1 = sequence[t - 1] if t - 1 >= 0 else 0
        t2 = sequence[t - 2] if t - 2 >= 0 else 0
        return (t2 * 14918087 * 18408749 + t1 * 14918087) % buckets

    def load_dataset(path, pad_size):
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

                # fasttext ngram
                buckets = config.n_gram_vocab
                bigram = []
                trigram = []
                # ------ngram------
                for i in range(pad_size):
                    bigram.append(biGramHash(sequence, i, buckets))
                    trigram.append(triGramHash(sequence, i, buckets))
                # -----------------
                texts.append((sequence, int(label), seq_len, bigram, trigram))
        return texts
    train = load_dataset(config.train_path, config.pad_size)
    dev = load_dataset(config.dev_path, config.pad_size)
    test = load_dataset(config.test_path, config.pad_size)
    return vocab, train, dev, test


class DatasetIterater(object):
    def __init__(self, batches, batch_size, device):
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size
        self.residue = False  # 记录batch数量是否为整数 
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device

    def _to_tensor(self, datas):
        # xx = [xxx[2] for xxx in datas]
        # indexx = np.argsort(xx)[::-1]
        # datas = np.array(datas)[indexx]
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        y = torch.LongTensor([_[1] for _ in datas]).to(self.device)
        bigram = torch.LongTensor([_[3] for _ in datas]).to(self.device)
        trigram = torch.LongTensor([_[4] for _ in datas]).to(self.device)

        # pad前的长度(超过pad_size的设为pad_size)
        seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        return (x, seq_len, bigram, trigram), y

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index > self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def build_iterator(dataset, config):
    iter = DatasetIterater(dataset, config.batch_size, config.device)
    return iter


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))

