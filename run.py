# coding: UTF-8
import time
import torch
import numpy as np
from Train_Test.train import train, init_network
from importlib import import_module
import argparse

parser = argparse.ArgumentParser(description='Text-Classification')
parser.add_argument('--model', type=str, required=True, help='Model:TextCNN,TextDCNN,TextDPCNN,TextRCNN,TextRNN,TextRNN_Attention,Transformer,FastText')
args = parser.parse_args()


if __name__ == '__main__':

    #预训练词向量---搜狗新闻
    embedding = 'embedding_SougouNews.npz'
    model_name = args.model
    if model_name == 'FastText':
        from Utils.utilsfasttext import generate_dataset, build_iterator, get_time_dif
        embedding = 'random'
    else:
        from Utils.utils import generate_dataset, build_iterator, get_time_dif

    x = import_module('Models.' + model_name)
    config = x.Config(embedding)

    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True

    start_time = time.time()
    print("---------LOADING Datas---------")
    vocab, train_data, dev_data, test_data = generate_dataset(config)
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # train
    config.n_vocab = len(vocab)
    model = x.Model(config).to(config.device)
    if model_name != 'Transformer':
        init_network(model)
    print(model.parameters)
    train(config, model, train_iter, dev_iter, test_iter)

