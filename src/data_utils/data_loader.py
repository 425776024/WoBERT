#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/12/29 9:50 上午
# @File    : data_set.py
import sys, os, pathlib

# project_path = pathlib.Path(__file__).resolve().parents[2]
# sys.path.append(project_path)
import pickle as pkl
# from sklearn.model_selection import train_test_split
from collections import Counter
from src.data_utils.train_test_split import train_test_split
from config.configs_interface import configs
import pandas as pd
from src.utils.logers import LOGS
import torch
import numpy as np
import random


def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(7)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


set_seed()


class DataLoader:
    def __init__(self):
        set_seed()
        self.batch_size = configs.train_args.batch_size
        self.data_sentences, self.data_labels, self.labels, self.label_to_index, \
        self.index_to_label, self.label_weight = self.get_data()
        self.laebl_size = len(self.label_to_index)
        self.train_batch_count = 0
        self.test_batch_count = 0
        self.batch_train_inputs = []
        self.batch_train_targets = []
        self.batch_test_inputs = []
        self.batch_test_targets = []
        self.process_data(self.data_sentences, self.data_labels)
        LOGS.log.debug(f'all data_sentences:{len(self.data_sentences)}, all data_labels:{len(self.data_labels)}')
        LOGS.log.debug(f'train_batch_count:{self.train_batch_count}, batch_size:{self.batch_size}')
        LOGS.log.debug(f'test_batch_count:{self.test_batch_count}')
        LOGS.log.debug(f'batch_train_inputs len:{len(self.batch_train_inputs)}')
        LOGS.log.debug(f'batch_test_inputs len:{len(self.batch_test_inputs)}')
        # all data process

    def process_data(self, data_sentences, data_labels):

        self.train_inputs, self.test_inputs, self.train_targets, self.test_targets = train_test_split(
            data_sentences,
            data_labels,
            test_size=configs.train_args.test_date_rate,
            random_state=1,
            banance=None)
        self.batch_size = configs.train_args.batch_size

        # train data batch process
        # 总数/batch_size = batch的数量
        self.train_batch_count = int(len(self.train_inputs) / self.batch_size)

        self.batch_train_inputs, self.batch_train_targets = [], []
        for i in range(self.train_batch_count):
            self.batch_train_inputs.append(self.train_inputs[i * self.batch_size: (i + 1) * self.batch_size])
            self.batch_train_targets.append(self.train_targets[i * self.batch_size: (i + 1) * self.batch_size])

        # test data batch process
        self.test_batch_count = int(len(self.test_inputs) / self.batch_size)

        self.batch_test_inputs, self.batch_test_targets = [], []
        for i in range(self.test_batch_count):
            self.batch_test_inputs.append(self.test_inputs[i * self.batch_size: (i + 1) * self.batch_size])
            self.batch_test_targets.append(self.test_targets[i * self.batch_size: (i + 1) * self.batch_size])

    def get_data(self, min_label_count=100):
        '''
        更具 min_label_count 进行过滤，（不分析特点类别，全部混合训练）
        :return: data_sentences, data_labels, label_to_index, index_to_label
        '''
        train_df = pd.read_csv(configs.data.train_data)
        LOGS.log.debug(f"Train set shape:{train_df.shape}")
        label_count = train_df['label'].value_counts()
        # data_sentences = train_df['text'].values
        LOGS.log.debug(f'原始类别数据：{label_count}')
        # 只分析这些label

        clear_labels = []
        for li in label_count.index:
            if label_count[li] < min_label_count:
                clear_labels.append(li)
        print('这些类别过滤删除：', clear_labels)
        df_clear = train_df[~train_df['label'].isin(clear_labels)]

        LOGS.log.debug(f"Train set shape:{df_clear.shape}")
        clear_label_count = df_clear['label'].value_counts()
        LOGS.log.debug(f'过滤后数据：{clear_label_count}')

        # 所有数据
        data_sentences = df_clear['text'].values.tolist()
        data_labels = df_clear['label'].values.tolist()
        labels = clear_label_count.keys()

        label_to_index = {
            str(k): i for i, k in enumerate(labels)
        }
        index_to_label = {
            i: str(k) for i, k in enumerate(labels)
        }
        data_labels = [label_to_index[i] for i in data_labels]

        assert len(data_sentences) == len(data_labels)
        label_counter = Counter(data_labels)
        count_vales = label_counter.values()
        # max_count = max(count_vales)
        min_count = min(count_vales)
        label_weight = []
        for label_i in range(len(label_to_index)):
            wi = min_count * 1.0 / label_counter[label_i]
            label_weight.append(wi)

        assert len(data_sentences) == len(data_labels)
        return data_sentences, data_labels, labels, label_to_index, index_to_label, label_weight


if __name__ == '__main__':
    d = DataLoader()
