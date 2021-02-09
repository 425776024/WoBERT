#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/12/28 3:16 下午
# @File    : data_process.py

import pandas as pd
from config.configs_interface import project_path
import os

'''
把xlsx洗出来常规训练数据
pip install xlrd==1.2.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
'''
path = os.path.join(project_path, 'data/新闻分类样本-终极版本.xlsx')


def data1():
    data_1 = pd.read_excel(path, sheet_name='分类错误率占比')
    labels = data_1['所有分类'].tolist()

    print(data_1.head(), data_1.shape)
    print(len(labels), labels)
    with open(os.path.join(project_path, 'data/labels.txt'), encoding='utf-8', mode='w') as wf:
        # wf.write('类别' + '\n')
        for label in labels:
            wf.write(label + '\n')


def data2():
    data_2 = pd.read_excel(path, sheet_name='分类数据汇总')
    labels2 = data_2['分类'].value_counts().keys()

    print(len(labels2), labels2)

    print(data_2.head(), data_2.shape)
    del data_2['ds']

    def return_label(x):
        if x['是否异常'] == '否':
            x['label'] = x['分类']
        else:
            x['label'] = x['正确分类']
        return x

    data_2 = data_2.apply(lambda x: return_label(x), axis=1)
    data_2.to_csv(path.replace('xlsx', 'csv'), index=False)

    train_data = pd.DataFrame()
    train_data['gid'] = data_2['gid']
    train_data['label'] = data_2['label']
    train_data['text'] = data_2['新闻标题']
    train_data.to_csv(os.path.join(project_path, 'data/train.txt'), index=False)


data2()
