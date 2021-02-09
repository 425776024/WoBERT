#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/12/31 3:16 下午
# @File    : train_test_split.py
from typing import List, Any, Tuple
from collections import Counter, defaultdict
import random
from src.utils.logers import LOGS
random.seed(0)


def train_test_sample(data: List[Any], test_num=0, seed=0):
    random.seed(seed)
    new_data = data.copy()
    random.shuffle(new_data)
    test_data = new_data[:test_num]
    train_data = new_data[test_num:]
    return test_data, train_data


def train_test_split(X: List[Any], Y: List[Any], test_size=0.2, random_state=1, banance=None) -> Tuple[
    List[Any], List[Any], List[Any], List[Any]]:
    '''
    - 根据类别比例，等比例随机划分train,test
    - 区别于：sklearn train_test_split的完全随机
    :param X: 数据
    :param Y: 数据label
    :param test_size: 划分比（每个类别数据，拿出这么多做test）
    :param random_state:
    :param banance:如果有数字，则每个类最多会 banance大小
    :return:x_train, x_test, y_train, y_test
    '''
    assert len(X) == len(Y), '数据不一致'
    all_num = len(Y)
    y_counter = Counter(Y)
    data = defaultdict(list)
    for i, y in enumerate(Y):
        data[y].append(X[i])

    # 如果有数字，则每个类最多会 banance大小
    if banance is not None:
        for yi in y_counter:
            data[yi] = data[yi][:banance]
            LOGS.log.debug(f'{yi} banance to: len={len(data[yi][:banance])}')

    test = []
    train = []
    for yi in y_counter:
        yi_test_num = int(test_size * y_counter[yi])
        xy_data = list(zip(data[yi], [yi] * y_counter[yi]))
        xy_test, xy_train = train_test_sample(xy_data, yi_test_num, random_state)

        test.extend(xy_test)
        train.extend(xy_train)

    random.shuffle(test)
    random.shuffle(train)
    x_train, x_test, y_train, y_test = [], [], [], []
    for i in range(len(train)):
        x_train.append(train[i][0])
        y_train.append(train[i][1])

    for i in range(len(test)):
        x_test.append(test[i][0])
        y_test.append(test[i][1])
    return x_train, x_test, y_train, y_test
