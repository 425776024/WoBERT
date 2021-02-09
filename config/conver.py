#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    : conver.py

import transformers.convert_bert_original_tf_checkpoint_to_pytorch as con

tf_wobert_path = '/Users/jiang/Documents/pre_train_models/chinese_wobert_L-12_H-768_A-12'

con.convert_tf_checkpoint_to_pytorch(
    f'{tf_wobert_path}/bert_model.ckpt',
    f'{tf_wobert_path}/bert_config.json',
    f'{tf_wobert_path}/pytorch_bert.bin'
)
