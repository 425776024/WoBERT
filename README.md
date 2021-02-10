# pytorch实现的[WoBERT](https://github.com/ZhuiyiTechnology/WoBERT) + 新闻多分类样例

## 1.安装依赖

```
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
```

## 2.模型下载

### 方式1. 

从https://github.com/ZhuiyiTechnology/WoBERT 下载，然后执行转换
```
python config/conver.py
```

### 方式2.

直接下载 : 链接: https://pan.baidu.com/s/1KbLvVkXriF_x9XZnaYvDhg  密码: 94au
    


## 3.配置参数

```
vim config/configs.yaml
```

## 4.train\test

```
python train.py
```

## 5.案例数据
``data/train.csv``


## 细节

除了transformers转换模型，主要是tokenizer的变化
- 1.稍微自定义了个专门的``分词``，见src/models/bert_model.py:``WoBertTokenizer``
- 2.稍微自定义了个专门的``BERT分类模型``，见src/models/bert_model.py:``WoBertClassificationModel``