import torch
import math
from torch import nn, optim
from transformers import BertModel, BertTokenizer
from transformers import AdamW
from src.data_utils.data_loader import DataLoader
from config.configs_interface import TrainArgs
import re, itertools, jieba, os


class WoBertTokenizer(BertTokenizer):
    def _tokenize(self, text):
        # split_tokens = []
        # if self.do_basic_tokenize:
        #     for token in self.basic_tokenizer.tokenize(text, never_split=self.all_special_tokens):
        #         for sub_token in self.wordpiece_tokenizer.tokenize(token):
        #             split_tokens.append(sub_token)
        # else:
        #     split_tokens = self.wordpiece_tokenizer.tokenize(text)
        # return split_tokens
        cut_words = jieba.lcut(text)
        return_words = []
        for w in cut_words:
            if w in self.vocab:
                # will not [UNK]
                return_words.append(w)
            else:
                # will be [UNK]
                w = list(w)
                return_words.extend(w)

        return return_words

    def tokenize(self, text, **kwargs):
        all_special_tokens = self.all_special_tokens
        text = self.prepare_for_tokenization(text, **kwargs)

        def lowercase_text(t):
            # convert non-special tokens to lowercase
            escaped_special_toks = [re.escape(s_tok) for s_tok in all_special_tokens]
            pattern = r"(" + r"|".join(escaped_special_toks) + r")|" + r"(.+?)"
            return re.sub(pattern, lambda m: m.groups()[0] or m.groups()[1].lower(), t)

        if self.init_kwargs.get("do_lower_case", False):
            text = lowercase_text(text)

        def split_on_token(tok, text):
            result = []
            split_text = text.split(tok)
            for i, sub_text in enumerate(split_text):
                sub_text = sub_text.rstrip()
                if i == 0 and not sub_text:
                    result += [tok]
                elif i == len(split_text) - 1:
                    if sub_text:
                        result += [sub_text]
                    else:
                        pass
                else:
                    if sub_text:
                        result += [sub_text]
                    result += [tok]
            return result

        def split_on_tokens(tok_list, text):
            if not text.strip():
                return []
            if not tok_list:
                return self._tokenize(text)

            tokenized_text = []
            text_list = [text]
            for tok in tok_list:
                tokenized_text = []
                for sub_text in text_list:
                    if sub_text not in self.unique_added_tokens_encoder:
                        tokenized_text += split_on_token(tok, sub_text)
                    else:
                        tokenized_text += [sub_text]
                text_list = tokenized_text

            return list(
                itertools.chain.from_iterable(
                    (
                        self._tokenize(token) if token not in self.unique_added_tokens_encoder else [token]
                        for token in tokenized_text
                    )
                )
            )

        added_tokens = self.unique_added_tokens_encoder
        tokenized_text = split_on_tokens(added_tokens, text)
        return tokenized_text


class WoBertClassificationModel(nn.Module):
    def __init__(self, dataLoader: DataLoader, train_args: TrainArgs):
        '''
        bert 多分类模型
        '''
        super(WoBertClassificationModel, self).__init__()
        self.device = train_args.device
        self.max_length = train_args.max_length
        jieba.load_userdict(os.path.join(train_args.pretrained_weights_path, 'vocab.txt'))
        self.tokenizer = WoBertTokenizer.from_pretrained(train_args.pretrained_weights_path)

        '''
        测试词汇级分词是否成功：
        text = '全球性的厂里冠心病很严重'
        cc = self.tokenizer.tokenize(text=text)
        cc1 = self.tokenizer.encode(text=text)
        cc2 = self.tokenizer.decode(token_ids=cc1)
        '''

        self.bert = BertModel.from_pretrained(train_args.pretrained_weights_path)
        # train_args.pretrained_weights_path/config.json
        self.dropout = nn.Dropout(self.bert.config.hidden_dropout_prob)
        # CrossEntropyLoss 不需要加softmax
        self.classifier = nn.Linear(self.bert.config.hidden_size, dataLoader.laebl_size)
        self.to(self.device)

    def forward(self, batch_sentences):
        batch_tokenized = self.tokenizer.batch_encode_plus(batch_sentences,
                                                           add_special_tokens=True,
                                                           max_length=self.max_length,
                                                           pad_to_max_length=True)
        input_ids = torch.tensor(batch_tokenized['input_ids'], device=self.device)
        attention_mask = torch.tensor(batch_tokenized['attention_mask'], device=self.device)
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        # 提取[CLS]对应的隐藏状态
        output = outputs[0][:, 0, :]
        # 实验：提取CLS位置更高分数，池化后的更差
        # 提取迟化
        # output = outputs[1]

        # pooled_output = self.dropout(pooled_output)
        logits = self.classifier(output)
        return logits


def build_optim(_model, train_args, checkpoint=None):
    saved_optimizer_state_dict = None
    if checkpoint:
        optim = checkpoint['optim']
        saved_optimizer_state_dict = optim.state_dict()
    else:
        optim = AdamW(_model.parameters(), lr=train_args.lr, eps=1e-8)

    if train_args.train_from is not None:
        optim.load_state_dict(saved_optimizer_state_dict)
        if train_args.device != 'cpu':
            for state in optim.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda(device=train_args.device)
    return optim


def load_init_model(dataLoader: DataLoader, train_args: TrainArgs):
    _model = WoBertClassificationModel(dataLoader, train_args)
    checkpoint = None
    step = 0
    if train_args.train_from is not None:
        checkpoint = torch.load(train_args.train_from, map_location=train_args.device)
        step = int(train_args.train_from.split('_')[-1].split('.')[0])
        _model.load_state_dict(checkpoint['model'], strict=True)
    _optim = build_optim(_model, train_args, checkpoint)

    return _model, _optim, step
