import torch
from torch import nn, optim
from transformers import BertModel, BertTokenizer
from src.utils.logers import LOGS
from src.data_utils.data_loader import DataLoader
from config.configs_interface import TrainArgs


class RoBertaClassificationModel(nn.Module):
    def __init__(self, dataLoader: DataLoader, train_args: TrainArgs):
        '''
        bert 多分类模型
        '''
        super(RoBertaClassificationModel, self).__init__()  # -1,或者设备不支持GPU
        self.device = train_args.device
        self.max_length = train_args.max_length
        self.tokenizer = BertTokenizer.from_pretrained(train_args.pretrained_weights_path)
        self.bert = BertModel.from_pretrained(train_args.pretrained_weights_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        # 默认的隐藏单元数是self.bert.config.hidden_size， 输出单元是标签数量，表示 二/多 分类
        self.dense = nn.Linear(self.bert.config.hidden_size, len(dataLoader.label_to_index))
        # self.softmax = nn.Softmax(dim=1)
        # self.dropout = nn.Dropout(dropout)
        self.to(self.device)

    def forward(self, batch_sentences):
        batch_tokenized = self.tokenizer.batch_encode_plus(batch_sentences, add_special_tokens=True,
                                                           max_length=self.max_length,
                                                           pad_to_max_length=True)
        input_ids = torch.tensor(batch_tokenized['input_ids'], device=self.device)
        attention_mask = torch.tensor(batch_tokenized['attention_mask'], device=self.device)
        bert_output = self.bert(input_ids, attention_mask=attention_mask)
        bert_cls_hidden_state = bert_output[0][:, 0, :]  # 提取[CLS]对应的隐藏状态
        linear_output = self.dense(bert_cls_hidden_state)
        # out = self.dropout(linear_output)
        # linear_output = self.dense2(out)
        # softmax_out = self.softmax(linear_output)
        return linear_output


def load_train_from(model, optimizer, train_args: TrainArgs):
    checkpoint = torch.load(train_args.train_from, map_location=train_args.device)
    model.load_state_dict(checkpoint['model'], strict=True)

    optimizer.load_state_dict(checkpoint['optim'])
    # 启用GPU
    if train_args.device != 'cpu':
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda(device=train_args.device)
    LOGS.log.debug(f'加载模型：{train_args.train_from}')
    return model, optimizer


def load_init_model(dataLoader: DataLoader, train_args: TrainArgs):
    _model = RoBertaClassificationModel(dataLoader, train_args)
    _optim = optim.Adam(_model.parameters(), lr=train_args.lr)
    return _model, _optim
