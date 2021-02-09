import os, sys
import datetime
import torch
from torch import nn
from src.utils.logers import LOGS
from tqdm import tqdm
import numpy as np
import random
from config.configs_interface import TrainArgs, configs
from src.data_utils.data_loader import DataLoader
from src.models import load_init_model

LOGS.init(os.path.join(configs.project.PROJECT_DIR, f'{configs.log.log_dir}/{configs.log.log_file_name}'))


def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(7)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


set_seed()


class Trainer:
    def __init__(self, dataLoader: DataLoader, train_args: TrainArgs):
        self.dataLoader = dataLoader
        self.train_args = train_args
        # 初始化模型
        _model, _optim, step = load_init_model(dataLoader=dataLoader, train_args=train_args)
        # 加载训练好的模型，断点训练
        if train_args.train_from:
            LOGS.log.debug(f'加载 step:{step}的模型')
            self.step = step + 1
        else:
            self.step = 0
        self.Accuracy = -1
        self.model = _model
        self.optimizer = _optim
        # 加权label，差不多效果
        # label_weight = torch.tensor(dataLoader.label_weight, device=train_args.device)
        # self.criterion = nn.CrossEntropyLoss(weight=label_weight)
        self.criterion = nn.CrossEntropyLoss()

    def _save(self, tag):

        checkpoint = {
            'model': self.model.state_dict(),
            'optim': self.optimizer,
        }

        checkpoint_path = os.path.join(self.train_args.save_model_dir,
                                       self.train_args.save_model_name + f'{tag}.pt')
        LOGS.log.debug("Saving checkpoint %s" % checkpoint_path)
        if not os.path.exists(checkpoint_path):
            # 1.7版本必须加 _usexx
            torch.save(checkpoint, checkpoint_path, _use_new_zipfile_serialization=False)
            del checkpoint

    def train(self):
        if self.train_args.train_from:
            Accuracy = self.batch_test()
            LOGS.log.debug(f'加载{self.train_args.train_from}模型，起始Accuracy：{Accuracy}')
        for epoch in range(self.train_args.epochs):
            print_avg_loss = 0.0
            for i in range(self.dataLoader.train_batch_count):
                inputs = self.dataLoader.batch_train_inputs[i].copy()
                self.model.train(mode=True)
                labels = torch.tensor(self.dataLoader.batch_train_targets[i].copy(), device=self.train_args.device)
                logits = self.model(inputs)
                # _logits = logits.view(-1, self.dataLoader.laebl_size)
                loss = self.criterion(logits, labels)
                # (loss.sum() / loss.numel()).backward()
                loss.backward()
                al = loss.cpu().detach().numpy().tolist()
                print_avg_loss += al
                self.optimizer.step()
                self.model.zero_grad()
                if i % self.train_args.print_every_batch == (self.train_args.print_every_batch - 1):
                    LOGS.log.debug(
                        f"Epoch:{epoch} ｜ Batch: {(i + 1)}, Loss: {print_avg_loss / self.train_args.print_every_batch}")
                    print_avg_loss = 0.0
                if self.step % self.train_args.save_checkpoint_steps == 0 and self.step >= self.train_args.save_checkpoint_steps:
                    tag = ''
                    better = False
                    if configs.train_args.on_eval:
                        Accuracy = self.batch_test()
                        if self.Accuracy == -1 or Accuracy > self.Accuracy:
                            self.Accuracy = Accuracy
                            better = True
                            LOGS.log.debug(f'better Accuracy:{Accuracy},step:{self.step}', email=False)
                        tag = f'_Accuracy_{round(Accuracy, 4)}'
                        tag = tag.replace('.', '_')
                    tag += '_step_' + str(self.step)
                    if better:
                        self._save(tag)
                self.step += 1
            # 显存增加风险
            del labels, logits, loss, print_avg_loss

    def test_cmd(self):
        self.model.train(mode=False)
        with torch.no_grad():
            LOGS.log.debug(f'输入：')
            inp = input()
            while inp != 'exit':
                if len(inp.strip()) < 3:
                    inp = input()
                    continue
                inputs = [inp]
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs, 1)
                predict_label = self.dataLoader.index_to_label[predicted.item()]
                prob = torch.softmax(outputs, dim=1)
                probs = prob.cpu().detach().numpy().tolist()[0]
                logs = [li + f"({probs[i]}" for i, li in enumerate(dataLoader.labels)]
                LOGS.log.debug(f'标签：{logs}')
                LOGS.log.debug(f'max prob label:{predict_label}, | prob :{probs[predicted.item()]}')
                inp = input()
                del outputs, predicted, prob

    def batch_test(self):
        eval = configs.train_args.on_eval
        if configs.train_args.mode == 'test':
            eval = False
        total = self.dataLoader.test_batch_count * self.dataLoader.batch_size
        if not eval:
            LOGS.log.debug(f'全部测试数据量：{total}')
            LOGS.log.debug(f'测试数据批次：{self.dataLoader.test_batch_count}')
        hit = 0
        # self.model.eval()
        self.model.zero_grad()
        self.optimizer.zero_grad()
        self.model.train(mode=False)
        # 显存增加风险
        with torch.no_grad():
            bad_case_data = []
            for i in tqdm(range(self.dataLoader.test_batch_count)):
                outputs = self.model(self.dataLoader.batch_test_inputs[i].copy())
                _, predicted = torch.max(outputs, 1)
                if configs.data.bad_case_data and not eval:
                    prob = torch.softmax(outputs, dim=1)
                    probs = prob.cpu().detach().numpy().tolist()
                predict_labels = [self.dataLoader.index_to_label[pi.item()] for pi in predicted]
                target_labels = [self.dataLoader.index_to_label[pi] for pi in self.dataLoader.batch_test_targets[i]]
                # if not eval and i % self.train_args.print_every_batch == 0:
                #     LOGS.log.debug(f"{i}  , target_label{target_labels} __  predict_labels:{predict_labels}")
                for j, _ in enumerate(predict_labels):
                    if predict_labels[j] == target_labels[j]:
                        hit += 1
                    # 统计bad_case，且不是训练模型中的评价
                    elif configs.data.bad_case_data and not eval:
                        input_ij = self.dataLoader.batch_test_inputs[i][j]
                        input_ij_predict = predict_labels[j]
                        input_ij_target = target_labels[j]
                        input_ij_prob = str(round(probs[j][self.dataLoader.label_to_index[input_ij_predict]], 4))
                        bad_case_data.append([input_ij, input_ij_predict, input_ij_target, input_ij_prob])

            if not eval and configs.data.bad_case_data:
                with open(configs.data.bad_case_data, mode='w', encoding='utf-8') as wf:
                    wf.write('预测' + ',' + '标注' + ',' + '标题' + ',' + '概率' + '\n')
                    for bad_case in bad_case_data:
                        input_ij = bad_case[0]
                        input_ij_predict = bad_case[1]
                        input_ij_target = bad_case[2]
                        input_ij_prob = bad_case[3]
                        wf.write(
                            input_ij_predict + ',' + input_ij_target + ',' + input_ij + ',' + input_ij_prob + '\n')
                LOGS.log.debug(f"all badcase num: {len(bad_case_data)}")
        Accuracy = hit / total * 100
        LOGS.log.debug(f"step:{self.step} model,Accuracy: {Accuracy}%")
        # 评估完，可以继续训练
        self.model.train(mode=True)
        return Accuracy


if __name__ == '__main__':
    set_seed()

    dataLoader = DataLoader()
    trainer = Trainer(dataLoader=dataLoader, train_args=configs.train_args)
    LOGS.log.debug(f'程序运行开始，{datetime.datetime.now()}')
    try:
        if configs.train_args.mode == 'train':
            trainer.train()
        elif configs.train_args.mode == 'test':
            trainer.batch_test()
        elif configs.train_args.mode == 'cmd':
            trainer.test_cmd()
    except Exception as e:
        LOGS.log.error(f'程序意外退出，{datetime.datetime.now()},error:{e}')
    LOGS.log.debug(f'程序运行结束，{datetime.datetime.now()}')
