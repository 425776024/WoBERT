from typing import Optional

from pydantic import BaseModel
import yaml, pathlib, os
from pydantic import validator
import torch
from src.utils.logers import LOGS


class LogConfigs(BaseModel):
    log_dir: str
    log_file_name: str


class ProjectConfigs(BaseModel):
    PROJECT_DIR: str = pathlib.Path(__file__).resolve().parents[1]
    PROJECT_NAME: str = "news"
    VERSION: str = "beta1"


class DataConfigs(BaseModel):
    train_data: str = 'data/train.csv'
    bad_case_data: str = 'data/bad_case.txt'

    @validator("train_data", pre=False, always=True)
    def set_train_data(cls, v, values):
        project_path = pathlib.Path(__file__).resolve().parents[1]
        v = os.path.join(project_path, v)
        return v


    @validator("bad_case_data", pre=False, always=True)
    def set_bad_case_data(cls, v, values):
        project_path = pathlib.Path(__file__).resolve().parents[1]
        v = os.path.join(project_path, v)
        return v


class TrainArgs(BaseModel):
    mode: str
    gpus: str
    device: str = ''
    save_model_dir: str
    save_model_name: str

    train_from: str = None
    pretrained_weights_path: str = None
    test_date_rate: float
    lr: float
    epochs: int
    warmup_steps: int = 0
    weight_decay: int = 0
    print_every_batch: int
    save_checkpoint_steps: int
    max_length: int
    batch_size: int
    on_eval: bool

    @validator("device", pre=False, always=True)
    def set_device(cls, v, values):
        device = "cpu" if values['gpus'] == 'cpu' or torch.cuda.is_available() == False else values['gpus']
        return device

    @validator("save_model_dir", pre=False, always=True)
    def set_save_model_dir(cls, v, values):
        project_path = pathlib.Path(__file__).resolve().parents[1]
        v = os.path.join(project_path, v)
        if not os.path.exists(v):
            os.mkdir(v)
        return v

    @validator("train_from", pre=False, always=True)
    def set_train_from(cls, v, values):
        if v is None:
            return v
        v = os.path.join(values['save_model_dir'], v)
        if not os.path.exists(v):
            os.mkdir(v)
        return v


class Configs(BaseModel):
    log: LogConfigs
    project: ProjectConfigs
    data: DataConfigs
    train_args: TrainArgs


def read_yaml(file_path):
    with open(file_path) as f:
        return yaml.safe_load(f)


project_path = pathlib.Path(__file__).resolve().parents[1]
curr_conf_path = os.path.join(project_path, 'config')

configs_yaml = os.path.join(curr_conf_path, 'configs.yaml')
configs = Configs(**read_yaml(str(configs_yaml)))

LOGS.init(os.path.join(configs.project.PROJECT_DIR, f'{configs.log.log_dir}/{configs.log.log_file_name}'))

LOGS.log.debug(configs.dict())
