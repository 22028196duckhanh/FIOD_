import os

import numpy as np
import torch
from torch.optim import lr_scheduler

from config.config import args
from yolov9_main.models.yolo import Model
from yolov9_main.train import parse_opt
from yolov9_main.utils.dataloaders import create_dataloader
from yolov9_main.utils.general import yaml_load, colorstr, one_cycle, one_flat_cycle, check_amp
from yolov9_main.utils.loss import hyp, ComputeLoss
from yolov9_main.utils.metrics import fitness
from yolov9_main.utils.torch_utils import smart_optimizer, ModelEMA, EarlyStopping
import yolov9_main.val as validate


def intersect_dicts(da, db, exclude=()):
    return {k: v for k, v in da.items() if k in db and all(x not in k for x in exclude) and v.shape == db[k].shape}

if __name__ == "__main__":
    print(torch.__version__)
    
    # opt = parse_opt()
    # hyp = yaml_load('yolov9_main/data/hyps/hyp.scratch-high.yaml')
    # optimizer = smart_optimizer(model, opt.optimizer, hyp['lr0'], hyp['momentum'], hyp['weight_decay'])
    # RANK = int(os.getenv('RANK', -1))
    # save_dir = os.path.join(os.path.dirname(__file__), 'results')
    # print(save_dir)
    # RANK = int(os.getenv('RANK', -1))
    # hyp = yaml_load('yolov9_main/data/hyps/hyp.scratch-high.yaml')
    # opt = parse_opt()
    # nc = 2
    # names = {0: 'person', 2: 'car'}

    # model = Model(cfg='yolov9_main/models/detect/yolov9-s.yaml')
    # checkpoint = torch.load('yolov9-s.pt', map_location='cpu')
    # model.load_state_dict(checkpoint, strict = False)

    checkpoint_path = r"E:\lab\FIOD_\yolov9-s.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = Model(checkpoint['model'].yaml).to("cpu")
    csd = checkpoint['model'].float().state_dict()
    csd = intersect_dicts(csd, model.state_dict(), exclude=())
    model.load_state_dict(csd, strict=False)

    print(model.parameters())


