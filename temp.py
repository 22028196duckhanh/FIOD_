import os

from yolov9_main.models.yolo import Model
from yolov9_main.train import parse_opt
from yolov9_main.utils.general import yaml_load
from yolov9_main.utils.torch_utils import smart_optimizer
model = Model(cfg = 'yolov9_main/models/detect/yolov9-s.yaml')
opt = parse_opt()
hyp = yaml_load('yolov9_main/data/hyps/hyp.scratch-high.yaml')
optimizer = smart_optimizer(model, opt.optimizer, hyp['lr0'], hyp['momentum'], hyp['weight_decay'])
RANK = int(os.getenv('RANK', -1))
print(RANK)