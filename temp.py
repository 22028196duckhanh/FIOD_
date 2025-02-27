import torch
import sys
path = r"E:\lab\FIOD_\yolov9_main"
sys.path.insert(0, path)

from models.yolo import Model

def intersect_dicts(da, db, exclude=()):
    return {k: v for k, v in da.items() if k in db and all(x not in k for x in exclude) and v.shape == db[k].shape}

checkpoint_path = r"E:\lab\FIOD_\yolov9-s.pt"
checkpoint = torch.load(checkpoint_path, map_location="cpu")
model = Model(checkpoint['model'].yaml).to("cpu")
csd = checkpoint['model'].float().state_dict()
csd = intersect_dicts(csd, model.state_dict(), exclude=())
model.load_state_dict(csd, strict=False)

print()