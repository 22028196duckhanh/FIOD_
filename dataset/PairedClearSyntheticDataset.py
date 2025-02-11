import os
import os.path as osp
import random

import numpy as np
import torch
from torch.utils import data

import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

from PIL import Image

class PairedClearSyntheticDataset(data.Dataset):
    def __init__(self, src_root, trg_root, set='train', max_iters=None, mean=(128, 128, 128)):
        """
        Args:
            src_root (str): Đường dẫn đến thư mục gốc của dataset foggy.
            trg_root (str): Đường dẫn đến thư mục gốc của dataset clear.
            set (str): 'train' hoặc 'val'.
            max_iters (int, optional): Số lượng tối đa các mẫu (nếu None, sử dụng tất cả).
            mean (tuple): Giá trị mean để chuẩn hóa ảnh.
        """
        self.src_root = src_root
        self.trg_root = trg_root
        self.set = set
        self.mean = mean

        # Đọc danh sách ảnh từ thư mục images
        self.src_image_dir = osp.join(src_root, set, 'images')
        self.trg_image_dir = osp.join(trg_root, set, 'images')
        self.label_dir = osp.join(trg_root, set, 'labels')

        self.img_ids = [f for f in os.listdir(self.src_image_dir) if f.endswith('.jpg') or f.endswith('.png')]

        if max_iters is not None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))

        self.files = []
        for img_id in self.img_ids:
            src_img_file = osp.join(self.src_image_dir, img_id)
            trg_img_file = osp.join(self.trg_image_dir, img_id)
            label_file = osp.join(self.label_dir, img_id.replace('.jpg', '.txt').replace('.png', '.txt'))
            self.files.append({
                "src_img": src_img_file,
                "trg_img": trg_img_file,
                "label": label_file,
                "name": img_id
            })

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]
        src_image = Image.open(datafiles["src_img"]).convert('RGB')
        trg_image = Image.open(datafiles["trg_img"]).convert('RGB')
        label_path = datafiles["label"]
        name = datafiles["name"]

        # Đọc label (bounding boxes và class IDs)
        with open(label_path, 'r') as f:
            lines = f.readlines()
        boxes = []
        labels = []
        for line in lines:
            class_id, x_center, y_center, width, height = map(float, line.strip().split())
            boxes.append([x_center, y_center, width, height])
            labels.append(int(class_id))

        # Chuyển đổi sang tensor
        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        # Áp dụng transform (nếu có)
        src_image, trg_image, boxes = self._apply_transform(src_image, trg_image, boxes)

        # Chuẩn hóa ảnh
        src_image = np.asarray(src_image, np.float32)
        trg_image = np.asarray(trg_image, np.float32)
        src_image = src_image[:, :, ::-1].copy()  # RGB to BGR, create a copy here
        src_image -= self.mean
        src_image = src_image.transpose((2, 0, 1))
        trg_image = trg_image[:, :, ::-1].copy()  # RGB to BGR, create a copy here
        trg_image -= self.mean
        trg_image = trg_image.transpose((2, 0, 1))

        src_image = torch.from_numpy(src_image)
        trg_image = torch.from_numpy(trg_image)

        return src_image, trg_image, boxes, labels, name

    def _apply_transform(self, src_image, trg_image, boxes, scale=(0.7, 1.3), crop_size=160):
        """
        Áp dụng transform lên ảnh và bounding boxes.
        """
        (W, H) = src_image.size[:2]
        if isinstance(scale, tuple):
            scale = random.random() * 0.6 + 0.7

        # Resize
        new_W, new_H = int(W * scale), int(H * scale)
        src_image = TF.resize(src_image, (new_H, new_W))
        trg_image = TF.resize(trg_image, (new_H, new_W))

        # Random crop
        i, j, h, w = transforms.RandomCrop.get_params(src_image, output_size=(crop_size, crop_size))
        src_image = TF.crop(src_image, i, j, h, w)
        trg_image = TF.crop(trg_image, i, j, h, w)

        # Random horizontal flip
        if random.random() > 0.5:
            src_image = TF.hflip(src_image)
            trg_image = TF.hflip(trg_image)

        # Adjust bounding boxes only if boxes exist
        if len(boxes.shape) > 1 and boxes.shape[0] > 0: # Check if boxes tensor has more than one dimension and contains boxes
            boxes[:, 0] = (boxes[:, 0] * W - j) / w  # x_center
            boxes[:, 1] = (boxes[:, 1] * H - i) / h  # y_center
            boxes[:, 2] = boxes[:, 2] * W / w        # width
            boxes[:, 3] = boxes[:, 3] * H / h        # height

        return src_image, trg_image, boxes

    def collate_fn(self, batch):
        """
        Hàm này được sử dụng để gom các mẫu trong batch lại với nhau.
        """
        src_images, trg_images, boxes, labels, names = zip(*batch)
        src_images = torch.stack(src_images, 0)
        trg_images = torch.stack(trg_images, 0)

        # Tìm max length trong batch
        max_boxes = max(len(b) for b in boxes)
        max_labels = max(len(l) for l in labels)

        # Pad và stack boxes
        padded_boxes = []
        padded_labels = []

        for box, label in zip(boxes, labels):
            # Nếu box rỗng, tạo tensor rỗng với shape phù hợp
            if len(box) == 0:
                padded_box = torch.zeros((max_boxes, 4), dtype=torch.float32)  # Giả sử bbox có 4 giá trị (x, y, w, h)
            else:
                pad_size = max_boxes - len(box)
                if pad_size > 0:
                    padded_box = torch.cat([box, torch.zeros((pad_size, box.size(1)), dtype=box.dtype)])
                else:
                    padded_box = box
            padded_boxes.append(padded_box)

            # Nếu label rỗng, tạo tensor rỗng với shape phù hợp
            if len(label) == 0:
                padded_label = torch.full((max_labels,), -1, dtype=torch.long)
            else:
                pad_size = max_labels - len(label)
                if pad_size > 0:
                    padded_label = torch.cat([label, torch.full((pad_size,), -1, dtype=label.dtype)])
                else:
                    padded_label = label
            padded_labels.append(padded_label)

            # Stack tất cả
        if padded_boxes:
            boxes = torch.stack(padded_boxes, 0)
        else:
            boxes = torch.empty((0, max_boxes, 4), dtype=torch.float32)  # Nếu không có box nào trong batch

        if padded_labels:
            labels = torch.stack(padded_labels, 0)
        else:
            labels = torch.empty((0, max_labels), dtype=torch.long)  # Nếu không có label nào trong batch

        return src_images, trg_images, boxes, labels, names