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
        self.src_image_dir = osp.join(src_root, set, '')
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

    def _apply_transform(self, src_image, trg_image, boxes, target_size=640):
        """
        Resize ảnh về kích thước (640, 640) và điều chỉnh bounding boxes.
        """
        # Lấy kích thước ảnh gốc
        W, H = src_image.size[:2]

        # Resize ảnh về (640, 640)
        src_image = TF.resize(src_image, (target_size, target_size))
        trg_image = TF.resize(trg_image, (target_size, target_size))

        # Điều chỉnh bounding boxes nếu có
        # if len(boxes.shape) > 1 and boxes.shape[0] > 0:
        #     scale_x = target_size / W
        #     scale_y = target_size / H

        #     boxes[:, 0] = boxes[:, 0] * scale_x  # x_center
        #     boxes[:, 1] = boxes[:, 1] * scale_y  # y_center
        #     boxes[:, 2] = boxes[:, 2] * scale_x  # width
        #     boxes[:, 3] = boxes[:, 3] * scale_y  # height

        return src_image, trg_image, boxes

    def collate_fn(self, batch):
        """
        Gom nhóm batch và chuẩn hóa bounding boxes theo định dạng (batch_idx, label, x, y, w, h).
        """
        src_images, trg_images, boxes_list, labels_list, names = zip(*batch)

        # Stack ảnh thành batch
        src_images = torch.stack(src_images, 0)
        trg_images = torch.stack(trg_images, 0)

        # Danh sách chứa boxes đã xử lý
        all_boxes = []

        # print(names)

        # Duyệt từng ảnh trong batch để xử lý boxes
        for batch_idx, (boxes, labels) in enumerate(zip(boxes_list, labels_list)):
            if len(boxes) > 0:
                # print("$$$$$$$$$$$$$$")
                # print("name: ", names[batch_idx])
                # print(labels)
                # print(boxes)

                # Tạo tensor batch index có cùng số lượng boxes
                batch_indices = torch.full((len(boxes), 1), batch_idx, dtype=torch.float32)
                labels = labels.unsqueeze(1)  # Chuyển labels thành shape (num_bb, 1)

                # Nối các thông tin lại thành (batch_idx, label, x, y, w, h)
                new_boxes = torch.cat([batch_indices, labels, boxes], dim=1)
                all_boxes.append(new_boxes)

        # Gộp tất cả bounding boxes lại
        if all_boxes:
            boxes = torch.cat(all_boxes, dim=0)  # (total_boxes, 6)
        else:
            boxes = torch.empty((0, 6), dtype=torch.float32)  # Nếu batch không có box nào

        return src_images, trg_images, boxes, names
