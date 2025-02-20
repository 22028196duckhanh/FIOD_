import os.path as osp
import random

import numpy as np
import torch
from torch.utils import data

import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

from PIL import Image

class RealFogDataset(data.Dataset):
    def __init__(self, fog_root, list_file, max_iters=None):
        """
        Args:
            fog_root (str): Đường dẫn đến thư mục gốc của dataset Foggy_Driving.
            list_file (str): Đường dẫn đến file chứa danh sách các ảnh (ví dụ: leftImg8bit_testall_filenames.txt).
            max_iters (int, optional): Số lượng tối đa các mẫu (nếu None, sử dụng tất cả).
            mean (tuple): Giá trị mean để chuẩn hóa ảnh.
        """
        self.fog_root = fog_root

        # Đọc danh sách ảnh từ file
        with open(list_file, 'r') as f:
            self.img_ids = [line.strip() for line in f.readlines()]

        if max_iters is not None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))

        self.files = []
        for img_id in self.img_ids:
            fog_img_file = osp.join(self.fog_root, img_id)
            self.files.append({
                "fog_img": fog_img_file,
                "name": osp.basename(img_id)  # Lấy tên file từ đường dẫn
            })

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]
        fog_image = Image.open(datafiles["fog_img"]).convert('RGB')
        name = datafiles["name"]

        fog_image = self._apply_transform(fog_image)
        fog_image = np.asarray(fog_image, np.float32) / 255.0


        fog_image = fog_image[:, :, ::-1].copy()  # RGB to BGR

        mean = np.array([104.00698793, 116.66876762, 122.67891434], dtype=np.float32) / 255.0
        fog_image -= mean

        fog_image = fog_image.transpose((2, 0, 1))

        fog_image = torch.from_numpy(fog_image)

        return fog_image, name

    def _apply_transform(self, fog_image, target_size=640):
        """
        Resize ảnh về kích thước cố định 640x640.
        """
        fog_image = TF.resize(fog_image, (target_size, target_size))
        return fog_image

    def collate_fn(self, batch):
        """
        Hàm này được sử dụng để gom các mẫu trong batch lại với nhau.
        """
        fog_images, names = zip(*batch)
        fog_images = torch.stack(fog_images, 0)
        return fog_images, names