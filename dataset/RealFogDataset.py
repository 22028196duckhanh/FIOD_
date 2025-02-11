import os.path as osp
import random

import numpy as np
import torch
from torch.utils import data

import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

from PIL import Image

class RealFogDataset(data.Dataset):
    def __init__(self, fog_root, list_file, max_iters=None, mean=(128, 128, 128)):
        """
        Args:
            fog_root (str): Đường dẫn đến thư mục gốc của dataset Foggy_Driving.
            list_file (str): Đường dẫn đến file chứa danh sách các ảnh (ví dụ: leftImg8bit_testall_filenames.txt).
            max_iters (int, optional): Số lượng tối đa các mẫu (nếu None, sử dụng tất cả).
            mean (tuple): Giá trị mean để chuẩn hóa ảnh.
        """
        self.fog_root = fog_root
        self.mean = mean

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

        # Áp dụng transform (nếu có)
        fog_image = self._apply_transform(fog_image)

        # Chuẩn hóa ảnh
        fog_image = np.asarray(fog_image, np.float32)
        fog_image = fog_image[:, :, ::-1].copy()  # RGB to BGR
        fog_image -= self.mean
        fog_image = fog_image.transpose((2, 0, 1))

        fog_image = torch.from_numpy(fog_image)

        return fog_image, name

    def _apply_transform(self, fog_image, scale=(0.7, 1.3), crop_size=160):
        (W, H) = fog_image.size[:2]
        if isinstance(scale, tuple):
            scale = random.random() * 0.6 + 0.7

        # Calculate minimum scale needed to reach crop_size
        scale_w = max(crop_size / W, scale)
        scale_h = max(crop_size / H, scale)
        final_scale = max(scale_w, scale_h)

        # Resize
        new_W, new_H = int(W * final_scale), int(H * final_scale)
        fog_image = TF.resize(fog_image, (new_H, new_W))

        # Random crop
        i, j, h, w = transforms.RandomCrop.get_params(fog_image, output_size=(crop_size, crop_size))
        fog_image = TF.crop(fog_image, i, j, crop_size, crop_size)

        # Random horizontal flip
        if random.random() > 0.5:
            fog_image = TF.hflip(fog_image)

        return fog_image

    def collate_fn(self, batch):
        """
        Hàm này được sử dụng để gom các mẫu trong batch lại với nhau.
        """
        fog_images, names = zip(*batch)
        fog_images = torch.stack(fog_images, 0)
        return fog_images, names