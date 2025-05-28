import json
import os
from pathlib import Path
from typing import Callable
from collections import defaultdict

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
from PIL import Image
from tqdm.auto import tqdm


class ExperimentDataset(Dataset):
    def __init__(
        self,
        exp_name,
        dataset,
        filter_subset: Callable=None,
    ):
        super().__init__()
        self.exp_root = Path(exp_name)
        assert self.exp_root.exists(), 'No such experiment'
        
        self.dataset = dataset
        self.images_root = Path(self.dataset.img_root)
        self.filter_subset = filter_subset
        
        image_names = [img for img in os.listdir(self.exp_root) if img.endswith(".png")] 
        self.exp_images = self._sort_exp_images(image_names) # sort files according to *idx*_.png

        if self.filter_subset is not None:
            filtered_ids = [idx for idx, (id_photo, prompt) in enumerate(self.dataset) if self.filter_subset(id_photo, prompt)]
            self.ids = filtered_ids
        else:
            self.ids = list(range(len(self.dataset)))

        assert len(self.dataset) == len(self.exp_images), "dataset and exp_images must have the same length, but got {} and {}".format(len(self.dataset), len(self.exp_images))
        self.img_to_idx = {img: idx for idx, img in enumerate(self.exp_images)}
        
        self.to_tensor = ToTensor()

    def update_target_exp(self, exp_name):
        self.exp_root = Path(exp_name)
        assert self.exp_root.exists(), 'No such experiment'
        image_names = [img for img in os.listdir(self.exp_root) if img.endswith(".png")] 
        self.exp_images = self._sort_exp_images(image_names) # sort files according to *idx*_.png
        assert len(self.dataset) == len(self.exp_images), "dataset and exp_images must have the same length"
        self.img_to_idx = {img: idx for idx, img in enumerate(self.exp_images)}
        
    def __len__(self):
        return len(self.ids)

    def _sort_exp_images(self, image_names):
        idxs = [int(imgn.split('.')[0].split("_")[0]) for imgn in image_names]
        table = {i: name for i, name in zip(idxs, image_names)} 
        sorted_images = [table[i] for i in sorted(idxs)]
        return sorted_images
    
    def __getitem__(self, idx):
        _, prompt = self.dataset[self.ids[idx]]
        ref_img_path, out_img_papth = self.get_imgs_paths(idx)
        ref_img, out_img = self._get_images(ref_img_path, out_img_papth)
        return idx, prompt, ref_img, out_img
    
    def _process_image(self, image):
        tensor_im = self.to_tensor(image)
        return (tensor_im - 0.5) * 2

    def get_imgs_paths(self, idx):
        ref_img_name, _ = self.dataset[self.ids[idx]]
        out_img_name = self.exp_images[self.ids[idx]]
        
        ref_img_path = self.images_root / ref_img_name
        out_img_path = self.exp_root / out_img_name
        
        return ref_img_path, out_img_path
    
    def _get_images(self, ref_img_path, out_img_path):
        ref_img = Image.open(ref_img_path)
        out_img = Image.open(out_img_path)
        ref_img_tensor = self._process_image(ref_img)
        out_img_tensor = self._process_image(out_img)
        return ref_img_tensor, out_img_tensor


class BaseMetric:
    def __init__(
        self, 
        exp_name,
        dataset,
        filter_subset: Callable=None,
        batch_size=8,
        device='cuda:0',
        **kwargs
    ):
        self.exp_root = Path(exp_name)
        self.batch_size = batch_size
        assert self.exp_root.exists(), 'No such experiment'
        
        self.dataset = ExperimentDataset(
            exp_name,
            dataset,
            filter_subset=filter_subset
        )
        self.loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)
        self.device = device

        if os.path.exists(self.exp_root / "ratios.json"):
            self.ratios = json.load(open(self.exp_root / "ratios.json"))

    def update_target_exp(self, exp_name):
        self.exp_root = Path(exp_name)
        assert self.exp_root.exists(), 'No such experiment'
        self.dataset.update_target_exp(exp_name)
        self.loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)
        if os.path.exists(self.exp_root / "ratios.json"):
            self.ratios = json.load(open(self.exp_root / "ratios.json"))
    
    @torch.no_grad()
    def __call__(self, rthresh=None):
        metric_result = defaultdict(list)
        for ids, prompts, ref_imgs, out_imgs in tqdm(self.loader, desc=self.exp_root.stem, leave=True):
            if hasattr(self, 'ratios'):
                filtered_ids = [idx for idx in ids if self.ratios[idx] >= rthresh] if rthresh is not None else ids
            else:
                filtered_ids = ids
            batch_metrics = self.calc_object(filtered_ids, prompts, ref_imgs, out_imgs)
            for m in batch_metrics.keys():
                metric_result[m] += batch_metrics[m]
        return metric_result
    
    def calc_object(
        self,
        ids,
        prompts, 
        ref_imgs, 
        out_imgs
    ):
    
        raise NotImplementedError('Implement self.calc_object()')
