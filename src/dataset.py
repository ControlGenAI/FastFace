import os
from itertools import product

from torch.utils.data import Dataset
from PIL import Image

DEFAULT_NEG_PROMPT = ""


def check_images(root_dir: str):
    """utility function to check that all images are readable"""
    img_files = [f for f in os.listdir(root_dir) if f.endswith(".png")]
    broken_images = []
    for imf in img_files:
        try:
            Image.open(os.path.join(root_dir, imf))
        except Exception as e:
            broken_images.append(imf)

    if len(broken_images) > 0:
        raise Exception(f"Error: several images in dataset are not readable: {broken_images}")
    

class FaceIdDataset(Dataset):
    """Evaluation dataset for ID adapters"""
    def __init__(
        self, 
        data_dir, 
        prompts_set="full",
        neg_prompt="", 
    ):
        self.data_dir = data_dir
        self.img_root = os.path.join(data_dir, "id_photos")
        self.neg_prompt = ""

        self.prompts_set = prompts_set
        self.prompts = self.read_prompts(data_dir)
        self.id_photos = [img for img in os.listdir(self.img_root) if img.endswith(".png")] 

        check_images(self.img_root)
        
        self.prompts = self.prompts # usefull for metrics analysis

        print(f"photos: {len(self.id_photos)}, prompts: {len(self.prompts)}")
        self.dataset = list(product(self.id_photos, self.prompts))

    def read_prompts(self, data_dir):
        set2file = {
            "full": "prompts.txt",
            "realistic": "prompts_nostyle.txt",
            "style": "prompts_style.txt",
        }
        prompts_file = set2file[self.prompts_set]
        with open(os.path.join(data_dir, prompts_file), "r+") as f:
            prompts = f.readlines()
        prompts = [p.strip().lower() for p in prompts]
        return prompts
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        id_photo, prompt = self.dataset[idx]        
        return id_photo, prompt
