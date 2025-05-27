import clip
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.transforms import ToTensor

from .metrics import BaseMetric


def load_clip(model_name: str, device: str):
    """
    Get clip model with preprocess which can process torch images in value range of [-1, 1]

    Parameters
    ----------
    model_name : str
        CLIP-encoder type

    device : str
        Device for clip-encoder

    Returns
    -------
    model : nn.Module
        torch model of downloaded clip-encoder

    preprocess : torchvision.transforms.transforms
        image preprocess for images from stylegan2 space to clip input image space
            - value range of [-1, 1] -> clip normalized space
            - resize to 224x224
    """
    
    model, preprocess = clip.load(model_name, device=device)
    preprocess = transforms.Compose([
        transforms.Normalize(mean=[-1.0, -1.0, -1.0], std=[2.0, 2.0, 2.0]),
        *preprocess.transforms[:2],
        preprocess.transforms[-1]
    ])

    return model, preprocess


class ClipEncoder:
    def __init__(self, visual_encoder, device):
        self.model, self.preprocess = load_clip(visual_encoder, device)
        self.device = device
    
    def encode_text(self, text: str):
        tokens = clip.tokenize(text).to(self.device)
        text_features = self.model.encode_text(tokens).detach()
        text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features
    
    def encode_image(self, image: Image.Image):
        image_features = self.model.encode_image(self.preprocess(image))
        image_features /= image_features.clone().norm(dim=-1, keepdim=True)
        return image_features


class CLIPMetric(BaseMetric):
    def __init__(self, exp_name, dataset, filter_subset, **kwargs):
        super().__init__(exp_name, dataset, filter_subset, **kwargs)
        
        assert 'device' in kwargs, 'set device up'
        
        if 'vit_b_path' in kwargs:
            self.vit_b_32 = ClipEncoder(kwargs['vit_b_path'], kwargs['device'])
        else:
            self.vit_b_32 = ClipEncoder('ViT-B/32', kwargs['device'])

        if 'vit_l_path' in kwargs:
            self.vit_l_14 = ClipEncoder(kwargs['vit_l_path'], kwargs['device'])
        else:
            self.vit_l_14 = ClipEncoder('ViT-L/14', kwargs['device'])

        self.to_tensor = ToTensor()
        self.sim = nn.CosineSimilarity()
    
    def calc_object(
        self, 
        ids,
        prompts, 
        ref_imgs, 
        out_imgs
    ):
        
        bs = len(prompts)
        out_imgs = out_imgs.to(self.device)
        
        b_32_im_features = self.vit_b_32.encode_image(out_imgs)
        b_32_text_features = self.vit_b_32.encode_text(prompts)
        b_32_scores = self.sim(b_32_im_features, b_32_text_features).squeeze()
        
        l_14_im_features = self.vit_l_14.encode_image(out_imgs)
        l_14_text_features = self.vit_l_14.encode_text(prompts)
        l_14_scores = self.sim(l_14_im_features, l_14_text_features).squeeze()
        
        if bs > 1:
            l_14_scores = l_14_scores.chunk(bs)
            b_32_scores = b_32_scores.chunk(bs)
        else:
            l_14_scores = [l_14_scores]
            b_32_scores = [b_32_scores]
        

        return {
            "l_14_scores": [metric.item() for metric in l_14_scores],
            "b_32_scores": [metric.item() for metric in b_32_scores]
        }