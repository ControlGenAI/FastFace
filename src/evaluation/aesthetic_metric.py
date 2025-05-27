import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
import ImageReward as RM

from .metrics import BaseMetric
from .clip_metric import ClipEncoder


class ACUModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.layers(x)

    
def normalized(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)


class AestheticMetric(BaseMetric):
    def __init__(self, exp_name, dataset, filter_subset, **kwargs):
        super().__init__(exp_name, dataset, filter_subset, **kwargs)
        assert 'device' in kwargs, 'set device up'
        self.device = kwargs['device']
        model_path = kwargs['model_path'] if 'model_path' in kwargs else 'models_cache/ir/ava+logos-l14-linearMSE.pth'
        self.score_model = self._load_model_acu(768, model_path, kwargs['device'])
        if 'vit_l_path' in kwargs:
            self.clip_model = ClipEncoder(kwargs["vit_l_path"], device=kwargs['device'])
        else:
            self.clip_model = ClipEncoder("ViT-L/14", device=kwargs['device'])
        
        self.ir_model = self._load_model_ir(**kwargs)
        
    def _load_model_ir(self, **kwargs):
        return RM.load("ImageReward-v1.0" if 'ir_path' not in kwargs else kwargs['ir_path'])
    
    def _load_model_acu(self, input_dim, model_path, device='cuda'):        
        model = ACUModel(input_dim)
        ckpt = torch.load(model_path)
        model.load_state_dict(ckpt)
        model.to(device)
        model.eval()
        return model
    
    def get_scores_acu(self, trg_images):
        image_features = self.clip_model.encode_image(trg_images)
        im_emb_arr = normalized(image_features.cpu().detach().numpy())
        prediction = self.score_model(torch.from_numpy(im_emb_arr).to(self.device).type(torch.cuda.FloatTensor))
        prediction = prediction.chunk(prediction.size(0), dim=0)
        prediction = [p.cpu().item() for p in prediction]
        return prediction
    
    def get_single_score_ir(self, prompt, image):
        score = self.ir_model.score(
            prompt,
            image
        )
        return score
    
    def get_scores_ir(self, prompts, trg_images):
        images = [transforms.ToPILImage()((im + 1)/2) for im in trg_images]
        scores = [self.get_single_score_ir(p, im) for p, im in zip(prompts, images)]
        return scores
        
    def calc_object(
        self,
        ids,
        prompts,
        ref_imgs,
        out_imgs
    ):
        out_imgs = out_imgs.to(self.device)
        acu_scores = self.get_scores_acu(out_imgs)
        ir_scores = self.get_scores_ir(prompts, out_imgs)
                
        return {
            "acu_score": acu_scores,
            "ir_scores": ir_scores
        }