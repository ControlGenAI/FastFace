import gc
import json
import os
from typing import List
from collections import defaultdict


from tqdm.auto import tqdm
from insightface.app import FaceAnalysis
import torch
import numpy as np
import cv2
import torch.nn as nn
from PIL import Image
import torchvision.transforms.functional as TF
from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization

from .metrics import BaseMetric, ExperimentDataset
from .clip_metric import CLIPMetric

class NoFaceFoundError(Exception):
    def __init__(self, message="face was not detected!"):
        super().__init__(message)
        self.message = message

    def __str__(self):
        return self.message


def pt_to_numpy(imgt):
    img_npy = imgt.detach().cpu().permute(0, 2, 3, 1).float().numpy()
    return img_npy


class FaceFeatExtract:
    def __init__(self, pretrained_ckpt='vggface2', im_size=(160, 160), device=torch.device("cuda"), dtype=torch.float32):
        self.im_size = im_size
        self.device = device
        self.mtcnn = MTCNN(image_size=im_size, device=device)
        self.resnet = InceptionResnetV1(pretrained=pretrained_ckpt, classify=False, device=self.device).to(dtype).eval()
        
        for param in self.resnet.parameters():
            param.requires_grad = False

    def get_face_bb(self, img_tensor: torch.Tensor, diff_out=False):
        tensor = img_tensor.detach().clone()
        if diff_out:
            tensor *= 255 # mtcnn expects 0-255 range
        batch_boxes, _ = self.mtcnn.detect(pt_to_numpy(tensor), landmarks=False)
        if batch_boxes.shape == (1,):
            raise NoFaceFoundError()
        return batch_boxes[0] # first face is largest, unless select_largest=False

    def extract_face_embed(self, img_tensor: torch.Tensor, bb:List=None, diff_out=True):
        if len(img_tensor.shape) == 3:
            img_tensor = img_tensor.unsqueeze(0)
        if bb is None:
            bb = self.get_face_bb(img_tensor)[0]
        # crop and process
        bb = [int(b) for b in bb]
        img_tensor = img_tensor[:, :, bb[1]:bb[3], bb[0]:bb[2]]
        if not diff_out:
            img_tensor = fixed_image_standardization(img_tensor)
        embed = self.resnet(img_tensor)
        return embed


def get_face_ratios(exp_data: ExperimentDataset, backend_model="buffalo_l", save=True):
    """computes a ratio of face bb to img size for each image, saves ratios.json file in experiment directory"""
    ratios = dict()
    face_app = FaceAnalysis(name=backend_model, providers=['CUDAExecutionProvider'])

    for idx, _, _, out_img in tqdm(iter(exp_data), desc= "building ratios.json for " + str(exp_data.exp_root), leave=True):
        # try to detect face
        _, out_img_path = exp_data.get_imgs_paths(idx)
        img = cv2.imread(out_img_path)
        for size in [(size, size) for size in range(640, 256, -64)]:
            face_app.det_model.input_size = size
            faces = face_app.get(img)
        
            if len(faces) == 0:
                continue
        if len(faces) == 0:
            ratios[idx] = 0. # 0/x = 0
            continue
        bb = faces[0].bbox
        bw, bh = bb[2] - bb[0], bb[3] - bb[1]
        bs = bw * bh
        ims = out_img.shape[-1] * out_img.shape[-2]
        ratios[idx] = bs / ims

    if save:
        with open(exp_data.exp_root / "ratios.json", "w") as f:
            json.dump(ratios, f)

    del face_app
    gc.collect()

    return ratios


def eval_metric_per_bins(metric: BaseMetric, bin_size=50):
    """sort images by ratios, compute mean fsim in bins of bin_size images, used to find evaluation threshold"""
    ratios = metric.dataset.ratios
    sorted_ids = np.array([k for k, v in sorted(ratios.items(), key=lambda kv: kv[1])])

    def chunker(seq, size):
        return (seq[pos:pos + size] for pos in range(0, len(seq), size))

    ratios_per_bins = dict()
    for ids_bin in chunker(sorted_ids, bin_size):
        face_dist_metrics, face_fail_cnt = metric(ids=ids_bin.astype(int))
        mean_bb_ratio = np.mean([ratios[idx] for idx in ids_bin])
        ratios_per_bins[mean_bb_ratio] = (np.mean(list(face_dist_metrics.values())), face_fail_cnt)

    return ratios_per_bins


class FaceDistanceMetric(BaseMetric):
    """Metric used ot estimate facial similarity in generated and original images"""
    def __init__(self, exp_name, dataset, filter_subset, **kwargs):
        super().__init__(exp_name, dataset, filter_subset, **kwargs)
        self.backend_model = "buffalo_l"
        device_id = torch.cuda.current_device()
        # TODO: remove this
        device_id = self.device.index
        self.face_app = FaceAnalysis(
            name=self.backend_model, 
            providers=['CUDAExecutionProvider'],
            provider_options=[{"device_id": device_id}])
        self.face_app.prepare(ctx_id=device_id, det_size=(640, 640))
        self.sim = nn.CosineSimilarity(dim=0)
        self.extract_fail_cnt = 0

        if os.path.exists(self.exp_root / "ratios.json"):
            self.ratios = json.load(open(self.exp_root / "ratios.json"))

    def extract_face_embed(self, image_path: str = "", image=None):
        if image is None:
            img = cv2.imread(image_path)
        else:
            img = image

        for size in [(size, size) for size in range(640, 256, -64)]:
            self.face_app.det_model.input_size = size
            faces = self.face_app.get(img)
        
            if len(faces) == 0:
                continue
            face_embed = faces[0].embedding
            face_embed /= np.linalg.norm(face_embed)
            return torch.from_numpy(face_embed).to(self.device)
        
        print(f"face-det failed: no face found in the image")
        return None

    def calc_object(self, idx, prompt, ref_img, out_img):
        ref_img_path, out_img_path = self.dataset.get_imgs_paths(idx)

        embedding1 = self.extract_face_embed(ref_img_path)
        embedding2 = self.extract_face_embed(out_img_path)

        if embedding1 is not None and embedding2 is not None:
            embedding1 = embedding1.to(self.device)
            embedding2 = embedding2.to(self.device)
            cosdist = self.sim(embedding1, embedding2)
            cosdist = cosdist.cpu().item()
            return {f"cossim/{self.backend_model}": [cosdist]}
        else:
            self.extract_fail_cnt += 1
            return {f"cossim/{self.backend_model}": []} # don't include stats from this image

    def eval(self, id_embedding, image):
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        embed = self.extract_face_embed(image=image)

        cosdist = self.sim(id_embedding.to(self.device), embed.to(self.device))
        cosdist = cosdist.cpu().item()
        return cosdist

    @torch.no_grad()
    def __call__(self, ids=None, rthresh=None):
        metric_result = defaultdict(list)
        if ids is not None:
            data = [self.dataset[i] for i in ids]
        else:
            data = self.dataset
        for idx, prompt, ref_img, out_img in tqdm(iter(data), desc=self.exp_root.stem, leave=True):
            if rthresh is not None and self.dataset.ratios[idx] < rthresh:
                continue 
            batch_metrics = self.calc_object(idx, prompt, ref_img, out_img)
            for m in batch_metrics.keys():
                metric_result[m] += batch_metrics[m]
        face_fail_cnt = self.extract_fail_cnt
        self.extract_fail_cnt = 0
        return metric_result, face_fail_cnt


class FaceClipScore(BaseMetric):
    def __init__(self, exp_name, dataset, filter_subset, **kwargs):
        super().__init__(exp_name, dataset, filter_subset, **kwargs)
        self.backend_model = "buffalo_l" 
        self.face_app = FaceFeatExtract(device=self.device)
        self.extract_fail_cnt = 0
        self.clip_metric = CLIPMetric(
            exp_name,
            dataset,
            filter_subset,
            **kwargs
        )

        if os.path.exists(self.exp_root / "ratios.json"):
            self.ratios = json.load(open(self.exp_root / "ratios.json"))

    def extract_faces(self, image_path: str = ""):
        img = Image.open(image_path)
        img_tensor = TF.pil_to_tensor(img)
        try:
            bb = self.face_app.get_face_bb(img_tensor.unsqueeze(0))[0]
            bb = [int(b) for b in bb]
            img_tensor = img_tensor[:, bb[1]:bb[3], bb[0]:bb[2]].float()
            if img_tensor.shape[-1] > 0 and img_tensor.shape[-2] > 0:
                return img_tensor
            return None
        except NoFaceFoundError:    
            return None

    def calc_object(self, idx, prompt, ref_img, out_img):
        ref_img_path, out_img_path = self.dataset.get_imgs_paths(idx)

        face_source = self.extract_faces(ref_img_path)
        face_pred = self.extract_faces(out_img_path)

        if face_source is not None and face_pred is not None:
            clip_scores = self.clip_metric.calc_object(
                [idx],
                [prompt],
                face_source.unsqueeze(0),
                face_pred.unsqueeze(0)
            )
            return {f"face_clipscore": clip_scores["l_14_scores"]}
        else:
            self.extract_fail_cnt += 1
            return {f"face_clipscore": []} # don't include stats from this image

    @torch.no_grad()
    def __call__(self, ids=None, rthresh=None):
        metric_result = defaultdict(list)
        if ids is not None:
            data = [self.dataset[i] for i in ids]
        else:
            data = self.dataset
        for idx, prompt, ref_img, out_img in tqdm(iter(data), desc=self.exp_root.stem, leave=True):
            # keep only style desrcription in prompt
            if ";" in prompt:
                prompt = prompt.split(";")[0]
            if rthresh is not None and self.dataset.ratios[idx] < rthresh:
                continue 
            batch_metrics = self.calc_object(idx, prompt, ref_img, out_img)
            for m in batch_metrics.keys():
                metric_result[m] += batch_metrics[m]
        face_fail_cnt = self.extract_fail_cnt
        self.extract_fail_cnt = 0
        return metric_result, face_fail_cnt