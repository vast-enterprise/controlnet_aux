# Openpose
# Original from CMU https://github.com/CMU-Perceptual-Computing-Lab/openpose
# 2nd Edited by https://github.com/Hzzone/pytorch-openpose
# 3rd Edited by ControlNet
# 4th Edited by ControlNet (added face and correct hands)

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import cv2
import json
import torch
import numpy as np
from PIL import Image

from ..util import HWC3, resize_image
from . import util

from .animalpose import AnimalPoseImage
from .types import PoseResult, HandResult, FaceResult, AnimalPoseResult


from typing import Tuple, List, Callable, Union, Optional

def draw_animalposes(animals: list[list[Keypoint]], H: int, W: int) -> np.ndarray:
    canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8)
    for animal_pose in animals:
        canvas = draw_animalpose(canvas, animal_pose)
    return canvas


def draw_animalpose(canvas: np.ndarray, keypoints: list[Keypoint]) -> np.ndarray:
    # order of the keypoints for AP10k and a standardized list of colors for limbs
    keypointPairsList = [
        (1, 2),
        (2, 3),
        (1, 3),
        (3, 4),
        (4, 9),
        (9, 10),
        (10, 11),
        (4, 6),
        (6, 7),
        (7, 8),
        (4, 5),
        (5, 15),
        (15, 16),
        (16, 17),
        (5, 12),
        (12, 13),
        (13, 14),
    ]
    colorsList = [
        (255, 255, 255),
        (100, 255, 100),
        (150, 255, 255),
        (100, 50, 255),
        (50, 150, 200),
        (0, 255, 255),
        (0, 150, 0),
        (0, 0, 255),
        (0, 0, 150),
        (255, 50, 255),
        (255, 0, 255),
        (255, 0, 0),
        (150, 0, 0),
        (255, 255, 100),
        (0, 150, 0),
        (255, 255, 0),
        (150, 150, 150),
    ]  # 16 colors needed

    for ind, (i, j) in enumerate(keypointPairsList):
        p1 = keypoints[i - 1]
        p2 = keypoints[j - 1]

        if p1 is not None and p2 is not None:
            cv2.line(
                canvas,
                (int(p1.x), int(p1.y)),
                (int(p2.x), int(p2.y)),
                colorsList[ind],
                5,
            )
    return canvas

def draw_pose(pose, H, W):
    bodies = pose['bodies']
    faces = pose['faces']
    hands = pose['hands']
    candidate = bodies['candidate']
    subset = bodies['subset']
    
    canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8)
    canvas = util.draw_bodypose(canvas, candidate, subset)
    canvas = util.draw_handpose(canvas, hands)
    canvas = util.draw_facepose(canvas, faces)

    return canvas

class DWposeDetector:
    def __init__(self, det_config=None, det_ckpt=None, pose_config=None, pose_ckpt=None, device="cpu"):
        from .wholebody import Wholebody

        self.pose_estimation = Wholebody(det_config, det_ckpt, pose_config, pose_ckpt, device)
    
    def to(self, device):
        self.pose_estimation.to(device)
        return self
    
    def __call__(self, input_image, detect_resolution=512, image_resolution=512, output_type="pil", **kwargs):
        
        input_image = cv2.cvtColor(np.array(input_image, dtype=np.uint8), cv2.COLOR_RGB2BGR)

        input_image = HWC3(input_image)
        input_image = resize_image(input_image, detect_resolution)
        H, W, C = input_image.shape
        
        with torch.no_grad():
            candidate, subset = self.pose_estimation(input_image)
            nums, keys, locs = candidate.shape
            candidate[..., 0] /= float(W)
            candidate[..., 1] /= float(H)
            body = candidate[:,:18].copy()
            body = body.reshape(nums*18, locs)
            score = subset[:,:18]
            
            for i in range(len(score)):
                for j in range(len(score[i])):
                    if score[i][j] > 0.3:
                        score[i][j] = int(18*i+j)
                    else:
                        score[i][j] = -1

            un_visible = subset<0.3
            candidate[un_visible] = -1

            foot = candidate[:,18:24]

            faces = candidate[:,24:92]

            hands = candidate[:,92:113]
            hands = np.vstack([hands, candidate[:,113:]])
            
            bodies = dict(candidate=body, subset=score)
            pose = dict(bodies=bodies, hands=hands, faces=faces)
            
            detected_map = draw_pose(pose, H, W)
            detected_map = HWC3(detected_map)
            
            img = resize_image(input_image, image_resolution)
            H, W, C = img.shape

            detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)

            if output_type == "pil":
                detected_map = Image.fromarray(detected_map)
                
            return detected_map

global_cached_animalpose = AnimalPoseImage()
class AnimalposeDetector:
    """
    A class for detecting animal poses in images using the RTMPose AP10k model.

    Attributes:
        model_dir (str): Path to the directory where the pose models are stored.
    """
    def __init__(self, animal_pose_estimation):
        self.animal_pose_estimation = animal_pose_estimation
    
    @classmethod
    def from_pretrained(cls, pretrained_model_or_path, pretrained_det_model_or_path=None, det_filename="yolox_l.onnx", pose_filename="dw-ll_ucoco_384.onnx", torchscript_device="cuda"):
        global global_cached_animalpose
        # det_model_path = custom_hf_download(pretrained_det_model_or_path, det_filename)
        # pose_model_path = custom_hf_download(pretrained_model_or_path, pose_filename)
        
        det_model_path = det_filename
        pose_model_path = pose_filename

        print(f"\nAnimalPose: Using {det_filename} for bbox detection and {pose_filename} for pose estimation")
        if global_cached_animalpose.det is None or global_cached_animalpose.det_filename != det_filename:
            t = AnimalPoseImage(det_model_path, None, torchscript_device=torchscript_device)
            t.pose = global_cached_animalpose.pose
            t.pose_filename = global_cached_animalpose.pose
            global_cached_animalpose = t
        
        if global_cached_animalpose.pose is None or global_cached_animalpose.pose_filename != pose_filename:
            t = AnimalPoseImage(None, pose_model_path, torchscript_device=torchscript_device)
            t.det = global_cached_animalpose.det
            t.det_filename = global_cached_animalpose.det_filename
            global_cached_animalpose = t
        return cls(global_cached_animalpose)
    
    def __call__(self, input_image, detect_resolution=512, output_type="pil", image_and_json=False, upscale_method="INTER_CUBIC", **kwargs):
        input_image, output_type = common_input_validate(input_image, output_type, **kwargs)
        input_image, remove_pad = resize_image_with_pad(input_image, detect_resolution, upscale_method)
        result = self.animal_pose_estimation(input_image)
        if result is None:
            detected_map = np.zeros_like(input_image)
            openpose_dict = {
                'version': 'ap10k',
                'animals': [],
                'canvas_height': input_image.shape[0],
                'canvas_width': input_image.shape[1]
            }
        else:
            detected_map, openpose_dict = result
        detected_map = remove_pad(detected_map)
        if output_type == "pil":
            detected_map = Image.fromarray(detected_map)
        
        if image_and_json:
            return (detected_map, openpose_dict)

        return detected_map