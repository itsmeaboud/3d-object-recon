from pathlib import Path
import sys
from typing import List, Tuple, Optional, Literal, Sequence
from dataclasses import dataclass
import logging

import numpy as np
import torch
from jaxtyping import Float32, Float
import time

VGGT_ROOT = Path(__file__).resolve().parents[2] / "third_party" / "vggt"
if VGGT_ROOT.is_dir() and str(VGGT_ROOT) not in sys.path:
    sys.path.insert(0, str(VGGT_ROOT))

from vggt.models.vggt import VGGT
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.load_fn import load_and_preprocess_images


logger = logging.getLogger(__name__)

PreProcessMode = Literal['crop', 'pad']

def depth_edge(depth: np.ndarray, rtol: float = 0.03, kernel_size: int = 3) -> np.ndarray:
    depth = np.asarray(depth)
    original_shape = depth.shape
    depth = depth.reshape(-1, *original_shape[-2:])

    pad = kernel_size // 2
    padded = np.pad(depth, ((0, 0), (pad, pad), (pad, pad)), mode="edge")
    depth_max = np.full_like(depth, -np.inf)
    depth_min = np.full_like(depth, np.inf)

    for y in range(kernel_size):
        for x in range(kernel_size):
            window = padded[:, y : y + depth.shape[-2], x : x + depth.shape[-1]]
            depth_max = np.maximum(depth_max, window)
            depth_min = np.minimum(depth_min, window)

    relative_jump = (depth_max - depth_min) / np.maximum(np.abs(depth), 1e-6)
    return (relative_jump > rtol).reshape(original_shape)




def tensor_sequence_to_numpy(
        tensor: Float32[torch.Tensor, "1 S ..."] | Float32[np.ndarray, "1 S ..."]
        ) -> Float32[np.ndarray, "S ..."]:
    
    """converting [1, S, ...]output tensor of model to a numpy array [S, ...]"""
    if isinstance(tensor, torch.Tensor):
        array = tensor.detach().cpu().numpy()
    else:
        array = tensor
    if array.ndim > 0 and array.shape[0] == 1:
        array = array[0]
    return array

@dataclass
class VGGTOutput: 
    world_points: Float32[np.ndarray, "S H W 3"]
    world_points_conf: Float32[np.ndarray, "S H W"]
    depth: Float32[np.ndarray, "S H W 1"]
    depth_conf: Float32[np.ndarray, "S H W"]
    images: Float32[np.ndarray, "S H W 3"]
    extrinsic: Float32[np.ndarray, "S 3 4"]
    intrinsic: Float32[np.ndarray, "S 3 3"]
    shape: Tuple[int, int]
    frames: int
    pose_enc: Float32[np.ndarray, "S 9"]


class VGGTInferencePipeline:

    def __init__(
            self, 
            model_name: str = "facebook/VGGT-1B", 
            device: str = 'cpu',
            preprocess_mode: PreProcessMode = "crop"
            ):
        
        self.device = torch.device(device)
        self.preprocess_mode = preprocess_mode
        self.model = VGGT.from_pretrained(model_name).to(self.device)
        self.model.eval()
        logger.info("VGGT model loaded on %s", self.device)


    def predict(self, image_paths: Sequence[str | Path]) -> VGGTOutput :

        paths = [Path(path)for path in image_paths]
        if not paths:
            raise ValueError("At least one image is required")

        # Prepare the image tensor [S, C, H, W] for feed forward
        batch_images: Float32[torch.Tensor, "S 3 H W"] = load_and_preprocess_images(paths, mode = self.preprocess_mode)
        batch_images = batch_images.to(self.device, non_blocking = True)
        height, width = batch_images.shape[-2:]
        frames = batch_images.shape[0]

        logger.info("Inference started for %d frame(s)", frames)
        start = time.perf_counter()
        with torch.inference_mode():
            #prediction keys ->['pose_enc', 'pose_enc_list', 'depth', 'depth_conf', 'world_points', 'world_points_conf'])
            predictions = self.model(batch_images)
        logger.info("Inference finished in %.2fs", time.perf_counter() - start)

        # [S, C, H, W] -> [S, H, W, C] for visulaization
        images = batch_images.detach().cpu().permute(0, 2, 3, 1).numpy()

        extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions['pose_enc'], batch_images[0].shape[-2:])
        depth_conf = predictions['depth_conf'].clone().detach()
        

        depth_conf[depth_edge(predictions['depth'][..., 0], rtol=0.03)] = 0.0
        predictions["depth_conf"] = depth_conf
        depth = tensor_sequence_to_numpy(predictions['depth'])
        depth_conf = tensor_sequence_to_numpy(predictions['depth_conf'])
        extrinsic = tensor_sequence_to_numpy(extrinsic)
        intrinsic = tensor_sequence_to_numpy(intrinsic)
        world_points_conf = tensor_sequence_to_numpy(predictions['world_points_conf'])
        pose_enc = tensor_sequence_to_numpy(predictions['pose_enc'])
        world_points = unproject_depth_map_to_point_map(depth, extrinsic, intrinsic)


        return VGGTOutput(
            world_points = world_points,
            world_points_conf = world_points_conf,
            depth = depth,
            depth_conf = depth_conf,
            images = images,
            extrinsic = extrinsic,
            intrinsic = intrinsic,
            shape = (height, width),
            frames = frames,
            pose_enc = pose_enc
        )




if __name__ == "__main__":
    pass







        


        


        
    
