from dataclasses import dataclass, replace
import numpy as np
import torch

from .data_types import ObjectPointCloud, ReconstructionRequest, PathLike
from .vggt_predictor import VGGTInferencePipeline, VGGTOutput, PreProcessMode
from .sam3_segmenter import SAM3Segmenter

from typing import Optional, Sequence
from jaxtyping import Bool
from pathlib import Path


@dataclass(frozen = True)
class ReconstructionConfig:
    
    device: Optional[str] = None
    sam_confidence: float = 0.5
    sam_weights_path: Optional[Path] = None
    preprocess_mode: PreProcessMode = 'crop'

def images_for_sam(images: np.ndarray) -> np.ndarray:

    if images.size == 0:
        return images
    
    if np.nanmax(images) <= 1.0:
        return images * 255.0
    
    return images

def filter_by_confidence(
        result: ObjectPointCloud,
        percentile: float
) -> ObjectPointCloud:
    
    """Return a filtered copy of reconstruction result"""

    threshold = np.nanpercentile(result.confidence, percentile)
    conf_mask = result.confidence >= threshold
    seg_mask = result.masks[conf_mask]

    filtered_points = result.points[conf_mask][seg_mask]
    filtered_colors = result.colors[conf_mask][seg_mask]


    return replace(
        result,
        filtered_points = filtered_points,
        filtered_colors = filtered_colors
    )

def get_rotation_matrix(angle):

    theta = np.radians(angle)
    sin_t, cos_t = np.sin(theta), np.cos(theta)
    R = np.array([
        [cos_t, 0, 1],
        [0, 1, 0],
        [-sin_t, 0, cos_t]
    ], dtype = np.float32)

    return R

def cv_to_world_frame(data: ObjectPointCloud):

    R_cv_to_world = np.array([
        [1, 0, 0],
        [0, 0, 1],
        [0, -1, 0]
    ])

    data.points = data.points @ R_cv_to_world.T

    #translation
    data.extrinsic[:, :3, 3] = data.extrinsic[:, :3, 3] @ R_cv_to_world.T
    #rotation
    data.extrinsic[:, :3, :3] =  R_cv_to_world @ data.extrinsic[:, :3, :3] 
    return data


class ObjectReconstructor:

    def __init__(
            self,
            cfg: Optional[ReconstructionConfig] = None,
            recon_model: Optional[VGGTInferencePipeline] = None,
            seg_model: Optional[SAM3Segmenter] = None
    ):
        
        self.cfg = cfg or ReconstructionConfig()
        self.device = self.cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.recon_model = recon_model or VGGTInferencePipeline(
            device = self.device,
            preprocess_mode = self.cfg.preprocess_mode
        )

        self.seg_model = seg_model or SAM3Segmenter(
            conf = self.cfg.sam_confidence,
            device = self.cfg.device,
            weights_path = self.cfg.sam_weights_path
        )


    def reoconstruct(
            self,
            image_paths: Sequence[PathLike],
            prompt: str,
    ) -> ObjectPointCloud:
        
        output: VGGTOutput = self.recon_model.predict(image_paths = image_paths)
        masks: Bool[np.ndarray, "S H W"] = self.seg_model.predict(
            images = images_for_sam(output.images),
            prompt = prompt
        )

        return cv_to_world_frame(ObjectPointCloud(
            points = output.world_points.reshape(-1, 3),
            colors = output.images.reshape(-1, 3),
            confidence = output.depth_conf.reshape(-1),
            masks = masks.reshape(-1),
            source_images = output.images,
            image_shape = output.shape,
            extrinsic = output.extrinsic,
            intrinsic = output.intrinsic
        ))
    
    def reconstruct_and_filter(
            self,
            request: ReconstructionRequest,
        ) -> ObjectPointCloud:
        
        result = self.reoconstruct(
            image_paths = request.image_paths,
            prompt = request.prompt,
            )
        
        return filter_by_confidence(
            result = result,
            percentile = request.confidence_perc
            )
    


