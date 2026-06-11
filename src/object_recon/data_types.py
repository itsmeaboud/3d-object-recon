from dataclasses import dataclass
from jaxtyping import Float32, Bool
from typing import Tuple, Optional, Union, Sequence
from pathlib import Path

import numpy as np

PathLike = Union[str, Path]

@dataclass
class ReconstructionRequest:
    image_paths: Sequence[PathLike]
    prompt: str
    confidence_perc: float = 20.0



@dataclass
class ObjectPointCloud:
    """Results prdouced by vggt + sam3"""

    points: Float32[np.ndarray, "N 3"]
    colors: Float32[np.ndarray, "N 3"]
    confidence: Float32[np.ndarray, "N 3"]
    masks: Bool[np.ndarray, "N"]
    source_images: Float32[np.ndarray, "S H W 3"]
    image_shape: Tuple[int, int]
    extrinsic: Float32[np.ndarray, "S 3 4"]
    intrinsic: Float32[np.ndarray, "S 3 3"]
    filtered_points: Optional[Float32[np.ndarray, "M 3"]] = None
    filtered_colors: Optional[Float32[np.ndarray, "M 3"]] = None
 

    @property
    def has_filtered_points(self) -> bool:
        return self.filtered_points is not None and self.filtered_colors is not None
    
    @property
    def visible_points(self) -> Float32[np.ndarray, "K 3"]:
        if self.filtered_points is not None:
            return self.filtered_points
        return self.points
    
    @property
    def visible_colors(self) -> Float32[np.ndarray, "K 3"]:
        if self.filtered_colors is not None:
            return self.filtered_colors
        return self.colors
    

