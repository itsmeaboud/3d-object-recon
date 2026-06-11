import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from jaxtyping import Bool


logger = logging.getLogger(__name__)
DEFAULT_WEIGHTS_PATH = Path("weights/sam3.1_multiplex.pt")


class SAM3Segmenter:
    """Segment a text-prompted object in each frame with SAM3."""

    def __init__(
        self,
        conf: float = 0.7,
        device: str | None = "cpu",
        weights_path: Path | str | None = DEFAULT_WEIGHTS_PATH,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.weights_path = Path(weights_path) if weights_path is not None else None
        self.masks: Bool[np.ndarray, "S H W"] | None = None
        self.cfg = self._build_config(conf=conf)

        self.predictor = self._load_predictor()
        logger.info("SAM3 loaded on %s", self.device)

    def predict(
        self,
        images: np.ndarray,
        prompt: str = "main object",
    ) -> Bool[np.ndarray, "S H W"]:
        """Return one boolean mask per input frame."""
        images = self._validate_images(images)
        if not prompt:
            raise ValueError("prompt must be a non-empty string")

        frame_count, height, width = images.shape[:3]
        masks: list[Bool[np.ndarray, "H W"]] = []

        for frame_index in range(frame_count):
            logger.info("SAM3 processing frame %d", frame_index)
            mask = self._predict_frame(
                image=images[frame_index],
                prompt=prompt,
                image_shape=(height, width),
            )
            masks.append(mask)

        self.masks = np.stack(masks, axis=0).astype(bool, copy=False)
        return self.masks

    def _build_config(self, conf: float) -> dict[str, object]:
        return {
            "conf": conf,
            "task": "segment",
            "mode": "predict",
            "model": self.weights_path,
            "save": False,
            "device": self.device,
        }

    def _load_predictor(self) -> Any:
        try:
            from ultralytics.models.sam import SAM3SemanticPredictor

            return SAM3SemanticPredictor(overrides=self.cfg)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load SAM3 predictor from {self.weights_path}"
            ) from exc

    @staticmethod
    def _validate_images(images: np.ndarray) -> np.ndarray:
        images = np.asarray(images)
        if images.ndim != 4:
            raise ValueError(f"Expected images with shape (S, H, W, 3), got {images.shape}")
        if images.shape[-1] != 3:
            raise ValueError(f"Expected RGB images with 3 channels, got {images.shape[-1]}")
        if images.shape[0] == 0:
            raise ValueError("At least one image is required")
        return images

    def _predict_frame(
        self,
        image: np.ndarray,
        prompt: str,
        image_shape: tuple[int, int],
    ) -> Bool[np.ndarray, "H W"]:
        image = np.ascontiguousarray(image)
        self.predictor.set_image(image)

        prediction = self.predictor(text=[prompt])
        if not prediction:
            return self._empty_mask(image_shape, prompt)

        output = prediction[0]
        if output.masks is None or output.masks.data is None or len(output.masks.data) == 0:
            return self._empty_mask(image_shape, prompt)

        logger.info("Object %s was found", prompt)
        mask: torch.Tensor = output.masks.data[0]
        return mask.detach().cpu().numpy().astype(bool, copy=False)

    @staticmethod
    def _empty_mask(image_shape: tuple[int, int], prompt: str) -> Bool[np.ndarray, "H W"]:
        logger.info("Object %s was not found; using an empty mask", prompt)
        return np.zeros(image_shape, dtype=bool)
