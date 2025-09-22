"""BiRefNet 기반 배경 제거 유틸리티."""

from __future__ import annotations

import math
import os
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

try:  # Pillow>=10 대응
    _RESAMPLE_BILINEAR = Image.Resampling.BILINEAR  # type: ignore[attr-defined]
except AttributeError:  # pragma: no cover - Pillow<10
    _RESAMPLE_BILINEAR = Image.BILINEAR

from diref.models.birefnet import BiRefNet
from diref.utils import check_state_dict


def _ensure_rgb(image: Image.Image) -> Image.Image:
    """입력 이미지를 RGB 모드로 맞춘다."""
    return image.convert("RGB") if image.mode != "RGB" else image


class BackgroundRemoverBiRef:
    """BiRefNet을 이용한 배경제거 헬퍼."""

    def __init__(
        self,
        model_path: str = os.path.join("saved_models", "BiRefNet-DIS-epoch_590.pth"),
        device: Optional[torch.device] = None,
    ) -> None:
        self.model_path = model_path
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[torch.nn.Module] = None
        self._preprocess = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        self._load_model()

    def _load_model(self) -> None:
        """저장된 BiRefNet 가중치를 불러온다."""
        if self.model is not None:
            return

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"BiRefNet 가중치를 찾을 수 없습니다: {self.model_path}."
            )

        model = BiRefNet(bb_pretrained=False)
        state_dict = torch.load(self.model_path, map_location="cpu", weights_only=False)
        state_dict = check_state_dict(state_dict)
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()

        # 혼합 정밀도/장치 설정은 직접 제어한다.
        model.config.mixed_precision = "no"
        model.config.device = self.device

        self.model = model

    @staticmethod
    def _align_to_multiple_of_32(size: Tuple[int, int]) -> Tuple[int, int]:
        width, height = size
        target_w = max(32, math.ceil(width / 32) * 32)
        target_h = max(32, math.ceil(height / 32) * 32)
        return target_w, target_h

    def _prepare_tensor(self, image: Image.Image) -> Tuple[torch.Tensor, Tuple[int, int]]:
        image = _ensure_rgb(image)
        orig_size = image.size
        target_size = self._align_to_multiple_of_32(orig_size)

        resized_image = image.resize(target_size, _RESAMPLE_BILINEAR)
        tensor = self._preprocess(resized_image).unsqueeze(0).to(self.device)
        return tensor, orig_size

    def remove_background_from_image(
        self,
        image: Image.Image,
        background_color: Tuple[int, int, int] = (0, 0, 0),
    ) -> Optional[Image.Image]:
        """입력 이미지를 받아 선택한 배경색으로 합성된 이미지를 반환한다."""
        if self.model is None:
            self._load_model()

        assert self.model is not None  # 타입 안정성

        try:
            inputs, orig_size = self._prepare_tensor(image)

            with torch.no_grad():
                outputs = self.model(inputs)
                if isinstance(outputs, (list, tuple)):
                    pred = outputs[-1]
                else:
                    pred = outputs
                pred = torch.sigmoid(pred)

            pred = F.interpolate(
                pred,
                size=(orig_size[1], orig_size[0]),
                mode="bilinear",
                align_corners=False,
            )

            mask = pred.squeeze(0).squeeze(0).cpu().numpy()
            mask = np.clip(mask, 0.0, 1.0)

            image_rgb = _ensure_rgb(image)
            image_np = np.array(image_rgb, dtype=np.float32)

            background_np = np.empty_like(image_np, dtype=np.float32)
            background_np[..., 0] = background_color[0]
            background_np[..., 1] = background_color[1]
            background_np[..., 2] = background_color[2]

            mask_3c = mask[..., None]
            result = image_np * mask_3c + background_np * (1.0 - mask_3c)
            result = np.clip(result, 0, 255).astype(np.uint8)

            return Image.fromarray(result)
        except Exception as exc:  # noqa: BLE001
            print(f"BiRefNet 배경 제거 중 오류 발생: {exc}")
            return None

    def remove_background_from_url(
        self,
        image_url: str,
        background_color: Tuple[int, int, int] = (0, 0, 0),
        timeout: float = 10.0,
    ) -> Optional[Image.Image]:
        """URL 이미지를 다운로드하여 배경을 제거한다."""
        import requests
        from io import BytesIO

        try:
            response = requests.get(image_url, timeout=timeout)
            response.raise_for_status()
            with Image.open(BytesIO(response.content)) as img:
                return self.remove_background_from_image(img, background_color=background_color)
        except Exception as exc:  # noqa: BLE001
            print(f"이미지 URL 처리 실패: {image_url} -> {exc}")
            return None


def remove_background_batch_biref(
    product_datas,
    background_color: Tuple[int, int, int] = (0, 0, 0),
) -> list:
    """(id, url) 쌍 배열을 일괄 처리한다."""
    remover = BackgroundRemoverBiRef()
    results = []
    for product_id, image_url in product_datas:
        print(f"상품 {product_id} 처리 중...")
        processed_image = remover.remove_background_from_url(
            image_url,
            background_color=background_color,
        )
        if processed_image is not None:
            results.append((product_id, processed_image))
        else:
            print(f"상품 {product_id} 처리 실패")
    return results
