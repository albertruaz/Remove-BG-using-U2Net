import os
from io import BytesIO
from typing import Optional, Tuple

import numpy as np
import requests
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms.functional import normalize

from isnet_is import ISNetDIS


def _ensure_rgb(image: Image.Image) -> Image.Image:
    """이미지가 RGB가 아니면 RGB로 변환한다."""
    return image.convert('RGB') if image.mode != 'RGB' else image


class BackgroundRemoverIS:
    """IS-Net을 이용한 배경제거 도우미 클래스."""

    def __init__(self, model_path: str = 'saved_models/isnet-general-use.pth', input_size=(1024, 1024)):
        self.model_path = model_path
        self.input_size = input_size
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.net = None
        self._load_model()

    def _load_model(self) -> None:
        """사전 학습된 IS-Net 가중치를 불러온다."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"IS-Net 가중치 파일을 찾을 수 없습니다: {self.model_path}")

        print("IS-Net 모델 로딩 중...")
        self.net = ISNetDIS()

        state_dict = torch.load(self.model_path, map_location=self.device, weights_only=False)
        self.net.load_state_dict(state_dict)
        self.net = self.net.to(self.device)

        self.net.eval()
        print("IS-Net 모델 로딩 완료")

    @staticmethod
    def _norm_pred(pred: torch.Tensor) -> torch.Tensor:
        """예측 결과를 [0,1] 범위로 정규화한다."""
        ma = torch.max(pred)
        mi = torch.min(pred)
        denom = ma - mi
        if denom < 1e-8:
            return torch.zeros_like(pred)
        return (pred - mi) / denom

    def _preprocess(self, image: Image.Image) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """이미지를 사전 처리하여 모델 입력 형태로 변환한다."""
        image = _ensure_rgb(image)
        orig_w, orig_h = image.size

        image_np = np.array(image).astype(np.float32)
        tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0)
        tensor = tensor.float()
        tensor = F.interpolate(tensor, size=self.input_size, mode='bilinear', align_corners=False)
        tensor = tensor / 255.0
        tensor = normalize(tensor, [0.5, 0.5, 0.5], [1.0, 1.0, 1.0])

        tensor = tensor.to(self.device)
        return tensor, (orig_h, orig_w)

    def _postprocess(
        self,
        pred: torch.Tensor,
        image: Image.Image,
        orig_shape: Tuple[int, int],
        background_color: Tuple[int, int, int] = (255, 255, 255),
    ) -> Image.Image:
        """모델 출력을 이용해 지정한 배경색이 적용된 이미지를 생성한다."""
        orig_h, orig_w = orig_shape
        pred = F.interpolate(pred, size=(orig_h, orig_w), mode='bilinear', align_corners=False)
        pred = pred.squeeze(0).squeeze(0)
        pred = self._norm_pred(pred)

        mask_np = pred.cpu().numpy()
        mask_np = np.clip(mask_np, 0.0, 1.0)

        image_rgb = _ensure_rgb(image)
        image_np = np.array(image_rgb).astype(np.float32)
        background_np = np.empty_like(image_np, dtype=np.float32)
        background_np[:, :, 0] = background_color[0]
        background_np[:, :, 1] = background_color[1]
        background_np[:, :, 2] = background_color[2]

        mask_3c = mask_np[..., None]
        result = image_np * mask_3c + background_np * (1.0 - mask_3c)
        result = np.clip(result, 0, 255).astype(np.uint8)

        return Image.fromarray(result)

    def remove_background_from_url(
        self,
        image_url: str,
        background_color: Tuple[int, int, int] = (255, 255, 255),
    ) -> Optional[Image.Image]:
        """이미지 URL을 받아 배경을 제거하고 지정한 배경색을 적용한다."""
        try:
            response = requests.get(image_url, timeout=10)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content))
            return self.remove_background_from_image(image, background_color=background_color)
        except Exception as exc:  # noqa: BLE001
            print(f"URL 처리 중 오류 발생: {image_url} -> {exc}")
            return None

    def remove_background_from_image(
        self,
        image: Image.Image,
        background_color: Tuple[int, int, int] = (255, 255, 255),
    ) -> Optional[Image.Image]:
        """PIL 이미지를 입력으로 받아 배경을 제거하고 지정한 배경색을 적용한다."""
        try:
            inputs, orig_shape = self._preprocess(image)

            with torch.no_grad():
                preds, _ = self.net(inputs)
                pred = preds[0]

            result_image = self._postprocess(pred, image, orig_shape, background_color=background_color)

            del preds, pred, inputs
            if self.device == 'cuda':
                torch.cuda.empty_cache()

            return result_image
        except Exception as exc:  # noqa: BLE001
            print(f"배경 제거 중 오류 발생: {exc}")
            return None


def remove_background_batch_is(product_datas):
    """배열 형태의 (id, url) 목록을 받아 일괄 처리한다."""
    remover = BackgroundRemoverIS()
    results = []

    for product_id, image_url in product_datas:
        print(f"상품 {product_id} 처리 중...")
        processed_image = remover.remove_background_from_url(image_url)
        if processed_image is not None:
            results.append((product_id, processed_image))
        else:
            print(f"상품 {product_id} 처리 실패")
    return results


def main() -> None:
    """test_images 폴더 전체를 처리하여 ISnet_results 폴더에 저장한다."""
    test_image_dir = os.path.join("test_data", "test_images_white_bg")
    output_dir = os.path.join("test_data", "ISnet_results")
    use_black_background = True  # False로 변경하면 흰색 배경으로 복구됩니다.
    background_color = (0, 0, 0) if use_black_background else (255, 255, 255)

    if not os.path.isdir(test_image_dir):
        print(f"테스트 이미지 폴더를 찾을 수 없습니다: {test_image_dir}")
        return

    import glob

    image_files = sorted(glob.glob(os.path.join(test_image_dir, "*")))
    if not image_files:
        print("처리할 이미지가 없습니다.")
        return

    os.makedirs(output_dir, exist_ok=True)

    remover = BackgroundRemoverIS()

    for img_path in image_files:
        print(f"처리 중: {img_path}")
        try:
            with Image.open(img_path) as image:
                result = remover.remove_background_from_image(image, background_color=background_color)
        except Exception as exc:  # noqa: BLE001
            print(f"이미지 로딩 실패: {img_path} -> {exc}")
            continue

        if result is None:
            print(f"배경 제거 실패: {img_path}")
            continue

        base_name = os.path.splitext(os.path.basename(img_path))[0]
        suffix = "blackbg" if use_black_background else "whitebg"
        output_path = os.path.join(output_dir, f"{base_name}_{suffix}.png")
        result.save(output_path)
        print(f"결과 저장: {output_path}")


if __name__ == "__main__":
    main()
