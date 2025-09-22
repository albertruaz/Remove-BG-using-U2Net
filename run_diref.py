"""BiRefNet 배경 제거 배치 스크립트."""

import os
from typing import Tuple

from PIL import Image

from biref_diref import BackgroundRemoverBiRef


def main() -> None:
    """test_images 폴더 전체를 처리하여 diref_results 폴더에 저장한다."""
    test_image_dir = os.path.join("test_data", "test_images_white_bg")
    output_dir = os.path.join("test_data", "diref_results")

    use_black_background = True  # False로 변경하면 흰색 배경으로 복구됩니다.
    background_color: Tuple[int, int, int] = (0, 0, 0) if use_black_background else (255, 255, 255)

    model_path = os.path.join("saved_models", "BiRefNet-DIS-epoch_590.pth")

    if not os.path.isdir(test_image_dir):
        print(f"테스트 이미지 폴더를 찾을 수 없습니다: {test_image_dir}")
        return

    import glob

    image_files = sorted(glob.glob(os.path.join(test_image_dir, "*")))
    if not image_files:
        print("처리할 이미지가 없습니다.")
        return

    if not os.path.exists(model_path):
        print(f"BiRefNet 가중치를 찾을 수 없습니다: {model_path}")
        return

    os.makedirs(output_dir, exist_ok=True)

    remover = BackgroundRemoverBiRef(model_path=model_path)

    for img_path in image_files:
        print(f"처리 중: {img_path}")
        try:
            with Image.open(img_path) as image:
                result = remover.remove_background_from_image(
                    image,
                    background_color=background_color,
                )
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
