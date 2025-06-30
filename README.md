# U2-Net 환경 설정 및 실행 가이드

## 개요

U2-Net을 이용한 이미지 배경 제거 도구입니다. 입력 이미지의 전경 객체를 추출하고 흰색 배경으로 변경합니다.

## 환경 요구사항

- Python 3.8
- Conda 환경 권장

## 설치 방법

### 1. 전체 환경 복원 (권장)

#### 또는 현재 작동하는 환경으로 복원:

```bash
conda create -n u2net python=3.8
conda activate u2net
pip install -r setting/requirements_current.txt
```

### 2. 최소 환경 설정

필수 패키지만 설치하려면:

```bash
conda create -n remove-bg python=3.8
conda activate remove-bg
pip install -r setting/requirements_minimal.txt
```

## 실행 방법

```bash
python run.py
```

### 입력/출력

- **입력**: `test_data/test_images/` 폴더의 이미지들
- **출력**: `test_data/u2net_results/` 폴더에 `[파일명]_whitebg.png` 형태로 저장

## 주요 패키지 버전

- Python: 3.8
- PyTorch: 2.3.0+ (CPU 버전)
- TorchVision: 0.18.0+
- scikit-image: 0.20.0+
- NumPy: 1.24.0+
- Pillow: 10.0.0+
- matplotlib: 3.7.0+

## 수정사항

- PyTorch 경고 메시지 해결 (`weights_only=False` 명시)
- Deprecated 함수 교체 (`F.upsample` → `F.interpolate`)
- matplotlib 의존성 추가

## 사용법

1. 테스트할 이미지를 `test_data/test_images/` 폴더에 넣기
2. `python run.py` 실행
3. `test_data/u2net_results/` 폴더에서 결과 확인
