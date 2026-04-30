# Orange Pi 5 설치 가이드 (RK3588)

## 1) 시스템 패키지 설치

```bash
sudo apt update
sudo apt install -y \
  python3 python3-pip python3-venv \
  build-essential g++ cmake \
  libopencv-dev libgl1-mesa-glx \
  v4l-utils
```

## 2) Python 패키지 설치

```bash
python3 -m pip install --user --upgrade pip
python3 -m pip install --user numpy opencv-python flask dlib
python3 -m pip install --user rknn-toolkit-lite2
```

`rknn-toolkit-lite2` 설치 실패 시 Python 버전에 맞는 wheel(cp310/cp311/cp312)을 사용하세요.

## 3) 설치 확인

```bash
python3 --version
uname -m
python3 -c "from rknnlite.api import RKNNLite; print('rknnlite OK')"
python3 -c "import cv2, numpy, dlib, flask; print('python deps OK')"
```

## 4) 실행

```bash
cd dms_v6_board
chmod +x run.sh
./run.sh
```

로컬 창 모드:

```bash
./run.sh --no-web
```
