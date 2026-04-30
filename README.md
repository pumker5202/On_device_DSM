# On_device_DSM (Orange Pi 5 / RK3588)

Orange Pi 5(RK3588)에서 RKNN 기반으로 동작하는 운전자 졸음 감지(DSM) 프로젝트입니다.
웹 모드와 로컬 창 모드를 모두 지원합니다.

## Features

- RKNN 모델(`eye_classifier.rknn`) 기반 온디바이스 추론
- 얼굴 랜드마크 + EAR/MAR + CNN 결과 결합 판정
- 웹 스트리밍 모드(Flask) / 로컬 디스플레이 모드(`--no-web`)
- 실행 스크립트(`run.sh`) 제공

## Project Structure

```text
dms_v6_board/
|- dms_v6_eye_unified.py
|- eye_classifier.rknn
|- shape_predictor_68_face_landmarks.dat
`- run.sh
```

## Quick Start

```bash
cd dms_v6_board
chmod +x run.sh
./run.sh
```

로컬 창 모드:

```bash
./run.sh --no-web
```

직접 실행:

```bash
python3 dms_v6_eye_unified.py
python3 dms_v6_eye_unified.py --no-web
```

## Environment

- Board: Orange Pi 5 / RK3588 (aarch64)
- OS: Ubuntu/Debian 계열
- Python: 3.10 권장 (3.7~3.12 가능)
- 필수 런타임: `rknn-toolkit-lite2`, `opencv-python`, `numpy`, `dlib`, `flask`

자세한 설치는 `docs/INSTALL_ORANGEPI5.md` 참고.

## Required Files

실행 폴더에 아래 파일이 반드시 있어야 합니다.

- `eye_classifier.rknn`
- `shape_predictor_68_face_landmarks.dat`
- `dms_v6_eye_unified.py`

## Verification

```bash
python3 -c "from rknnlite.api import RKNNLite; print('rknnlite OK')"
python3 -c "import cv2, numpy, dlib; print('deps OK')"
```

## Common Errors

- `No module named 'rknnlite'`: `rknn-toolkit-lite2` 미설치 또는 Python 버전 불일치
- `No module named 'dlib'`: 빌드 도구(`cmake`, `build-essential`, `g++`) 누락 가능
- 웹캠 인식 실패: `v4l2-ctl --list-devices`로 장치 확인 후 카메라 인덱스 조정

상세는 `docs/TROUBLESHOOTING.md` 참고.

## License

MIT