# Troubleshooting

## `No module named 'rknnlite'` 또는 `No module named 'rknn'`

원인:
- RKNN 런타임 미설치
- Python 버전과 wheel 태그 불일치

해결:

```bash
python3 -m pip install --user --upgrade pip
python3 -m pip install --user rknn-toolkit-lite2
python3 -c "from rknnlite.api import RKNNLite; print('OK')"
```

## `No module named 'dlib'`

원인:
- 빌드 도구가 부족하거나 설치 실패

해결:

```bash
sudo apt update
sudo apt install -y build-essential g++ cmake
python3 -m pip install --user dlib
```

## `No module named 'cv2'`

해결:

```bash
python3 -m pip install --user opencv-python
```

## 웹캠이 열리지 않음 (`웹캠 열기 실패`)

원인:
- 장치 인덱스 불일치
- 권한/연결 문제

해결:

```bash
v4l2-ctl --list-devices
```

카메라 번호를 확인한 뒤 코드의 `cv2.VideoCapture(0)` 인덱스를 맞춰주세요.

## Flask 웹 모드 접속 안 됨

원인:
- Flask 미설치
- 포트 충돌

해결:

```bash
python3 -m pip install --user flask
```

기본 주소: `http://0.0.0.0:5000`

## 실행 파일/모델 누락

아래 파일이 같은 폴더에 있는지 확인:

- `dms_v6_eye_unified.py`
- `eye_classifier.rknn`
- `shape_predictor_68_face_landmarks.dat`
