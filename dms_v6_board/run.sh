#!/bin/bash
# DMS v6 (eye_classifier) 실행 스크립트
# 압축 해제 후: chmod +x run.sh && ./run.sh

cd "$(dirname "$0")"

echo "🔄 DMS 로딩 중..."
python3 dms_v6_eye_unified.py "$@"
