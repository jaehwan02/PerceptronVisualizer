# PerceptronVisualizer
## 프로젝트 개요
PerceptronVisualizer는 다층 퍼셉트론(MLP) 신경망의 작동 원리를 직관적으로 이해할 수 있는 시각화 도구입니다.
사용자는 웹 캔버스에서 숫자를 입력하고, 신경망의 입력, 은닉, 출력 레이어를 시각적으로 탐구할 수 있습니다.

![화면 기록 2024-12-23 오전 4 36 50](https://github.com/user-attachments/assets/cd73ec70-ebe2-4605-880a-a0c49af69376)

---

## 기술 스택
- 웹 프레임워크: FastAPI, Jinja2
- 머신러닝: PyTorch, Scikit-learn
- 데이터 전처리 및 시각화: Pillow, Matplotlib
- 프론트엔드: HTML, CSS, JavaScript

---

## 설치 및 실행
1. 환경 설정
   ```pip install -r requirements.txt```
2. 모델 학습
   ```python train.py```
3. 서버 실행
   ```uvicorn main:app --reload```
4. 웹 애플리케이션 접속
브라우저에서 http://localhost:8000로 접속.

---

파이썬 수행평가 시간 이슈로 인해 생성형 ai의 코드가 다량 포함되어 있음을 알립니다.
