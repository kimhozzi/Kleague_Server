# 파이썬 3.10 환경 (캡처본의 pycache 버전에 맞춤)
FROM python:3.12

# 작업 디렉토리 설정
WORKDIR /code

# 라이브러리 목록 복사 및 설치
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# 나머지 모든 파일(모델, CSV, 파이썬 코드 등) 복사
COPY . /code

# FastAPI 서버 실행 (메인 파일 이름이 main.py 이므로 main:app 으로 설정)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]