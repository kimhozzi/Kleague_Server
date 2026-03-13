# main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import json
import os

app = FastAPI()

# for huggingface code
@app.get("/")
def read_root():
    try:
        return {"message": " K리그 승패 예측 서버가 정상 작동 중입니다! API 문서: /docs"}
    except Exception as e:
        print(f"Error: {e}")
    raise HTTPException(status_code=500, detail=f"서버 에러: {str(e)}")


# ----------------------------------------------------------------
# 1. CORS 설정
# ----------------------------------------------------------------
origins = [
    "http://localhost:3000",
    "http://localhost:5173",
    "*" 
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 2. 전역 변수 및 매핑 정보

artifacts = {}
# 한글변수명을 코드명으로 매핑
KOREAN_TO_CODE = {
    "울산": "ULS", "수원삼성": "SSB", "포항": "POH", "제주": "JEJ",
    "전북": "JEO", "성남": "SNG", "서울": "SEO", "대구": "DAE",
    "인천": "INC", "강원": "GAN", "광주": "GWA", "수원FC": "SFC",
    "김천": "GIM", "대전": "DJN"
}

# train_model.py와 동일한 피처 가공 함수 추가
def features_v1(history_values):
   
    # 데이터가 리스트로 들어올 경우를 대비해 numpy array로 변환
    data = np.array(history_values)
    
    # 1. 가중 평균 최근 5경기를 기준으로 하여... + 가중 평균
    weights = np.array([0.1, 0.15, 0.2, 0.25, 0.3]) 
    weighted_mean = np.average(data, axis=0, weights=weights)
    
    # 2. 추세 기울기
    x = np.arange(len(data))
    slopes = []
    for i in range(data.shape[1]):
        y = data[:, i]
        slope, _ = np.polyfit(x, y, 1) 
        slopes.append(slope)
    slopes = np.array(slopes)
    
    # 3. 변동성 - 표준편차 추가
    std_dev = np.std(data, axis=0)
    
    return np.concatenate([weighted_mean, slopes, std_dev])


@app.on_event("startup")
def load_artifacts():
    global artifacts
    print(">>> [System] 모델 및 데이터 로딩 시작...")
    try:
        # 모델 로드 list -> dict
        artifacts['lgb'] = joblib.load('lgb_model.pkl')
        artifacts['lstm'] = load_model('lstm_model.keras')
        
        with open('team_recent_data.json', 'r', encoding='utf-8') as f:
            artifacts['stats'] = json.load(f)
            
        print(">>> [System] 모델 및 데이터 로딩 완료")    
    except Exception as e:
        print(f" [Error] 로딩 실패: {e}")

# ----------------------------------------------------------------
# 3. API 요청 모델
# ----------------------------------------------------------------
class PredictRequest(BaseModel):
    home_team: str 
    away_team: str 

# ----------------------------------------------------------------
# 4. 예측 로직 (수정됨)
# ----------------------------------------------------------------
@app.post("/api/predict")
async def predict_match(req: PredictRequest):
    home_name = req.home_team
    away_name = req.away_team
    
    # 1. 매핑 및 키 찾기
    home_code = KOREAN_TO_CODE.get(home_name, home_name)
    away_code = KOREAN_TO_CODE.get(away_name, away_name)

    stats = artifacts.get('stats', {})

    home_key = home_code if home_code in stats else (home_name if home_name in stats else None)
    away_key = away_code if away_code in stats else (away_name if away_name in stats else None)

    if not home_key:
        raise HTTPException(status_code=404, detail=f"홈 팀 '{home_name}' 데이터 없음")
    if not away_key:
        raise HTTPException(status_code=404, detail=f"원정 팀 '{away_name}' 데이터 없음")

    try:
        # JSON에서 꺼낸 원본 데이터 (5, 13)
        # home_seq = np.array(stats[home_key]) 
        # away_seq = np.array(stats[away_key]) 
        h_data = stats[home_key]
        a_data = stats[away_key]
        # 1. LightGBM 입력: features_v1을 통과시켜 39개씩 만든 뒤 합침 (총 78개)
        # home_features = features_v1(home_seq)
        # away_features = features_v1(away_seq)
        
        #input_lgb = np.concatenate([home_features, away_features]).reshape(1, -1)

        # 2. LSTM 입력: (1, 5, 13) - 그대로 유지
        # Shape: (1, 5, 13)
        input_lstm_h = np.array(h_data['recent_5']).reshape(1, 5, -1)
        input_lstm_a = np.array(a_data['recent_5']).reshape(1, 5, -1)

        # 2-1. LightGBM 입력 (최근 기세 + 시즌 평균)
        # 39(기세) + 13(평균) + 39(기세) + 13(평균) = 104
        h_v1 = features_v1(h_data['recent_5'])
        h_avg = np.array(h_data['season_avg'])
        a_v1 = features_v1(a_data['recent_5'])
        a_avg = np.array(a_data['season_avg'])
        input_lgb = np.concatenate([h_v1, h_avg, a_v1, a_avg]).reshape(1, -1)

        # ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
        
        # 디버깅 로그
        print(f" LGBM 입력 Shape: {input_lgb.shape} (기대값: (1, 78))")

        # 예측 실행
        lgb_prob = artifacts['lgb'].predict_proba(input_lgb)[0]
        lstm_prob = artifacts['lstm'].predict([input_lstm_h, input_lstm_a], verbose=0)[0]

        # 앙상블 (50:50)
        final_prob = (lgb_prob * 0.5) + (lstm_prob * 0.5)
        
        idx = np.argmax(final_prob)
        pred_text = "승 (Win)" if idx == 2 else ("패 (Loss)" if idx == 0 else "무 (Draw)")

        return {
            "home_team": home_name,
            "away_team": away_name,
            "prediction": pred_text,
            "probability": {
                "win": float(final_prob[2]),
                "draw": float(final_prob[1]),
                "lose": float(final_prob[0])
            }
        }

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=f"서버 에러: {str(e)}")
    



    # Websocket.io 로 방문자 기능 추가하기 + 로그인 버튼도 빼버리셈 react에서 ㅇㅇ 