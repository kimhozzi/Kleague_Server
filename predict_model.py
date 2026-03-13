import pandas as pd
import numpy as np
import joblib
import json
import tensorflow as tf
from tensorflow.keras.models import load_model

# --- 1. 모델 및 데이터 로드 ---
print(">>> 모델 및 데이터 로딩 중...")

# LightGBM 모델 로드
lgb_model = joblib.load('lgb_model.pkl')

# LSTM 모델 로드
lstm_model = load_model('lstm_model.keras')

# 최근 데이터(이미 스케일링 된 상태) 로드
with open('team_recent_data.json', 'r', encoding='utf-8') as f:
    team_data = json.load(f)

# 클래스 매핑 (학습 코드 기준: 0=패, 1=무, 2=승)
classes = {0: '패 (원정승)', 1: '무승부', 2: '승 (홈승)'}

# --- 2. 예측 함수 정의 ---
def predict_match(home_team, away_team):
    # 팀 데이터 존재 여부 확인
    if home_team not in team_data or away_team not in team_data:
        print(f"오류: {home_team} 또는 {away_team}의 최근 데이터가 없습니다.")
        return

    # 리스트 형태의 데이터를 numpy 배열로 변환
    # JSON에서 불러오면 리스트이므로 np.array 변환 필수
    home_seq = np.array(team_data[home_team])  # Shape: (5, 13)
    away_seq = np.array(team_data[away_team])  # Shape: (5, 13)

    # -------------------------------
    # [입력 데이터 가공]
    # -------------------------------
    
    # 1) LightGBM용 입력 (2D): 평균값 + [1]
    # 학습 코드: np.concatenate([home_mean, away_mean, [1]])
    input_lgb = np.concatenate([
        np.mean(home_seq, axis=0), 
        np.mean(away_seq, axis=0), 
        [1]  # 학습 때 넣었던 상수 1 (중요!)
    ]).reshape(1, -1)

    # 2) LSTM용 입력 (3D): (1, 5, 13)
    # 학습 코드와 동일하게 차원 확장
    input_lstm_h = home_seq.reshape(1, 5, -1)
    input_lstm_a = away_seq.reshape(1, 5, -1)

    # -------------------------------
    # [모델 예측]
    # -------------------------------
    
    # LightGBM 예측 (확률)
    lgb_prob = lgb_model.predict_proba(input_lgb)[0]
    
    # LSTM 예측 (확률)
    # verbose=0으로 로그 출력 끔
    lstm_prob = lstm_model.predict([input_lstm_h, input_lstm_a], verbose=0)[0]

    # 두 모델의 평균 확률 계산 (앙상블)
    avg_prob = (lgb_prob + lstm_prob) / 2
    final_pred = np.argmax(avg_prob) # 확률이 가장 높은 인덱스

    # --- 3. 결과 출력 --- only log -> 
    
    print(f"\n⚽ [{home_team}] vs [{away_team}] 예측 결과 ⚽")
    print("-" * 40)
    
    print(f"[LightGBM] 패: {lgb_prob[0]:.1%} | 무: {lgb_prob[1]:.1%} | 승: {lgb_prob[2]:.1%}")
    print(f"[LSTM    ] 패: {lstm_prob[0]:.1%} | 무: {lstm_prob[1]:.1%} | 승: {lstm_prob[2]:.1%}")
    print("-" * 40)
    print(f"📊 [종합 예측] >> {classes[final_pred]} (확률: {avg_prob[final_pred]:.1%})")
    print(f"   (종합 확률: 패 {avg_prob[0]:.1%} / 무 {avg_prob[1]:.1%} / 승 {avg_prob[2]:.1%})")

# --- 4. 실행 예시 ---
# team_recent_data.json에 있는 실제 팀명을 입력해야 합니다.
# 예시 팀명입니다. 실제 데이터에 있는 팀명으로 바꿔서 테스트해보세요.
# if __name__ == "__main__":
#     # 데이터에 있는 팀 목록 출력 (참고용)
#     print(f"가능한 팀 목록: {list(team_data.keys())[:5]} ...")
    
#     # 테스트 (실제 팀 이름으로 변경 필요)
#     # 예: predict_match('울산', '전북')
    
#     # 사용자 입력을 받아 실행하려면 아래 주석을 해제하세요. 이걸 print 가 아닌 
#     h_team = input("\n홈팀을 입력하세요: ")
#     a_team = input("원정팀을 입력하세요: ")
#     predict_match(h_team, a_team)

