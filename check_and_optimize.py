import json
import numpy as np
import joblib
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import load_model

# -------------------------------------------------------------------------
# 1. 데이터 로드 및 타입 확인 (가장 중요한 부분)
# -------------------------------------------------------------------------
print(">>> [1/5] 데이터 로드 및 구조 확인 시작...")

try:
    with open('team_recent_data.json', 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
except FileNotFoundError:
    print("❌ 에러: 'team_recent_data.json' 파일이 없습니다.")
    exit()

# 첫 번째 데이터로 타입 확인
first_team_key = list(raw_data.keys())[0]
first_match_data = raw_data[first_team_key][0]

print("\n" + "="*50)
print(f"👀 데이터 구조 확인 (첫 번째 데이터 샘플)")
print("="*50)
print(f"Type: {type(first_match_data)}")
print(f"Data: {first_match_data}")
print("="*50 + "\n")

# 데이터 타입에 따라 처리 방식 결정 플래그
IS_DICT = isinstance(first_match_data, dict)

if IS_DICT:
    print("✅ 데이터가 '딕셔너리(Dictionary)' 형태입니다. (Key로 접근)")
else:
    print("✅ 데이터가 '리스트(List)' 형태입니다. (Index로 접근)")
    # 리스트일 경우 순서 매핑 (날짜, 상대팀, 승패, 특성값 순서라고 가정)
    # [날짜, 상대, 결과, [특성...]]
    IDX_DATE = 0
    IDX_OPP = 1
    IDX_RES = 2
    IDX_FEAT = 3

# -------------------------------------------------------------------------
# 2. 데이터 전처리 (train_model.py 로직 + 타입 대응)
# -------------------------------------------------------------------------
print(">>> [2/5] 데이터 전처리 진행 중...")

X_lgb_list = []      
X_lstm_h_list = []   
X_lstm_a_list = []   
y_list = []          

for team_code, matches in raw_data.items():
    if len(matches) < 5: 
        continue
        
    for i in range(len(matches)):
        # 과거 5경기가 없으면 패스
        if i < 5: 
            continue
            
        match = matches[i]
        
        # 타입에 따라 데이터 추출
        if IS_DICT:
            match_date = match['date']
            opp_code = match['opponent']
            match_result = match['result']
            home_recent = [m['features'] for m in matches[i-5:i]]
        else:
            match_date = match[IDX_DATE]
            opp_code = match[IDX_OPP]
            match_result = match[IDX_RES]
            home_recent = [m[IDX_FEAT] for m in matches[i-5:i]]

        # 상대팀 데이터 찾기
        if opp_code not in raw_data:
            continue
            
        opp_all_matches = raw_data[opp_code]
        
        # 날짜 매칭되는 상대팀 경기 인덱스 찾기
        opp_idx = -1
        
        if IS_DICT:
             # 딕셔너리 구조일 때
             for idx, m in enumerate(opp_all_matches):
                 if m['date'] == match_date:
                     opp_idx = idx
                     break
        else:
             # 리스트 구조일 때
             for idx, m in enumerate(opp_all_matches):
                 if m[IDX_DATE] == match_date:
                     opp_idx = idx
                     break
        
        # 못 찾았거나, 상대팀도 과거 5경기가 부족하면 패스
        if opp_idx == -1 or opp_idx < 5: 
            continue

        # 상대팀 최근 5경기 추출
        if IS_DICT:
            away_recent = [m['features'] for m in opp_all_matches[opp_idx-5:opp_idx]]
        else:
            away_recent = [m[IDX_FEAT] for m in opp_all_matches[opp_idx-5:opp_idx]]
            
        # 데이터 조립
        home_seq = np.array(home_recent)
        away_seq = np.array(away_recent)
        
        # LightGBM용 (평균)
        home_mean = np.mean(home_seq, axis=0)
        away_mean = np.mean(away_seq, axis=0)
        lgb_row = np.concatenate([home_mean, away_mean, [1]]) # 상수항 1
        
        # 결과 라벨링
        if match_result == 'win': label = 2
        elif match_result == 'lose': label = 0
        else: label = 1
        
        X_lgb_list.append(lgb_row)
        X_lstm_h_list.append(home_seq)
        X_lstm_a_list.append(away_seq)
        y_list.append(label)

X_lgb = np.array(X_lgb_list)
X_lstm_h = np.array(X_lstm_h_list)
X_lstm_a = np.array(X_lstm_a_list)
y = np.array(y_list)

print(f"✅ 총 데이터 개수: {len(y)}개")

# 테스트셋 분리 (검증용)
_, X_test_lgb, _, y_test = train_test_split(X_lgb, y, test_size=0.2, random_state=42)
_, X_test_lstm_h, _, _ = train_test_split(X_lstm_h, y, test_size=0.2, random_state=42)
_, X_test_lstm_a, _, _ = train_test_split(X_lstm_a, y, test_size=0.2, random_state=42)

# -------------------------------------------------------------------------
# 3. 가중치 최적화 수행
# -------------------------------------------------------------------------
print(">>> [3/5] 모델 로딩 중...")
try:
    lgb_model = joblib.load('lgb_model.pkl')
    lstm_model = load_model('lstm_model.keras')
except Exception as e:
    print(f"❌ 모델 로드 실패: {e}")
    print("같은 폴더에 lgb_model.pkl, lstm_model.keras 파일이 있는지 확인하세요.")
    exit()

print(">>> [4/5] 예측 수행 중...")
pred_lgb = lgb_model.predict_proba(X_test_lgb)
pred_lstm = lstm_model.predict([X_test_lstm_h, X_test_lstm_a], verbose=0)

print(">>> [5/5] 최적 가중치 계산 중...")
best_acc = 0
best_w = 0.5

# 0.0 ~ 1.0 까지 0.01 단위로 테스트
for w in np.arange(0.0, 1.01, 0.01):
    final_prob = (pred_lgb * w) + (pred_lstm * (1 - w))
    final_pred = np.argmax(final_prob, axis=1)
    acc = accuracy_score(y_test, final_pred)
    
    if acc > best_acc:
        best_acc = acc
        best_w = w

print("\n" + "="*50)
print(f"🏆 최종 결과 (최고 정확도: {best_acc*100:.2f}%)")
print("="*50)
print(f"LGBM (정형) 가중치 : {best_w:.2f}")
print(f"LSTM (시계열) 가중치 : {1.0 - best_w:.2f}")
print("="*50)