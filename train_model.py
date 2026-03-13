# train_model.py
import pandas as pd
import numpy as np
import joblib
import json
import lightgbm as lgb
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Concatenate, Dropout
from scipy.stats import linregress 

# 데이터 로드 > 2025년 시즌까지 포함
df = pd.read_csv('K리그_통합데이터_result(~2025).csv')

# --- 1. 전처리 ---
## 하위 컬럼 합병  to 단일 행
row_0 = df.iloc[0]
new_columns = []
for col, val in zip(df.columns, row_0):
    col_clean = str(col).split('.')[0]
    if col == 'Rnd.': col_clean = 'Rnd'
    if str(val) in ['시도', '성공', '성공%']:
        new_columns.append(f"{col_clean}_{str(val)}")
    else:
        new_columns.append(col_clean)

df.columns = new_columns
df = df.drop(0).reset_index(drop=True)
df = df[df['Rnd'] != 'Rnd.']

cols_to_numeric = df.columns.drop(['대회', 'H/A', '팀명', '시즌'])
for col in cols_to_numeric:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df = df.dropna(subset=['Rnd', '득점'])
df = df.sort_values(['시즌', 'Rnd']) # 시계열 정렬

# --- 2. 피처 선정 및 스케일링 ---
features = [
    '득점', '도움', '슈팅', '유효 슈팅', 'PA내 슈팅',
    '패스_성공%', '키패스', '공격진영 패스_성공',
    '경합 지상_성공%', '경합 공중_성공%', 
    '인터셉트', '차단', '파울'
]

scaler = StandardScaler()
df[features] = scaler.fit_transform(df[features])

# (이름 변경: features_v1) ---
def features_v1(history_values):
    """
    Input: 최근 5경기 데이터 (numpy array or dataframe values) shape=(5, 13)
    Output: 1차원 벡터 (Feature수 x 3 = 39)
    """
    # 데이터프레임이 들어오면 values로 변환, 아니면 그대로 사용
    data = history_values.values if hasattr(history_values, 'values') else history_values
    
    # 1. 가중 평균 (Weighted Mean)
    weights = np.array([0.1, 0.15, 0.2, 0.25, 0.3]) 
    weighted_mean = np.average(data, axis=0, weights=weights)
    
    # 2. 추세 기울기 (Slope)
    x = np.arange(len(data))
    slopes = []
    for i in range(data.shape[1]):
        y = data[:, i]
        slope, _ = np.polyfit(x, y, 1) 
        slopes.append(slope)
    slopes = np.array(slopes)
    
    # 3. 변동성 (Standard Deviation)
    std_dev = np.std(data, axis=0)
    
    return np.concatenate([weighted_mean, slopes, std_dev])


# --- 3. 데이터셋 생성 ---
def create_dataset(df, window_size=5):
    print( f" 학습용 데이터셋 : {window_size}")
    X_home_seq, X_away_seq, X_2d, y = [], [], [], []
    
    matches = pd.merge(
        df[df['H/A'] == 'HOME'],
        df[df['H/A'] == 'AWAY'],
        on=['시즌', 'Rnd', '대회'],
        suffixes=('', '_opp')
    )
    
    print(f" 총 {len(matches)} 경기 데이터")

    for _, row in matches.iterrows():
        home_team, away_team = row['팀명'], row['팀명_opp']
        season, rnd = row['시즌'], row['Rnd']
        
        # 과거 5경기 조회
        home_hist = df[(df['팀명'] == home_team) & (df['시즌'] == season) & (df['Rnd'] < rnd)].tail(window_size)
        away_hist = df[(df['팀명'] == away_team) & (df['시즌'] == season) & (df['Rnd'] < rnd)].tail(window_size)
        
        home_whole = df[(df['팀명'] == home_team) & (df['시즌'] == season) & (df['Rnd'] < rnd)]
        away_whole = df[(df['팀명'] == away_team) & (df['시즌'] == season) & (df['Rnd'] < rnd)]

        if len(home_hist) == window_size and len(away_hist) == window_size: #5경기 이상 있어야 함
            # 1) LSTM용 (Sequence) 최근 5경기
            X_home_seq.append(home_hist[features].values)
            X_away_seq.append(away_hist[features].values)


            # 2) LightGBM용 (2D) - features_v1 적용 최근 5경기 + 전체 시즌 평균으로 수정
            h_v1 = features_v1(home_hist[features])
            h_avg = home_whole[features].mean().values
            a_v1 = features_v1(away_hist[features])
            a_avg = away_whole[features].mean().values

            X_2d.append(np.concatenate([h_v1, a_v1, h_avg, a_avg])) 
            
            # Target
            if row['득점'] > row['득점_opp']: target = 2   # 승
            elif row['득점'] < row['득점_opp']: target = 0  # 패
            else: target = 1  # 무
            y.append(target)
            
    return np.array(X_home_seq), np.array(X_away_seq), np.array(X_2d), np.array(y)

X_h, X_a, X_2d, y = create_dataset(df)

# 학습 데이터 분할
X_h_tr, X_h_te, X_a_tr, X_a_te, X_2d_tr, X_2d_te, y_tr, y_te = train_test_split(
    X_h, X_a, X_2d, y, test_size=0.2, random_state=42, stratify=y
)


## 과적합 피하기 위해서 dense 층 줄임, dropouㅅ
print(">>> LightGBM 학습 중...") 
lgb_model = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05, verbosity=-1)
lgb_model.fit(X_2d_tr, y_tr)

print(">>> LSTM 학습 중...") 
input_h = Input(shape=(5, len(features)), name='Home_Input')
lstm_h = LSTM(32)(input_h)
input_a = Input(shape=(5, len(features)), name='Away_Input')
lstm_a = LSTM(32)(input_a)

# LightGBM + LSTM 병합 후 학습해야 함 dropout 추가 
merged = Concatenate()([lstm_h, lstm_a])
x = Dense(64, activation='relu')(merged)
x = Dropout(0.3)(x)
x = Dense(32, activation='relu')(x)
x = Dropout(0.2)(x)
output = Dense(3, activation='softmax')(x)

lstm_model = Model(inputs=[input_h, input_a], outputs=output)
lstm_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
lstm_model.fit([X_h_tr, X_a_tr], y_tr, epochs=30, batch_size=16, verbose=0)

# --- 5. 저장 --- dump 및 save
print(">>> 파일 저장 중...")
joblib.dump(lgb_model, 'lgb_model.pkl')
lstm_model.save('lstm_model.keras')
joblib.dump(scaler, 'scaler.pkl')

# 추론용 데이터 저장 (가공 전 원본 Sequence 저장) (최근 5경기 + 시즌 전체 평균)
latest_data = {}
#각 팀의 경기를
for team in df['팀명'].unique():
    # 해당 팀의 가장 최근 시즌 데이터 가져오기 - 최근 시즌의 5경기
    team_df = df[df['팀명'] == team].sort_values(['시즌', 'Rnd'])
    
    # 1. 최근 5경기 (LSTM & features_v1용)
    last_5 = team_df.tail(5)[features].values
    
    # 2. 이번 시즌 전체 평균 (avg용)
    # 가장 마지막 경기가 포함된 '시즌'의 전체 평균을 구함
    if not team_df.empty:
        last_season = team_df.iloc[-1]['시즌']
        season_whole = team_df[team_df['시즌'] == last_season][features]
        season_avg = season_whole.mean().values
    else:
        season_avg = np.zeros(len(features))

    # 5경기가 꽉 찼을 때만 저장
    if len(last_5) == 5:
        latest_data[team] = {
            'recent_5': last_5.tolist(),   # 리스트로 변환
            'season_avg': season_avg.tolist() # 리스트로 변환
        }

with open('team_recent_data.json', 'w', encoding='utf-8') as f:
    json.dump(latest_data, f, ensure_ascii=False)

print(f">>> 완료! 총 {len(latest_data)}개 팀 데이터 저장됨.")



# ## 다음
#  main.py 수정 가이드
# main.py에 다음 두 가지를 적용해야 합니다.

# features_v1 함수 추가: train_model.py에 있는 함수를 그대로 복사해 옵니다.

# predict_match 내부 로직 변경: 기존의 단순 평균(np.mean) 방식을 지우고, features_v1을 통과시키는 로직으로 교체합니다.
