# model_v3.py – V3 без признаков, создающих сдвиг
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.metrics import average_precision_score
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')

print("="*50)
print("V3 FIXED – без сдвиговых признаков")
print("="*50)

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
train['timestamp'] = pd.to_datetime(train['timestamp'])
test['timestamp'] = pd.to_datetime(test['timestamp'])

def create_features(df):
    df = df.copy()
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5,6]).astype(int)
    df['is_night'] = ((df['hour']>=23)|(df['hour']<=5)).astype(int)
    
    df['amount_log'] = np.log1p(df['amount'])
    df['amount_round10'] = (df['amount']%10==0).astype(int)
    df['amount_round100'] = (df['amount']%100==0).astype(int)
    
    df['high_risk_channel'] = ((df['channel'].isin(['online','mobile'])) & (df['card_present']==0)).astype(int)
    df['foreign_online'] = ((df['is_foreign']==1) & (df['channel'].isin(['online','mobile']))).astype(int)
    df['high_amount'] = (df['amount']>300).astype(int)
    
    df['income_bracket'] = df['income_bracket'].fillna('unknown')
    return df

train = create_features(train)
test = create_features(test)

# Агрегаты – ТОЛЬКО частотные, без средних сумм
cust = train.groupby('customer_id').agg(
    cust_count=('amount','count'),
    cust_foreign_pct=('is_foreign','mean'),
    cust_highrisk_pct=('high_risk_channel','mean'),
    cust_night_pct=('is_night','mean'),
    cust_last_time=('timestamp','max')
).reset_index()
cust['cust_days_since_last'] = (train['timestamp'].max() - cust['cust_last_time']).dt.days
cust.drop('cust_last_time', axis=1, inplace=True)

merch = train.groupby('merchant_id').agg(
    merch_count=('amount','count'),
    merch_highrisk_pct=('high_risk_channel','mean')
).reset_index()

cat = train.groupby('merchant_category').agg(
    cat_count=('amount','count'),
    cat_highrisk_pct=('high_risk_channel','mean')
).reset_index()

for df in [train, test]:
    df = df.merge(cust, on='customer_id', how='left')
    df = df.merge(merch, on='merchant_id', how='left')
    df = df.merge(cat, on='merchant_category', how='left')
    for c in df.columns:
        if df[c].dtype in ['float64','int64']:
            df[c] = df[c].fillna(0)

features = [
    'amount','amount_log','amount_round10','amount_round100',
    'is_foreign','card_present','age','account_age_days',
    'hour','day_of_week','is_weekend','is_night',
    'high_risk_channel','foreign_online','high_amount',
    'cust_count','cust_foreign_pct','cust_highrisk_pct','cust_night_pct',
    'cust_days_since_last',
    'merch_count','merch_highrisk_pct',
    'cat_count','cat_highrisk_pct'
]
cat_features = ['merchant_category','channel','region','income_bracket']

# Валидация (нечестная, как в V3, но даёт ориентир)
train = train.sort_values('timestamp')
tscv = TimeSeriesSplit(n_splits=5)
scores = []

for fold, (tr_idx, val_idx) in enumerate(tscv.split(train)):
    X_tr = train.iloc[tr_idx][features + cat_features]
    y_tr = train.iloc[tr_idx]['is_fraud']
    X_val = train.iloc[val_idx][features + cat_features]
    y_val = train.iloc[val_idx]['is_fraud']
    if y_val.sum()==0: continue
    
    model = CatBoostClassifier(iterations=1000, learning_rate=0.03, depth=7,
                                cat_features=cat_features, verbose=False, 
                                random_seed=42, class_weights=[1,20])
    model.fit(X_tr, y_tr)
    scores.append(average_precision_score(y_val, model.predict_proba(X_val)[:,1]))
    print(f"   Fold {fold+1}: {scores[-1]:.4f}")

print(f"\nCV: {np.mean(scores):.4f}")

final_model = CatBoostClassifier(iterations=1000, learning_rate=0.03, depth=7,
                                  cat_features=cat_features, verbose=200, 
                                  random_seed=42, class_weights=[1,20])
final_model.fit(train[features + cat_features], train['is_fraud'])
test_pred = final_model.predict_proba(test[features + cat_features])[:,1]

pd.DataFrame({'transaction_id':test['transaction_id'], 'fraud_proba':test_pred}).to_csv('submission_v3_fixed.csv', index=False)
print(f"✅ submission_v3_fixed.csv сохранён")