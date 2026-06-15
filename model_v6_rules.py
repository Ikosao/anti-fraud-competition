# model_v6_rules.py – исправленная версия
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.metrics import average_precision_score
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("МОДЕЛЬ V6 – С БИЗНЕС-ПРАВИЛАМИ")
print("="*60)

# ========== 1. Загрузка ==========
print("\n1. Загрузка...")
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
train['timestamp'] = pd.to_datetime(train['timestamp'])
test['timestamp'] = pd.to_datetime(test['timestamp'])
print(f"Train: {train.shape[0]} | Fraud: {train['is_fraud'].mean():.4f}")

# ========== 2. Признаки + правила ==========
def create_features(df):
    df = df.copy()
    
    # Временные
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5,6]).astype(int)
    df['is_night'] = ((df['hour']>=23)|(df['hour']<=5)).astype(int)
    
    # Сумма
    df['amount_log'] = np.log1p(df['amount'])
    
    # Риск-флаги
    df['high_risk_channel'] = ((df['channel'].isin(['online','mobile'])) & (df['card_present']==0)).astype(int)
    df['foreign_online'] = ((df['is_foreign']==1) & (df['channel'].isin(['online','mobile']))).astype(int)
    
    # Бизнес-правила
    df['rule_in_store'] = ((df['channel']=='in_store') | (df['card_present']==1)).astype(int)
    safe_cats = ['clothing','fuel','health','grocery','restaurant','utilities']
    df['rule_safe_cat'] = df['merchant_category'].isin(safe_cats).astype(int)
    df['rule_gambling_online'] = ((df['merchant_category']=='gambling') & (df['channel']=='online')).astype(int)
    risky_cats = ['gambling','electronics','travel','online_services']
    df['rule_risky_cat'] = df['merchant_category'].isin(risky_cats).astype(int)
    df['rule_risky_cat_online'] = (df['rule_risky_cat'] & (df['channel'].isin(['online','mobile']))).astype(int)
    
    # device_id
    df['has_device'] = df['device_id'].notna().astype(int)
    
    df['income_bracket'] = df['income_bracket'].fillna('unknown')
    return df

print("2. Создание признаков...")
train = create_features(train)
test = create_features(test)

# ========== 3. Агрегаты ==========
print("3. Агрегаты...")
cust = train.groupby('customer_id').agg(
    cust_count=('amount','count'),
    cust_foreign_pct=('is_foreign','mean'),
    cust_highrisk_pct=('high_risk_channel','mean'),
    cust_risky_cat_pct=('rule_risky_cat','mean'),
    last_time=('timestamp','max')
).reset_index()
cust['cust_days_since_last'] = (train['timestamp'].max() - cust['last_time']).dt.days
cust.drop('last_time', axis=1, inplace=True)

# Присоединяем агрегаты
train = train.merge(cust, on='customer_id', how='left')
test = test.merge(cust, on='customer_id', how='left')

for df in [train, test]:
    for c in df.columns:
        if df[c].dtype in ['float64','int64']:
            df[c] = df[c].fillna(0)

# ========== 4. Признаки ==========
features = [
    'amount','amount_log',
    'is_foreign','card_present','age','account_age_days',
    'hour','day_of_week','is_weekend','is_night',
    'high_risk_channel','foreign_online',
    'rule_in_store','rule_safe_cat','rule_gambling_online',
    'rule_risky_cat','rule_risky_cat_online','has_device',
    'cust_count','cust_foreign_pct','cust_highrisk_pct',
    'cust_risky_cat_pct','cust_days_since_last'
]
cat_features = ['merchant_category','channel','region','income_bracket']
print(f"Признаков: {len(features)} + {len(cat_features)} кат.")

# ========== 5. Валидация ==========
print("\n4. Валидация...")
train = train.sort_values('timestamp')
tscv = TimeSeriesSplit(n_splits=5)
scores = []

for fold, (tr_idx, val_idx) in enumerate(tscv.split(train)):
    X_tr = train.iloc[tr_idx][features + cat_features]
    y_tr = train.iloc[tr_idx]['is_fraud']
    X_val = train.iloc[val_idx][features + cat_features]
    y_val = train.iloc[val_idx]['is_fraud']
    if y_val.sum()==0: continue
    
    model = CatBoostClassifier(
        iterations=1000, learning_rate=0.03, depth=6,
        cat_features=cat_features, verbose=False,
        random_seed=42, class_weights=[1,15]
    )
    model.fit(X_tr, y_tr)
    pred = model.predict_proba(X_val)[:,1]
    scores.append(average_precision_score(y_val, pred))
    print(f"   Fold {fold+1}: {scores[-1]:.4f}")

print(f"\n   Средний CV: {np.mean(scores):.4f}")

# ========== 6. Финальная модель ==========
print("\n5. Финальное обучение...")
final_model = CatBoostClassifier(
    iterations=1000, learning_rate=0.03, depth=6,
    cat_features=cat_features, verbose=100,
    random_seed=42, class_weights=[1,15]
)
final_model.fit(train[features + cat_features], train['is_fraud'])
test_pred = final_model.predict_proba(test[features + cat_features])[:,1]

print(f"\n   Средняя: {test_pred.mean():.4f} | Макс: {test_pred.max():.4f}")
print(f"   >0.5: {(test_pred>0.5).sum()} | >0.1: {(test_pred>0.1).sum()}")

pd.DataFrame({
    'transaction_id': test['transaction_id'],
    'fraud_proba': test_pred
}).to_csv('submission_v6_rules.csv', index=False)

print(f"\n submission_v6_rules.csv сохранён")
print(f"CV PR-AUC: {np.mean(scores):.4f}")