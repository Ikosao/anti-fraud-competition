# model_v5.py – правильная работа с временными признаками
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit, train_test_split
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("МОДЕЛЬ АНТИФРОДА v5 – УСТОЙЧИВЫЕ ПРИЗНАКИ")
print("="*60)

# ========== 1. Загрузка ==========
print("\n1. Загрузка...")
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
train['timestamp'] = pd.to_datetime(train['timestamp'])
test['timestamp']  = pd.to_datetime(test['timestamp'])
print(f"Train: {train.shape[0]}  |  Test: {test.shape[0]}  |  Fraud: {train['is_fraud'].mean():.4f}")

# ========== 2. Признаки с циклическим временем ==========
def create_features(df):
    df = df.copy()
    
    # Циклические временные признаки
    df['hour_sin'] = np.sin(2 * np.pi * df['timestamp'].dt.hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['timestamp'].dt.hour / 24)
    df['dow_sin'] = np.sin(2 * np.pi * df['timestamp'].dt.dayofweek / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['timestamp'].dt.dayofweek / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['timestamp'].dt.month / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['timestamp'].dt.month / 12)
    df['dom_sin'] = np.sin(2 * np.pi * df['timestamp'].dt.day / 31)
    df['dom_cos'] = np.cos(2 * np.pi * df['timestamp'].dt.day / 31)
    
    # Флаги (они меньше подвержены сдвигу)
    df['is_weekend'] = df['timestamp'].dt.dayofweek.isin([5,6]).astype(int)
    df['is_night'] = ((df['timestamp'].dt.hour>=23)|(df['timestamp'].dt.hour<=5)).astype(int)
    df['is_morning'] = df['timestamp'].dt.hour.between(6,10).astype(int)
    
    # Относительное время (день от начала данных)
    min_ts = pd.Timestamp('2025-01-01')  # фиксированная точка отсчёта
    df['day_number'] = (df['timestamp'] - min_ts).dt.days
    
    # Сумма
    df['amount_log'] = np.log1p(df['amount'])
    df['amount_sqrt'] = np.sqrt(df['amount'])
    df['amount_round10'] = (df['amount']%10==0).astype(int)
    df['amount_round100'] = (df['amount']%100==0).astype(int)
    
    # Риски
    df['high_risk_channel'] = ((df['channel'].isin(['online','mobile'])) & (df['card_present']==0)).astype(int)
    df['foreign_online'] = ((df['is_foreign']==1) & (df['channel'].isin(['online','mobile']))).astype(int)
    df['high_amount'] = (df['amount']>300).astype(int)
    df['very_high_amount'] = (df['amount']>1000).astype(int)
    
    # device_id
    df['device_id'] = df['device_id'].fillna('missing').astype(str)
    df['device_enc'] = pd.factorize(df['device_id'])[0]
    df['has_device'] = (df['device_id'] != 'missing').astype(int)
    
    df['income_bracket'] = df['income_bracket'].fillna('unknown')
    return df

print("2. Создание признаков...")
train = create_features(train)
test = create_features(test)

# ========== 3. Агрегаты с относительным временем ==========
def build_aggregates(base, target_df):
    # Клиенты
    cust = base.groupby('customer_id').agg(
        cust_avg_amount=('amount','mean'),
        cust_std_amount=('amount','std'),
        cust_max_amount=('amount','max'),
        cust_min_amount=('amount','min'),
        cust_count=('amount','count'),
        cust_foreign_pct=('is_foreign','mean'),
        cust_highrisk_pct=('high_risk_channel','mean'),
        cust_night_pct=('is_night','mean'),
        cust_weekend_pct=('is_weekend','mean'),
        last_day=('day_number','max'),
        first_day=('day_number','min')
    ).reset_index()
    
    cust['cust_active_days'] = cust['last_day'] - cust['first_day']
    cust['cust_daily_freq'] = cust['cust_count'] / (cust['cust_active_days'] + 1)
    cust['cust_days_since_last'] = base['day_number'].max() - cust['last_day']
    cust.drop(['last_day','first_day'], axis=1, inplace=True)
    cust['cust_std_amount'] = cust['cust_std_amount'].fillna(0)
    
    # Мерчанты (только частотные)
    merch = base.groupby('merchant_id').agg(
        merch_count=('amount','count'),
        merch_foreign_pct=('is_foreign','mean'),
        merch_highrisk_pct=('high_risk_channel','mean')
    ).reset_index()
    
    # Категории
    cat = base.groupby('merchant_category').agg(
        cat_avg_amount=('amount','mean'),
        cat_count=('amount','count'),
        cat_highrisk_pct=('high_risk_channel','mean'),
        cat_foreign_pct=('is_foreign','mean')
    ).reset_index()
    
    # Регион
    region = base.groupby('region').agg(
        region_avg_amount=('amount','mean'),
        region_highrisk_pct=('high_risk_channel','mean')
    ).reset_index()
    
    out = target_df.merge(cust, on='customer_id', how='left')
    out = out.merge(merch, on='merchant_id', how='left')
    out = out.merge(cat, on='merchant_category', how='left')
    out = out.merge(region, on='region', how='left')
    for c in out.columns:
        if out[c].dtype in ['float64','int64'] and c!='is_fraud':
            out[c] = out[c].fillna(0)
    return out

def add_combinations(df):
    df['amount_to_cust_avg'] = df['amount']/(df['cust_avg_amount']+1)
    df['amount_to_cat_avg'] = df['amount']/(df['cat_avg_amount']+1)
    df['amount_zscore_cust'] = (df['amount']-df['cust_avg_amount'])/(df['cust_std_amount']+1)
    df['is_new_customer'] = ((df['cust_count']==0)|df['cust_count'].isna()).astype(int)
    df['is_rare_merchant'] = (df['merch_count']<5).astype(int)
    df['unusual_amount'] = (df['amount']>df['cust_avg_amount']*3).astype(int)
    df['unusual_time'] = ((df['is_night']==1)&(df['cust_night_pct']<0.1)).astype(int)
    return df

# ========== 4. Признаки ==========
base_features = [
    'amount','amount_log','amount_sqrt','amount_round10','amount_round100',
    'is_foreign','card_present','age','account_age_days',
    'hour_sin','hour_cos','dow_sin','dow_cos',
    'month_sin','month_cos','dom_sin','dom_cos',
    'day_number',
    'is_weekend','is_night','is_morning',
    'high_risk_channel','foreign_online','high_amount','very_high_amount',
    'device_enc','has_device'
]
agg_features = [
    'cust_avg_amount','cust_std_amount','cust_max_amount','cust_min_amount',
    'cust_count','cust_foreign_pct','cust_highrisk_pct',
    'cust_night_pct','cust_weekend_pct',
    'cust_active_days','cust_daily_freq','cust_days_since_last',
    'merch_count','merch_foreign_pct','merch_highrisk_pct',
    'cat_avg_amount','cat_count','cat_highrisk_pct','cat_foreign_pct',
    'region_avg_amount','region_highrisk_pct'
]
combo_features = [
    'amount_to_cust_avg','amount_to_cat_avg',
    'amount_zscore_cust',
    'is_new_customer','is_rare_merchant','unusual_amount','unusual_time'
]
cat_features = ['merchant_category','channel','region','income_bracket']
all_features = base_features + agg_features + combo_features
print(f"Признаков: {len(all_features)} + {len(cat_features)} кат.")

# ========== 5. Валидация ==========
print("\n3. Честная валидация...")
train = train.sort_values('timestamp')
tscv = TimeSeriesSplit(n_splits=5)
scores = []

for fold, (tr_idx, val_idx) in enumerate(tscv.split(train)):
    train_fold = train.iloc[tr_idx].copy()
    val_fold = train.iloc[val_idx].copy()
    if val_fold['is_fraud'].sum()==0: continue
    
    train_fold = build_aggregates(train_fold, train_fold)
    val_fold = build_aggregates(train_fold, val_fold)
    train_fold = add_combinations(train_fold)
    val_fold = add_combinations(val_fold)

    model = CatBoostClassifier(
        iterations=2000, learning_rate=0.02, depth=8,
        cat_features=cat_features, verbose=False,
        random_seed=42, class_weights=[1,25]
    )
    model.fit(train_fold[all_features + cat_features], train_fold['is_fraud'])
    pred = model.predict_proba(val_fold[all_features + cat_features])[:,1]
    sc = average_precision_score(val_fold['is_fraud'], pred)
    scores.append(sc)
    print(f"   Fold {fold+1}: PR-AUC = {sc:.4f}")

print(f"\n   Средний честный PR-AUC: {np.mean(scores):.4f}")

# ========== 6. Adversarial ==========
print("\n4. Adversarial validation...")
train_adv = train.copy(); train_adv['is_test'] = 0
test_adv = test.copy(); test_adv['is_test'] = 1
combined = pd.concat([train_adv, test_adv], ignore_index=True)
combined = create_features(combined)
combined = build_aggregates(train, combined)
combined = add_combinations(combined)

X_adv = combined[all_features + cat_features]
y_adv = combined['is_test']
X_tr_adv, X_val_adv, y_tr_adv, y_val_adv = train_test_split(
    X_adv, y_adv, test_size=0.3, random_state=42, stratify=y_adv
)
adv_model = CatBoostClassifier(
    iterations=500, learning_rate=0.1, depth=5,
    cat_features=cat_features, verbose=False, random_seed=42
)
adv_model.fit(X_tr_adv, y_tr_adv)
auc_adv = roc_auc_score(y_val_adv, adv_model.predict_proba(X_val_adv)[:,1])
print(f"   ROC-AUC отличия: {auc_adv:.4f}")

# ========== 7. Финальная модель ==========
print("\n5. Финальное обучение...")
train_full = build_aggregates(train, train)
test_full = build_aggregates(train, test)
train_full = add_combinations(train_full)
test_full = add_combinations(test_full)

final_model = CatBoostClassifier(
    iterations=2000, learning_rate=0.02, depth=8,
    cat_features=cat_features, verbose=200,
    random_seed=42, class_weights=[1,25]
)
final_model.fit(train_full[all_features + cat_features], train_full['is_fraud'])
test_pred = final_model.predict_proba(test_full[all_features + cat_features])[:,1]

print(f"\n   Средняя: {test_pred.mean():.4f} | Макс: {test_pred.max():.4f} | >0.5: {(test_pred>0.5).sum()}")

sub = pd.DataFrame({'transaction_id':test['transaction_id'], 'fraud_proba':test_pred})
sub.to_csv('submission_v5.csv', index=False)
print(f"\n submission_v5.csv сохранён. CV: {np.mean(scores):.4f}")