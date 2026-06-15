# model_v2.py - Исправленная версия с ансамблем
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
import lightgbm as lgb
from sklearn.metrics import average_precision_score
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')

print("="*50)
print("МОДЕЛЬ АНТИФРОДА v2 - АНСАМБЛЬ")
print("="*50)

# ========== 1. ЗАГРУЗКА ==========
print("\n1. Загружаем данные...")
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

print(f"Train: {train.shape[0]}, Test: {test.shape[0]}")
print(f"Fraud: {train['is_fraud'].mean():.4f}")

# ========== 2. ОБРАБОТКА ==========
train['timestamp'] = pd.to_datetime(train['timestamp'])
test['timestamp'] = pd.to_datetime(test['timestamp'])

# ========== 3. ВСЕ ПРИЗНАКИ ==========
print("\n2. Создаем расширенные признаки...")

def create_all_features(df):
    """Создание всех признаков"""
    df = df.copy()
    
    # --- Временные ---
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['day_of_month'] = df['timestamp'].dt.day
    df['month'] = df['timestamp'].dt.month
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['is_night'] = ((df['hour'] >= 23) | (df['hour'] <= 5)).astype(int)
    df['is_morning'] = df['hour'].between(6, 10).astype(int)
    df['is_business_hours'] = df['hour'].between(9, 18).astype(int)
    
    # --- Сумма ---
    df['amount_log'] = np.log1p(df['amount'])
    df['amount_sqrt'] = np.sqrt(df['amount'])
    df['amount_round_10'] = (df['amount'] % 10 == 0).astype(int)
    df['amount_round_100'] = (df['amount'] % 100 == 0).astype(int)
    df['amount_round_50'] = (df['amount'] % 50 == 0).astype(int)
    
    # --- Риск-флаги ---
    df['high_risk_channel'] = (
        (df['channel'].isin(['online', 'mobile'])) & 
        (df['card_present'] == 0)
    ).astype(int)
    
    df['foreign_online'] = (
        (df['is_foreign'] == 1) & 
        (df['channel'].isin(['online', 'mobile']))
    ).astype(int)
    
    df['high_amount'] = (df['amount'] > 500).astype(int)
    df['very_high_amount'] = (df['amount'] > 1000).astype(int)
    
    # --- Обработка пропусков ---
    df['income_bracket'] = df['income_bracket'].fillna('unknown')
    
    return df

# Создаем признаки
train = create_all_features(train)
test = create_all_features(test)

# ========== 4. АГРЕГАЦИИ ==========
print("3. Агрегации клиентов...")

# Клиенты
cust_agg = train.groupby('customer_id').agg(
    cust_avg_amount=('amount', 'mean'),
    cust_std_amount=('amount', 'std'),
    cust_max_amount=('amount', 'max'),
    cust_min_amount=('amount', 'min'),
    cust_median_amount=('amount', 'median'),
    cust_count=('amount', 'count'),
    cust_amount_sum=('amount', 'sum'),
    cust_foreign_pct=('is_foreign', 'mean'),
    cust_card_present_pct=('card_present', 'mean'),
    cust_high_risk_pct=('high_risk_channel', 'mean'),
    cust_night_pct=('is_night', 'mean'),
    cust_last_time=('timestamp', 'max'),
    cust_first_time=('timestamp', 'min')
).reset_index()

max_time = train['timestamp'].max()
cust_agg['cust_days_since_last'] = (max_time - cust_agg['cust_last_time']).dt.days
cust_agg['cust_active_days'] = (
    cust_agg['cust_last_time'] - cust_agg['cust_first_time']
).dt.days
cust_agg['cust_daily_freq'] = cust_agg['cust_count'] / (cust_agg['cust_active_days'] + 1)
cust_agg['cust_amount_trend'] = cust_agg['cust_amount_sum'] / (cust_agg['cust_active_days'] + 1)

cust_agg.drop(['cust_last_time', 'cust_first_time'], axis=1, inplace=True)
cust_agg['cust_std_amount'] = cust_agg['cust_std_amount'].fillna(0)

# Мерчанты
print("4. Агрегации мерчантов...")
merch_agg = train.groupby('merchant_id').agg(
    merch_avg_amount=('amount', 'mean'),
    merch_std_amount=('amount', 'std'),
    merch_max_amount=('amount', 'max'),
    merch_count=('amount', 'count'),
    merch_foreign_pct=('is_foreign', 'mean'),
    merch_high_risk_pct=('high_risk_channel', 'mean'),
    merch_card_present_pct=('card_present', 'mean')
).reset_index()

merch_agg['merch_std_amount'] = merch_agg['merch_std_amount'].fillna(0)

# Категории
print("5. Агрегации категорий...")
cat_agg = train.groupby('merchant_category').agg(
    cat_avg_amount=('amount', 'mean'),
    cat_median_amount=('amount', 'median'),
    cat_count=('amount', 'count'),
    cat_high_risk_pct=('high_risk_channel', 'mean'),
    cat_foreign_pct=('is_foreign', 'mean')
).reset_index()

# Объединяем
print("6. Объединяем данные...")
train = train.merge(cust_agg, on='customer_id', how='left')
test = test.merge(cust_agg, on='customer_id', how='left')

train = train.merge(merch_agg, on='merchant_id', how='left')
test = test.merge(merch_agg, on='merchant_id', how='left')

train = train.merge(cat_agg, on='merchant_category', how='left')
test = test.merge(cat_agg, on='merchant_category', how='left')

# Заполняем пропуски в test
for col in test.columns:
    if test[col].dtype in ['float64', 'int64'] and col != 'is_fraud':
        test[col] = test[col].fillna(0)

# ========== 5. ДОПОЛНИТЕЛЬНЫЕ ПРИЗНАКИ ==========
print("7. Комбинированные признаки...")
for df in [train, test]:
    df['amount_vs_cust_avg'] = df['amount'] / (df['cust_avg_amount'] + 1)
    df['amount_vs_cust_max'] = df['amount'] / (df['cust_max_amount'] + 1)
    df['amount_vs_merch_avg'] = df['amount'] / (df['merch_avg_amount'] + 1)
    df['amount_vs_cat_avg'] = df['amount'] / (df['cat_avg_amount'] + 1)
    
    df['amount_zscore_cust'] = (
        (df['amount'] - df['cust_avg_amount']) / (df['cust_std_amount'] + 1)
    )
    
    df['is_new_customer'] = ((df['cust_count'] == 0) | df['cust_count'].isna()).astype(int)
    df['is_rare_merchant'] = (df['merch_count'] < 10).astype(int)
    df['is_unusual_time'] = (
        (df['is_night'] == 1) & (df['cust_night_pct'] < 0.1)
    ).astype(int)
    df['amount_spike'] = (
        df['amount'] > df['cust_avg_amount'] * 2
    ).astype(int)

# ========== 6. ПРИЗНАКИ ДЛЯ МОДЕЛЕЙ ==========
# Для LightGBM - все числовые
features_lgb = [
    'amount', 'amount_log', 'amount_sqrt',
    'amount_round_10', 'amount_round_100', 'amount_round_50',
    'is_foreign', 'card_present', 'age', 'account_age_days',
    'hour', 'day_of_week', 'day_of_month', 'month',
    'is_weekend', 'is_night', 'is_morning', 'is_business_hours',
    'high_risk_channel', 'foreign_online', 
    'high_amount', 'very_high_amount',
    'cust_avg_amount', 'cust_std_amount', 'cust_max_amount', 
    'cust_min_amount', 'cust_median_amount', 'cust_count', 
    'cust_amount_sum', 'cust_foreign_pct', 'cust_card_present_pct',
    'cust_high_risk_pct', 'cust_night_pct',
    'cust_days_since_last', 'cust_active_days',
    'cust_daily_freq', 'cust_amount_trend',
    'merch_avg_amount', 'merch_std_amount', 'merch_max_amount',
    'merch_count', 'merch_foreign_pct', 'merch_high_risk_pct',
    'merch_card_present_pct',
    'cat_avg_amount', 'cat_median_amount', 'cat_count',
    'cat_high_risk_pct', 'cat_foreign_pct',
    'amount_vs_cust_avg', 'amount_vs_cust_max',
    'amount_vs_merch_avg', 'amount_vs_cat_avg',
    'amount_zscore_cust',
    'is_new_customer', 'is_rare_merchant',
    'is_unusual_time', 'amount_spike'
]

# Для CatBoost - добавляем строковые категории
features_cb = features_lgb + ['merchant_category', 'channel', 'region', 'income_bracket']
cat_features_cb = ['merchant_category', 'channel', 'region', 'income_bracket']

print(f"Признаков: {len(features_lgb)} числовых + 4 категориальных")

# ========== 7. ВАЛИДАЦИЯ ==========
print("\n8. Временная валидация...")
train = train.sort_values('timestamp')
tscv = TimeSeriesSplit(n_splits=5)

cb_scores = []
lgb_scores = []
ensemble_scores = []

for fold, (tr_idx, val_idx) in enumerate(tscv.split(train)):
    X_tr_lgb = train.iloc[tr_idx][features_lgb]
    y_tr = train.iloc[tr_idx]['is_fraud']
    X_val_lgb = train.iloc[val_idx][features_lgb]
    y_val = train.iloc[val_idx]['is_fraud']
    
    X_tr_cb = train.iloc[tr_idx][features_cb]
    X_val_cb = train.iloc[val_idx][features_cb]
    
    if y_val.sum() == 0:
        continue
    
    # CatBoost
    cb_model = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.03,
        depth=6,
        cat_features=cat_features_cb,
        verbose=False,
        random_seed=42,
        class_weights=[1, 20]
    )
    cb_model.fit(X_tr_cb, y_tr)
    cb_pred = cb_model.predict_proba(X_val_cb)[:, 1]
    cb_score = average_precision_score(y_val, cb_pred)
    cb_scores.append(cb_score)
    
    # LightGBM
    lgb_model = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=63,
        max_depth=8,
        min_child_samples=50,
        subsample=0.8,
        colsample_bytree=0.8,
        verbose=-1,
        random_state=42,
        class_weight='balanced'
    )
    lgb_model.fit(X_tr_lgb, y_tr)
    lgb_pred = lgb_model.predict_proba(X_val_lgb)[:, 1]
    lgb_score = average_precision_score(y_val, lgb_pred)
    lgb_scores.append(lgb_score)
    
    # Ансамбль
    ensemble_pred = 0.6 * cb_pred + 0.4 * lgb_pred
    ensemble_score = average_precision_score(y_val, ensemble_pred)
    ensemble_scores.append(ensemble_score)
    
    print(f"   Fold {fold+1}: CB={cb_score:.4f}, LGB={lgb_score:.4f}, Ens={ensemble_score:.4f}")

print(f"\n   Средний CatBoost: {np.mean(cb_scores):.4f}")
print(f"   Средний LightGBM: {np.mean(lgb_scores):.4f}")
print(f"   Средний Ансамбль: {np.mean(ensemble_scores):.4f}")

# ========== 8. ФИНАЛЬНЫЕ МОДЕЛИ ==========
print("\n9. Обучение финальных моделей...")

print("   CatBoost...")
cb_final = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.03,
    depth=6,
    cat_features=cat_features_cb,
    verbose=200,
    random_seed=42,
    class_weights=[1, 20]
)
cb_final.fit(train[features_cb], train['is_fraud'])

print("   LightGBM...")
lgb_final = lgb.LGBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=63,
    max_depth=8,
    min_child_samples=50,
    subsample=0.8,
    colsample_bytree=0.8,
    verbose=100,
    random_state=42,
    class_weight='balanced'
)
lgb_final.fit(train[features_lgb], train['is_fraud'])

# ========== 9. ПРЕДСКАЗАНИЯ ==========
print("\n10. Предсказания...")
cb_test_pred = cb_final.predict_proba(test[features_cb])[:, 1]
lgb_test_pred = lgb_final.predict_proba(test[features_lgb])[:, 1]

final_pred = 0.6 * cb_test_pred + 0.4 * lgb_test_pred

print(f"   Средняя: {final_pred.mean():.4f}")
print(f"   Макс: {final_pred.max():.4f}")
print(f"   > 0.5: {(final_pred > 0.5).sum()} шт")
print(f"   > 0.1: {(final_pred > 0.1).sum()} шт")

# ========== 10. СОХРАНЕНИЕ ==========
submission = pd.DataFrame({
    'transaction_id': test['transaction_id'],
    'fraud_proba': final_pred
})

submission.to_csv('submission_v2.csv', index=False)

print("\n" + "="*50)
print("ГОТОВО!")
print(f"Файл: submission_v2.csv")
print(f"CV PR-AUC: {np.mean(ensemble_scores):.4f}")
print("="*50)