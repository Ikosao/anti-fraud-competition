# model_v3.py - Оптимизированный CatBoost
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import average_precision_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.isotonic import IsotonicRegression
import warnings
warnings.filterwarnings('ignore')

print("="*50)
print("МОДЕЛЬ АНТИФРОДА v3 - ОПТИМИЗАЦИЯ")
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

# ========== 3. ВСЕ ПРИЗНАКИ (расширенные) ==========
print("\n2. Создаем признаки...")

def create_features(df):
    df = df.copy()
    
    # Временные
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['day_of_month'] = df['timestamp'].dt.day
    df['month'] = df['timestamp'].dt.month
    df['week_of_year'] = df['timestamp'].dt.isocalendar().week.astype(int)
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['is_night'] = ((df['hour'] >= 23) | (df['hour'] <= 5)).astype(int)
    df['is_morning'] = df['hour'].between(6, 10).astype(int)
    df['is_lunch'] = df['hour'].between(11, 14).astype(int)
    df['is_evening'] = df['hour'].between(18, 22).astype(int)
    
    # Сумма
    df['amount_log'] = np.log1p(df['amount'])
    df['amount_sqrt'] = np.sqrt(df['amount'])
    df['amount_round_10'] = (df['amount'] % 10 == 0).astype(int)
    df['amount_round_100'] = (df['amount'] % 100 == 0).astype(int)
    
    # Бины суммы
    df['amount_bin'] = pd.cut(df['amount'], 
                               bins=[0, 10, 25, 50, 100, 250, 500, 1000, np.inf], 
                               labels=['0-10', '10-25', '25-50', '50-100', 
                                      '100-250', '250-500', '500-1000', '1000+'])
    
    # Риск-флаги
    df['high_risk_channel'] = (
        (df['channel'].isin(['online', 'mobile'])) & (df['card_present'] == 0)
    ).astype(int)
    
    df['foreign_tx'] = df['is_foreign'].astype(int)
    df['no_card'] = (df['card_present'] == 0).astype(int)
    
    df['foreign_online'] = (
        (df['is_foreign'] == 1) & (df['channel'].isin(['online', 'mobile']))
    ).astype(int)
    
    df['foreign_no_card'] = (
        (df['is_foreign'] == 1) & (df['card_present'] == 0)
    ).astype(int)
    
    df['high_amount'] = (df['amount'] > 300).astype(int)
    df['very_high_amount'] = (df['amount'] > 1000).astype(int)
    
    # Обработка пропусков
    df['income_bracket'] = df['income_bracket'].fillna('unknown')
    
    return df

train = create_features(train)
test = create_features(test)

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
    cust_no_card_pct=('no_card', 'mean'),
    cust_high_risk_pct=('high_risk_channel', 'mean'),
    cust_night_pct=('is_night', 'mean'),
    cust_weekend_pct=('is_weekend', 'mean'),
    cust_last_time=('timestamp', 'max'),
    cust_first_time=('timestamp', 'min')
).reset_index()

max_time = train['timestamp'].max()
cust_agg['cust_days_since_last'] = (max_time - cust_agg['cust_last_time']).dt.days
cust_agg['cust_active_days'] = (cust_agg['cust_last_time'] - cust_agg['cust_first_time']).dt.days
cust_agg['cust_daily_freq'] = cust_agg['cust_count'] / (cust_agg['cust_active_days'] + 1)
cust_agg['cust_amount_trend'] = cust_agg['cust_amount_sum'] / (cust_agg['cust_active_days'] + 1)
cust_agg['cust_avg_per_day'] = cust_agg['cust_amount_sum'] / (cust_agg['cust_active_days'] + 1)

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
    merch_no_card_pct=('no_card', 'mean')
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

# Регион
print("6. Агрегации регионов...")
region_agg = train.groupby('region').agg(
    region_avg_amount=('amount', 'mean'),
    region_fraud_pct=('is_fraud', 'mean'),  # Это не утечка т.к. смотрим на train статистику
    region_count=('amount', 'count')
).reset_index()

# Объединяем
train = train.merge(cust_agg, on='customer_id', how='left')
test = test.merge(cust_agg, on='customer_id', how='left')

train = train.merge(merch_agg, on='merchant_id', how='left')
test = test.merge(merch_agg, on='merchant_id', how='left')

train = train.merge(cat_agg, on='merchant_category', how='left')
test = test.merge(cat_agg, on='merchant_category', how='left')

train = train.merge(region_agg, on='region', how='left')
test = test.merge(region_agg, on='region', how='left')

# Заполняем пропуски
for col in test.columns:
    if test[col].dtype in ['float64', 'int64'] and col != 'is_fraud':
        test[col] = test[col].fillna(0)

# ========== 5. КОМБИНИРОВАННЫЕ ПРИЗНАКИ ==========
print("7. Комбинированные признаки...")
for df in [train, test]:
    # Отношения
    df['amount_to_cust_avg'] = df['amount'] / (df['cust_avg_amount'] + 1)
    df['amount_to_cust_max'] = df['amount'] / (df['cust_max_amount'] + 1)
    df['amount_to_merch_avg'] = df['amount'] / (df['merch_avg_amount'] + 1)
    df['amount_to_cat_avg'] = df['amount'] / (df['cat_avg_amount'] + 1)
    df['amount_to_region_avg'] = df['amount'] / (df['region_avg_amount'] + 1)
    
    # Z-score
    df['amount_zscore_cust'] = (df['amount'] - df['cust_avg_amount']) / (df['cust_std_amount'] + 1)
    df['amount_zscore_merch'] = (df['amount'] - df['merch_avg_amount']) / (df['merch_std_amount'] + 1)
    
    # Флаги
    df['is_new_customer'] = ((df['cust_count'] == 0) | df['cust_count'].isna()).astype(int)
    df['is_rare_merchant'] = (df['merch_count'] < 5).astype(int)
    df['is_first_transaction'] = (df['cust_count'] == 1).astype(int)
    
    # Аномалии
    df['unusual_amount'] = (df['amount'] > df['cust_avg_amount'] * 3).astype(int)
    df['unusual_time'] = ((df['is_night'] == 1) & (df['cust_night_pct'] < 0.1)).astype(int)
    df['unusual_merchant'] = ((df['merch_count'] < 3) & (df['cust_count'] > 10)).astype(int)
    
    # Комбинации риска
    df['risk_score'] = (
        df['high_risk_channel'] * 2 + 
        df['is_foreign'] * 2 + 
        df['no_card'] * 1 + 
        df['high_amount'] * 1 +
        df['is_night'] * 1
    )

# ========== 6. ПРИЗНАКИ ДЛЯ МОДЕЛИ ==========
features = [
    # Базовые
    'amount', 'amount_log', 'amount_sqrt',
    'amount_round_10', 'amount_round_100',
    'is_foreign', 'card_present', 'age', 'account_age_days',
    # Временные
    'hour', 'day_of_week', 'day_of_month', 'month', 'week_of_year',
    'is_weekend', 'is_night', 'is_morning', 'is_lunch', 'is_evening',
    # Риск-флаги
    'high_risk_channel', 'foreign_online', 'foreign_no_card',
    'high_amount', 'very_high_amount', 'foreign_tx', 'no_card',
    # Клиент
    'cust_avg_amount', 'cust_std_amount', 'cust_max_amount', 
    'cust_min_amount', 'cust_median_amount', 'cust_count', 
    'cust_amount_sum', 'cust_foreign_pct', 'cust_no_card_pct',
    'cust_high_risk_pct', 'cust_night_pct', 'cust_weekend_pct',
    'cust_days_since_last', 'cust_active_days',
    'cust_daily_freq', 'cust_amount_trend', 'cust_avg_per_day',
    # Мерчант
    'merch_avg_amount', 'merch_std_amount', 'merch_max_amount',
    'merch_count', 'merch_foreign_pct', 'merch_high_risk_pct',
    'merch_no_card_pct',
    # Категория
    'cat_avg_amount', 'cat_median_amount', 'cat_count',
    'cat_high_risk_pct', 'cat_foreign_pct',
    # Регион
    'region_avg_amount', 'region_fraud_pct', 'region_count',
    # Комбинации
    'amount_to_cust_avg', 'amount_to_cust_max',
    'amount_to_merch_avg', 'amount_to_cat_avg', 'amount_to_region_avg',
    'amount_zscore_cust', 'amount_zscore_merch',
    'is_new_customer', 'is_rare_merchant', 'is_first_transaction',
    'unusual_amount', 'unusual_time', 'unusual_merchant',
    'risk_score'
]

# Категориальные признаки для CatBoost
cat_features = [
    'merchant_category', 'channel', 'region', 
    'income_bracket', 'amount_bin'
]

print(f"\n8. Признаков: {len(features)} + {len(cat_features)} категориальных")

# ========== 7. ВАЛИДАЦИЯ ==========
print("\n9. Временная валидация...")
train = train.sort_values('timestamp')
tscv = TimeSeriesSplit(n_splits=5)

scores = []

for fold, (tr_idx, val_idx) in enumerate(tscv.split(train)):
    X_tr = train.iloc[tr_idx][features + cat_features]
    y_tr = train.iloc[tr_idx]['is_fraud']
    X_val = train.iloc[val_idx][features + cat_features]
    y_val = train.iloc[val_idx]['is_fraud']
    
    if y_val.sum() == 0:
        continue
    
    # CatBoost с оптимизированными параметрами
    model = CatBoostClassifier(
        iterations=2000,
        learning_rate=0.02,
        depth=8,
        l2_leaf_reg=3,
        border_count=128,
        cat_features=cat_features,
        loss_function='Logloss',
        eval_metric='PRAUC',
        verbose=False,
        random_seed=42,
        class_weights=[1, 25],  # Увеличил вес fraud класса
        early_stopping_rounds=100
    )
    
    model.fit(
        X_tr, y_tr,
        eval_set=(X_val, y_val),
        verbose=False
    )
    
    pred = model.predict_proba(X_val)[:, 1]
    score = average_precision_score(y_val, pred)
    scores.append(score)
    print(f"   Fold {fold+1}: PR-AUC = {score:.4f}")

mean_cv = np.mean(scores)
print(f"\n   Средний PR-AUC: {mean_cv:.4f} (+/- {np.std(scores)*2:.4f})")

# ========== 8. ФИНАЛЬНАЯ МОДЕЛЬ ==========
print("\n10. Обучение финальной модели...")
final_model = CatBoostClassifier(
    iterations=2000,
    learning_rate=0.02,
    depth=8,
    l2_leaf_reg=3,
    border_count=128,
    cat_features=cat_features,
    loss_function='Logloss',
    verbose=200,
    random_seed=42,
    class_weights=[1, 25],
    early_stopping_rounds=100
)

final_model.fit(train[features + cat_features], train['is_fraud'])

# ========== 9. ПРЕДСКАЗАНИЯ ==========
print("\n11. Базовые предсказания...")
test_pred_raw = final_model.predict_proba(test[features + cat_features])[:, 1]

# Калибровка вероятностей
print("12. Калибровка вероятностей...")
# Обучаем калибратор на out-of-fold предсказаниях
train_preds = []
train_targets = []

for tr_idx, val_idx in tscv.split(train):
    X_tr = train.iloc[tr_idx][features + cat_features]
    y_tr = train.iloc[tr_idx]['is_fraud']
    X_val = train.iloc[val_idx][features + cat_features]
    y_val = train.iloc[val_idx]['is_fraud']
    
    cal_model = CatBoostClassifier(
        iterations=2000,
        learning_rate=0.02,
        depth=8,
        cat_features=cat_features,
        verbose=False,
        random_seed=42,
        class_weights=[1, 25]
    )
    cal_model.fit(X_tr, y_tr)
    preds = cal_model.predict_proba(X_val)[:, 1]
    train_preds.extend(preds)
    train_targets.extend(y_val)

# Isotonic Regression для калибровки
iso_reg = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip')
iso_reg.fit(train_preds, train_targets)

# Калибруем предсказания
test_pred_calibrated = iso_reg.predict(test_pred_raw)

# ========== 10. СТАТИСТИКА ==========
print(f"\n   Сырые предсказания:")
print(f"   Средняя: {test_pred_raw.mean():.4f}")
print(f"   Макс: {test_pred_raw.max():.4f}")
print(f"   > 0.5: {(test_pred_raw > 0.5).sum()} шт")
print(f"   > 0.1: {(test_pred_raw > 0.1).sum()} шт")

print(f"\n   Калиброванные предсказания:")
print(f"   Средняя: {test_pred_calibrated.mean():.4f}")
print(f"   Макс: {test_pred_calibrated.max():.4f}")
print(f"   > 0.5: {(test_pred_calibrated > 0.5).sum()} шт")
print(f"   > 0.1: {(test_pred_calibrated > 0.1).sum()} шт")

# ========== 11. СОХРАНЕНИЕ ==========
# Сохраняем обе версии
submission_raw = pd.DataFrame({
    'transaction_id': test['transaction_id'],
    'fraud_proba': test_pred_raw
})
submission_raw.to_csv('submission_v3_raw.csv', index=False)

submission_cal = pd.DataFrame({
    'transaction_id': test['transaction_id'],
    'fraud_proba': test_pred_calibrated
})
submission_cal.to_csv('submission_v3_calibrated.csv', index=False)

print("\n" + "="*50)
print("ГОТОВО!")
print(f"CV PR-AUC: {mean_cv:.4f}")
print(f"Файлы:")
print(f"  - submission_v3_raw.csv (сырые предсказания)")
print(f"  - submission_v3_calibrated.csv (калиброванные)")
print("="*50)

# ========== 12. ВАЖНОСТЬ ПРИЗНАКОВ ==========
print("\n13. Топ-20 важных признаков:")
feature_importance = pd.DataFrame({
    'feature': features + cat_features,
    'importance': final_model.feature_importances_
}).sort_values('importance', ascending=False)

print(feature_importance.head(20).to_string(index=False))
