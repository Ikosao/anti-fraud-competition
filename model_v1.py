# model_v1.py - Финальная версия без утечек
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.metrics import average_precision_score
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')

print("="*50)
print("МОДЕЛЬ АНТИФРОДА v2")
print("="*50)

# ========== 1. ЗАГРУЗКА ==========
print("\n1. Загружаем данные...")
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

print(f"Train: {train.shape[0]}, Test: {test.shape[0]}")
print(f"Fraud: {train['is_fraud'].mean():.4f}")

# ========== 2. ОБРАБОТКА ВРЕМЕНИ ==========
train['timestamp'] = pd.to_datetime(train['timestamp'])
test['timestamp'] = pd.to_datetime(test['timestamp'])

print(f"Train: {train['timestamp'].min()} -> {train['timestamp'].max()}")
print(f"Test: {test['timestamp'].min()} -> {test['timestamp'].max()}")

# ========== 3. БАЗОВЫЕ ПРИЗНАКИ ==========
print("\n2. Создаем признаки...")

for df in [train, test]:
    # Время
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['day_of_month'] = df['timestamp'].dt.day
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['is_night'] = ((df['hour'] >= 23) | (df['hour'] <= 5)).astype(int)
    
    # Сумма
    df['amount_log'] = np.log1p(df['amount'])
    df['amount_round'] = (df['amount'] % 10 == 0).astype(int)
    
    # Канал + присутствие карты
    df['online_no_card'] = ((df['channel'] == 'online') & (df['card_present'] == 0)).astype(int)
    df['mobile_no_card'] = ((df['channel'] == 'mobile') & (df['card_present'] == 0)).astype(int)

# Заполняем пропуски
for df in [train, test]:
    df['income_bracket'] = df['income_bracket'].fillna('unknown')

# ========== 4. АГРЕГАЦИИ (БЕЗ целевой переменной!) ==========
print("3. Агрегации клиентов...")

# Только статистики доступные в момент транзакции
cust_agg = train.groupby('customer_id').agg(
    cust_avg_amount=('amount', 'mean'),
    cust_max_amount=('amount', 'max'),
    cust_min_amount=('amount', 'min'),
    cust_count=('amount', 'count'),
    cust_foreign_pct=('is_foreign', 'mean'),
    cust_card_present_pct=('card_present', 'mean'),
    cust_last_time=('timestamp', 'max')
).reset_index()

# Дни с последней транзакции
cust_agg['cust_days_since_last'] = (
    (train['timestamp'].max() - cust_agg['cust_last_time']).dt.days
)
# Частота транзакций
cust_agg['cust_daily_freq'] = (
    cust_agg['cust_count'] / (cust_agg['cust_days_since_last'] + 1)
)
cust_agg.drop('cust_last_time', axis=1, inplace=True)

train = train.merge(cust_agg, on='customer_id', how='left')
test = test.merge(cust_agg, on='customer_id', how='left')

# Агрегации мерчантов
print("4. Агрегации мерчантов...")
merch_agg = train.groupby('merchant_id').agg(
    merch_avg_amount=('amount', 'mean'),
    merch_count=('amount', 'count'),
    merch_foreign_pct=('is_foreign', 'mean')
).reset_index()

train = train.merge(merch_agg, on='merchant_id', how='left')
test = test.merge(merch_agg, on='merchant_id', how='left')

# Агрегации категорий
print("5. Агрегации категорий...")
cat_agg = train.groupby('merchant_category').agg(
    cat_avg_amount=('amount', 'mean'),
    cat_online_pct=('channel', lambda x: ((x == 'online') | (x == 'mobile')).mean())
).reset_index()

train = train.merge(cat_agg, on='merchant_category', how='left')
test = test.merge(cat_agg, on='merchant_category', how='left')

# Заполняем NaN для новых сущностей в test
for col in test.columns:
    if test[col].dtype in ['float64', 'int64'] and col != 'is_fraud':
        test[col] = test[col].fillna(test[col].median() if test[col].notna().any() else 0)

# ========== 5. ДОПОЛНИТЕЛЬНЫЕ ПРИЗНАКИ ==========
print("6. Комбинированные признаки...")
for df in [train, test]:
    # Отклонения от средних
    df['amount_to_cust_avg'] = df['amount'] / (df['cust_avg_amount'] + 1)
    df['amount_to_merch_avg'] = df['amount'] / (df['merch_avg_amount'] + 1)
    
    # Новый клиент/мерчант
    df['is_new_customer'] = (df['cust_count'].isna() | (df['cust_count'] == 0)).astype(int)
    df['is_rare_merchant'] = (df['merch_count'] < 5).astype(int)
    
    # Сумма относительно возраста аккаунта
    df['amount_to_acc_age'] = df['amount'] / (df['account_age_days'] + 1)
    
    # Комбинации риска
    df['high_risk_channel'] = (
        (df['channel'].isin(['online', 'mobile'])) & 
        (df['card_present'] == 0)
    ).astype(int)
    
    df['unusual_amount'] = (
        (df['amount'] > df['cust_avg_amount'] * 3)
    ).astype(int)

# ========== 6. ФИНАЛЬНЫЙ НАБОР ПРИЗНАКОВ ==========
features = [
    # Базовые
    'amount', 'amount_log', 'amount_round',
    'merchant_category', 'channel', 'is_foreign', 'card_present',
    'region', 'income_bracket', 'age', 'account_age_days',
    # Временные
    'hour', 'day_of_week', 'day_of_month', 'is_weekend', 'is_night',
    # Клиент
    'cust_avg_amount', 'cust_max_amount', 'cust_min_amount',
    'cust_count', 'cust_foreign_pct', 'cust_card_present_pct',
    'cust_days_since_last', 'cust_daily_freq',
    # Мерчант
    'merch_avg_amount', 'merch_count', 'merch_foreign_pct',
    # Категория
    'cat_avg_amount', 'cat_online_pct',
    # Комбинации
    'amount_to_cust_avg', 'amount_to_merch_avg',
    'is_new_customer', 'is_rare_merchant',
    'amount_to_acc_age', 'high_risk_channel', 'unusual_amount',
    'online_no_card', 'mobile_no_card'
]

cat_features = ['merchant_category', 'channel', 'region', 'income_bracket']

print(f"\n7. Всего признаков: {len(features)}")

# ========== 7. ВАЛИДАЦИЯ ==========
print("\n8. Временная валидация...")
train = train.sort_values('timestamp')
tscv = TimeSeriesSplit(n_splits=5)

scores = []
for fold, (tr_idx, val_idx) in enumerate(tscv.split(train)):
    X_tr = train.iloc[tr_idx][features]
    y_tr = train.iloc[tr_idx]['is_fraud']
    X_val = train.iloc[val_idx][features]
    y_val = train.iloc[val_idx]['is_fraud']
    
    if y_val.sum() == 0:
        continue
    
    model = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.03,
        depth=6,
        cat_features=cat_features,
        verbose=False,
        random_seed=42,
        class_weights=[1, 15]  # балансировка классов
    )
    
    model.fit(X_tr, y_tr)
    pred = model.predict_proba(X_val)[:, 1]
    score = average_precision_score(y_val, pred)
    scores.append(score)
    print(f"   Fold {fold+1}: PR-AUC = {score:.4f}")

mean_cv = np.mean(scores)
print(f"\n   Средний PR-AUC: {mean_cv:.4f}")

# ========== 8. ФИНАЛЬНАЯ МОДЕЛЬ ==========
print("\n9. Обучение финальной модели...")
final_model = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.03,
    depth=6,
    cat_features=cat_features,
    verbose=100,
    random_seed=42,
    class_weights=[1, 15]
)

final_model.fit(train[features], train['is_fraud'])

# ========== 9. ПРЕДСКАЗАНИЯ ==========
print("\n10. Предсказания...")
pred = final_model.predict_proba(test[features])[:, 1]

print(f"   Средняя вероятность: {pred.mean():.4f}")
print(f"   Максимальная: {pred.max():.4f}")
print(f"   > 0.5: {(pred > 0.5).sum()} шт")
print(f"   > 0.1: {(pred > 0.1).sum()} шт")

# ========== 10. СОХРАНЕНИЕ ==========
submission = pd.DataFrame({
    'transaction_id': test['transaction_id'],
    'fraud_proba': pred
})

submission.to_csv('submission_v1.csv', index=False)

print("\n" + "="*50)
print("ГОТОВО!")
print(f"CV PR-AUC: {mean_cv:.4f}")
print(f"Файл: submission_v1.csv")
print("="*50)