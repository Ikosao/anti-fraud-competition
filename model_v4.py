# model_v4_fixed.py – исправленный adversarial + улучшенные признаки
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit, train_test_split
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("МОДЕЛЬ АНТИФРОДА v4.1 – ИСПРАВЛЕННАЯ")
print("="*60)

# ========== 1. Загрузка ==========
print("\n1. Загрузка...")
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
train['timestamp'] = pd.to_datetime(train['timestamp'])
test['timestamp']  = pd.to_datetime(test['timestamp'])
print(f"Train: {train.shape[0]}  |  Test: {test.shape[0]}  |  Fraud: {train['is_fraud'].mean():.4f}")

# ========== 2. Признаки ==========
def create_features(df):
    df = df.copy()
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['day_of_month'] = df['timestamp'].dt.day
    df['month'] = df['timestamp'].dt.month
    df['is_weekend'] = df['day_of_week'].isin([5,6]).astype(int)
    df['is_night'] = ((df['hour']>=23)|(df['hour']<=5)).astype(int)
    df['is_morning'] = df['hour'].between(6,10).astype(int)

    df['amount_log'] = np.log1p(df['amount'])
    df['amount_sqrt'] = np.sqrt(df['amount'])
    df['amount_round10'] = (df['amount']%10==0).astype(int)
    df['amount_round100'] = (df['amount']%100==0).astype(int)

    df['high_risk_channel'] = ((df['channel'].isin(['online','mobile'])) & (df['card_present']==0)).astype(int)
    df['foreign_online'] = ((df['is_foreign']==1) & (df['channel'].isin(['online','mobile']))).astype(int)
    df['high_amount'] = (df['amount']>300).astype(int)
    df['very_high_amount'] = (df['amount']>1000).astype(int)

    # device_id
    df['device_id'] = df['device_id'].fillna('missing').astype(str)
    df['device_enc'] = pd.factorize(df['device_id'])[0]

    # категориальные
    df['income_bracket'] = df['income_bracket'].fillna('unknown')
    df['merchant_category'] = df['merchant_category'].astype(str)
    df['channel'] = df['channel'].astype(str)
    df['region'] = df['region'].astype(str)
    return df

print("2. Создание признаков...")
train = create_features(train)
test = create_features(test)

# ========== 3. Агрегаты (без fraud_rate!) ==========
def build_aggregates(base, target_df):
    # клиенты
    cust = base.groupby('customer_id').agg(
        cust_avg_amount=('amount','mean'),
        cust_std_amount=('amount','std'),
        cust_max_amount=('amount','max'),
        cust_min_amount=('amount','min'),
        cust_count=('amount','count'),
        cust_foreign_pct=('is_foreign','mean'),
        cust_highrisk_pct=('high_risk_channel','mean'),
        cust_night_pct=('is_night','mean'),
        cust_device_unique=('device_enc','nunique'),
        last_time=('timestamp','max')
    ).reset_index()
    max_t = base['timestamp'].max()
    cust['cust_days_since_last'] = (max_t - cust['last_time']).dt.days
    cust.drop('last_time', axis=1, inplace=True)
    cust['cust_std_amount'] = cust['cust_std_amount'].fillna(0)

    # мерчанты
    merch = base.groupby('merchant_id').agg(
        merch_avg_amount=('amount','mean'),
        merch_std_amount=('amount','std'),
        merch_max_amount=('amount','max'),
        merch_count=('amount','count'),
        merch_foreign_pct=('is_foreign','mean'),
        merch_highrisk_pct=('high_risk_channel','mean')
    ).reset_index()
    merch['merch_std_amount'] = merch['merch_std_amount'].fillna(0)

    # категории
    cat = base.groupby('merchant_category').agg(
        cat_avg_amount=('amount','mean'),
        cat_count=('amount','count'),
        cat_highrisk_pct=('high_risk_channel','mean'),
        cat_foreign_pct=('is_foreign','mean')
    ).reset_index()

    # регион
    region = base.groupby('region').agg(
        region_avg_amount=('amount','mean'),
        region_highrisk_pct=('high_risk_channel','mean')
    ).reset_index()

    # объединение
    out = target_df.merge(cust, on='customer_id', how='left')
    out = out.merge(merch, on='merchant_id', how='left')
    out = out.merge(cat, on='merchant_category', how='left')
    out = out.merge(region, on='region', how='left')
    for c in out.columns:
        if out[c].dtype in ['float64','int64'] and c!='is_fraud':
            out[c] = out[c].fillna(0)
    return out

def add_combination_features(df):
    df['amount_to_cust_avg'] = df['amount']/(df['cust_avg_amount']+1)
    df['amount_to_merch_avg'] = df['amount']/(df['merch_avg_amount']+1)
    df['amount_to_cat_avg'] = df['amount']/(df['cat_avg_amount']+1)
    df['amount_zscore_cust'] = (df['amount']-df['cust_avg_amount'])/(df['cust_std_amount']+1)
    df['amount_zscore_merch'] = (df['amount']-df['merch_avg_amount'])/(df['merch_std_amount']+1)
    df['is_new_customer'] = ((df['cust_count']==0)|df['cust_count'].isna()).astype(int)
    df['is_rare_merchant'] = (df['merch_count']<5).astype(int)
    df['unusual_amount'] = (df['amount']>df['cust_avg_amount']*3).astype(int)
    df['unusual_time'] = ((df['is_night']==1)&(df['cust_night_pct']<0.1)).astype(int)
    return df

# ========== 4. Честная валидация ==========
print("\n3. Честная временная валидация...")
train = train.sort_values('timestamp')
tscv = TimeSeriesSplit(n_splits=5)
base_features = [
    'amount','amount_log','amount_sqrt','amount_round10','amount_round100',
    'is_foreign','card_present','age','account_age_days',
    'hour','day_of_week','day_of_month','month','is_weekend','is_night','is_morning',
    'high_risk_channel','foreign_online','high_amount','very_high_amount','device_enc'
]
agg_features = [
    'cust_avg_amount','cust_std_amount','cust_max_amount','cust_min_amount',
    'cust_count','cust_foreign_pct','cust_highrisk_pct','cust_night_pct',
    'cust_device_unique','cust_days_since_last',
    'merch_avg_amount','merch_std_amount','merch_max_amount','merch_count',
    'merch_foreign_pct','merch_highrisk_pct',
    'cat_avg_amount','cat_count','cat_highrisk_pct','cat_foreign_pct',
    'region_avg_amount','region_highrisk_pct'
]
combo_features = [
    'amount_to_cust_avg','amount_to_merch_avg','amount_to_cat_avg',
    'amount_zscore_cust','amount_zscore_merch',
    'is_new_customer','is_rare_merchant','unusual_amount','unusual_time'
]
cat_features = ['merchant_category','channel','region','income_bracket']
all_features = base_features + agg_features + combo_features
print(f"Всего признаков: {len(all_features)} + {len(cat_features)} кат.")

scores = []
for fold, (tr_idx, val_idx) in enumerate(tscv.split(train)):
    train_fold = train.iloc[tr_idx].copy()
    val_fold = train.iloc[val_idx].copy()
    if val_fold['is_fraud'].sum()==0: continue
    train_fold = build_aggregates(train_fold, train_fold)
    val_fold = build_aggregates(train_fold, val_fold)
    train_fold = add_combination_features(train_fold)
    val_fold = add_combination_features(val_fold)

    X_tr = train_fold[all_features + cat_features]
    y_tr = train_fold['is_fraud']
    X_va = val_fold[all_features + cat_features]
    y_va = val_fold['is_fraud']

    model = CatBoostClassifier(
        iterations=2000, learning_rate=0.02, depth=8,
        cat_features=cat_features, verbose=False,
        random_seed=42, class_weights=[1,25]
    )
    model.fit(X_tr, y_tr)
    pred = model.predict_proba(X_va)[:,1]
    scores.append(average_precision_score(y_va, pred))
    print(f"   Fold {fold+1}: PR-AUC = {scores[-1]:.4f}")

print(f"\n   Средний честный PR-AUC: {np.mean(scores):.4f}")

# ========== 5. Adversarial validation (исправлено) ==========
print("\n4. Adversarial validation...")
# Объединяем train и test, метим is_test, перемешиваем
train_adv = train.copy()
train_adv['is_test'] = 0
test_adv = test.copy()
test_adv['is_test'] = 1
combined = pd.concat([train_adv, test_adv], ignore_index=True)
combined = create_features(combined)
combined = build_aggregates(train, combined)  # база – train
combined = add_combination_features(combined)

X_adv = combined[all_features + cat_features]
y_adv = combined['is_test']

# Разделяем перемешанные данные, чтобы в train были оба класса
X_tr_adv, X_val_adv, y_tr_adv, y_val_adv = train_test_split(
    X_adv, y_adv, test_size=0.3, random_state=42, stratify=y_adv
)
adv_model = CatBoostClassifier(
    iterations=500, learning_rate=0.1, depth=5,
    cat_features=cat_features, verbose=False, random_seed=42
)
adv_model.fit(X_tr_adv, y_tr_adv)
pred_adv = adv_model.predict_proba(X_val_adv)[:,1]
auc_adv = roc_auc_score(y_val_adv, pred_adv)
print(f"   ROC-AUC отличия train от test: {auc_adv:.4f}")
if auc_adv > 0.7:
    print("    Сильный сдвиг! Топ-10 важных признаков:")
    imp = pd.DataFrame({'feat':all_features+cat_features, 'imp':adv_model.feature_importances_})
    print(imp.sort_values('imp',ascending=False).head(10))
else:
    print("   ✓ Сдвиг небольшой, перенос хороший.")

# ========== 6. Финальная модель ==========
print("\n5. Финальное обучение...")
train_full = build_aggregates(train, train)
test_full = build_aggregates(train, test)
train_full = add_combination_features(train_full)
test_full = add_combination_features(test_full)

final_model = CatBoostClassifier(
    iterations=2000, learning_rate=0.02, depth=8,
    cat_features=cat_features, verbose=200,
    random_seed=42, class_weights=[1,25]
)
final_model.fit(train_full[all_features + cat_features], train_full['is_fraud'])
test_pred = final_model.predict_proba(test_full[all_features + cat_features])[:,1]

print(f"\n   Средняя вероятность fraud: {test_pred.mean():.4f}")
print(f"   Макс: {test_pred.max():.4f}")
print(f"   >0.5: {(test_pred>0.5).sum()} шт")

sub = pd.DataFrame({'transaction_id':test['transaction_id'], 'fraud_proba':test_pred})
sub.to_csv('submission_v4.csv', index=False)
print("\n submission_v4.csv сохранён.")
print(f"Честный CV PR-AUC: {np.mean(scores):.4f}")