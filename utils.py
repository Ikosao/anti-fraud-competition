# utils.py
import pandas as pd
import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')

def add_basic_features(df):
    """Добавляем простые признаки"""
    df = df.copy()
    
    # Временные признаки
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 6)).astype(int)
    
    # Признаки суммы
    df['amount_log'] = np.log1p(df['amount'])
    df['amount_round'] = (df['amount'] % 10 == 0).astype(int)
    
    # Пропуски
    df['income_bracket'] = df['income_bracket'].fillna('unknown')
    df['device_id'] = df['device_id'].fillna('no_device')
    
    return df

def add_customer_stats(train, test):
    """Статистики по клиентам"""
    # Считаем статистики только на train
    stats = train.groupby('customer_id').agg({
        'amount': ['mean', 'max', 'count'],
        'is_fraud': 'mean',
        'timestamp': 'max'
    }).reset_index()
    
    stats.columns = ['customer_id', 'avg_amount', 'max_amount', 
                     'trans_count', 'fraud_rate', 'last_time']
    
    # Дни с последней транзакции
    stats['days_since_last'] = (
        (train['timestamp'].max() - stats['last_time']).dt.days
    )
    stats.drop('last_time', axis=1, inplace=True)
    
    # Добавляем к данным
    train = train.merge(stats, on='customer_id', how='left')
    test = test.merge(stats, on='customer_id', how='left')
    
    # Заполняем пропуски для новых клиентов
    for col in ['avg_amount', 'max_amount', 'trans_count', 'fraud_rate', 'days_since_last']:
        test[col].fillna(0, inplace=True)
    
    return train, test

def validate_time_series(train, model, features, n_splits=5):
    """Проверка модели по времени"""
    train = train.sort_values('timestamp')
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    scores = []
    for fold, (train_idx, val_idx) in enumerate(tscv.split(train)):
        X_train = train.iloc[train_idx][features]
        y_train = train.iloc[train_idx]['is_fraud']
        X_val = train.iloc[val_idx][features]
        y_val = train.iloc[val_idx]['is_fraud']
        
        model.fit(X_train, y_train)
        pred = model.predict_proba(X_val)[:, 1]
        score = average_precision_score(y_val, pred)
        scores.append(score)
        print(f"  Fold {fold+1}: PR-AUC = {score:.4f}")
    
    print(f"  Average: {np.mean(scores):.4f} (+/- {np.std(scores)*2:.4f})")
    return np.mean(scores)