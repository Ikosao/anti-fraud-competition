# ensemble_v3_v6.py – ансамбль двух подходов
import pandas as pd
import numpy as np

print("="*50)
print("АНСАМБЛЬ V3 + V6")
print("="*50)

# Загружаем предсказания
v3 = pd.read_csv('submission_v3_calibrated.csv')
v6 = pd.read_csv('submission_v6_rules.csv')

print(f"V3: средняя={v3['fraud_proba'].mean():.4f}, макс={v3['fraud_proba'].max():.4f}")
print(f"V6: средняя={v6['fraud_proba'].mean():.4f}, макс={v6['fraud_proba'].max():.4f}")

# Проверяем что одинаковые ID
assert (v3['transaction_id'] == v6['transaction_id']).all(), "ID не совпадают!"

# Пробуем разные веса
weights = [
    (0.5, 0.5, "50-50"),
    (0.6, 0.4, "60-40 V3"),
    (0.4, 0.6, "40-60 V6"),
    (0.7, 0.3, "70-30 V3"),
    (0.3, 0.7, "30-70 V6"),
]

for w3, w6, name in weights:
    ensemble = w3 * v3['fraud_proba'] + w6 * v6['fraud_proba']
    
    submission = pd.DataFrame({
        'transaction_id': v3['transaction_id'],
        'fraud_proba': ensemble
    })
    
    filename = f'submission_ensemble_{name.replace(" ", "_").replace("-", "_")}.csv'
    submission.to_csv(filename, index=False)
    
    print(f"\n{name}:")
    print(f"  Средняя: {ensemble.mean():.4f}")
    print(f"  Макс: {ensemble.max():.4f}")
    print(f"  >0.5: {(ensemble>0.5).sum()} шт")
    print(f"  >0.1: {(ensemble>0.1).sum()} шт")
    print(f"  Сохранён: {filename}")

# Дополнительно: смешивание с учётом уверенности
print("\n" + "="*50)
print("Адаптивный ансамбль:")
print("="*50)

# Где V6 увереннее (правила сработали) – больше веса V6
v6_confidence = (
    (v6['fraud_proba'] > 0.1) | (v6['fraud_proba'] < 0.001)
).astype(float)

# Динамический вес
dynamic_weight = 0.4 + 0.3 * v6_confidence  # 0.4-0.7 для V6
ensemble_adaptive = (1 - dynamic_weight) * v3['fraud_proba'] + dynamic_weight * v6['fraud_proba']

submission_adaptive = pd.DataFrame({
    'transaction_id': v3['transaction_id'],
    'fraud_proba': ensemble_adaptive
})
submission_adaptive.to_csv('submission_ensemble_adaptive.csv', index=False)

print(f"  Средняя: {ensemble_adaptive.mean():.4f}")
print(f"  Макс: {ensemble_adaptive.max():.4f}")
print(f"  >0.5: {(ensemble_adaptive>0.5).sum()} шт")
print(f"  Сохранён: submission_ensemble_adaptive.csv")

print("\n Все ансамбли готовы!")
print("Рекомендую отправить на лидерборд:")
print("  1. submission_v3_calibrated.csv")
print("  2. submission_v6_rules.csv")
print("  3. submission_ensemble_adaptive.csv")