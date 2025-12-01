import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score

# 1. ФИНАНСОВЫЕ ДАННЫЕ
print("ФИНАНСОВЫЕ ДАННЫЕ")
finance_data = pd.DataFrame({
    'цена': [102, 105, 98, 107, 103, 110, 95, 108, 100, 112],
    'объем': [1200, 1500, 800, 1600, 1400, 1800, 700, 1700, 1100, 1900],
    'волатильность': [2.1, 3.2, 5.1, 2.8, 2.5, 4.1, 6.2, 3.1, 2.3, 4.5]
})

# Задача регрессии: предсказать цену
X_fin = finance_data[['объем', 'волатильность']]
y_fin_reg = finance_data['цена']

# Задача классификации: определить тренд (рост/падение)
finance_data['тренд'] = (finance_data['цена'] > finance_data['цена'].shift(1)).fillna(0).astype(int)
y_fin_clf = finance_data['тренд']

# Разделение данных
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_fin, y_fin_reg, test_size=0.3, random_state=42)
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(X_fin, y_fin_clf, test_size=0.3, random_state=42)

# Обучение моделей
reg_fin = RandomForestRegressor(n_estimators=50, random_state=42)
reg_fin.fit(X_train_reg, y_train_reg)

clf_fin = RandomForestClassifier(n_estimators=50, random_state=42)
clf_fin.fit(X_train_clf, y_train_clf)

# Оценка моделей
y_pred_reg = reg_fin.predict(X_test_reg)
y_pred_clf = clf_fin.predict(X_test_clf)

print(f"Регрессия: {mean_squared_error(y_test_reg, y_pred_reg):.2f}")
print(f"Классификация: {accuracy_score(y_test_clf, y_pred_clf):.2f}")





# 2. МЕДИЦИНСКИЕ ДАННЫЕ
print("\n2. МЕДИЦИНСКИЕ ДАННЫЕ")
medical_data = pd.DataFrame({
    'возраст': [25, 47, 62, 33, 58, 41, 29, 53, 35, 67],
    'давление': [118, 142, 158, 124, 148, 135, 122, 146, 128, 162],
    'холестерин': [182, 228, 281, 195, 245, 212, 188, 238, 202, 268]
})

# Задача регрессии: предсказать уровень холестерина
X_med = medical_data[['возраст', 'давление']]
y_med_reg = medical_data['холестерин']

# Задача классификации: определить риск (высокий/низкий)
medical_data['риск'] = (medical_data['холестерин'] > 220).astype(int)
y_med_clf = medical_data['риск']

# Разделение данных
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_med, y_med_reg, test_size=0.3, random_state=42)
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(X_med, y_med_clf, test_size=0.3, random_state=42)

# Обучение моделей
reg_med = RandomForestRegressor(n_estimators=50, random_state=42)
reg_med.fit(X_train_reg, y_train_reg)

clf_med = RandomForestClassifier(n_estimators=50, random_state=42)
clf_med.fit(X_train_clf, y_train_clf)

# Оценка моделей
y_pred_reg = reg_med.predict(X_test_reg)
y_pred_clf = clf_med.predict(X_test_clf)

print(f"Регрессия: {mean_squared_error(y_test_reg, y_pred_reg):.2f}")
print(f"Классификация: {accuracy_score(y_test_clf, y_pred_clf):.2f}")






# 3. ДАННЫЕ НЕДВИЖИМОСТИ
print("\n3. ДАННЫЕ НЕДВИЖИМОСТИ")
real_estate_data = pd.DataFrame({
    'площадь': [48, 75, 105, 58, 92, 68, 115, 52, 88, 125],
    'комнаты': [1, 2, 3, 2, 3, 2, 4, 1, 3, 4],
    'этаж': [3, 7, 12, 2, 9, 5, 15, 1, 8, 18],
    'цена': [52, 78, 145, 62, 95, 72, 168, 48, 92, 185]
})

# Задача регрессии: предсказать цену
X_est = real_estate_data[['площадь', 'комнаты', 'этаж']]
y_est_reg = real_estate_data['цена']

# Задача классификации: определить категорию (элитная/стандартная)
real_estate_data['категория'] = (real_estate_data['цена'] > 100000).astype(int)
y_est_clf = real_estate_data['категория']

# Разделение данных
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_est, y_est_reg, test_size=0.3, random_state=42)
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(X_est, y_est_clf, test_size=0.3, random_state=42)

# Обучение моделей
reg_est = RandomForestRegressor(n_estimators=50, random_state=42)
reg_est.fit(X_train_reg, y_train_reg)

clf_est = RandomForestClassifier(n_estimators=50, random_state=42)
clf_est.fit(X_train_clf, y_train_clf)

# Оценка моделей
y_pred_reg = reg_est.predict(X_test_reg)
y_pred_clf = clf_est.predict(X_test_clf)

print(f"Регрессия: {mean_squared_error(y_test_reg, y_pred_reg):.2f}")
print(f"Классификация: {accuracy_score(y_test_clf, y_pred_clf):.2f}")
