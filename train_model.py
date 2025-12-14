# complete_model_training.py
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
import joblib

# 1. Veriyi yükleme
df = pd.read_csv('hackathon_train_set.csv', sep=';', encoding='utf-8-sig')

print(f"Orijinal veri boyutu: {df.shape}")
print(f"İlk 5 satır:\n{df.head()}")

# 2. Price sütununu temizleme
def clean_price(price):
    if isinstance(price, str):
        # TL, nokta ve virgülleri temizle
        price = str(price).replace('TL', '').replace(' ', '').strip()
        # Binlik ayırıcı nokta ve virgülleri kaldır
        price = price.replace('.', '').replace(',', '')
        try:
            return float(price)
        except:
            return np.nan
    elif pd.isna(price):
        return np.nan
    else:
        try:
            return float(price)
        except:
            return np.nan

df['Price'] = df['Price'].apply(clean_price)

# 3. Eksik fiyatları kaldır
df = df.dropna(subset=['Price'])
print(f"Fiyat temizleme sonrası satır sayısı: {len(df)}")

# 4. Feature seçimi - ÖNEMLİ KISIM
# Sadece en önemli feature'ları seç
important_features = [
    'District', 'Neighborhood', 'm² (Gross)', 'm² (Net)', 
    'Number of rooms', 'Building Age', 'Floor location',
    'Number of floors', 'Heating', 'Number of bathrooms',
    'Balcony', 'Furnished', 'Using status', 'Available for Loan',
    'Front South', 'Elevator', 'Parking Lot', 'Air conditioning',
    'Furniture', 'Swimming Pool (Open)', 'Sea', 'Nature',
    'Metro', 'Bus stop', 'Hospital', 'Market', 'Gym'
]

# DataFrame'de var olan feature'ları filtrele
selected_features = [col for col in important_features if col in df.columns]
print(f"Seçilen feature sayısı: {len(selected_features)}")
print(f"Seçilen feature'lar: {selected_features}")

# 5. X ve y'yi hazırla
X = df[selected_features].copy()
y = df['Price'].copy()

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

# 6. Kategorik değişkenleri encode et
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
print(f"Kategorik sütunlar: {categorical_cols}")

label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    # NaN değerleri 'Unknown' ile doldur
    X[col] = X[col].fillna('Unknown').astype(str)
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le
    print(f"{col} için {len(le.classes_)} unique değer")

# 7. Sayısal değişkenlerdeki NaN'leri doldur
numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
for col in numeric_cols:
    X[col] = X[col].fillna(X[col].median())

# 8. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

print(f"Train shape: {X_train.shape}")
print(f"Test shape: {X_test.shape}")

# 9. Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 10. Model eğitimi
print("\nModel eğitiliyor...")
model = RandomForestRegressor(
    n_estimators=200,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train_scaled, y_train)

# 11. Model değerlendirme
from sklearn.metrics import mean_absolute_error, r2_score

y_pred_train = model.predict(X_train_scaled)
y_pred_test = model.predict(X_test_scaled)

mae_train = mean_absolute_error(y_train, y_pred_train)
mae_test = mean_absolute_error(y_test, y_pred_test)
r2_train = r2_score(y_train, y_pred_train)
r2_test = r2_score(y_test, y_pred_test)

print("\n" + "="*50)
print("MODEL PERFORMANSI:")
print("="*50)
print(f"Train MAE: {mae_train:,.0f} TL")
print(f"Test MAE: {mae_test:,.0f} TL")
print(f"Train R² Score: {r2_train:.4f}")
print(f"Test R² Score: {r2_test:.4f}")
print("="*50)

# 12. Feature importance
feature_importance = pd.DataFrame({
    'feature': selected_features,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nEN ÖNEMLİ 10 FEATURE:")
print(feature_importance.head(10))

# 13. Modeli ve preprocessing araçlarını kaydet
joblib.dump(model, 'real_estate_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(label_encoders, 'label_encoders.pkl')

# Feature listesini de kaydet
with open('selected_features.txt', 'w') as f:
    for feature in selected_features:
        f.write(f"{feature}\n")

print("\nModel başarıyla kaydedildi!")
print(f"Model dosyası: real_estate_model.pkl")
print(f"Scaler dosyası: scaler.pkl")
print(f"Label encoders dosyası: label_encoders.pkl")# complete_model_training.py
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
import joblib

# 1. Veriyi yükleme
df = pd.read_csv('hackathon_train_set.csv', sep=';', encoding='utf-8-sig')

print(f"Orijinal veri boyutu: {df.shape}")
print(f"İlk 5 satır:\n{df.head()}")

# 2. Price sütununu temizleme
def clean_price(price):
    if isinstance(price, str):
        # TL, nokta ve virgülleri temizle
        price = str(price).replace('TL', '').replace(' ', '').strip()
        # Binlik ayırıcı nokta ve virgülleri kaldır
        price = price.replace('.', '').replace(',', '')
        try:
            return float(price)
        except:
            return np.nan
    elif pd.isna(price):
        return np.nan
    else:
        try:
            return float(price)
        except:
            return np.nan

df['Price'] = df['Price'].apply(clean_price)

# 3. Eksik fiyatları kaldır
df = df.dropna(subset=['Price'])
print(f"Fiyat temizleme sonrası satır sayısı: {len(df)}")

# 4. Feature seçimi - ÖNEMLİ KISIM
# Sadece en önemli feature'ları seç
important_features = [
    'District', 'Neighborhood', 'm² (Gross)', 'm² (Net)', 
    'Number of rooms', 'Building Age', 'Floor location',
    'Number of floors', 'Heating', 'Number of bathrooms',
    'Balcony', 'Furnished', 'Using status', 'Available for Loan',
    'Front South', 'Elevator', 'Parking Lot', 'Air conditioning',
    'Furniture', 'Swimming Pool (Open)', 'Sea', 'Nature',
    'Metro', 'Bus stop', 'Hospital', 'Market', 'Gym'
]

# DataFrame'de var olan feature'ları filtrele
selected_features = [col for col in important_features if col in df.columns]
print(f"Seçilen feature sayısı: {len(selected_features)}")
print(f"Seçilen feature'lar: {selected_features}")

# 5. X ve y'yi hazırla
X = df[selected_features].copy()
y = df['Price'].copy()

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

# 6. Kategorik değişkenleri encode et
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
print(f"Kategorik sütunlar: {categorical_cols}")

label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    # NaN değerleri 'Unknown' ile doldur
    X[col] = X[col].fillna('Unknown').astype(str)
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le
    print(f"{col} için {len(le.classes_)} unique değer")

# 7. Sayısal değişkenlerdeki NaN'leri doldur
numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
for col in numeric_cols:
    X[col] = X[col].fillna(X[col].median())

# 8. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

print(f"Train shape: {X_train.shape}")
print(f"Test shape: {X_test.shape}")

# 9. Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 10. Model eğitimi
print("\nModel eğitiliyor...")
model = RandomForestRegressor(
    n_estimators=200,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train_scaled, y_train)

# 11. Model değerlendirme
from sklearn.metrics import mean_absolute_error, r2_score

y_pred_train = model.predict(X_train_scaled)
y_pred_test = model.predict(X_test_scaled)

mae_train = mean_absolute_error(y_train, y_pred_train)
mae_test = mean_absolute_error(y_test, y_pred_test)
r2_train = r2_score(y_train, y_pred_train)
r2_test = r2_score(y_test, y_pred_test)

print("\n" + "="*50)
print("MODEL PERFORMANSI:")
print("="*50)
print(f"Train MAE: {mae_train:,.0f} TL")
print(f"Test MAE: {mae_test:,.0f} TL")
print(f"Train R² Score: {r2_train:.4f}")
print(f"Test R² Score: {r2_test:.4f}")
print("="*50)

# 12. Feature importance
feature_importance = pd.DataFrame({
    'feature': selected_features,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nEN ÖNEMLİ 10 FEATURE:")
print(feature_importance.head(10))

# 13. Modeli ve preprocessing araçlarını kaydet
joblib.dump(model, 'real_estate_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(label_encoders, 'label_encoders.pkl')

# Feature listesini de kaydet
with open('selected_features.txt', 'w') as f:
    for feature in selected_features:
        f.write(f"{feature}\n")

print("\nModel başarıyla kaydedildi!")
print(f"Model dosyası: real_estate_model.pkl")
print(f"Scaler dosyası: scaler.pkl")
print(f"Label encoders dosyası: label_encoders.pkl")