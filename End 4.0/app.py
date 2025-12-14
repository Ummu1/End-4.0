from flask import Flask, render_template, request
import pandas as pd
import pickle
import os
import re

app = Flask(__name__)

# =================================================
# 1️⃣ MODELİ YÜKLE
# =================================================
MODEL_PATH = "model.pkl"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("❌ model.pkl bulunamadı")

with open(MODEL_PATH, "rb") as f:
    loaded = pickle.load(f)

if isinstance(loaded, tuple):
    model, model_features = loaded
else:
    model = loaded
    model_features = model.feature_names_in_

print("✅ Model yüklendi")

# =================================================
# 2️⃣ CSV → İLÇE / MAHALLE
# =================================================
DATA_PATH = "data.csv"

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError("❌ data.csv bulunamadı")

df_loc = pd.read_csv(DATA_PATH)
df_loc = df_loc[["District", "Neighborhood"]].dropna()

district_map = (
    df_loc.groupby("District")["Neighborhood"]
    .unique()
    .apply(list)
    .to_dict()
)

district_list = sorted(district_map.keys())

print("✅ İlçe–mahalle verisi yüklendi")

df_full = pd.read_csv("hackathon_train_set.csv", sep=";")

# ✅ Isıtma türlerini manuel ekliyoruz
heating_list = [
    "Doğalgaz",
    "Kombi",
    "Merkezi Sistem",
    "Soba",
    "Klima",
    "Jeotermal",
    "Yerden Isıtma",
    "Isıtma Yok"
]

# =================================================
# 3️⃣ ROUTE
# =================================================
@app.route("/", methods=["GET", "POST"])
def index():
    fair_value = None
    advice = None

    if request.method == "POST":
        try:
            ilan_fiyati = float(request.form["price"])
            district = request.form["district"]
            neighborhood = request.form["neighborhood"]
            gross = float(request.form["gross"])
            net = float(request.form["net"])
            rooms = request.form["rooms"]
            age = int(request.form["age"])
            bath = int(request.form["bath"])
            heating = request.form["heating"]

            # ===============================
            # Backend doğrulamalar
            # ===============================
            if ilan_fiyati <= 0:
                advice = "❌ İlan fiyatı 0 veya negatif olamaz!"
                return render_template("index.html",
                    district_list=district_list,
                    district_map=district_map,
                    heating_list=heating_list,
                    fair_value=None,
                    advice=advice
                )

            if gross <= 0 or net <= 0:
                advice = "❌ Brüt ve net m² pozitif olmalıdır!"
                return render_template("index.html",
                    district_list=district_list,
                    district_map=district_map,
                    heating_list=heating_list,
                    fair_value=None,
                    advice=advice
                )

            if age < 0:
                advice = "❌ Bina yaşı negatif olamaz!"
                return render_template("index.html",
                    district_list=district_list,
                    district_map=district_map,
                    heating_list=heating_list,
                    fair_value=None,
                    advice=advice
                )

            if bath < 0:
                advice = "❌ Banyo sayısı negatif olamaz!"
                return render_template("index.html",
                    district_list=district_list,
                    district_map=district_map,
                    heating_list=heating_list,
                    fair_value=None,
                    advice=advice
                )

            if not re.match(r"^[0-9]+\+[0-9]+$", rooms):
                advice = "❌ Oda sayısı '2+1' formatında olmalıdır!"
                return render_template("index.html",
                    district_list=district_list,
                    district_map=district_map,
                    heating_list=heating_list,
                    fair_value=None,
                    advice=advice
                )

            if heating not in heating_list:
                advice = "❌ Geçersiz ısıtma türü!"
                return render_template("index.html",
                    district_list=district_list,
                    district_map=district_map,
                    heating_list=heating_list,
                    fair_value=None,
                    advice=advice
                )

            # ===============================
            # MODEL GİRİŞİ
            # ===============================
            input_data = {
                "District": district,
                "Neighborhood": neighborhood,
                "m² (Gross)": gross,
                "m² (Net)": net,
                "Number of rooms": rooms,
                "Building Age": age,
                "Number of bathrooms": bath,
                "Heating": heating
            }

            df_input = pd.DataFrame([input_data])
            df_input = pd.get_dummies(df_input)

            for col in model_features:
                if col not in df_input.columns:
                    df_input[col] = 0

            df_input = df_input[model_features]

            fair_value = float(model.predict(df_input)[0])

            fark = (fair_value - ilan_fiyati) / fair_value

            if fark > 0.15:
                advice = "🟢 FIRSAT – Piyasa değerinin altında"
            elif fark < -0.15:
                advice = "🔴 PAHALI – Piyasa değerinin üstünde"
            else:
                advice = "🟡 NORMAL – Piyasa fiyatında"

        except Exception as e:
            advice = f"❌ Hata: {e}"

    return render_template(
        "index.html",
        district_list=district_list,
        district_map=district_map,
        heating_list=heating_list,
        fair_value=fair_value,
        advice=advice
    )


# =================================================
# 4️⃣ ÇALIŞTIR
# =================================================
if __name__ == "__main__":
    # host=0.0.0.0 → ağdaki diğer cihazlar da erişebilir
    # port=80 → HTTP için varsayılan port
    app.run(debug=False, host="127.0.0.1", port=80)