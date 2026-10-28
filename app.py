import json
from pathlib import Path

import altair as alt
import joblib
import pandas as pd
import streamlit as st

BASE = Path(__file__).parent
DATA = BASE / "data" / "mpg.csv"
MODEL_PATH = BASE / "models" / "rf_mpg.joblib"
METRICS_PATH = BASE / "models" / "metrics.json"

st.set_page_config(page_title="Données automobiles", page_icon="🚗", layout="wide")

@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA).rename(columns={"model_year": "year"})
    if df["year"].max() < 100:
        df["year"] += 1900
    df["weight_kg"] = (df["weight"] * 0.45359237).round(2)          # lb -> kg
    df["consumption_l_100km"] = (235.215 / df["mpg"]).round(2)      # mpg -> L/100 km
    return df


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None


df = load_data()
COLS = ["name", "origin", "year", "cylinders", "mpg", "consumption_l_100km",
        "horsepower", "weight_kg", "acceleration"]

st.sidebar.header("Filtres")
origin = st.sidebar.selectbox("Origine", ["Toutes"] + sorted(df["origin"].unique()))
cyl_all = sorted(df["cylinders"].unique())
cylinders = st.sidebar.multiselect("Nombre de cylindres", cyl_all, default=cyl_all)
y_min, y_max = int(df["year"].min()), int(df["year"].max())
years = st.sidebar.slider("Année du modèle", y_min, y_max, (y_min, y_max))
mpg_floor = float(df["mpg"].min())
mpg_min = st.sidebar.slider("Efficacité minimale (mpg)", mpg_floor, float(df["mpg"].max()), mpg_floor)

mask = (
    df["cylinders"].isin(cylinders)
    & df["year"].between(*years)
    & (df["mpg"].isna() | (df["mpg"] >= mpg_min))
)
if origin != "Toutes":
    mask &= df["origin"] == origin
fdf = df[mask][COLS].reset_index(drop=True)

st.title("🚗 Exploration des données automobiles")
st.caption("Dataset historique Auto MPG · copie CSV publique du dépôt Seaborn")

tab1, tab2 = st.tabs(["📊 Analyse descriptive", "🤖 Prédiction (Random Forest)"])

with tab1:
    st.info("mpg élevé = meilleure efficacité. L/100 km faible = consommation moindre. "
            "Analyse descriptive, sans modèle IA ni prédiction.")

    st.header("1 — Indicateurs et données filtrées")
    st.write(f"{len(fdf)} observations sur {len(df)}")
    if fdf.empty:
        st.warning("Aucune donnée ne correspond aux filtres.")
        st.stop()

    c1, c2, c3 = st.columns(3)
    c1.metric("Efficacité moyenne", f"{fdf['mpg'].mean():.1f} mpg")
    c2.metric("Consommation moyenne", f"{fdf['consumption_l_100km'].mean():.1f} L/100 km")
    c3.metric("Puissance moyenne", f"{fdf['horsepower'].mean():.1f} hp")
    st.caption("Les moyennes ignorent les valeurs manquantes. La consommation moyenne est la "
               "moyenne des conversions individuelles, pas la conversion du mpg moyen.")
    st.dataframe(fdf, use_container_width=True)

    st.header("2 — Choisir un véhicule")
    labels = {i: f"{r['name']} — {r['year']} (ligne {i})" for i, r in fdf.iterrows()}
    choice = st.selectbox("Véhicule", list(labels), format_func=labels.get)
    st.dataframe(fdf.loc[[choice]])

    st.header("3 — Comparaisons graphiques")
    color = alt.Color("origin:N", title="Origine")

    scatter = (alt.Chart(fdf.dropna(subset=["mpg", "weight_kg"]))
               .mark_circle(size=40)
               .encode(x=alt.X("weight_kg:Q", title="Poids (kg)", scale=alt.Scale(zero=False)),
                       y=alt.Y("mpg:Q", title="Efficacité (mpg)"),
                       color=color,
                       tooltip=["name", "origin", "year", "weight_kg", "mpg", "consumption_l_100km"])
               .properties(title="Poids et efficacité énergétique", height=350))
    st.altair_chart(scatter, use_container_width=True)

    by_year = fdf.groupby("year", as_index=False)["mpg"].mean()
    line = (alt.Chart(by_year).mark_line(point=True)
            .encode(x=alt.X("year:O", title="Année du modèle"),
                    y=alt.Y("mpg:Q", title="Efficacité moyenne (mpg)", scale=alt.Scale(zero=False)))
            .properties(title="Efficacité moyenne par année de modèle", height=300))
    st.altair_chart(line, use_container_width=True)

    left, right = st.columns(2)
    count = fdf.groupby("origin", as_index=False).size().rename(columns={"size": "Nombre"})
    left.altair_chart(
        alt.Chart(count).mark_bar()
        .encode(x=alt.X("origin:N", title="Origine"), y="Nombre:Q", color=color)
        .properties(title="Nombre d'observations par origine", height=300),
        use_container_width=True)

    comp = fdf.groupby("origin", as_index=False).agg(
        mpg=("mpg", "mean"), conso=("consumption_l_100km", "mean"), hp=("horsepower", "mean"))
    right.altair_chart(
        alt.Chart(comp).mark_bar()
        .encode(x=alt.X("origin:N", title="Origine"),
                y=alt.Y("conso:Q", title="Consommation moyenne (L/100 km)"), color=color)
        .properties(title="Consommation moyenne par origine", height=300),
        use_container_width=True)
    st.subheader("Moyennes par origine")
    st.dataframe(comp.round(2).rename(columns={"mpg": "mpg moyen", "conso": "L/100 km moyen",
                                               "hp": "puissance moyenne (hp)"}))

with tab2:
    st.header("Estimer la consommation d'un véhicule")
    model = load_model()
    if model is None:
        st.error("Modèle introuvable. Lance d'abord : `python train_model.py`")
        st.stop()

    with st.form("prediction"):
        a, b, c = st.columns(3)
        cyl = a.selectbox("Cylindres", cyl_all, index=cyl_all.index(4))
        disp = b.number_input("Cylindrée (cu. in)", 60.0, 500.0, 150.0, step=5.0)
        hp = c.number_input("Puissance (hp)", 40.0, 250.0, 100.0, step=5.0)
        d, e, f = st.columns(3)
        weight = d.number_input("Poids (kg)", 600.0, 2600.0, 1200.0, step=10.0)
        acc = e.number_input("Accélération (0-60 mph, s)", 7.0, 26.0, 15.0, step=0.5)
        year = f.slider("Année du modèle", y_min, y_max + 5, 1978)
        org = st.selectbox("Origine", sorted(df["origin"].unique()))
        go = st.form_submit_button("Prédire")

    if go:
        x = pd.DataFrame([{"cylinders": cyl, "displacement": disp, "horsepower": hp,
                           "weight_kg": weight, "acceleration": acc, "year": year, "origin": org}])
        mpg_pred = float(model.predict(x)[0])
        r1, r2 = st.columns(2)
        r1.metric("Efficacité estimée", f"{mpg_pred:.1f} mpg")
        r2.metric("Consommation estimée", f"{235.215 / mpg_pred:.1f} L/100 km")

    if METRICS_PATH.exists():
        m = json.loads(METRICS_PATH.read_text())
        st.caption(f"Performance sur le jeu de test ({m['n_test']} véhicules) : "
                   f"MAE = {m['MAE']:.2f} mpg · RMSE = {m['RMSE']:.2f} mpg · R² = {m['R2']:.3f}")