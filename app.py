import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, silhouette_score
from statsmodels.tsa.arima.model import ARIMA

# ==========================================
# 1. KONFIGURASI HALAMAN & CSS
# ==========================================
st.set_page_config(
    page_title="Telang Weather Command Center",
    page_icon="🌦️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-title {font-size: 2.2rem; color: #1E88E5; font-weight: bold;}
    .section-header {font-size: 1.2rem; color: #333; border-bottom: 2px solid #ddd; margin-top: 16px; margin-bottom: 10px;}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. LOAD DATA
# ==========================================
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("cuaca_telang_perjam.csv")
        # Pastikan kolom tanggal/waktu ada
        if 'tanggal' not in df.columns or 'waktu' not in df.columns:
            st.error("Dataset harus memiliki kolom 'tanggal' dan 'waktu'.")
            return None

        # Buat datetime index
        df["datetime"] = pd.to_datetime(df["tanggal"].astype(str) + " " + df["waktu"].astype(str), errors='coerce')
        df = df.set_index("datetime")
        # Buat kolom bantu
        # Pastikan kolom 'tanggal' sebagai datetime juga (untuk mengambil tahun/bulan tanpa waktu)
        df["tanggal_only"] = pd.to_datetime(df["tanggal"], errors='coerce')
        df["tahun"] = df["tanggal_only"].dt.year
        df["bulan"] = df["tanggal_only"].dt.month
        df["jam"] = df.index.hour
        return df
    except Exception as e:
        st.error(f"Load data error: {e}")
        return None

df = load_data()

# ==========================================
# FUNGSI WARNA MARKER LEAFLET
# ==========================================
def color_marker(val, var):
    if var == "suhu (°C)":
        if val < 24: return "blue"
        elif val < 29: return "green"
        else: return "red"
    elif var == "curah_hujan (mm)":
        if val <= 0.5: return "yellow"
        elif val < 5: return "lightblue"
        else: return "darkblue"
    elif var == "kelembapan (%)":
        return "orange" if val < 60 else "cyan"
    return "gray"

# ==========================================
# 3. SIDEBAR
# ==========================================
st.sidebar.title("🎛️ Pusat Kontrol")

if df is not None:
    # Target utama
    available_targets = [c for c in ["suhu (°C)", "curah_hujan (mm)", "kelembapan (%)"] if c in df.columns]
    if not available_targets:
        st.error("Kolom target (suhu / curah_hujan / kelembapan) tidak ditemukan di dataset.")
        st.stop()
    target_col = st.sidebar.selectbox("Pilih variabel utama", available_targets)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Parameter GMM")

    # Default features: pilih dari actual columns
    possible_feats = [c for c in df.columns if c not in ['tanggal', 'waktu', 'kode_cuaca', 'jam', 'tanggal_only', 'tahun', 'bulan']]
    default_feats = [f for f in ["suhu (°C)", "kelembapan (%)", "radiasi_matahari (MJ/m²)", "radiasi_matahari (MJ/m2)"] if f in possible_feats]
    if not default_feats:
        # fallback to first three numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        default_feats = numeric_cols[:3]

    gmm_feats = st.sidebar.multiselect("Fitur clustering", possible_feats, default=default_feats)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Filter Waktu (GMM & Peta)")

    # Tahun range based on data available but limited to 2015-2025
    years_available = sorted(df["tahun"].dropna().unique().astype(int))
    years_available = [y for y in years_available if 2015 <= y <= 2025]
    if not years_available:
        years_available = list(range(2015, 2026))
    tahun_default = max(years_available)
    tahun_pilihan = st.sidebar.slider("Pilih Tahun", min_value=min(years_available), max_value=max(years_available), value=tahun_default)

    bulan_options = ["Semua Bulan"] + [f"{m}" for m in range(1,13)]
    bulan_choice = st.sidebar.selectbox("Pilih Bulan", bulan_options, index=0)
    bulan_pilihan = 0 if bulan_choice == "Semua Bulan" else int(bulan_choice)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Parameter ARIMA")
    p = st.sidebar.number_input("p", 0, 5, 1)
    d = st.sidebar.number_input("d", 0, 2, 1)
    q = st.sidebar.number_input("q", 0, 5, 1)

    # ==========================================
    # 4. DASHBOARD UTAMA
    # ==========================================
    st.markdown('<div class="main-title">🌦️ Telang Weather Command Center</div>', unsafe_allow_html=True)
    st.write("Sistem terintegrasi: Prediksi, Clustering (per tahun/bulan), dan Peta interaktif.")

    # ======================================================
    # =============== BAGIAN 1 — PETA LEAFLET (PREDIKSI) ==============
    # ======================================================
    st.markdown('<div class="section-header">🌍 Peta Prediksi Cuaca (Leaflet)</div>', unsafe_allow_html=True)
    colA, colB = st.columns([1, 3])

    with colA:
        st.info("Prediksi cuaca ke depan menggunakan ARIMA atau Random Forest")

        method_option = st.selectbox("Metode Prediksi", ["ARIMA (Time Series)", "Random Forest (Regression)"])
        hours_ahead = st.slider("Prediksi (jam ke depan):", 1, 24, 1)

        train_series = df[target_col].dropna().iloc[-336:]
        if len(train_series) == 0:
            st.warning("Data target tidak cukup untuk prediksi.")
            forecast_val = np.nan
        else:
            last_value = train_series.iloc[-1]
            forecast_val = last_value
            try:
                if method_option == "ARIMA (Time Series)":
                    model = ARIMA(train_series, order=(p, d, q))
                    fit = model.fit()
                    pred = fit.forecast(hours_ahead)
                    forecast_val = pred.iloc[-1]
                else:
                    temp = pd.DataFrame({"target": train_series})
                    temp["hour"] = temp.index.hour
                    temp["ts"] = temp.index.astype(np.int64) // 10**9
                    X = temp[["hour", "ts"]]
                    y = temp["target"]
                    rf = RandomForestRegressor(n_estimators=60, max_depth=12, random_state=42)
                    rf.fit(X, y)
                    future = train_series.index[-1] + pd.Timedelta(hours=hours_ahead)
                    X_future = pd.DataFrame({"hour": [future.hour], "ts": [int(future.timestamp())]})
                    forecast_val = rf.predict(X_future)[0]
            except Exception as e:
                st.warning(f"Prediksi gagal: {e}. Menggunakan nilai terakhir.")
                forecast_val = last_value

        delta_text = ""
        try:
            delta_text = f"{forecast_val - last_value:.2f}"
        except:
            delta_text = ""
        st.metric(f"Prediksi {target_col}", f"{forecast_val:.2f}" if not pd.isna(forecast_val) else "N/A", delta=delta_text)

    # Map with central marker showing forecast
    with colB:
        lat_center, lon_center = (-7.1222358, 112.7211714)
        m_pred = folium.Map(location=[lat_center, lon_center], zoom_start=13)
        folium.CircleMarker(
            location=[lat_center, lon_center],
            radius=25,
            color=color_marker(forecast_val if not pd.isna(forecast_val) else 0, target_col),
            fill=True,
            fill_opacity=0.8,
            tooltip=f"{target_col}: {forecast_val:.2f} (pred +{hours_ahead}h)"
        ).add_to(m_pred)
        st_folium(m_pred, height=420, width=760)

    # ======================================================
    # =============== BAGIAN 2 — GMM per Tahun & Bulan =======================
    # ======================================================
    st.markdown('<div class="section-header">🧩 Klasifikasi Pola Cuaca (GMM) per Tahun & Bulan</div>', unsafe_allow_html=True)

    if not gmm_feats:
        st.warning("Pilih minimal 1 fitur untuk clustering di sidebar.")
    else:
        # Filter per tahun & bulan
        df_filtered = df.copy()
        df_filtered = df_filtered[df_filtered["tahun"] == int(tahun_pilihan)]
        if bulan_pilihan != 0:
            df_filtered = df_filtered[df_filtered["bulan"] == int(bulan_pilihan)]

        st.write(f"Menampilkan data untuk Tahun **{tahun_pilihan}** — Bulan **{'Semua' if bulan_pilihan==0 else bulan_pilihan}** (jumlah: {len(df_filtered)})")

        if len(df_filtered) < 10:
            st.warning("Data terlalu sedikit untuk clustering pada filter ini.")
        else:
            # select gmm features ensuring they exist
            gmm_feats_present = [c for c in gmm_feats if c in df_filtered.columns]
            if not gmm_feats_present:
                st.error("Fitur clustering tidak ditemukan di dataset.")
            else:
                X_full = df_filtered[gmm_feats_present].dropna()
                if X_full.empty:
                    st.warning("Data fitur terpilih mengandung banyak NA — tidak ada data setelah dropna.")
                else:
                    # sample for speed
                    sample = X_full.sample(n=4000, random_state=42) if len(X_full) > 4000 else X_full.copy()

                    scaler = StandardScaler()
                    Xs = scaler.fit_transform(sample)

                    # Fit GMM
                    try:
                        gmm = GaussianMixture(n_components=3, random_state=42)
                        labels = gmm.fit_predict(Xs)
                    except Exception as e:
                        st.error(f"GMM gagal: {e}")
                        labels = np.zeros(len(Xs), dtype=int)

                    df_gmm = sample.copy()
                    df_gmm["cluster"] = labels
                    # Silhouette (jika memungkinkan)
                    try:
                        sil = silhouette_score(Xs, labels)
                        st.metric("Silhouette Score", f"{sil:.4f}")
                    except:
                        pass

                    # Mapping cluster -> Dingin/Normal/Panas if 'suhu (°C)' in features
                    if "suhu (°C)" in gmm_feats_present:
                        # find index of suhu in features to sort by means
                        suhu_idx = gmm_feats_present.index("suhu (°C)")
                        means = gmm.means_[:, suhu_idx]
                        order = np.argsort(means)
                        mapping = { order[0]: "Dingin", order[1]: "Normal", order[2]: "Panas" }
                    else:
                        # fallback: use first feature means
                        means = gmm.means_[:, 0]
                        order = np.argsort(means)
                        mapping = { order[0]: "Low", order[1]: "Mid", order[2]: "High" }

                    df_gmm["kategori"] = df_gmm["cluster"].map(mapping)

                    # colors
                    color_map = {
                        "Panas": "red",
                        "Normal": "green",
                        "Dingin": "blue",
                        "High": "red",
                        "Mid": "green",
                        "Low": "blue"
                    }

                    # show cluster statistics
                    st.write("**Rata-rata per cluster (sample):**")
                    st.dataframe(df_gmm.groupby("kategori")[gmm_feats_present + ["cluster"]].mean().round(3))

                    # SCATTER PLOT: if at least 2 features available
                    st.subheader(f"Visualisasi Clustering Tahun {tahun_pilihan} — {'Semua Bulan' if bulan_pilihan==0 else 'Bulan '+str(bulan_pilihan)}")

                    if len(gmm_feats_present) >= 2:
                        x_feat, y_feat = gmm_feats_present[0], gmm_feats_present[1]
                        try:
                            # Ambil lebar layar browser
                            cw = st.get_option("browser.clientWidth")
                            if cw and cw > 0:
                                # versi kecil: min 3 inch, max 6 inch
                                fig_width = max(3, min(cw / 300, 6))
                            else:
                                fig_width = 4
                        except:
                            fig_width = 4
                        fig_height = fig_width * 0.65  # proporsi responsif
                        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
                        for k, grp in df_gmm.groupby("kategori"):
                            ax.scatter(
                                grp[x_feat],
                                grp[y_feat],
                                label=k,
                                s=8,                 # TITIK KECIL
                                alpha=0.7,
                                c=color_map.get(k, "gray")
                            )

                        ax.set_xlabel(x_feat)
                        ax.set_ylabel(y_feat)
                        ax.legend(title="Kategori")
                        ax.grid(alpha=0.3, linestyle="--")
                        st.pyplot(fig)
                    else:
                        st.info("Butuh minimal 2 fitur untuk scatterplot (pilih 2 fitur di sidebar).")

                    st.markdown("### Distribusi Kategori Cuaca")
                    # Urutan kategori agar warna konsisten
                    ordered_cats = ["Panas", "Normal", "Dingin", "High", "Mid", "Low"]

                    # Ambil hanya yang muncul
                    dist_total = df_gmm["kategori"].value_counts().reindex(ordered_cats).dropna()

                    # Mapping warna konsisten
                    color_map_plot = {
                        "Panas": "red",
                        "Normal": "green",
                        "Dingin": "blue",
                        "High": "red",
                        "Mid": "green",
                        "Low": "blue"
                    }
                    colors_used = [color_map_plot[k] for k in dist_total.index]

                    colA, colB = st.columns(2)

                    with colA:
                        st.write("**Distribusi (%)**")
                        st.dataframe(
                            (dist_total / dist_total.sum() * 100).round(2).rename("Persentase (%)")
                        )

                    with colB:
                        # PIE CHART dengan warna konsisten
                        fig_pie, ax_pie = plt.subplots(figsize=(4, 4))
                        ax_pie.pie(
                            dist_total.values,
                            labels=dist_total.index,
                            autopct="%1.1f%%",
                            colors=colors_used       # <<< PENTING: WARNA KATEGORI FIX
                        )
                        ax_pie.set_title("Persentase Kategori")
                        st.pyplot(fig_pie)

                    if bulan_pilihan == 0:
                        st.markdown("### Distribusi Per Bulan (Jan–Des)")

                        monthly_dist = (
                            df_gmm.assign(bulan=df_filtered["bulan"].loc[df_gmm.index])
                                .groupby(["bulan", "kategori"])
                                .size()
                                .unstack(fill_value=0)
                                .reindex(columns=dist_total.index)
                        )

                        st.dataframe(monthly_dist)

                        # BAR CHART dengan warna konsisten per kategori
                        fig_bar, ax_bar = plt.subplots(figsize=(10, 4))

                        for cat in monthly_dist.columns:
                            ax_bar.bar(
                                monthly_dist.index,
                                monthly_dist[cat],
                                label=cat,
                                color=color_map_plot[cat]    # <<< WARNA SAMA DENGAN SCATTER
                            )

                        ax_bar.set_xlabel("Bulan")
                        ax_bar.set_ylabel("Jumlah")
                        ax_bar.set_title("Distribusi Kategori per Bulan")
                        ax_bar.legend(title="Kategori")
                        st.pyplot(fig_bar)

                    # show counts
                    st.write("### Jumlah titik per kategori (sample):")
                    st.write(df_gmm["kategori"].value_counts())

    # ======================================================
    # =============== BAGIAN 3 — RF & ARIMA FULL ===========
    # ======================================================
    st.markdown('<div class="section-header">📈 Prediksi Lanjutan (RF & ARIMA)</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    # ------------------ RANDOM FOREST ---------------------
    with col1:
        st.subheader("🌲 Random Forest")

        if st.button("Training Model RF"):
            try:
                cols_remove = [c for c in ["curah_hujan (mm)", "suhu (°C)", "kelembapan (%)"] if c != target_col and c in df.columns]
                X = df.drop(columns=cols_remove + ["tanggal", "waktu", "kode_cuaca", "jam", "tanggal_only", "tahun", "bulan"], errors='ignore')
                y = df[target_col].astype(float)

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                num_cols = X.select_dtypes(include=np.number).columns

                pre = ColumnTransformer([("num", SimpleImputer(strategy="median"), num_cols)])

                model = Pipeline([("pre", pre), ("rf", RandomForestRegressor(n_estimators=70, random_state=42))])
                model.fit(X_train, y_train)
                pred = model.predict(X_test)

                r2 = r2_score(y_test, pred)
                rmse = np.sqrt(mean_squared_error(y_test, pred))

                st.metric("R2", f"{r2:.3f}")
                st.metric("RMSE", f"{rmse:.3f}")

                fig_rf, ax_rf = plt.subplots(figsize=(8,4))
                ax_rf.scatter(y_test, pred, alpha=0.4)
                ax_rf.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
                ax_rf.set_xlabel("Actual"); ax_rf.set_ylabel("Predicted")
                st.pyplot(fig_rf)
            except Exception as e:
                st.error(f"RF gagal: {e}")

    # ------------------ ARIMA ---------------------
    with col2:
        st.subheader("📈 ARIMA Forecast")
        try:
            series = df[target_col].resample("D").mean().fillna(method="ffill")
            n = int(len(series) * 0.9)
            train, test = series[:n], series[n:]
            model = ARIMA(train, order=(p, d, q)).fit()
            fc = model.forecast(len(test))
            future = model.forecast(len(test) + 14)

            fig, ax = plt.subplots(figsize=(10,5))
            ax.plot(test.index, test, label="Test", color="green")
            ax.plot(test.index, fc, label="Pred", color="red")
            ax.plot(pd.date_range(test.index[-1], periods=15)[1:], future[-14:], label="Future", color="orange")
            ax.legend()
            st.pyplot(fig)
        except Exception as e:
            st.error(f"ARIMA gagal: {e}")

    # ======================================================
    # =============== BAGIAN 4 — ANOMALI ===================
    # ======================================================
    st.markdown('<div class="section-header">🚨 Deteksi Anomali</div>', unsafe_allow_html=True)
    thresh = st.slider("Batas Z-Score", 1.5, 5.0, 3.0, 0.5)

    ser = df[target_col].dropna()
    mean_val, std_val = ser.mean(), ser.std()

    if std_val > 0:
        z = (ser - mean_val) / std_val
        anom = z.abs() > thresh
        st.metric("Jumlah Anomali", int(anom.sum()))
        fig, ax = plt.subplots(figsize=(12,3))
        ax.plot(ser.index, ser, color="gray")
        ax.scatter(ser.index[anom], ser[anom], color="red")
        st.pyplot(fig)
    else:
        st.warning("Variasi terlalu kecil untuk hitung Z-score")

else:
    st.warning("File dataset belum ditemukan.")
