import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# =====================================
# KONFIGURASI APLIKASI
# =====================================

st.set_page_config(
    page_title="SIPKEM Pematangsiantar",
    page_icon="🌿",
    layout="wide"
)

# =====================================
# TAMPILAN APLIKASI
# =====================================

st.markdown("""
<style>
.stApp {
    background-color: #f3f8f2;
}

[data-testid="stSidebar"] {
    background-color: #dcebdc;
}

h1, h2, h3 {
    color: #285c3b;
}

div[data-testid="stMetric"] {
    background-color: white;
    padding: 15px;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# =====================================
# MEMBACA DATASET
# =====================================

@st.cache_data
def load_data():
    return pd.read_csv(
        "dataset_kemiskinan_siantar_sintetis.csv"
    )

try:
    df = load_data()
except FileNotFoundError:
    st.error("Dataset belum ditemukan!")
    st.info("Jalankan kode_dataset.py terlebih dahulu.")
    st.stop()

# =====================================
# MENENTUKAN FITUR
# =====================================

kolom_kategori = [
    'Pendidikan_Tertinggi',
    'Status_Ketenagakerjaan',
    'Status_Kepemilikan_Rumah',
    'Jenis_Lantai',
    'Jenis_Dinding',
    'Sumber_Air_Minum',
    'Fasilitas_Jamban',
    'Tangki_Septik_IPAL'
]

kolom_angka = [
    'Luas_Lantai_perkapita_m2',
    'Pengeluaran_perkapita_Rp',
    'Penerima_BPNT',
    'Penerima_PKH',
    'Penerima_Jaminan_Sosial',
    'Penggunaan_Jaminan_Kesehatan'
]

fitur = kolom_kategori + kolom_angka

X = df[fitur]
y = df['Status_Kemiskinan_Ekstrem']

# =====================================
# PREPROCESSING DATA
# =====================================

preprocessing = ColumnTransformer(
    transformers=[
        (
            'kategori',
            OneHotEncoder(handle_unknown='ignore'),
            kolom_kategori
        ),
        (
            'angka',
            'passthrough',
            kolom_angka
        )
    ]
)

# =====================================
# MEMBAGI DATA TRAINING DAN TESTING
# =====================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =====================================
# MEMBANGUN MODEL RANDOM FOREST
# =====================================

model = Pipeline([
    ('preprocessing', preprocessing),

    ('random_forest', RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    ))
])

# =====================================
# TRAINING MODEL
# =====================================

@st.cache_resource
def training_model(_model, X_train, y_train):
    _model.fit(X_train, y_train)
    return _model

model = training_model(model, X_train, y_train)

# =====================================
# EVALUASI MODEL
# =====================================

y_pred = model.predict(X_test)

akurasi = accuracy_score(y_test, y_pred)
presisi = precision_score(
    y_test, y_pred, zero_division=0
)
recall = recall_score(
    y_test, y_pred, zero_division=0
)
f1 = f1_score(
    y_test, y_pred, zero_division=0
)

# =====================================
# SIDEBAR MENU
# =====================================

st.sidebar.title("🌿 SIPKEM")
st.sidebar.write("Pematangsiantar")

menu = st.sidebar.radio(
    "Menu Utama",
    [
        "Beranda",
        "Data Kemiskinan",
        "Prediksi",
        "Evaluasi Model"
    ]
)

st.sidebar.markdown("---")
st.sidebar.caption("Metode: Random Forest")
st.sidebar.caption("Tahun: 2026")

# =====================================
# HALAMAN BERANDA
# =====================================

if menu == "Beranda":

    st.title("SIPKEM PEMATANGSIANTAR")

    st.subheader(
        "Sistem Informasi Klasifikasi "
        "Tingkat Kerentanan Kemiskinan"
    )

    st.write(
        "Aplikasi ini digunakan untuk melakukan "
        "klasifikasi tingkat kerentanan kemiskinan "
        "menggunakan algoritma Random Forest."
    )

    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total Data",
        len(df)
    )

    col2.metric(
        "Data Miskin",
        int((y == 1).sum())
    )

    col3.metric(
        "Data Tidak Miskin",
        int((y == 0).sum())
    )

    st.markdown("---")

    st.subheader("Distribusi Status Kemiskinan")

    distribusi = df[
        'Status_Kemiskinan_Ekstrem'
    ].value_counts()

    st.bar_chart(distribusi)

    st.info(
        "Dataset bawaan merupakan data sintetis "
        "untuk simulasi penelitian, bukan data "
        "mikro resmi BPS."
    )

# =====================================
# HALAMAN DATA KEMISKINAN
# =====================================

elif menu == "Data Kemiskinan":

    st.title("Data Kemiskinan")

    st.write(
        "Berikut merupakan dataset yang digunakan "
        "dalam proses klasifikasi."
    )

    st.metric(
        "Jumlah Data Rumah Tangga",
        len(df)
    )

    st.dataframe(
        df,
        use_container_width=True
    )

    st.subheader("Informasi Dataset")

    st.write("Jumlah Baris:", df.shape[0])
    st.write("Jumlah Kolom:", df.shape[1])

    st.subheader("Distribusi Kelas")

    st.bar_chart(
        df['Status_Kemiskinan_Ekstrem'].value_counts()
    )

# =====================================
# HALAMAN PREDIKSI
# =====================================

elif menu == "Prediksi":

    st.title("Prediksi Kerentanan Kemiskinan")

    st.write(
        "Masukkan karakteristik rumah tangga "
        "untuk melakukan proses prediksi."
    )

    with st.form("form_prediksi"):

        col1, col2 = st.columns(2)

        with col1:

            pendidikan = st.selectbox(
                "Pendidikan Tertinggi",
                [
                    "Tidak_Ijazah",
                    "SD",
                    "SMP",
                    "SMA_Ke_Atas"
                ]
            )

            pekerjaan = st.selectbox(
                "Status Ketenagakerjaan",
                ["Bekerja", "Pengangguran"]
            )

            rumah = st.selectbox(
                "Kepemilikan Rumah",
                [
                    "Milik_Sendiri",
                    "Bukan_Milik_Sendiri"
                ]
            )

            lantai = st.selectbox(
                "Jenis Lantai",
                ["Bukan_Tanah", "Tanah"]
            )

            dinding = st.selectbox(
                "Jenis Dinding",
                ["Tembok", "Bukan_Tembok"]
            )

            air = st.selectbox(
                "Sumber Air Minum",
                ["Leding", "Non_Leding"]
            )

            jamban = st.selectbox(
                "Fasilitas Jamban",
                ["Sendiri", "Bukan_Sendiri"]
            )

        with col2:

            septik = st.selectbox(
                "Tangki Septik",
                ["Ada", "Tidak_Ada"]
            )

            luas = st.number_input(
                "Luas Lantai per Kapita (m2)",
                min_value=0.0,
                value=29.32
            )

            pengeluaran = st.number_input(
                "Pengeluaran per Kapita (Rp)",
                min_value=0,
                value=1500000
            )

            bpnt = st.selectbox(
                "Penerima BPNT",
                ["Tidak", "Ya"]
            )

            pkh = st.selectbox(
                "Penerima PKH",
                ["Tidak", "Ya"]
            )

            sosial = st.selectbox(
                "Penerima Jaminan Sosial",
                ["Tidak", "Ya"]
            )

            kesehatan = st.selectbox(
                "Penggunaan Jaminan Kesehatan",
                ["Tidak", "Ya"]
            )

        tombol = st.form_submit_button(
            "Proses Prediksi"
        )

    if tombol:

        data_baru = pd.DataFrame([{

            'Pendidikan_Tertinggi': pendidikan,
            'Status_Ketenagakerjaan': pekerjaan,
            'Status_Kepemilikan_Rumah': rumah,
            'Jenis_Lantai': lantai,
            'Jenis_Dinding': dinding,
            'Sumber_Air_Minum': air,
            'Fasilitas_Jamban': jamban,
            'Tangki_Septik_IPAL': septik,
            'Luas_Lantai_perkapita_m2': luas,
            'Pengeluaran_perkapita_Rp': pengeluaran,
            'Penerima_BPNT': 1 if bpnt == "Ya" else 0,
            'Penerima_PKH': 1 if pkh == "Ya" else 0,
            'Penerima_Jaminan_Sosial': 1 if sosial == "Ya" else 0,
            'Penggunaan_Jaminan_Kesehatan': 1 if kesehatan == "Ya" else 0

        }])

        hasil = model.predict(data_baru)[0]

        probabilitas = model.predict_proba(
            data_baru
        )[0]

        kelas = list(model.classes_).index(1)

        st.markdown("---")

        st.subheader("Hasil Prediksi")

        if hasil == 1:

            st.error(
                "Hasil: RENTAN / MISKIN (Kelas 1)"
            )

        else:

            st.success(
                "Hasil: TIDAK MISKIN (Kelas 0)"
            )

        st.metric(
            "Probabilitas Kelas Miskin",
            f"{probabilitas[kelas] * 100:.2f}%"
        )

        st.caption(
            "Hasil ini merupakan prediksi model, "
            "bukan penetapan resmi status kemiskinan."
        )

# =====================================
# HALAMAN EVALUASI MODEL
# =====================================

elif menu == "Evaluasi Model":

    st.title("Evaluasi Model Random Forest")

    st.write(
        "Hasil evaluasi model berdasarkan data testing."
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Akurasi",
        f"{akurasi * 100:.2f}%"
    )

    col2.metric(
        "Presisi",
        f"{presisi * 100:.2f}%"
    )

    col3.metric(
        "Recall",
        f"{recall * 100:.2f}%"
    )

    col4.metric(
        "F1-Score",
        f"{f1 * 100:.2f}%"
    )

    st.markdown("---")

    st.subheader("Confusion Matrix")

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=[0, 1]
    )

    fig, ax = plt.subplots(figsize=(6, 4))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Greens",
        xticklabels=[
            "Tidak Miskin",
            "Miskin"
        ],
        yticklabels=[
            "Tidak Miskin",
            "Miskin"
        ],
        ax=ax
    )

    ax.set_xlabel("Prediksi")
    ax.set_ylabel("Data Aktual")

    st.pyplot(fig)
    plt.close(fig)

    st.markdown("---")

    st.subheader("Classification Report")

    laporan = classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=[
            "Tidak Miskin",
            "Miskin"
        ],
        zero_division=0,
        output_dict=True
    )

    st.dataframe(
        pd.DataFrame(laporan).transpose(),
        use_container_width=True
    )

    st.warning(
        "Evaluasi ini menggunakan dataset sintetis. "
        "Hasilnya belum mewakili akurasi pada data "
        "kemiskinan nyata."
    )

# =====================================
# FOOTER
# =====================================

st.sidebar.markdown("---")
st.sidebar.caption(
    "© 2026 SIPKEM Pematangsiantar"
)