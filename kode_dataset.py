import pandas as pd
import numpy as np

np.random.seed(42)

#Jumlah Data
N = 2000

#1. Data Kategorikal
pendidikan = np.random.choice(
    ['Tidak_ijazah', ' SD', 'SMP', 'SMA_Ke_Atas'],
    N, 
    p=[0.0278, 0.1007, 0.1773, 0.6942]
)

status_kerja = np.random.choice(
    ['Bekerja', 'Pengangguran',],
    N, 
    p=[0.9138, 0.0862]
)


kepemilikan = np.random.choice(
    ['Milik_Sendiri', 'Bukan_Milik_Sendiri'],
    N,
    p=[0.4790, 0.5210]
)

lantai = np.random.choice(
    ['Bukan_Tanah', 'Tanah'],
    N,
    p=[0.9836, 0.0164]
)

dinding = np.random.choice(
    ['Tembok', 'Bukan_Tembok'],
    N,
    p=[0.8259, 0.1741]
)

air = np.random.choice(
    ['Leding', 'Non_Leding'],
    N,
    p=[0.8296, 0.1704]
)

jamban = np.random.choice(
    ['Sendiri', 'Bukan_Sendiri'],
    N,
    p=[0.9656, 0.0344]
)

septik = np.random.choice(
    ['Ada', 'Tidak_Ada'],
    N,
    p=[0.9290, 0.0710]
)

# 2. Data bantuan sosial
bpnt = np.random.choice([1, 0], N, p=[0.2837, 0.7163])
pkh = np.random.choice([1, 0], N, p=[0.1476, 0.8524])
jaminan_sosial = np.random.choice([1, 0], N, p=[0.1534, 0.8466])
jaminan_kesehatan = np.random.choice([1, 0], N, p=[0.6725, 0.3275])

# 3. Data numerik
luas_lantai = np.random.normal(
    loc=29.32, scale=8.0, size=N
).clip(min=5)

pengeluaran = np.random.lognormal(
    mean=np.log(1200000), sigma=0.6, size=N
)

pengeluaran = (
    pengeluaran / pengeluaran.mean()
) * 1527154

# 4. Menentukan tingkat risiko
risk_score = np.zeros(N)

risk_score += np.where(
    pendidikan == 'Tidak_Ijazah', 4,
    np.where(pendidikan == 'SD', 2,
             np.where(pendidikan == 'SMP', 1, 0))
)

risk_score += np.where(status_kerja == 'Pengangguran', 3, 0)
risk_score += np.where(lantai == 'Tanah', 5, 0)
risk_score += np.where(dinding == 'Bukan_Tembok', 2, 0)

risk_score += np.where(
    kepemilikan == 'Bukan_Milik_Sendiri', 1, 0
)

risk_score += np.where(
    pengeluaran < np.percentile(pengeluaran, 15), 3, 0
)

# 5. Membuat target kemiskinan
jumlah_miskin = int(N * 0.0724)

indeks = np.argsort(risk_score)[::-1]

target = np.zeros(N, dtype=int)
target[indeks[:jumlah_miskin]] = 1

# 6. Mengacak data
acak = np.random.permutation(N)

df = pd.DataFrame({
    'Pendidikan_Tertinggi': pendidikan[acak],
    'Status_Ketenagakerjaan': status_kerja[acak],
    'Status_Kepemilikan_Rumah': kepemilikan[acak],
    'Jenis_Lantai': lantai[acak],
    'Jenis_Dinding': dinding[acak],
    'Sumber_Air_Minum': air[acak],
    'Fasilitas_Jamban': jamban[acak],
    'Tangki_Septik_IPAL': septik[acak],
    'Luas_Lantai_perkapita_m2': np.round(luas_lantai[acak], 2),
    'Pengeluaran_perkapita_Rp': np.round(pengeluaran[acak], 0),
    'Penerima_BPNT': bpnt[acak],
    'Penerima_PKH': pkh[acak],
    'Penerima_Jaminan_Sosial': jaminan_sosial[acak],
    'Penggunaan_Jaminan_Kesehatan': jaminan_kesehatan[acak],
    'Status_Kemiskinan_Ekstrem': target[acak]
})

# 7. Menyimpan dataset
df.to_csv(
    'dataset_kemiskinan_siantar_sintetis.csv',
    index=False
)

print("Dataset berhasil dibuat!")
print("Jumlah data:", len(df))
print(df.head())

