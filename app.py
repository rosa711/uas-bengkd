import streamlit as st
import pandas as pd
import numpy as np
import itertools
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# 1. Pengumpulan Data
dir = 'hungarian.data'
with open(dir, encoding='Latin1') as file:
    lines = [line.strip() for line in file]

# 2. Menelaah Data
data = itertools.takewhile(
    lambda x: len(x) == 76,
    (' '.join(lines[i:(i + 10)]).split() for i in range(0, len(lines), 10))
)
df = pd.DataFrame.from_records(data)

df = df.iloc[:, :-1]
df = df.drop(df.columns[0], axis=1)

# 3. Validasi Data
df.replace(-9.0, np.nan, inplace=True)

# 4. Menentukan Objek Data (ambil 14 fitur)
kolom_ygdipilih = [2, 3, 8, 9, 11, 15, 18, 31, 37, 39, 40, 43, 50, 57]
objek_data = df[kolom_ygdipilih]
objek_data.columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
objek_data = objek_data.astype(float)
objek_data.head()

objek_data.drop(columns=['slope', 'ca', 'thal'], inplace=True)

# 5. Membersihkan dataset
duplicate = objek_data[objek_data.duplicated()]
nilai_duplikat = objek_data.drop_duplicates(inplace=True)
imputer = SimpleImputer(strategy='mean')
objek_data = pd.DataFrame(imputer.fit_transform(objek_data), columns=objek_data.columns)

X = objek_data.drop('target', axis=1)
y = objek_data['target']
oversample = SMOTE()
Xos, yos = oversample.fit_resample(X, y)

# Split data menjadi training dan testing sets
X_train, X_test, y_train, y_test = train_test_split(Xos, yos, test_size=0.2, random_state=42)
knn = KNeighborsClassifier(n_neighbors=5)

# standarkan data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# train data
knn.fit(X_train_scaled, y_train)

# Prediksi pada testing set
y_predict_knn = knn.predict(X_test_scaled)
akurasi_knn = accuracy_score(y_test, y_predict_knn)
akurasi_knn = akurasi_knn * 100

# Inisialisasi Streamlit
st.header('Prediksi Penyakit Jantung Menggunakan KNN')
st.subheader('Masukkan Data Pasien:')

age = st.number_input('Usia', min_value=0, max_value=150, value=50)
sex = st.selectbox('Jenis Kelamin', ['Wanita', 'Pria'])
cp = st.selectbox('Jenis Nyeri Dada', ['Tipe 0', 'Tipe 1', 'Tipe 2', 'Tipe 3'])
trestbps = st.number_input('Tekanan Darah (mm Hg)', min_value=0, value=120)
chol = st.number_input('Kolesterol Serum (mg/dl)', min_value=0, value=200)
fbs = st.selectbox('Gula Darah Puasa > 120 mg/dl', ['Tidak', 'Ya'])
restecg = st.selectbox('Hasil EKG Istirahat', ['Normal', 'ST-T Abnormalitas', 'Hypertrofi Ventrikel Kiri'])
thalach = st.number_input('Detak Jantung Maksimum', min_value=0, value=150)
exang = st.selectbox('Angina yang Diinduksi Olahraga', ['Tidak', 'Ya'])
oldpeak = st.number_input('Depresi ST yang Diinduksi Olahraga', min_value=0.0, value=0.0)

# Fungsi untuk mengubah input pengguna menjadi dataframe
def user_input_to_dataframe():
    sex_mapping = {'Wanita': 0, 'Pria': 1}
    cp_mapping = {'Tipe 0': 0, 'Tipe 1': 1, 'Tipe 2': 2, 'Tipe 3': 3}
    fbs_mapping = {'Tidak': 0, 'Ya': 1}
    restecg_mapping = {'Normal': 0, 'ST-T Abnormalitas': 1, 'Hypertrofi Ventrikel Kiri': 2}
    exang_mapping = {'Tidak': 0, 'Ya': 1}

    user_data = {
        'age': age, 'sex': sex_mapping[sex], 'cp': cp_mapping[cp], 'trestbps': trestbps,
        'chol': chol, 'fbs': fbs_mapping[fbs], 'restecg': restecg_mapping[restecg], 'thalach': thalach,
        'exang': exang_mapping[exang], 'oldpeak': oldpeak
    }

    input_df = pd.DataFrame([user_data])
    return input_df

# Fungsi untuk menampilkan prediksi
def predict():
    input_data = user_input_to_dataframe()
    input_data_scaled = scaler.transform(input_data)
    prediction = knn.predict(input_data_scaled)
    return prediction[0]

# Tombol untuk prediksi
if st.button('Prediksi'):
    prediction = predict()
    st.success(f'Hasil Prediksi: {"Positif" if prediction == 1 else "Negatif"}')

# Tampilkan akurasi model (opsional)
st.sidebar.subheader('Akurasi Model KNN')
akurasi_knn = accuracy_score(y_test, knn.predict(X_test_scaled))
st.sidebar.text(f'Akurasi: {akurasi_knn:.2f}')

# Tampilkan dataset (opsional)
if st.checkbox('Tampilkan Dataset'):
    st.write(objek_data)
