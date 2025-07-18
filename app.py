import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.preprocessing import LabelEncoder
import plotly.express as px
import io

# Konfigurasi Halaman
st.set_page_config(
    page_title="Prediksi Stok Obat - Apotek Barokah Farma",
    page_icon="⚕️",
    layout="wide"
)

# Fungsi untuk styling CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Membuat file CSS sementara untuk styling
with open("style.css", "w") as f:
    f.write("""
    /* Main background */
    .stApp {
        background-color: #ffffff;
    }

    /* Sidebar style */
    [data-testid="stSidebar"] {
        background-color: #a8e6cf;
    }
    
    /* Tombol */
    .stButton>button {
        border-radius: 12px;
        box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
        transition: 0.3s;
        font-family: 'sans-serif';
    }
    .stButton>button:hover {
        box-shadow: 0 8px 16px 0 rgba(0,0,0,0.2);
    }
    
    /* Box container (div) */
    .css-1r6slb0 { /* Kelas CSS Streamlit untuk kontainer utama */
        border-radius: 15px;
    }

    /* Tabel, Grafik, etc. */
    .stDataFrame, .stPlotlyChart {
        border-radius: 15px;
        box-shadow: 0 4px 8px 0 rgba(0,0,0,0.1);
        padding: 15px;
    }
    
    /* Header di sidebar */
    [data-testid="stSidebar"] .css-1d391kg {
        color: #1a5d57;
    }
    """)

local_css("style.css")


# --- FUNGSI UTAMA ---

def main_app():
    # === HEADER / SIDEBAR ===
    with st.sidebar:
        st.markdown(
            """
            <div style="padding: 20px; background-color: #dcedf7; border-radius: 15px; text-align: center;">
                <h1 style="color: #0b5394;">⚕️ Apotek Barokah Farma</h1>
                <p style="font-style: italic;">Solusi Cerdas Manajemen Stok</p>
            </div>
            """, unsafe_allow_html=True
        )
        st.title("Menu Navigasi")
        
        # Opsi menu
        page = st.radio("Pilih Halaman:", ["Dashboard & Prediksi", "Visualisasi Pohon Keputusan"])

        st.info("Aplikasi ini menggunakan Algoritma C4.5 (Decision Tree) untuk memprediksi kebutuhan restok obat.")

    # === KONTEN UTAMA ===
    st.header(f"📊 {page}")

    # --- UPLOAD DATASET ---
    st.subheader("1. Unggah Dataset Stok Obat")
    uploaded_file = st.file_uploader(
        "Pilih file CSV atau Excel",
        type=['csv', 'xlsx'],
        help="Pastikan file memiliki kolom: Nama Item, Jenis, Satuan, Harga, Stok Awal, Terjual, Sisa Stok, Status Kebutuhan"
    )

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
        except Exception as e:
            st.error(f"Error saat membaca file: {e}")
            return

        # Tampilkan data yang diunggah
        st.write("**Data Awal Anda:**")
        st.dataframe(df.head())

        # Preprocessing Data
        # Mengubah fitur kategorikal menjadi numerik
        categorical_cols = ['Jenis', 'Satuan']
        df_encoded = df.copy()
        label_encoders = {}
        for col in categorical_cols:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col])
            label_encoders[col] = le

        # Mendefinisikan fitur (X) dan target (y)
        features = ['Jenis', 'Satuan', 'Harga', 'Stok Awal', 'Terjual', 'Sisa Stok']
        target = 'Status Kebutuhan'
        
        X = df_encoded[features]
        y = df_encoded[target]

        # Membangun model Decision Tree (C4.5 menggunakan entropy)
        dt_classifier = DecisionTreeClassifier(criterion='entropy', random_state=42)
        dt_classifier.fit(X, y)

        # --- Tampilkan halaman sesuai pilihan di sidebar ---
        if page == "Dashboard & Prediksi":
            display_dashboard_and_prediction(df, dt_classifier, features, label_encoders)
        elif page == "Visualisasi Pohon Keputusan":
            display_decision_tree(dt_classifier, features, y.unique())

    else:
        st.info("Silakan unggah dataset untuk memulai analisis.")


def display_dashboard_and_prediction(df, model, features, encoders):
    # --- DASHBOARD RINGKASAN ---
    st.subheader("📈 Dashboard Ringkasan")
    
    total_obat = df.shape[0]
    perlu_restok = df[df['Status Kebutuhan'] == 'Perlu Restok'].shape[0]
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Jumlah Total Obat", f"{total_obat} Item")
    with col2:
        st.metric("Obat Perlu Restok", f"{perlu_restok} Item")

    # Notifikasi jika > 5 item perlu restok
    if perlu_restok > 5:
        st.warning(f"⚠️ Perhatian! Terdapat {perlu_restok} item yang perlu segera di-restok.")

    # Grafik Batang
    st.write("**Grafik Jumlah Obat Berdasarkan Status Kebutuhan**")
    status_counts = df['Status Kebutuhan'].value_counts().reset_index()
    status_counts.columns = ['Status Kebutuhan', 'Jumlah']
    
    fig = px.bar(
        status_counts, 
        x='Status Kebutuhan', 
        y='Jumlah', 
        color='Status Kebutuhan',
        color_discrete_map={'Perlu Restok': '#ff6961', 'Tidak Perlu Restok': '#77dd77'},
        text='Jumlah'
    )
    fig.update_layout(
        xaxis_title="Status Kebutuhan",
        yaxis_title="Jumlah Item Obat",
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    
    # --- PREDIKSI OBAT BARU ---
    st.subheader("🔮 Prediksi Kebutuhan Obat Baru")
    
    with st.form("prediction_form"):
        st.write("Isi detail obat baru di bawah ini:")
        
        # Mengambil pilihan unik dari data asli untuk dropdown
        jenis_options = df['Jenis'].unique()
        satuan_options = df['Satuan'].unique()
        
        col_form1, col_form2 = st.columns(2)
        
        with col_form1:
            jenis_input = st.selectbox("Jenis Obat", options=jenis_options)
            harga_input = st.number_input("Harga", min_value=0)
            terjual_input = st.number_input("Jumlah Terjual", min_value=0)
            
        with col_form2:
            satuan_input = st.selectbox("Satuan", options=satuan_options)
            stok_awal_input = st.number_input("Stok Awal", min_value=0)
            sisa_stok_input = st.number_input("Sisa Stok", min_value=0)
            
        submitted = st.form_submit_button("Lakukan Prediksi")

        if submitted:
            # Mengubah input form menjadi format yang bisa diprediksi
            input_data = {
                'Jenis': encoders['Jenis'].transform([jenis_input])[0],
                'Satuan': encoders['Satuan'].transform([satuan_input])[0],
                'Harga': harga_input,
                'Stok Awal': stok_awal_input,
                'Terjual': terjual_input,
                'Sisa Stok': sisa_stok_input
            }
            input_df = pd.DataFrame([input_data])
            
            # Melakukan prediksi
            prediction = model.predict(input_df[features])
            prediction_proba = model.predict_proba(input_df[features])
            
            # Menampilkan hasil
            st.write("**Hasil Prediksi:**")
            if prediction[0] == "Perlu Restok":
                st.error(f"**Status: {prediction[0]}** (Confidence: {np.max(prediction_proba)*100:.2f}%)")
            else:
                st.success(f"**Status: {prediction[0]}** (Confidence: {np.max(prediction_proba)*100:.2f}%)")

    st.markdown("---")

    # --- EXPORT HASIL ---
    st.subheader("📥 Export Hasil Analisis")
    st.write("Unduh data asli beserta hasil prediksinya dalam format Excel.")

    # Membuat prediksi untuk seluruh dataset sebagai contoh
    df_prediksi = df.copy()
    df_encoded = df_prediksi.copy()
    for col, le in encoders.items():
        df_encoded[col] = le.transform(df_encoded[col])

    df_prediksi['Prediksi Status'] = model.predict(df_encoded[features])

    # Fungsi untuk konversi ke Excel dalam memory
    @st.cache_data
    def to_excel(df_to_export):
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df_to_export.to_excel(writer, index=False, sheet_name='Prediksi_Stok')
        processed_data = output.getvalue()
        return processed_data

    excel_data = to_excel(df_prediksi)
    
    st.download_button(
        label="📥 Unduh File Excel",
        data=excel_data,
        file_name="hasil_prediksi_stok_obat.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    
    # Bonus: Rekomendasi Stok Aman
    st.info("""
    **💡 Rekomendasi Stok Aman:**
    - **Stok Pengaman (Safety Stock):** Selalu siapkan stok tambahan untuk mengantisipasi lonjakan permintaan atau keterlambatan pengiriman.
    - **Analisis Pola Penjualan:** Perhatikan obat mana yang paling cepat habis pada periode tertentu (misal: musim hujan untuk obat flu).
    - **Metode FIFO:** Terapkan prinsip *First-In, First-Out* untuk menghindari obat kedaluwarsa.
    """)

def display_decision_tree(model, features, class_names):
    # --- VISUALISASI POHON KEPUTUSAN ---
    st.subheader("🌳 Visualisasi Pohon Keputusan (C4.5)")
    st.write("""
    Pohon keputusan ini menggambarkan bagaimana model mengambil keputusan. Setiap node (kotak) mewakili sebuah 'pertanyaan' terhadap salah satu fitur data. 
    Berdasarkan jawaban ('ya' atau 'tidak'), model akan bergerak ke cabang berikutnya hingga mencapai daun (leaf node) yang merupakan keputusan akhir (prediksi status).
    - **entropy**: Ukuran ketidakpastian atau 'keragaman' dalam sebuah node. Nilai 0 berarti semua data di node tersebut termasuk dalam satu kelas (murni).
    - **samples**: Jumlah data yang ada di node tersebut.
    - **value**: Distribusi jumlah data untuk setiap kelas ([Jumlah 'Tidak Perlu Restok'], [Jumlah 'Perlu Restok']).
    - **class**: Kelas mayoritas pada node tersebut.
    """)
    
    try:
        from sklearn.tree import plot_tree
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(25, 15))
        plot_tree(
            model,
            feature_names=features,
            class_names=sorted(list(class_names)),
            filled=True,
            rounded=True,
            fontsize=10,
            ax=ax
        )
        st.pyplot(fig)
        
        # Menawarkan download gambar pohon keputusan
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=300, bbox_inches='tight')
        st.download_button(
            label="📥 Unduh Gambar Pohon Keputusan",
            data=buf.getvalue(),
            file_name="pohon_keputusan.png",
            mime="image/png"
        )

    except ImportError:
        st.error("Gagal membuat visualisasi. Pastikan `matplotlib` terinstall.")
    except Exception as e:
        st.error(f"Terjadi kesalahan saat membuat visualisasi: {e}")

# --- LOGIKA LOGIN (Sederhana) ---
def login_page():
    st.markdown(
        """
        <div style="text-align: center;">
            <h1 style="color: #0b5394;">⚕️ Selamat Datang di Aplikasi Prediksi Stok</h1>
            <h3 style="color: #1a5d57;">Apotek Barokah Farma</h3>
        </div>
        """, unsafe_allow_html=True
    )
    st.write("") # spasi

    with st.form("login_form"):
        username = st.text_input("Username", placeholder="admin")
        password = st.text_input("Password", type="password", placeholder="admin")
        login_button = st.form_submit_button("Login")

        if login_button:
            if username == "admin" and password == "admin":
                st.session_state['logged_in'] = True
                st.rerun() # Mengulang script untuk masuk ke main_app
            else:
                st.error("Username atau Password salah!")

# --- MAIN CONTROLLER ---
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

if st.session_state['logged_in']:
    main_app()
else:
    login_page()