import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import folium
from streamlit_folium import st_folium
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import base64
from io import BytesIO

# Konfigurasi halaman
st.set_page_config(layout="wide")
st.title("\U0001F4CA Dashboard Analisis E-Commerce")

# CSS untuk gambar
st.markdown("""
    <style>
    .centered-img {
        display: block;
        margin-left: auto;
        margin-right: auto;
        width: 500px;
        height: 500px;
    }
            
    
    </style>
""", unsafe_allow_html=True)

def tampilkan_gambar(fig):
    buf = BytesIO()
    fig.set_size_inches(4, 4)
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    st.markdown(f'<img src="data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}" class="centered-img"/>', unsafe_allow_html=True)

# Load Data
@st.cache_data
def muat_data():
    pesanan = pd.read_csv("./Ecom_datasets/orders_dataset.csv", parse_dates=["order_purchase_timestamp"])
    ulasan = pd.read_csv("./Ecom_datasets/order_reviews_dataset.csv")
    items = pd.read_csv("./Ecom_datasets/order_items_dataset.csv")
    pembayaran = pd.read_csv("./Ecom_datasets/order_payments_dataset.csv")
    produk = pd.read_csv("./Ecom_datasets/products_dataset.csv")
    geolokasi = pd.read_csv("./Ecom_datasets/geolocation_dataset.csv")
    return pesanan, ulasan, items, pembayaran, produk, geolokasi

pesanan, ulasan, items, pembayaran, produk, geolokasi = muat_data()

# --- Visualisasi Tren Pesanan dan Ulasan ---
pesanan['bulan_pembelian'] = pesanan['order_purchase_timestamp'].dt.to_period('M').astype(str)
jumlah_pesanan_per_bulan = pesanan.groupby('bulan_pembelian').size()
gabung_pesanan_ulasan = pd.merge(
    pesanan[['order_id', 'bulan_pembelian']],
    ulasan[['order_id', 'review_score']],
    on='order_id', how='inner')
rata2_skor_per_bulan = gabung_pesanan_ulasan.groupby('bulan_pembelian')['review_score'].mean()

st.markdown("## \U0001F4C8 Tren Pesanan dan Ulasan")
col1, col2 = st.columns(2)

with col1:
    st.markdown("### Jumlah Pesanan per Bulan")
    fig1, ax1 = plt.subplots()
    jumlah_pesanan_per_bulan.plot(kind='bar', color='skyblue', ax=ax1)
    max_idx = jumlah_pesanan_per_bulan.index.get_loc(jumlah_pesanan_per_bulan.idxmax())
    ax1.axvline(x=max_idx, color='red', linestyle='--', label='Tertinggi')
    ax1.set_xlabel("Bulan")
    ax1.set_ylabel("Jumlah")
    ax1.legend()
    tampilkan_gambar(fig1)

with col2:
    st.markdown("### Rata-rata Skor Ulasan per Bulan")
    fig2, ax2 = plt.subplots()
    rata2_skor_per_bulan.plot(marker='o', ax=ax2)
    ax2.set_xlabel("Bulan")
    ax2.set_ylabel("Skor Ulasan")
    tampilkan_gambar(fig2)

# --- Tipe Pembayaran dan Produk Terlaris ---
st.markdown("---")
st.markdown("## \U0001F6CD️ Analisis Pembayaran & Produk")
col3, col4 = st.columns(2)

with col3:
    st.markdown("### Distribusi Tipe Pembayaran")
    fig3, ax3 = plt.subplots()
    pembayaran['payment_type'].value_counts().plot(kind='bar', color='salmon', ax=ax3)
    ax3.set_xlabel("Tipe")
    ax3.set_ylabel("Jumlah")
    tampilkan_gambar(fig3)

with col4:
    items_dengan_nama = pd.merge(items, produk[['product_id', 'product_category_name']], on='product_id', how='left')
    produk_terlaris_nama = items_dengan_nama.groupby('product_category_name')['price'].sum().sort_values(ascending=False).head(10)
    st.markdown("### Top 10 Kategori Produk Terlaris")
    fig4, ax4 = plt.subplots()
    produk_terlaris_nama.plot(kind='bar', color='green', ax=ax4)
    ax4.set_ylabel("Total Penjualan")
    tampilkan_gambar(fig4)

# --- Segmentasi RFM ---
with st.expander("Segmentasi RFM", expanded=True):
    tanggal_acuan = pesanan['order_purchase_timestamp'].max()
    rfm = pesanan.groupby('customer_id').agg({
        'order_purchase_timestamp': lambda x: (tanggal_acuan - x.max()).days,
        'order_id': 'count'
    }).rename(columns={
        'order_purchase_timestamp': 'Hari_Sejak_Transaksi_Terakhir',
        'order_id': 'Frekuensi_Transaksi'
    })

    monetary = pd.merge(pesanan[['order_id', 'customer_id']], pembayaran[['order_id', 'payment_value']], on='order_id')
    rata2_monetary = monetary.groupby('customer_id')['payment_value'].sum().rename('Total_Pembayaran')
    rfm = rfm.join(rata2_monetary, how='left').fillna(0)

    rfm['R'] = pd.qcut(rfm['Hari_Sejak_Transaksi_Terakhir'], 4, labels=[4, 3, 2, 1]).astype(str)
    rfm['F'] = pd.qcut(rfm['Frekuensi_Transaksi'].rank(method='first'), 4, labels=[1, 2, 3, 4]).astype(str)
    rfm['M'] = pd.qcut(rfm['Total_Pembayaran'], 4, labels=[1, 2, 3, 4]).astype(str)
    rfm['RFM_Skor'] = rfm['R'] + rfm['F'] + rfm['M']

    def klasifikasi_rfm(row):
        if row['RFM_Skor'] == '444':
            return 'Best Customers'
        elif row['R'] == '4':
            return 'Loyal Customers'
        elif row['F'] == '4':
            return 'Frequent Buyers'
        elif row['M'] == '4':
            return 'Big Spenders'
        elif row['RFM_Skor'] == '111':
            return 'Tidak Aktif'
        return 'Others'

    rfm['Segment'] = rfm.apply(klasifikasi_rfm, axis=1)

    st.markdown("### Tabel Segmentasi Pelanggan")
    st.dataframe(rfm[['Hari_Sejak_Transaksi_Terakhir', 'Frekuensi_Transaksi', 'Total_Pembayaran', 'RFM_Skor', 'Segment']].head())

    fig5, ax5 = plt.subplots()
    rfm['Segment'].value_counts().plot(kind='bar', color='mediumseagreen', ax=ax5)
    ax5.set_title("Distribusi Pelanggan per Segmen")
    tampilkan_gambar(fig5)

    st.dataframe(rfm['Segment'].value_counts().reset_index().rename(columns={'index': 'Segment', 'Segment': 'Jumlah'}))

# --- Lokasi ---
st.markdown("---")
st.markdown("## Distribusi Lokasi & Klaster Pelanggan")

st.markdown("### Distribusi Geografis Pelanggan")
lokasi_kota = geolokasi.groupby('geolocation_city').agg({
    'geolocation_lat': 'mean',
    'geolocation_lng': 'mean'
}).reset_index()

peta = folium.Map(location=[-14.2350, -51.9253], zoom_start=4)
for _, row in lokasi_kota.iterrows():
    folium.CircleMarker(
        location=[row['geolocation_lat'], row['geolocation_lng']],
        radius=3,
        popup=row['geolocation_city'],
        color='blue',
        fill=True,
        fill_color='blue',
        fill_opacity=0.5
    ).add_to(peta)

col_center = st.columns([1, 2, 1])[1]
with col_center:
    st_data = st_folium(peta, width=700, height=600)

# --- Clustering ---
st.markdown("### Visualisasi Klaster Pelanggan Berdasarkan RFM")
rfm_clustering = rfm[['Hari_Sejak_Transaksi_Terakhir', 'Frekuensi_Transaksi', 'Total_Pembayaran']].copy()
rfm_log = np.log1p(rfm_clustering)
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm_log)

rfm_clustering['Skor_R'] = pd.qcut(rfm_clustering['Hari_Sejak_Transaksi_Terakhir'], 4, labels=[4, 3, 2, 1]).astype(int)
rfm_clustering['Skor_F'] = pd.qcut(rfm_clustering['Frekuensi_Transaksi'].rank(method='first'), 4, labels=[1, 2, 3, 4]).astype(int)
rfm_clustering['Skor_M'] = pd.qcut(rfm_clustering['Total_Pembayaran'], 4, labels=[1, 2, 3, 4]).astype(int)
rfm_clustering['Skor_Total'] = rfm_clustering['Skor_R'] + rfm_clustering['Skor_F'] + rfm_clustering['Skor_M']
rfm_clustering['Cluster'] = pd.cut(rfm_clustering['Skor_Total'], bins=[2, 6, 8, 10, 12], labels=[0, 1, 2, 3]).astype(int)

# Label Segmen
rfm_clustering['Segment'] = rfm_clustering['Cluster'].map({
    3: 'Best Customers',
    2: 'Loyal Buyers',
    1: 'Average Buyers',
    0: 'At-Risk Customers'
})

# PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(rfm_scaled)
rfm_clustering['PCA1'] = pca_result[:, 0]
rfm_clustering['PCA2'] = pca_result[:, 1]

colors = {
    'At-Risk Customers': 'red',
    'Average Buyers': 'orange',
    'Loyal Buyers': 'green',
    'Best Customers': 'blue'
}
fig, ax = plt.subplots()
for segment, color in colors.items():
    data = rfm_clustering[rfm_clustering['Segment'] == segment]
    ax.scatter(data['PCA1'], data['PCA2'], label=segment, color=color, alpha=0.6)

ax.set_title("PCA Visualisasi Klaster Pelanggan")
ax.set_xlabel("PCA 1")
ax.set_ylabel("PCA 2")
ax.legend()
tampilkan_gambar(fig)

# Ringkasan Cluster
st.markdown("## Ringkasan Tiap Cluster")
cluster_summary = rfm_clustering.groupby('Segment')[['Hari_Sejak_Transaksi_Terakhir', 'Frekuensi_Transaksi', 'Total_Pembayaran']].mean().round(2)
st.dataframe(cluster_summary)

st.markdown("## Distribusi Jumlah Pelanggan per Segmen")
st.dataframe(rfm_clustering['Segment'].value_counts().reset_index().rename(columns={'index': 'Segment', 'Segment': 'Jumlah'}))
