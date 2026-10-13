import streamlit as st
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import io
import os

try:
    from streamlit_image_coordinates import streamlit_image_coordinates
    HAS_IMAGE_COORDINATES = True
except ImportError:
    HAS_IMAGE_COORDINATES = False

# Konfigurasi Halaman Streamlit
st.set_page_config(
    page_title="PCD: Resize & Visualisasi Matriks Citra",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS untuk tampilan modern, rapi, dan elegan
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .badge-tag {
        display: inline-block;
        background: #EEF2FF;
        color: #4F46E5;
        font-weight: 600;
        font-size: 0.8rem;
        padding: 4px 10px;
        border-radius: 9999px;
        margin-bottom: 0.5rem;
        border: 1px solid #C7D2FE;
    }
    .metric-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 12px 16px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-card h4 {
        margin: 0;
        font-size: 0.85rem;
        color: #64748B;
        font-weight: 500;
    }
    .metric-card p {
        margin: 4px 0 0 0;
        font-size: 1.25rem;
        color: #0F172A;
        font-weight: 700;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0px 0px;
        padding: 8px 16px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Header Aplikasi
st.markdown('<span class="badge-tag">Materi Pengajaran PCD • Pertemuan 3</span>', unsafe_allow_html=True)
st.markdown('<div class="main-title">Aplikasi Resize Citra & Visualisasi Matriks Piksel</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Pelajari representasi numerik citra digital (Grayscale & Channel RGB) serta pengaruh operasi resize spasial dengan berbagai metode interpolasi.</div>', unsafe_allow_html=True)

# ==========================================
# SIDEBAR: PENGATURAN INPUT & RESIZE
# ==========================================
st.sidebar.header("⚙️ Konfigurasi Citra & Resize")

# 1. Pilihan Input Gambar
st.sidebar.subheader("1. Sumber Citra")
source_option = st.sidebar.radio(
    "Pilih cara input citra:",
    ["Upload Citra Sendiri", "Gunakan Citra Contoh PCD"],
    index=1
)

input_image = None
image_name = "sample"

if source_option == "Upload Citra Sendiri":
    uploaded_file = st.sidebar.file_uploader(
        "Unggah file gambar (PNG, JPG, JPEG, BMP):",
        type=["png", "jpg", "jpeg", "bmp", "webp"]
    )
    if uploaded_file is not None:
        pil_img = Image.open(uploaded_file)
        # Convert ke RGB jika format lain (RGBA / Grayscale)
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")
        input_image = np.array(pil_img)
        image_name = os.path.splitext(uploaded_file.name)[0]
    else:
        st.sidebar.info("Silakan unggah citra atau pilih 'Gunakan Citra Contoh PCD' untuk mencoba langsung.")
else:
    sample_files = {
        "8x8 Pola Matriks Edukasi (Ideal untuk inspeksi seluruh angka)": "sample_images/contoh_8x8_pola_matriks.png",
        "16x16 Gradien Warna RGB": "sample_images/contoh_16x16_gradien_warna.png",
        "128x128 Pola Geometris Lengkap": "sample_images/contoh_128x128_geometris.png"
    }
    selected_sample = st.sidebar.selectbox("Pilih citra contoh:", list(sample_files.keys()))
    sample_path = sample_files[selected_sample]
    if os.path.exists(sample_path):
        bgr = cv2.imread(sample_path)
        input_image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        image_name = os.path.splitext(os.path.basename(sample_path))[0]
    else:
        st.sidebar.error("File sampel tidak ditemukan.")

# Jika citra belum tersedia, stop
if input_image is None:
    st.info("👈 Silakan pilih citra pada menu di sidebar sebelah kiri untuk memulai.")
    st.stop()

# Info Citra Asli
orig_h, orig_w, orig_c = input_image.shape

# 2. Pengaturan Resize
st.sidebar.markdown("---")
st.sidebar.subheader("2. Pengaturan Resize Citra")

resize_mode = st.sidebar.radio(
    "Metode Penentuan Ukuran:",
    ["Faktor Skala (%)", "Dimensi Kustom (Piksel)"]
)

keep_aspect_ratio = st.sidebar.checkbox("Pertahankan Rasio Aspek (Aspect Ratio)", value=True)

if resize_mode == "Faktor Skala (%)":
    scale_percent = st.sidebar.slider(
        "Persentase Skala:",
        min_value=10,
        max_value=400,
        value=50 if orig_w > 16 else 200,
        step=5,
        help="Contoh: 50% berarti ukuran diperkecil menjadi setengahnya, 200% diperbesar 2x lipat."
    )
    new_w = max(1, int(orig_w * scale_percent / 100))
    new_h = max(1, int(orig_h * scale_percent / 100))
else:
    if keep_aspect_ratio:
        new_w = st.sidebar.number_input("Lebar Baru (Width):", min_value=1, max_value=2048, value=max(1, orig_w // 2))
        ratio = orig_h / orig_w
        new_h = max(1, int(new_w * ratio))
        st.sidebar.caption(f"Tinggi Otomatis: **{new_h} px** (Rasio: {orig_w}:{orig_h})")
    else:
        new_w = st.sidebar.number_input("Lebar Baru (Width):", min_value=1, max_value=2048, value=max(1, orig_w // 2))
        new_h = st.sidebar.number_input("Tinggi Baru (Height):", min_value=1, max_value=2048, value=max(1, orig_h // 2))

# 3. Metode Interpolasi
st.sidebar.markdown("---")
st.sidebar.subheader("3. Metode Interpolasi Spasial")

interp_dict = {
    "Nearest Neighbor (INTER_NEAREST)": cv2.INTER_NEAREST,
    "Bilinear (INTER_LINEAR - Standar)": cv2.INTER_LINEAR,
    "Bicubic (INTER_CUBIC - Halus 4x4)": cv2.INTER_CUBIC,
    "Area Resampling (INTER_AREA - Terbaik Downscale)": cv2.INTER_AREA,
    "Lanczos (INTER_LANCZOS4 - 8x8 Tetangga)": cv2.INTER_LANCZOS4
}

selected_interp_name = st.sidebar.selectbox(
    "Pilih Algoritma Interpolasi:",
    list(interp_dict.keys()),
    index=1
)
selected_interp = interp_dict[selected_interp_name]

# Penjelasan singkat interpolasi
interp_notes = {
    "Nearest Neighbor (INTER_NEAREST)": "Mengambil nilai piksel terdekat. Paling cepat, tidak mengubah nilai piksel (hanya menduplikasi), namun menghasilkan artefak bergerigi (pixelated).",
    "Bilinear (INTER_LINEAR - Standar)": "Menghitung rata-rata tertimbang dari 4 piksel tetangga (2x2). Memberikan hasil lebih halus daripada Nearest Neighbor.",
    "Bicubic (INTER_CUBIC - Halus 4x4)": "Menggunakan interpolasi spline polinomial kubik dari 16 piksel tetangga (4x4). Sangat halus dan tajam untuk pembesaran citra.",
    "Area Resampling (INTER_AREA - Terbaik Downscale)": "Resampling berbasis relasi luas piksel. Pilihan terbaik untuk memperkecil citra (downscaling) agar tidak timbul efek moiré.",
    "Lanczos (INTER_LANCZOS4 - 8x8 Tetangga)": "Interpolasi menggunakan fungsi sinc berbobot Lanczos atas 64 piksel tetangga (8x8). Kualitas tinggi namun komputasi lebih berat."
}
st.sidebar.info(interp_notes[selected_interp_name])

# ==========================================
# PROSES PENGOLAHAN CITRA
# ==========================================
# 1. Resize Citra
resized_image = cv2.resize(input_image, (new_w, new_h), interpolation=selected_interp)

# 2. Ekstrak Grayscale Citra Asli & Hasil Resize
# Rumus standar ITU-R BT.601: Y = 0.299 R + 0.587 G + 0.114 B
orig_gray = cv2.cvtColor(input_image, cv2.COLOR_RGB2GRAY)
resized_gray = cv2.cvtColor(resized_image, cv2.COLOR_RGB2GRAY)

# 3. Ekstrak Channel R, G, B terpisah
orig_r, orig_g, orig_b = input_image[:, :, 0], input_image[:, :, 1], input_image[:, :, 2]
resized_r, resized_g, resized_b = resized_image[:, :, 0], resized_image[:, :, 1], resized_image[:, :, 2]

# ==========================================
# METRICS OVERVIEW (RINGKASAN DATA CITRA)
# ==========================================
m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)

with m_col1:
    st.markdown(f"""
    <div class="metric-card">
        <h4>Dimensi Asli</h4>
        <p>{orig_w} × {orig_h} px</p>
    </div>
    """, unsafe_allow_html=True)

with m_col2:
    st.markdown(f"""
    <div class="metric-card">
        <h4>Dimensi Resize</h4>
        <p>{new_w} × {new_h} px</p>
    </div>
    """, unsafe_allow_html=True)

with m_col3:
    pixel_change = ((new_w * new_h) / (orig_w * orig_h)) * 100
    st.markdown(f"""
    <div class="metric-card">
        <h4>Rasio Jumlah Piksel</h4>
        <p>{pixel_change:.1f}%</p>
    </div>
    """, unsafe_allow_html=True)

with m_col4:
    st.markdown(f"""
    <div class="metric-card">
        <h4>Tipe Data Piksel</h4>
        <p>{str(input_image.dtype)} [0-255]</p>
    </div>
    """, unsafe_allow_html=True)

with m_col5:
    st.markdown(f"""
    <div class="metric-card">
        <h4>Channel Citra</h4>
        <p>3 (R, G, B)</p>
    </div>
    """, unsafe_allow_html=True)

st.write("")

# Helper function untuk visualisasi heatmap matriks
def plot_matrix_heatmap(matrix, title="Matriks", colormap="viridis", max_annot_dim=32):
    h, w = matrix.shape
    fig, ax = plt.subplots(figsize=(max(5, min(14, w * 0.7)), max(4, min(12, h * 0.6))))
    
    # Render heatmap dengan rasterized agar cepat untuk matriks besar
    im = ax.imshow(matrix, cmap=colormap, vmin=0, vmax=255, aspect="auto", rasterized=True)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Intensitas Piksel (0 - 255)", rotation=270, labelpad=15)
    
    ax.set_title(f"{title} ({h} baris × {w} kolom)", fontsize=12, fontweight='bold', pad=10)
    ax.set_xlabel("Indeks Kolom (X)")
    ax.set_ylabel("Indeks Baris (Y)")
    
    # Anotasi angka piksel jika ukuran matriks tergolong kecil / proporsional (<= 32x32)
    if h <= max_annot_dim and w <= max_annot_dim:
        for r in range(h):
            for c in range(w):
                val = int(matrix[r, c])
                # Menentukan kontras warna teks
                text_color = "white" if val < 128 else "black"
                ax.text(c, r, str(val), ha="center", va="center", color=text_color, fontsize=8, fontweight='medium')
        ax.set_xticks(range(w))
        ax.set_yticks(range(h))
    
    plt.tight_layout()
    return fig

# Helper function styling dataframe dengan penanganan aman untuk matriks besar
def format_matrix_dataframe(matrix, channel_type="gray"):
    df = pd.DataFrame(matrix)
    df.columns = [f"K{c}" for c in range(matrix.shape[1])]
    df.index = [f"B{r}" for r in range(matrix.shape[0])]
    
    # Batas aman pewarnaan CSS styling pandas (agar tidak terjadi memory overload atau StreamlitAPIException)
    MAX_STYLED_CELLS = 25000  # batas aman (~150x150 sel)
    
    if matrix.size <= MAX_STYLED_CELLS:
        color_map = {
            "gray": "Greys",
            "red": "Reds",
            "green": "Greens",
            "blue": "Blues"
        }.get(channel_type, "viridis")
        try:
            styled = df.style.background_gradient(cmap=color_map, vmin=0, vmax=255).format(precision=0)
            return styled, True
        except Exception:
            return df, False
            
    # Jika matriks lebih besar dari batas aman, gunakan DataFrame standar tanpa CSS Styler
    return df, False

# Helper function komparasi dua sub-matriks berdampingan untuk Operasi Titik
def render_side_by_side_matrix_diff(mat_before, mat_after, title_before="Sebelum", title_after="Sesudah"):
    m1 = cv2.cvtColor(mat_before, cv2.COLOR_RGB2GRAY) if len(mat_before.shape) == 3 else mat_before
    m2 = cv2.cvtColor(mat_after, cv2.COLOR_RGB2GRAY) if len(mat_after.shape) == 3 else mat_after
    
    h, w = m1.shape
    row_max = min(10, h)
    col_max = min(10, w)
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**Matriks {title_before} (Area {row_max}×{col_max})**")
        df1 = pd.DataFrame(m1[:row_max, :col_max])
        df1.columns = [f"K{c}" for c in range(col_max)]
        df1.index = [f"B{r}" for r in range(row_max)]
        st.dataframe(df1.style.background_gradient(cmap="Greys", vmin=0, vmax=255).format(precision=0), use_container_width=True)
    with c2:
        st.markdown(f"**Matriks {title_after} (Area {row_max}×{col_max})**")
        df2 = pd.DataFrame(m2[:row_max, :col_max])
        df2.columns = [f"K{c}" for c in range(col_max)]
        df2.index = [f"B{r}" for r in range(row_max)]
        st.dataframe(df2.style.background_gradient(cmap="Greys", vmin=0, vmax=255).format(precision=0), use_container_width=True)

# Helper function plot histogram dan kurva CDF
def plot_hist_cdf_comparison(img_before, img_after, title1="Sebelum", title2="Sesudah"):
    g1 = cv2.cvtColor(img_before, cv2.COLOR_RGB2GRAY) if len(img_before.shape) == 3 else img_before
    g2 = cv2.cvtColor(img_after, cv2.COLOR_RGB2GRAY) if len(img_after.shape) == 3 else img_after
    
    hist1, _ = np.histogram(g1.flatten(), 256, [0, 256])
    cdf1 = hist1.cumsum()
    cdf1_norm = cdf1 * float(hist1.max()) / max(1, cdf1.max())
    
    hist2, _ = np.histogram(g2.flatten(), 256, [0, 256])
    cdf2 = hist2.cumsum()
    cdf2_norm = cdf2 * float(hist2.max()) / max(1, cdf2.max())
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
    
    ax1.plot(cdf1_norm, color='#F59E0B', linewidth=2, label='CDF (Kumulatif)')
    ax1.hist(g1.flatten(), 256, [0, 256], color='#4F46E5', alpha=0.6, label='Histogram Intensitas')
    ax1.set_title(f"Histogram & CDF: {title1}", fontweight='bold')
    ax1.set_xlabel("Nilai Piksel (0 - 255)")
    ax1.set_ylabel("Frekuensi / CDF Skala")
    ax1.set_xlim([0, 256])
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(cdf2_norm, color='#F59E0B', linewidth=2, label='CDF (Kumulatif)')
    ax2.hist(g2.flatten(), 256, [0, 256], color='#10B981', alpha=0.6, label='Histogram Intensitas')
    ax2.set_title(f"Histogram & CDF: {title2}", fontweight='bold')
    ax2.set_xlabel("Nilai Piksel (0 - 255)")
    ax2.set_ylabel("Frekuensi / CDF Skala")
    ax2.set_xlim([0, 256])
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

# Helper functions untuk Inspeksi Interaktif Citra <-> Matriks
def draw_target_crosshair(img, target_x, target_y):
    marked = img.copy()
    h, w = marked.shape[:2]
    tx = max(0, min(w - 1, int(target_x)))
    ty = max(0, min(h - 1, int(target_y)))
    
    # Ukuran kotak penanda proporsional
    box_sz = max(1, min(12, int(max(w, h) * 0.04)))
    ch_len = max(5, box_sz * 3)
    
    line_th = max(1, int(max(w, h) / 250))
    # Garis bidik (crosshair) warna kuning terang
    cv2.line(marked, (max(0, tx - ch_len), ty), (min(w - 1, tx + ch_len), ty), (0, 255, 255), line_th)
    cv2.line(marked, (tx, max(0, ty - ch_len)), (tx, min(h - 1, ty + ch_len)), (0, 255, 255), line_th)
    # Kotak target warna merah terang
    cv2.rectangle(marked, (max(0, tx - box_sz), max(0, ty - box_sz)), (min(w - 1, tx + box_sz), min(h - 1, ty + box_sz)), (255, 0, 0), line_th + 1)
    return marked

def extract_zoomed_patch(img, target_x, target_y, radius=4, zoom=25):
    h, w = img.shape[:2]
    tx = max(0, min(w - 1, int(target_x)))
    ty = max(0, min(h - 1, int(target_y)))
    
    x1, x2 = max(0, tx - radius), min(w, tx + radius + 1)
    y1, y2 = max(0, ty - radius), min(h, ty + radius + 1)
    
    patch = img[y1:y2, x1:x2]
    ph, pw = patch.shape[:2]
    
    zh, zw = ph * zoom, pw * zoom
    zoomed = cv2.resize(patch, (zw, zh), interpolation=cv2.INTER_NEAREST)
    
    # Gambar garis grid antar-piksel
    for r in range(ph + 1):
        cv2.line(zoomed, (0, r * zoom), (zw, r * zoom), (100, 100, 100), 1)
    for c in range(pw + 1):
        cv2.line(zoomed, (c * zoom, 0), (c * zoom, zh), (100, 100, 100), 1)
        
    # Kotak merah tebal pada piksel tengah yang dipilih
    center_c = tx - x1
    center_r = ty - y1
    cx1, cy1 = center_c * zoom, center_r * zoom
    cx2, cy2 = cx1 + zoom, cy1 + zoom
    cv2.rectangle(zoomed, (cx1, cy1), (cx2, cy2), (255, 0, 0), 3)
    
    return zoomed, (y1, y2, x1, x2)

# ==========================================
# TABS UTAMA APLIKASI
# ==========================================
tab1, tab_insp, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🖼️ Komparasi Citra",
    "🎯 Inspeksi Interaktif (Citra ⇄ Matriks)",
    "🔢 Matriks Citra Asli",
    "🔄 Matriks Citra Hasil Resize",
    "✨ Operasi Titik (Point Operations)",
    "📊 Analisis & Statistik Matriks",
    "💾 Ekspor Hasil & Matriks"
])

# ------------------------------------------
# TAB 1: KOMPARASI CITRA
# ------------------------------------------
with tab1:
    st.subheader("Perbandingan Visual: Citra Asli vs Hasil Resize")
    
    col_view1, col_view2 = st.columns(2)
    
    with col_view1:
        st.markdown(f"**Citra Asli ({orig_w} × {orig_h} px)**")
        st.image(input_image, caption=f"Citra Asli: {image_name} ({orig_w}x{orig_h})", use_column_width=True)
        st.caption(f"Total Piksel: **{orig_w * orig_h:,} piksel** | Ukuran Memori Array: **{input_image.nbytes / 1024:.2f} KB**")
    
    with col_view2:
        st.markdown(f"**Citra Hasil Resize ({new_w} × {new_h} px)**")
        st.image(resized_image, caption=f"Hasil Resize: {selected_interp_name} ({new_w}x{new_h})", use_column_width=True)
        st.caption(f"Total Piksel: **{new_w * new_h:,} piksel** | Ukuran Memori Array: **{resized_image.nbytes / 1024:.2f} KB**")
    
    st.markdown("---")
    st.subheader("Dekomposisi Visual Berdasarkan Channel Warna (R, G, B & Grayscale)")
    st.write("Citra warna RGB tersusun dari 3 matriks independen: Merah, Hijau, dan Biru. Sedangkan citra Grayscale merepresentasikan intensitas cahaya tunggal.")
    
    # Tampilkan channel terpisah untuk citra asli
    st.markdown("#### Channel Citra Asli:")
    ch_col1, ch_col2, ch_col3, ch_col4 = st.columns(4)
    with ch_col1:
        st.image(orig_gray, caption="Grayscale (Intensitas)", use_column_width=True, clamp=True)
    with ch_col2:
        # Visualisasi channel merah
        r_vis = np.zeros_like(input_image)
        r_vis[:, :, 0] = orig_r
        st.image(r_vis, caption="Channel Merah (Red)", use_column_width=True)
    with ch_col3:
        # Visualisasi channel hijau
        g_vis = np.zeros_like(input_image)
        g_vis[:, :, 1] = orig_g
        st.image(g_vis, caption="Channel Hijau (Green)", use_column_width=True)
    with ch_col4:
        # Visualisasi channel biru
        b_vis = np.zeros_like(input_image)
        b_vis[:, :, 2] = orig_b
        st.image(b_vis, caption="Channel Biru (Blue)", use_column_width=True)
        
    st.markdown("#### Channel Citra Hasil Resize:")
    rch_col1, rch_col2, rch_col3, rch_col4 = st.columns(4)
    with rch_col1:
        st.image(resized_gray, caption="Grayscale (Hasil Resize)", use_column_width=True, clamp=True)
    with rch_col2:
        r_vis_res = np.zeros_like(resized_image)
        r_vis_res[:, :, 0] = resized_r
        st.image(r_vis_res, caption="Red (Hasil Resize)", use_column_width=True)
    with rch_col3:
        g_vis_res = np.zeros_like(resized_image)
        g_vis_res[:, :, 1] = resized_g
        st.image(g_vis_res, caption="Green (Hasil Resize)", use_column_width=True)
    with rch_col4:
        b_vis_res = np.zeros_like(resized_image)
        b_vis_res[:, :, 2] = resized_b
        st.image(b_vis_res, caption="Blue (Hasil Resize)", use_column_width=True)


# ------------------------------------------
# TAB INSPEKSI INTERAKTIF: CITRA <-> MATRIKS
# ------------------------------------------
with tab_insp:
    st.subheader("🎯 Inspeksi Interaktif Dua Arah: Hubungan Citra ⇄ Matriks Piksel")
    st.markdown("""
    Fitur ini menghubungkan secara langsung representasi visual citra dengan representasi numerik matriks:
    - 🖱️ **Klik pada Citra**: Klik sembarang titik pada citra di bawah, maka sistem akan langsung menampilkan koordinat piksel, nilai warna $(R, G, B, \text{Grayscale})$, dan matriks tetangganya.
    - 🔢 **Pilih Angka di Matriks**: Ubah nilai Kolom ($X$) dan Baris ($Y$), maka citra akan langsung menandai titik tersebut dengan garis bidik (*crosshair*) merah dan menampilkan pembesaran mikroskopis.
    """)
    
    # Pilihan Citra yang Diinspeksi
    insp_src_col1, insp_src_col2 = st.columns([2, 2])
    with insp_src_col1:
        insp_target_choice = st.radio(
            "Pilih Citra untuk Diinspeksi:",
            ["Citra Asli", "Citra Hasil Resize"],
            horizontal=True,
            key="insp_img_target"
        )
    with insp_src_col2:
        insp_channel_view = st.radio(
            "Channel Matriks Tetangga yang Ditampilkan:",
            ["Grayscale", "Red (R)", "Green (G)", "Blue (B)"],
            horizontal=True,
            key="insp_channel_mode"
        )
        
    curr_insp_img = input_image if insp_target_choice == "Citra Asli" else resized_image
    ci_h, ci_w = curr_insp_img.shape[:2]
    
    # Inisialisasi Session State Koordinat
    if "insp_x" not in st.session_state:
        st.session_state["insp_x"] = min(ci_w // 2, 4)
    if "insp_y" not in st.session_state:
        st.session_state["insp_y"] = min(ci_h // 2, 4)
        
    # Pastikan koordinat tetap dalam batas valid jika ukuran citra berubah
    st.session_state["insp_x"] = max(0, min(ci_w - 1, int(st.session_state["insp_x"])))
    st.session_state["insp_y"] = max(0, min(ci_h - 1, int(st.session_state["insp_y"])))
    
    st.markdown("---")
    
    col_insp_left, col_insp_right = st.columns([1, 1])
    
    with col_insp_left:
        st.markdown(f"#### 🖼️ 1. Citra Interaktif ({ci_w} × {ci_h} px)")
        st.caption("👇 **Klik langsung di atas gambar** untuk memilih piksel, atau amati posisi piksel aktif berpenanda merah:")
        
        # Penanganan Display Width & Scaling untuk Klik
        if ci_w < 250:
            scale_disp = max(1, 320 // ci_w)
            disp_w = ci_w * scale_disp
            disp_h = ci_h * scale_disp
            scaled_for_display = cv2.resize(curr_insp_img, (disp_w, disp_h), interpolation=cv2.INTER_NEAREST)
        else:
            disp_w = min(480, ci_w)
            scale_disp = disp_w / ci_w
            disp_h = int(ci_h * scale_disp)
            scaled_for_display = cv2.resize(curr_insp_img, (disp_w, disp_h), interpolation=cv2.INTER_NEAREST)
            
        # Widget Klik Interaktif
        if HAS_IMAGE_COORDINATES:
            click_event = streamlit_image_coordinates(
                Image.fromarray(scaled_for_display),
                width=disp_w,
                key="pcd_interactive_clicker"
            )
            if click_event is not None:
                new_cx = int(click_event["x"] * ci_w / disp_w)
                new_cy = int(click_event["y"] * ci_h / disp_h)
                new_cx = max(0, min(ci_w - 1, new_cx))
                new_cy = max(0, min(ci_h - 1, new_cy))
                
                # Update session state jika berbeda
                if (new_cx, new_cy) != (st.session_state["insp_x"], st.session_state["insp_y"]):
                    st.session_state["insp_x"] = new_cx
                    st.session_state["insp_y"] = new_cy
                    st.rerun()
        else:
            st.image(curr_insp_img, caption="Citra Aktif", width=disp_w)
            
        cur_x = st.session_state["insp_x"]
        cur_y = st.session_state["insp_y"]
        
        # Tampilkan Citra dengan Penanda Crosshair Target
        st.markdown(f"**Posisi Piksel Terpilih pada Citra:** Koordinat `(X={cur_x}, Y={cur_y})`")
        marked_img = draw_target_crosshair(curr_insp_img, cur_x, cur_y)
        st.image(marked_img, caption=f"Penanda Target: Kotak Merah & Garis Bidik Kuning di (X={cur_x}, Y={cur_y})", use_column_width=True)
        
        # Pembesaran Mikroskopis Patch Sekitar Piksel
        st.markdown("##### 🔬 Tampilan Mikroskopis Zoom (Piksel Fisik & Grid):")
        patch_zoom, (py1, py2, px1, px2) = extract_zoomed_patch(curr_insp_img, cur_x, cur_y, radius=4, zoom=25)
        st.image(patch_zoom, caption=f"Zoom-in 25x Area Tetangga [Y:{py1}..{py2-1}, X:{px1}..{px2-1}] (Kotak Merah = Piksel Terpilih)", use_column_width=True)

    with col_insp_right:
        st.markdown("#### 🔢 2. Nilai & Matriks Piksel")
        st.caption("Pilih koordinat secara manual di bawah, atau amati nilai dari piksel yang baru saja Anda klik di sebelah kiri:")
        
        # Pemilih Koordinat Number Input / Sliders (Dua Arah)
        coord_c1, coord_c2 = st.columns(2)
        with coord_c1:
            inp_x = st.number_input(
                f"Kolom X Matriks (0 .. {ci_w - 1}):",
                min_value=0,
                max_value=ci_w - 1,
                value=cur_x,
                step=1,
                key="manual_coord_x"
            )
        with coord_c2:
            inp_y = st.number_input(
                f"Baris Y Matriks (0 .. {ci_h - 1}):",
                min_value=0,
                max_value=ci_h - 1,
                value=cur_y,
                step=1,
                key="manual_coord_y"
            )
            
        if inp_x != cur_x or inp_y != cur_y:
            st.session_state["insp_x"] = inp_x
            st.session_state["insp_y"] = inp_y
            st.rerun()
            
        # Ekstrak Nilai Piksel pada Koordinat (cur_x, cur_y)
        pixel_r = int(curr_insp_img[cur_y, cur_x, 0])
        pixel_g = int(curr_insp_img[cur_y, cur_x, 1])
        pixel_b = int(curr_insp_img[cur_y, cur_x, 2])
        # Rumus Grayscale ITU-R BT.601
        pixel_gray = int(round(0.299 * pixel_r + 0.587 * pixel_g + 0.114 * pixel_b))
        hex_color = f"#{pixel_r:02x}{pixel_g:02x}{pixel_b:02x}".upper()
        
        # Kartu Ringkasan Informasi Piksel
        st.markdown(f"""
        <div style="background: #F1F5F9; border: 1px solid #CBD5E1; border-radius: 12px; padding: 14px; margin-bottom: 16px;">
            <div style="display: flex; align-items: center; gap: 16px;">
                <div style="width: 70px; height: 70px; background-color: rgb({pixel_r}, {pixel_g}, {pixel_b}); border-radius: 8px; border: 2px solid #64748B; box-shadow: 0 2px 4px rgba(0,0,0,0.1); flex-shrink: 0;" title="Warna Piksel Nyata"></div>
                <div style="flex-grow: 1;">
                    <h4 style="margin: 0; color: #1E293B; font-size: 1.1rem;">📍 Piksel Koordinat: (X={cur_x}, Y={cur_y})</h4>
                    <p style="margin: 4px 0 0 0; font-size: 0.9rem; color: #475569;">
                        <b>Kode Hex:</b> <code>{hex_color}</code> &bull; <b>Grayscale:</b> <span style="background: #E2E8F0; padding: 2px 6px; border-radius: 4px; font-weight: 600;">{pixel_gray}</span>
                    </p>
                    <div style="margin-top: 6px; font-size: 0.85rem; font-weight: 600; display: flex; gap: 10px;">
                        <span style="color: #DC2626;">Red (R): {pixel_r}</span>
                        <span style="color: #16A34A;">Green (G): {pixel_g}</span>
                        <span style="color: #2563EB;">Blue (B): {pixel_b}</span>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Tabel Matriks Tetangga (Local Neighborhood Window)
        st.markdown("##### 🪟 Matriks Tetangga di Sekitar Titik Terpilih:")
        win_size = st.select_slider(
            "Pilih Ukuran Jendela Matriks (Kernel Window):",
            options=[3, 5, 7, 9],
            value=5,
            format_func=lambda w: f"{w} × {w} piksel",
            key="insp_win_size"
        )
        
        rad = win_size // 2
        ny1, ny2 = max(0, cur_y - rad), min(ci_h, cur_y + rad + 1)
        nx1, nx2 = max(0, cur_x - rad), min(ci_w, cur_x + rad + 1)
        
        if insp_channel_view == "Grayscale":
            if len(curr_insp_img.shape) == 3:
                mat_source = cv2.cvtColor(curr_insp_img, cv2.COLOR_RGB2GRAY)
            else:
                mat_source = curr_insp_img
            cmap_name = "Greys"
        elif insp_channel_view == "Red (R)":
            mat_source = curr_insp_img[:, :, 0]
            cmap_name = "Reds"
        elif insp_channel_view == "Green (G)":
            mat_source = curr_insp_img[:, :, 1]
            cmap_name = "Greens"
        else:
            mat_source = curr_insp_img[:, :, 2]
            cmap_name = "Blues"
            
        sub_neighborhood = mat_source[ny1:ny2, nx1:nx2]
        
        df_neighbor = pd.DataFrame(sub_neighborhood)
        df_neighbor.columns = [f"X={c}" + (" 📍" if c == cur_x else "") for c in range(nx1, nx2)]
        df_neighbor.index = [f"Y={r}" + (" 📍" if r == cur_y else "") for r in range(ny1, ny2)]
        
        styled_neighbor = df_neighbor.style.background_gradient(
            cmap=cmap_name, vmin=0, vmax=255
        ).format(precision=0)
        
        st.dataframe(styled_neighbor, use_container_width=True)
        st.caption(f"🏷️ Baris & Kolom bertanda **📍** adalah piksel aktif yang dipilih `(X={cur_x}, Y={cur_y})`. Nilai pada sel tersebut adalah **{sub_neighborhood[cur_y - ny1, cur_x - nx1]}**.")


# ------------------------------------------
# TAB 2: MATRIKS CITRA ASLI
# ------------------------------------------
with tab2:
    st.subheader("Visualisasi Nilai Matriks Piksel: Citra Asli")
    st.write("Setiap piksel pada citra digital direpresentasikan sebagai nilai integer [0 - 255] (tipe `uint8`). Nilai 0 merepresentasikan intensitas hitam pekat, dan 255 putih/maksimum.")
    
    # Pemilihan Channel Matriks
    col_ch_sel, col_mode_sel = st.columns([2, 2])
    with col_ch_sel:
        target_channel_orig = st.selectbox(
            "Pilih Channel Matriks yang Ingin Ditampilkan:",
            ["Grayscale (2D: H × W)", "Red Channel (R)", "Green Channel (G)", "Blue Channel (B)", "Semua Channel Terpisah (R, G, B)"],
            key="orig_channel_select"
        )
    with col_mode_sel:
        view_type_orig = st.radio(
            "Tipe Tampilan:",
            ["Tabel Interaktif (DataFrame)", "Peta Panas (Heatmap Grafis)", "NumPy Raw Array"],
            horizontal=True,
            key="orig_view_type"
        )
    
    # Pengaturan Area Inspeksi (ROI / Sub-matriks)
    st.markdown("##### 🔍 Pengaturan Area Inspeksi Matriks:")
    if orig_h > 24 or orig_w > 24:
        st.caption(f"Citra asli berukuran **{orig_w} × {orig_h}** ({orig_w * orig_h:,} elemen). Anda dapat menggeser slider di bawah untuk memilih area (sub-matriks) yang ingin diinspeksi secara detail:")
        roi_col1, roi_col2 = st.columns(2)
        with roi_col1:
            row_range_orig = st.slider("Rentang Baris (Baris Y):", 0, orig_h, (0, min(16, orig_h)), key="orig_rows")
        with roi_col2:
            col_range_orig = st.slider("Rentang Kolom (Kolom X):", 0, orig_w, (0, min(16, orig_w)), key="orig_cols")
        
        show_all_orig = st.checkbox("Tampilkan seluruh matriks tanpa pemotongan", value=False, key="orig_full")
        if show_all_orig:
            r_slice_orig = slice(0, orig_h)
            c_slice_orig = slice(0, orig_w)
        else:
            r_slice_orig = slice(row_range_orig[0], row_range_orig[1])
            c_slice_orig = slice(col_range_orig[0], col_range_orig[1])
    else:
        st.success(f"Ukuran citra ({orig_w} × {orig_h}) cukup kecil untuk menampilkan seluruh matriks secara penuh!")
        r_slice_orig = slice(0, orig_h)
        c_slice_orig = slice(0, orig_w)

    def render_matrix_view(mat, channel_name, colormap, view_mode):
        sub_mat = mat[r_slice_orig, c_slice_orig]
        total_cells = sub_mat.size
        st.markdown(f"**Matriks {channel_name}** — Dimensi: `{sub_mat.shape[0]} baris × {sub_mat.shape[1]} kolom` ({total_cells:,} elemen)")
        
        if view_mode == "Tabel Interaktif (DataFrame)":
            styled_df, is_styled = format_matrix_dataframe(sub_mat, colormap)
            if not is_styled and total_cells > 25000:
                st.info(f"💡 **Mode Cepat (Native Arrow) Aktif**: Matriks berukuran besar ({sub_mat.shape[0]} × {sub_mat.shape[1]} = {total_cells:,} elemen). Tabel ditampilkan tanpa pewarnaan gradien CSS agar browser tetap ringan dan tidak lag. Anda dapat menggunakan slider rentang baris & kolom di atas jika ingin mengamati warna gradien pada area tertentu.")
            st.dataframe(styled_df, use_container_width=True)
        elif view_mode == "Peta Panas (Heatmap Grafis)":
            fig = plot_matrix_heatmap(sub_mat, title=f"Matriks {channel_name}", colormap="Reds" if colormap=="red" else "Greens" if colormap=="green" else "Blues" if colormap=="blue" else "viridis")
            st.pyplot(fig)
            plt.close(fig)
        elif view_mode == "NumPy Raw Array":
            if total_cells > 50000:
                st.info(f"💡 Matriks berukuran besar ({total_cells:,} elemen). Menampilkan representasi array NumPy dengan pemangkasan otomatis.")
            arr_str = np.array2string(sub_mat, separator=', ', precision=0, threshold=1000, edgeitems=3)
            st.code(f"np.array(\n{arr_str}\n, dtype=np.uint8)", language="python")

    if target_channel_orig == "Grayscale (2D: H × W)":
        render_matrix_view(orig_gray, "Grayscale", "gray", view_type_orig)
    elif target_channel_orig == "Red Channel (R)":
        render_matrix_view(orig_r, "Red Channel", "red", view_type_orig)
    elif target_channel_orig == "Green Channel (G)":
        render_matrix_view(orig_g, "Green Channel", "green", view_type_orig)
    elif target_channel_orig == "Blue Channel (B)":
        render_matrix_view(orig_b, "Blue Channel", "blue", view_type_orig)
    elif target_channel_orig == "Semua Channel Terpisah (R, G, B)":
        st.markdown("#### 🔴 Red Channel (R):")
        render_matrix_view(orig_r, "Red (R)", "red", view_type_orig)
        st.markdown("#### 🟢 Green Channel (G):")
        render_matrix_view(orig_g, "Green (G)", "green", view_type_orig)
        st.markdown("#### 🔵 Blue Channel (B):")
        render_matrix_view(orig_b, "Blue (B)", "blue", view_type_orig)


# ------------------------------------------
# TAB 3: MATRIKS CITRA HASIL RESIZE
# ------------------------------------------
with tab3:
    st.subheader("Visualisasi Nilai Matriks Piksel: Citra Hasil Resize")
    st.write(f"Berikut adalah matriks numerik citra setelah proses transformasi ukuran ({new_w} × {new_h} piksel) dengan metode **{selected_interp_name}**.")
    
    col_ch_sel_res, col_mode_sel_res = st.columns([2, 2])
    with col_ch_sel_res:
        target_channel_res = st.selectbox(
            "Pilih Channel Matriks yang Ingin Ditampilkan:",
            ["Grayscale (2D: H × W)", "Red Channel (R)", "Green Channel (G)", "Blue Channel (B)", "Semua Channel Terpisah (R, G, B)"],
            key="res_channel_select"
        )
    with col_mode_sel_res:
        view_type_res = st.radio(
            "Tipe Tampilan:",
            ["Tabel Interaktif (DataFrame)", "Peta Panas (Heatmap Grafis)", "NumPy Raw Array"],
            horizontal=True,
            key="res_view_type"
        )
    
    # Pengaturan Area Inspeksi Citra Hasil Resize
    st.markdown("##### 🔍 Pengaturan Area Inspeksi Matriks:")
    if new_h > 24 or new_w > 24:
        st.caption(f"Citra hasil resize berukuran **{new_w} × {new_h}** ({new_w * new_h:,} elemen). Anda dapat menggeser slider di bawah untuk memilih area yang ingin diinspeksi:")
        res_roi_col1, res_roi_col2 = st.columns(2)
        with res_roi_col1:
            row_range_res = st.slider("Rentang Baris (Baris Y):", 0, new_h, (0, min(16, new_h)), key="res_rows")
        with res_roi_col2:
            col_range_res = st.slider("Rentang Kolom (Kolom X):", 0, new_w, (0, min(16, new_w)), key="res_cols")
        
        show_all_res = st.checkbox("Tampilkan seluruh matriks tanpa pemotongan", value=False, key="res_full")
        if show_all_res:
            r_slice_res = slice(0, new_h)
            c_slice_res = slice(0, new_w)
        else:
            r_slice_res = slice(row_range_res[0], row_range_res[1])
            c_slice_res = slice(col_range_res[0], col_range_res[1])
    else:
        st.success(f"Ukuran citra hasil resize ({new_w} × {new_h}) sangat ringkas, ditampilkan penuh!")
        r_slice_res = slice(0, new_h)
        c_slice_res = slice(0, new_w)

    def render_matrix_view_res(mat, channel_name, colormap, view_mode):
        sub_mat = mat[r_slice_res, c_slice_res]
        total_cells = sub_mat.size
        st.markdown(f"**Matriks {channel_name} (Resize: {selected_interp_name})** — Dimensi: `{sub_mat.shape[0]} baris × {sub_mat.shape[1]} kolom` ({total_cells:,} elemen)")
        
        if view_mode == "Tabel Interaktif (DataFrame)":
            styled_df, is_styled = format_matrix_dataframe(sub_mat, colormap)
            if not is_styled and total_cells > 25000:
                st.info(f"💡 **Mode Cepat (Native Arrow) Aktif**: Matriks hasil resize berukuran besar ({sub_mat.shape[0]} × {sub_mat.shape[1]} = {total_cells:,} elemen). Tabel ditampilkan tanpa pewarnaan gradien CSS agar browser tetap cepat dan tidak lag. Anda dapat menggunakan slider rentang baris & kolom di atas jika ingin mengamati warna gradien pada area tertentu.")
            st.dataframe(styled_df, use_container_width=True)
        elif view_mode == "Peta Panas (Heatmap Grafis)":
            fig = plot_matrix_heatmap(sub_mat, title=f"Matriks Hasil Resize - {channel_name}", colormap="Reds" if colormap=="red" else "Greens" if colormap=="green" else "Blues" if colormap=="blue" else "viridis")
            st.pyplot(fig)
            plt.close(fig)
        elif view_mode == "NumPy Raw Array":
            if total_cells > 50000:
                st.info(f"💡 Matriks hasil resize berukuran besar ({total_cells:,} elemen). Menampilkan representasi array NumPy dengan pemangkasan otomatis.")
            arr_str = np.array2string(sub_mat, separator=', ', precision=0, threshold=1000, edgeitems=3)
            st.code(f"np.array(\n{arr_str}\n, dtype=np.uint8)", language="python")

    if target_channel_res == "Grayscale (2D: H × W)":
        render_matrix_view_res(resized_gray, "Grayscale", "gray", view_type_res)
    elif target_channel_res == "Red Channel (R)":
        render_matrix_view_res(resized_r, "Red Channel", "red", view_type_res)
    elif target_channel_res == "Green Channel (G)":
        render_matrix_view_res(resized_g, "Green Channel", "green", view_type_res)
    elif target_channel_res == "Blue Channel (B)":
        render_matrix_view_res(resized_b, "Blue Channel", "blue", view_type_res)
    elif target_channel_res == "Semua Channel Terpisah (R, G, B)":
        st.markdown("#### 🔴 Red Channel (R):")
        render_matrix_view_res(resized_r, "Red (R)", "red", view_type_res)
        st.markdown("#### 🟢 Green Channel (G):")
        render_matrix_view_res(resized_g, "Green (G)", "green", view_type_res)
        st.markdown("#### 🔵 Blue Channel (B):")
        render_matrix_view_res(resized_b, "Blue (B)", "blue", view_type_res)


# ------------------------------------------
# TAB 4: OPERASI TITIK (POINT OPERATIONS)
# ------------------------------------------
with tab4:
    st.subheader("Operasi Titik pada Citra Digital (Point Operations)")
    st.markdown("""
    **Operasi Titik** (*Point Processing*) adalah pemetaan intensitas di mana nilai piksel keluaran $g(x, y)$ pada koordinat $(x, y)$ **hanya bergantung** pada nilai piksel masukan $f(x, y)$ di koordinat tersebut tanpa dipengaruhi oleh piksel tetangganya:
    $$g(x, y) = T[f(x, y)]$$
    """)
    
    # Pemilihan Citra Masukan & Mode Warna
    col_pt_cfg1, col_pt_cfg2 = st.columns(2)
    with col_pt_cfg1:
        base_img_source = st.radio(
            "Gunakan Citra Acuan:",
            ["Citra Asli", "Citra Hasil Resize"],
            horizontal=True,
            key="pt_source_choice"
        )
    with col_pt_cfg2:
        base_color_mode = st.radio(
            "Format Citra:",
            ["Citra Berwarna (RGB)", "Citra Grayscale"],
            horizontal=True,
            key="pt_color_format"
        )
    
    # Tentukan input image
    raw_selected = input_image if base_img_source == "Citra Asli" else resized_image
    if base_color_mode == "Citra Grayscale":
        target_pt_input = cv2.cvtColor(raw_selected, cv2.COLOR_RGB2GRAY)
    else:
        target_pt_input = raw_selected
        
    st.markdown("---")
    
    # Sub-menu interaktif untuk Point Operations
    sub_pt1, sub_pt2, sub_pt3, sub_pt4 = st.tabs([
        "☀️ Kecerahan (Brightness)",
        "🌓 Kontras (Contrast)",
        "🎛️ Kombinasi Linear (α & β)",
        "📊 Ekualisasi Histogram (Histogram Equalization)"
    ])
    
    # --------------------------------------
    # 1. BRIGHTNESS (KECERAHAN)
    # --------------------------------------
    with sub_pt1:
        st.markdown("#### 1. Pengaturan Kecerahan (Brightness Adjustment)")
        st.markdown(r"""
        Operasi penambahan/pengurangan intensitas (*brightness*) dilakukan dengan menambahkan konstanta bias $\beta$ ke setiap piksel:
        $$g(x, y) = \text{clip}(f(x, y) + \beta, 0, 255)$$
        - Jika $\beta > 0$: Citra menjadi lebih terang.
        - Jika $\beta < 0$: Citra menjadi lebih gelap.
        - Nilai dibatasi (*clipping*) pada rentang $[0, 255]$ agar sesuai dengan format `uint8`.
        """)
        
        beta_val = st.slider("Nilai Bias Kecerahan (β):", min_value=-150, max_value=150, value=50, step=5, key="slider_beta")
        
        # Proses Kecerahan
        out_bright = cv2.convertScaleAbs(target_pt_input, alpha=1.0, beta=beta_val)
        
        # Tampilan Gambar Berdampingan
        b_c1, b_c2 = st.columns(2)
        with b_c1:
            st.image(target_pt_input, caption=f"Citra Sebelum (Bias β = 0)", use_column_width=True, clamp=True)
        with b_c2:
            st.image(out_bright, caption=f"Citra Sesudah (Bias β = {beta_val:+d})", use_column_width=True, clamp=True)
            
        # Histogram Komparasi
        st.markdown("##### 📈 Pergeseran Histogram Intensitas:")
        st.pyplot(plot_hist_cdf_comparison(target_pt_input, out_bright, "Sebelum (Normal)", f"Sesudah Brightness ({beta_val:+d})"))
        
        # Komparasi Nilai Matriks Piksel
        st.markdown("##### 🔢 Komparasi Nilai Matriks Piksel (Sebelum vs Sesudah):")
        st.caption(f"Perhatikan bagaimana setiap nilai piksel di bawah bertambah/berkurang sebesar **{beta_val:+d}** (dengan batas saturasi [0, 255]):")
        render_side_by_side_matrix_diff(target_pt_input, out_bright, "Sebelum", f"Sesudah ({beta_val:+d})")
        
    # --------------------------------------
    # 2. CONTRAST (KONTRAS)
    # --------------------------------------
    with sub_pt2:
        st.markdown("#### 2. Pengaturan Kontras (Contrast Adjustment)")
        st.markdown(r"""
        Kontras merepresentasikan sebaran perbedaan intensitas antara area terang dan area gelap.
        - **Metode Linear Scaling**:
          $$g(x, y) = \text{clip}(\alpha \cdot f(x, y), 0, 255)$$
        - **Metode Berpusat pada Midpoint 128 (Direkomendasikan)**:
          $$g(x, y) = \text{clip}(\alpha \cdot (f(x, y) - 128) + 128, 0, 255)$$
          Metode ini mempertahankan level abu-abu sedang (128) sebagai titik acuan tetap, sehingga kontras meregang secara simetris.
        """)
        
        cont_col1, cont_col2 = st.columns([2, 1])
        with cont_col1:
            alpha_val = st.slider("Faktor Pengali Kontras (α):", min_value=0.1, max_value=3.0, value=1.6, step=0.1, key="slider_alpha")
        with cont_col2:
            contrast_mode = st.radio("Metode Kontras:", ["Berpusat pada Midpoint 128", "Linear Scaling Murni"], key="contrast_mode_radio")
            
        # Proses Kontras
        if contrast_mode == "Berpusat pada Midpoint 128":
            res_c = alpha_val * (target_pt_input.astype(np.float32) - 128.0) + 128.0
            out_contrast = np.clip(res_c, 0, 255).astype(np.uint8)
        else:
            out_contrast = cv2.convertScaleAbs(target_pt_input, alpha=alpha_val, beta=0)
            
        # Tampilan Gambar Berdampingan
        c_v1, c_v2 = st.columns(2)
        with c_v1:
            st.image(target_pt_input, caption="Citra Sebelum (Kontras Normal α = 1.0)", use_column_width=True, clamp=True)
        with c_v2:
            st.image(out_contrast, caption=f"Citra Sesudah (Kontras α = {alpha_val:.1f}x)", use_column_width=True, clamp=True)
            
        # Histogram Komparasi
        st.markdown("##### 📈 Efek Peregangan / Kompresi Histogram:")
        st.pyplot(plot_hist_cdf_comparison(target_pt_input, out_contrast, "Sebelum (Normal)", f"Sesudah Kontras (α = {alpha_val})"))
        
        # Komparasi Nilai Matriks Piksel
        st.markdown("##### 🔢 Komparasi Nilai Matriks Piksel (Sebelum vs Sesudah):")
        render_side_by_side_matrix_diff(target_pt_input, out_contrast, "Sebelum (Normal)", f"Sesudah (α={alpha_val})")
        
    # --------------------------------------
    # 3. LINEAR COMBINATION (α & β)
    # --------------------------------------
    with sub_pt3:
        st.markdown("#### 3. Kombinasi Linear Kecerahan & Kontras (α & β)")
        st.markdown(r"""
        Transformasi linear gabungan adalah operasi titik yang paling sering digunakan:
        $$g(x, y) = \text{clip}(\alpha \cdot f(x, y) + \beta, 0, 255)$$
        - $\alpha$ mengontrol kecuraman gradien intensitas (**kontras**).
        - $\beta$ mengontrol pergeseran dasar intensitas (**kecerahan**).
        """)
        
        lin_c1, lin_c2 = st.columns(2)
        with lin_c1:
            lin_alpha = st.slider("Kontras (α):", min_value=0.1, max_value=3.0, value=1.4, step=0.1, key="lin_alpha")
        with lin_c2:
            lin_beta = st.slider("Kecerahan (β):", min_value=-100, max_value=100, value=25, step=5, key="lin_beta")
            
        out_linear = cv2.convertScaleAbs(target_pt_input, alpha=lin_alpha, beta=lin_beta)
        
        # Gambar
        lv1, lv2 = st.columns(2)
        with lv1:
            st.image(target_pt_input, caption="Citra Sebelum", use_column_width=True, clamp=True)
        with lv2:
            st.image(out_linear, caption=f"Citra Sesudah (α={lin_alpha}, β={lin_beta:+d})", use_column_width=True, clamp=True)
            
        st.markdown("##### 📈 Histogram Sebelum vs Sesudah:")
        st.pyplot(plot_hist_cdf_comparison(target_pt_input, out_linear, "Sebelum", f"Sesudah (α={lin_alpha}, β={lin_beta})"))
        
        st.markdown("##### 🔢 Komparasi Nilai Matriks Piksel:")
        render_side_by_side_matrix_diff(target_pt_input, out_linear, "Sebelum", f"Sesudah (α={lin_alpha}, β={lin_beta})")

    # --------------------------------------
    # 4. HISTOGRAM EQUALIZATION
    # --------------------------------------
    with sub_pt4:
        st.markdown("#### 4. Ekualisasi Histogram (Histogram Equalization)")
        st.markdown(r"""
        **Ekualisasi Histogram** adalah teknik peningkatan kontras (*contrast enhancement*) dengan meratakan distribusi probabilitas tingkat keabuan piksel.
        
        Secara matematis, transformasi ini menggunakan fungsi distribusi kumulatif (*Cumulative Distribution Function* / CDF):
        $$s_k = T(r_k) = \text{round}\left( \frac{L - 1}{M \cdot N} \sum_{j=0}^{k} n_j \right)$$
        - $L = 256$ (tingkat intensitas $[0, 255]$).
        - $M \cdot N$ adalah total jumlah piksel dalam citra.
        - $n_j$ adalah frekuensi kemunculan piksel dengan intensitas $j$.
        - **Ciri Khas**: Kurva CDF sesudah ekualisasi akan mendekati garis lurus diagonal $45^\circ$, menandakan intensitas terdistribusi merata ke seluruh rentang dinamis.
        """)
        
        he_cfg1, he_cfg2 = st.columns([2, 2])
        with he_cfg1:
            he_method = st.radio(
                "Pilih Algoritma Ekualisasi:",
                ["Global Histogram Equalization (cv2.equalizeHist)", "Adaptive CLAHE (Contrast Limited)"],
                key="he_method_radio"
            )
        
        if he_method == "Adaptive CLAHE (Contrast Limited)":
            with he_cfg2:
                clahe_clip = st.slider("CLAHE Clip Limit:", min_value=1.0, max_value=8.0, value=2.0, step=0.5, help="Membatasi kontras lokal untuk mencegah amplifikasi noise.")
                clahe_grid = st.select_slider("Ukuran Blok Grid (Tile Grid Size):", options=[4, 8, 16], value=8, format_func=lambda x: f"{x}×{x} piksel")
        else:
            with he_cfg2:
                st.info("💡 **Global Equalization** menghitung CDF dari seluruh citra sekaligus. Sangat cocok untuk citra dengan pencahayaan monoton (terlalu gelap/terlalu terang secara merata).")
                clahe_clip = 2.0
                clahe_grid = 8
                
        # Proses Ekualisasi Histogram
        if len(target_pt_input.shape) == 2:
            # Grayscale
            if he_method == "Adaptive CLAHE (Contrast Limited)":
                clahe_obj = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(clahe_grid, clahe_grid))
                out_he = clahe_obj.apply(target_pt_input)
            else:
                out_he = cv2.equalizeHist(target_pt_input)
        else:
            # RGB Color: Gunakan ruang warna YCrCb untuk mempertahankan saturasi warna
            if he_method == "Adaptive CLAHE (Contrast Limited)":
                lab = cv2.cvtColor(target_pt_input, cv2.COLOR_RGB2LAB)
                l, a, b_ch = cv2.split(lab)
                clahe_obj = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(clahe_grid, clahe_grid))
                cl = clahe_obj.apply(l)
                out_he = cv2.cvtColor(cv2.merge((cl, a, b_ch)), cv2.COLOR_LAB2RGB)
            else:
                ycrcb = cv2.cvtColor(target_pt_input, cv2.COLOR_RGB2YCrCb)
                y, cr, cb = cv2.split(ycrcb)
                eq_y = cv2.equalizeHist(y)
                out_he = cv2.cvtColor(cv2.merge((eq_y, cr, cb)), cv2.COLOR_YCrCb2RGB)
                
        # Tampilan Hasil Citra
        he_v1, he_v2 = st.columns(2)
        with he_v1:
            st.image(target_pt_input, caption="Citra Sebelum Ekualisasi", use_column_width=True, clamp=True)
        with he_v2:
            st.image(out_he, caption=f"Citra Sesudah Ekualisasi ({'CLAHE' if 'CLAHE' in he_method else 'Global'})", use_column_width=True, clamp=True)
            
        # Histogram & CDF Kurva
        st.markdown("##### 📈 Perbandingan Histogram & Kurva Kumulatif (CDF):")
        st.caption("Perhatikan bagaimana kurva CDF (garis oranye) sebelum ekualisasi melengkung tidak teratur, sedangkan sesudah ekualisasi menjadi garis linier naik yang merata:")
        st.pyplot(plot_hist_cdf_comparison(target_pt_input, out_he, "Sebelum Ekualisasi", "Sesudah Ekualisasi"))
        
        # Komparasi Matriks Piksel
        st.markdown("##### 🔢 Komparasi Nilai Matriks Piksel Sebelum vs Sesudah Ekualisasi:")
        render_side_by_side_matrix_diff(target_pt_input, out_he, "Sebelum", "Sesudah Equalize")


# ------------------------------------------
# TAB 5: ANALISIS & STATISTIK MATRIKS
# ------------------------------------------
with tab5:
    st.subheader("Analisis Statistik & Perbandingan Nilai Piksel")
    st.write("Melihat bagaimana operasi perbesaran/pengecilan citra dan algoritma interpolasi mempengaruhi distribusi nilai numerik piksel.")
    
    # Tabel Statistik Komparasi
    stat_data = {
        "Channel": ["Grayscale", "Red (R)", "Green (G)", "Blue (B)"],
        "Min (Asli)": [int(orig_gray.min()), int(orig_r.min()), int(orig_g.min()), int(orig_b.min())],
        "Min (Resize)": [int(resized_gray.min()), int(resized_r.min()), int(resized_g.min()), int(resized_b.min())],
        "Max (Asli)": [int(orig_gray.max()), int(orig_r.max()), int(orig_g.max()), int(orig_b.max())],
        "Max (Resize)": [int(resized_gray.max()), int(resized_r.max()), int(resized_g.max()), int(resized_b.max())],
        "Rata-rata (Asli)": [f"{orig_gray.mean():.2f}", f"{orig_r.mean():.2f}", f"{orig_g.mean():.2f}", f"{orig_b.mean():.2f}"],
        "Rata-rata (Resize)": [f"{resized_gray.mean():.2f}", f"{resized_r.mean():.2f}", f"{resized_g.mean():.2f}", f"{resized_b.mean():.2f}"],
        "Std Dev (Asli)": [f"{orig_gray.std():.2f}", f"{orig_r.std():.2f}", f"{orig_g.std():.2f}", f"{orig_b.std():.2f}"],
        "Std Dev (Resize)": [f"{resized_gray.std():.2f}", f"{resized_r.std():.2f}", f"{resized_g.std():.2f}", f"{resized_b.std():.2f}"]
    }
    st.table(pd.DataFrame(stat_data))
    
    st.markdown("---")
    st.markdown("#### 📈 Histogram Intensitas Piksel (Asli vs Resize):")
    
    fig_hist, (ax_h1, ax_h2) = plt.subplots(1, 2, figsize=(14, 4))
    
    # Histogram Asli
    ax_h1.hist(orig_r.ravel(), bins=32, color='red', alpha=0.5, label='Red')
    ax_h1.hist(orig_g.ravel(), bins=32, color='green', alpha=0.5, label='Green')
    ax_h1.hist(orig_b.ravel(), bins=32, color='blue', alpha=0.5, label='Blue')
    ax_h1.set_title(f"Histogram Citra Asli ({orig_w}x{orig_h})")
    ax_h1.set_xlabel("Nilai Piksel (0 - 255)")
    ax_h1.set_ylabel("Frekuensi")
    ax_h1.legend()
    ax_h1.grid(True, alpha=0.3)
    
    # Histogram Resize
    ax_h2.hist(resized_r.ravel(), bins=32, color='red', alpha=0.5, label='Red')
    ax_h2.hist(resized_g.ravel(), bins=32, color='green', alpha=0.5, label='Green')
    ax_h2.hist(resized_b.ravel(), bins=32, color='blue', alpha=0.5, label='Blue')
    ax_h2.set_title(f"Histogram Citra Resize ({new_w}x{new_h}) - {selected_interp_name}")
    ax_h2.set_xlabel("Nilai Piksel (0 - 255)")
    ax_h2.set_ylabel("Frekuensi")
    ax_h2.legend()
    ax_h2.grid(True, alpha=0.3)
    
    st.pyplot(fig_hist)
    plt.close(fig_hist)
    
    # Catatan Teori
    with st.expander("📚 Ringkasan Teori: Konsep Matriks Citra & Transformasi Spasial"):
        st.markdown(r"""
        1. **Representasi Citra Digital sebagai Matriks**:
           - Citra berukuran $M \times N$ dengan $C$ saluran dinyatakan sebagai fungsi matematis:
             $$f(x, y, c) \in \{0, 1, 2, \dots, 255\}$$
           - Indeks baris merepresentasikan koordinat vertikal $y$, sedangkan indeks kolom merepresentasikan koordinat horizontal $x$.
        2. **Transformasi Skala Spasial (Resize)**:
           - Secara matematis, pemetaan koordinat dari titik asli $(x, y)$ ke titik baru $(x', y')$ didefinisikan:
             $$\begin{bmatrix} x' \\ y' \end{bmatrix} = \begin{bmatrix} s_x & 0 \\ 0 & s_y \end{bmatrix} \begin{bmatrix} x \\ y \end{bmatrix}$$
             di mana $s_x = \frac{W_{baru}}{W_{asli}}$ dan $s_y = \frac{H_{baru}}{H_{asli}}$.
        3. **Peran Interpolasi**:
           - Pemetaan terbalik $(x, y) = (x'/s_x, y'/s_y)$ sering kali menghasilkan koordinat non-integer (bilangan riil).
           - **Nearest Neighbor**: Membulatkan ke koordinat bulat terdekat. Nilai piksel tidak berubah, tetapi terjadi efek tangga/tangga aliasing.
           - **Bilinear**: Menghitung rata-rata tertimbang dari 4 tetangga sekitar berdasarkan jarak euclidean fraksional.
           - **Bicubic**: Menggunakan fungsi pembobotan kubik 16 titik, mempertahankan kontur ketajaman tepi lebih baik.
        """)


# ------------------------------------------
# TAB 6: EKSPOR HASIL & MATRIKS
# ------------------------------------------
with tab6:
    st.subheader("Unduh Citra Hasil Resize & Data Matriks Numerik")
    st.write("Anda dapat mengekspor data hasil olahan citra untuk keperluan laporan praktikum, analisis tugas, atau pengolahan lebih lanjut.")
    
    exp_col1, exp_col2 = st.columns(2)
    
    with exp_col1:
        st.markdown("#### 1. Unduh File Citra Hasil Resize")
        # Encode gambar ke PNG buffer
        res_pil = Image.fromarray(resized_image)
        buf_img = io.BytesIO()
        res_pil.save(buf_img, format="PNG")
        byte_img = buf_img.getvalue()
        
        st.download_button(
            label=f"⬇️ Unduh Citra Resize ({new_w}x{new_h} PNG)",
            data=byte_img,
            file_name=f"{image_name}_resized_{new_w}x{new_h}.png",
            mime="image/png",
            use_container_width=True
        )
        
        # Opsi unduh Grayscale
        gray_pil = Image.fromarray(resized_gray)
        buf_gray = io.BytesIO()
        gray_pil.save(buf_gray, format="PNG")
        st.download_button(
            label=f"⬇️ Unduh Citra Grayscale Resize (PNG)",
            data=buf_gray.getvalue(),
            file_name=f"{image_name}_gray_resized_{new_w}x{new_h}.png",
            mime="image/png",
            use_container_width=True
        )

    with exp_col2:
        st.markdown("#### 2. Unduh Matriks Piksel (Format CSV)")
        # CSV Grayscale
        df_csv_gray = pd.DataFrame(resized_gray)
        csv_gray = df_csv_gray.to_csv(index=False, header=False).encode('utf-8')
        st.download_button(
            label="⬇️ Unduh Matriks Grayscale (CSV)",
            data=csv_gray,
            file_name=f"{image_name}_matriks_grayscale_{new_w}x{new_h}.csv",
            mime="text/csv",
            use_container_width=True
        )
        
        # CSV Red Channel
        df_csv_r = pd.DataFrame(resized_r)
        csv_r = df_csv_r.to_csv(index=False, header=False).encode('utf-8')
        st.download_button(
            label="⬇️ Unduh Matriks Red Channel (CSV)",
            data=csv_r,
            file_name=f"{image_name}_matriks_red_{new_w}x{new_h}.csv",
            mime="text/csv",
            use_container_width=True
        )
        
        # CSV Green Channel
        df_csv_g = pd.DataFrame(resized_g)
        csv_g = df_csv_g.to_csv(index=False, header=False).encode('utf-8')
        st.download_button(
            label="⬇️ Unduh Matriks Green Channel (CSV)",
            data=csv_g,
            file_name=f"{image_name}_matriks_green_{new_w}x{new_h}.csv",
            mime="text/csv",
            use_container_width=True
        )
        
        # CSV Blue Channel
        df_csv_b = pd.DataFrame(resized_b)
        csv_b = df_csv_b.to_csv(index=False, header=False).encode('utf-8')
        st.download_button(
            label="⬇️ Unduh Matriks Blue Channel (CSV)",
            data=csv_b,
            file_name=f"{image_name}_matriks_blue_{new_w}x{new_h}.csv",
            mime="text/csv",
            use_container_width=True
        )

st.markdown("---")
st.caption("Aplikasi Praktikum Pengolahan Citra Digital Pertemuan 3 • Menggunakan Python, OpenCV, NumPy, & Streamlit")
