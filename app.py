import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Atur konfigurasi halaman
st.set_page_config(page_title="Deteksi Tajwid PCD", layout="wide")

# ==========================================
# Fungsi-fungsi Pemrosesan Citra (dari Notebook)
# ==========================================

@st.cache_data
def process_tajwid_detection(image_bytes, sat_min, val_min, min_area, max_area_ratio, blur_size, kernel_size):
    # 1. Load image dari bytes Streamlit ke OpenCV format
    nparr = np.frombuffer(image_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img_bgr is None:
        return None, None, None, "Gagal memuat gambar."
        
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h_img, w_img = img_rgb.shape[:2]
    
    # 2. Buat mask non-latar
    hsv_blur = cv2.GaussianBlur(img_hsv, (blur_size, blur_size), 0)
    h, s, v = cv2.split(hsv_blur)
    mask_non_latar = ((s > sat_min) & (v > val_min)).astype(np.uint8) * 255

    # Gunakan kernel yang bisa diatur (membantu menyatukan huruf yang terputus)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask_non_latar = cv2.morphologyEx(mask_non_latar, cv2.MORPH_OPEN, kernel, iterations=1)
    mask_non_latar = cv2.morphologyEx(mask_non_latar, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # 3. Analisis Klaster Warna
    pixels_hsv = img_hsv[mask_non_latar > 0]
    if len(pixels_hsv) == 0:
        return img_rgb, img_rgb, [], "Tidak ada piksel berwarna yang terdeteksi (coba turunkan threshold Saturation/Value)."
        
    sample_maks = 20000
    if len(pixels_hsv) > sample_maks:
        idx = np.random.choice(len(pixels_hsv), sample_maks, replace=False)
        pixels_hsv = pixels_hsv[idx]
        
    data = pixels_hsv.astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.5)
    jumlah_kandidat = 7
    _, labels, centers = cv2.kmeans(data, jumlah_kandidat, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    
    labels = labels.flatten()
    kandidat_warna = []
    for i in range(jumlah_kandidat):
        count = int(np.sum(labels == i))
        center_hsv = centers[i].astype(np.uint8)
        center_rgb = cv2.cvtColor(np.uint8([[center_hsv]]), cv2.COLOR_HSV2RGB)[0, 0]
        kandidat_warna.append({
            "id": i, "jumlah_piksel": count, "hsv": center_hsv, "rgb": center_rgb
        })
        
    kandidat_warna.sort(key=lambda x: x["jumlah_piksel"], reverse=True)
    
    # 4. Gabungkan Klaster
    grup = []
    def jarak_hue(h1, h2):
        d = abs(int(h1) - int(h2))
        return min(d, 180 - d)

    ambang_h, ambang_s, ambang_v = 15, 60, 60
    for item in kandidat_warna:
        ditempatkan = False
        for g in grup:
            pusat = g[0]["hsv"]
            if (jarak_hue(item["hsv"][0], pusat[0]) <= ambang_h and 
                abs(int(item["hsv"][1]) - int(pusat[1])) <= ambang_s and 
                abs(int(item["hsv"][2]) - int(pusat[2])) <= ambang_v):
                g.append(item)
                ditempatkan = True
                break
        if not ditempatkan:
            grup.append([item])
            
    # Pemetaan Warna ke Tajwid Sederhana (berdasarkan Jarak RGB Euclidean)
    def get_tajwid_name(rgb):
        r, g, b = int(rgb[0]), int(rgb[1]), int(rgb[2])
        colors = {
            "Ikhfa (Biru)": (0, 112, 192),
            "Ikhfa Meem Saakin (Pink)": (220, 100, 150),
            "Qalqala (Merah)": (200, 30, 30),
            "Qalb (Ungu)": (100, 50, 150),
            "Idghaam (Hijau Tua)": (0, 100, 50),
            "Idghaam Meem Saakin (Hijau Muda)": (100, 200, 100),
            "Ghunna (Oranye)": (220, 100, 20)
        }
        min_dist = float('inf')
        best_name = "Tajwid"
        for name, color in colors.items():
            dist = (r - color[0])**2 + (g - color[1])**2 + (b - color[2])**2
            if dist < min_dist:
                min_dist = dist
                best_name = name
        return best_name

    kelas_stabil = []
    for idx, g in enumerate(grup, start=1):
        total = sum(item["jumlah_piksel"] for item in g)
        hsv_stack = np.array([item["hsv"] for item in g], dtype=np.float32)
        bobot = np.array([item["jumlah_piksel"] for item in g], dtype=np.float32)
        hsv_mean = np.average(hsv_stack, axis=0, weights=bobot).astype(np.uint8)
        rgb_mean = cv2.cvtColor(np.uint8([[hsv_mean]]), cv2.COLOR_HSV2RGB)[0, 0]
        
        kelas_stabil.append({
            "kelas": get_tajwid_name(rgb_mean),
            "jumlah_piksel": int(total),
            "hsv": hsv_mean,
            "rgb": rgb_mean,
        })
    kelas_stabil.sort(key=lambda x: x["jumlah_piksel"], reverse=True)

    # 5. Deteksi Kontur dan Bounding Box
    img_deteksi = img_rgb.copy()
    max_area = int(h_img * w_img * max_area_ratio) # Batas maksimal agar bingkai/border tidak terdeteksi
    
    for item in kelas_stabil:
        h0, s0, v0 = [int(x) for x in item["hsv"]]
        h_channel, s_channel, v_channel = cv2.split(img_hsv)
        
        delta_h = np.minimum(np.abs(h_channel.astype(int) - h0), 180 - np.abs(h_channel.astype(int) - h0))
        delta_s = np.abs(s_channel.astype(int) - s0)
        delta_v = np.abs(v_channel.astype(int) - v0)
        
        # Toleransi diringankan agar huruf lebih utuh
        mask_kelas = ((delta_h <= 12) & (delta_s <= 70) & (delta_v <= 70)).astype(np.uint8) * 255
        
        # Morphological Closing yang lebih kuat agar huruf/harakat yang terpisah menyatu jadi 1 kotak
        kernel_kelas = np.ones((kernel_size+2, kernel_size+2), np.uint8)
        mask_kelas = cv2.morphologyEx(mask_kelas, cv2.MORPH_CLOSE, kernel_kelas, iterations=3)
        mask_kelas = cv2.bitwise_and(mask_kelas, mask_non_latar)
        
        contours, _ = cv2.findContours(mask_kelas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        color_rgb = tuple(int(c) for c in item["rgb"])
        label_text = item['kelas']
        
        # Tentukan ukuran font yang dinamis berdasarkan lebar gambar (minimal 0.5)
        font_scale = max(0.5, w_img / 1800.0)
        thickness = max(1, int(font_scale * 2))

        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Filter noise (terlalu kecil) dan bingkai/border (terlalu besar)
            if min_area < area < max_area:
                x, y, w, h = cv2.boundingRect(cnt)
                
                # Filter rasio aspek (menghindari garis panjang horizontal/vertikal)
                aspect_ratio = w / float(h)
                if aspect_ratio > 15 or aspect_ratio < 0.1:
                    continue
                    
                cv2.rectangle(img_deteksi, (x, y), (x + w, y + h), color_rgb, 2)
                
                # Menghitung ukuran teks untuk membuat background
                (text_w, text_h), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                
                # Koordinat background teks (di atas kotak)
                bg_y1 = max(0, y - text_h - 10)
                bg_y2 = bg_y1 + text_h + 10
                
                # Menggambar kotak background dengan warna tajwid itu sendiri
                cv2.rectangle(img_deteksi, (x, bg_y1), (x + text_w + 4, bg_y2), color_rgb, -1)
                
                # Menentukan warna teks (putih atau hitam) agar selalu terbaca/kontras
                r_c, g_c, b_c = color_rgb
                brightness = (r_c * 0.299) + (g_c * 0.587) + (b_c * 0.114)
                text_color = (255, 255, 255) if brightness < 120 else (0, 0, 0)
                
                # Paksa warna hitam untuk tajwid yang warnanya terang secara visual
                if "Pink" in label_text or "Hijau Muda" in label_text or "Oranye" in label_text:
                    text_color = (0, 0, 0)
                
                # Menuliskan label tajwid di atas background
                cv2.putText(img_deteksi, label_text, (x + 2, bg_y2 - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness, cv2.LINE_AA)
                
    return img_rgb, img_deteksi, kelas_stabil, None

# ==========================================
# Desain Antarmuka Streamlit (Tema Islamic)
# ==========================================

custom_css = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Amiri:wght@400;700&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');

html, body, [class*="css"] { font-family: 'Poppins', sans-serif; }
h1, h2, h3 { font-family: 'Amiri', serif !important; }

/* Styling Tombol Utama (Warna Emas) */
div.stButton > button:first-child {
    background-color: #d4af37 !important; color: #0f4c3a !important; font-weight: bold; border-radius: 8px; border: none;
}
div.stButton > button:first-child:hover { background-color: #e6c55c !important; color: #0f4c3a !important; }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center;'>🕌 Aplikasi Deteksi Tajwid Al-Qur'an</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #e5dfc5;'>Mendeteksi penanda tajwid menggunakan <b>Segmentasi Warna HSV</b> & <b>K-Means Clustering</b>.</p>", unsafe_allow_html=True)
st.markdown("---")

st.sidebar.markdown("<h2>⚙️ Pengaturan Filter</h2>", unsafe_allow_html=True)
sat_min = st.sidebar.slider("Minimal Saturation (Warna)", 0, 255, 45, help="Semakin kecil, warna pudar ikut terdeteksi.")
val_min = st.sidebar.slider("Minimal Value (Kecerahan)", 0, 255, 40, help="Membedakan warna dari background krem/putih.")

st.sidebar.markdown("<hr style='border-color: #d4af37; opacity: 0.3;'>", unsafe_allow_html=True)
st.sidebar.markdown("<h2>🔧 Kerapian Box</h2>", unsafe_allow_html=True)
min_area = st.sidebar.slider("Minimal Luas Box (Noise)", 10, 1000, 150)
max_area_ratio = st.sidebar.slider("Maksimal Luas Box (%)", 0.01, 1.0, 0.1)
kernel_size = st.sidebar.slider("Kernel Morfologi", 1, 15, 5)
blur_size = st.sidebar.selectbox("Ukuran Blur (Gaussian)", [3, 5, 7], index=1)

st.sidebar.markdown("<hr style='border-color: #d4af37; opacity: 0.3;'>", unsafe_allow_html=True)
st.sidebar.markdown("<h2>📂 Unggah Gambar</h2>", unsafe_allow_html=True)
uploaded_file = st.sidebar.file_uploader("", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    st.sidebar.success("Gambar berhasil diunggah!")
    
    if st.button("Proses Gambar Sekarang", type="primary"):
        with st.spinner('Sedang memproses gambar... (Melakukan K-Means Clustering)'):
            bytes_data = uploaded_file.getvalue()
            img_asli, img_hasil, kelas_stabil, err_msg = process_tajwid_detection(
                bytes_data, sat_min, val_min, min_area, max_area_ratio, blur_size, kernel_size
            )
            
            if err_msg:
                st.error(err_msg)
            else:
                st.success("Selesai memproses gambar!")
                
                # Menampilkan Hasil
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Gambar Asli")
                    st.image(img_asli, use_container_width=True)
                with col2:
                    st.subheader("Hasil Deteksi (Bounding Box)")
                    st.image(img_hasil, use_container_width=True)
                    
                st.markdown("---")
                st.markdown("<h3 style='text-align: center;'>✨ Analisis Warna Tajwid ✨</h3>", unsafe_allow_html=True)
                st.write("")
                
                # Tampilkan palet warna
                if kelas_stabil:
                    cols = st.columns(len(kelas_stabil))
                    for i, item in enumerate(kelas_stabil):
                        with cols[i]:
                            r, g, b = item['rgb']
                            hex_color = f"#{r:02x}{g:02x}{b:02x}"
                            # Paksa semua teks di palet warna menjadi hitam pekat
                            text_color = 'black'
                            
                            # Modifikasi warna background khusus untuk Pink agar lebih jelas
                            if "Pink" in item['kelas']:
                                hex_color = "#FF69B4" # Hot Pink yang cerah dan jelas
                            st.markdown(
                                f"""
                                <div style="background-color: {hex_color}; padding: 15px; border-radius: 8px; text-align: center; border: 1px solid #d4af37;">
                                    <span style="color: {text_color} !important; font-weight: bold; font-size: 1.1em;">{item['kelas']}</span>
                                </div>
                                """, unsafe_allow_html=True
                            )
                            st.write(f"Luas Total: **{item['jumlah_piksel']} px**")

else:
    st.info("👈 Silakan unggah gambar dari sidebar di sebelah kiri untuk memulai demonstrasi aplikasi.")
