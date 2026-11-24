import cv2

# ---------- 1. Muat model Haar Cascade bawaan OpenCV ----------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
if face_cascade.empty():
    raise RuntimeError("Gagal memuat file cascade.")

# ---------- 2. Baca citra ----------
nama_file = "WC2026.jpg"
img = cv2.imread(nama_file)
if img is None:
    raise FileNotFoundError(
        f"Citra '{nama_file}' tidak ditemukan. "
        "Pastikan berada di folder yang sama dengan skrip."
    )

# ---------- 3. Pra-pemrosesan: grayscale ----------
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.equalizeHist(gray)  # opsional: meratakan kontras

# ---------- 4. Deteksi wajah ----------
faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.1,   # pengurangan ukuran jendela tiap skala
    minNeighbors=5,    # minimal tetangga tumpang tindih agar valid
    minSize=(30, 30)   # ukuran wajah minimum yang dicari
)

# ---------- 5. Gambar bounding box ----------
for i, (x, y, w, h) in enumerate(faces, start=1):
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(img, f"Wajah {i}", (x, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

print("Jumlah wajah terdeteksi:", len(faces))

# ---------- 6. Simpan & tampilkan hasil ----------
cv2.imwrite("hasil_deteksi.jpg", img)
cv2.imshow("Hasil Deteksi Wajah", img)
cv2.waitKey(0)           # tekan tombol apa saja untuk menutup
cv2.destroyAllWindows()