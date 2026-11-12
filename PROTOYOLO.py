import cv2
import zmq
import numpy as np
import time

# --- KONFIGURASI JARINGAN ---
# Ganti dengan alamat IP Raspberry Pi kamu saat ini
RASPI_IP = "10.60.69.8"
PORT = 5555

# ---------------------------------------------------------
# 1. SETUP ZEROMQ SUB (SUBSCRIBER)
# ---------------------------------------------------------
context = zmq.Context()
zmq_socket = context.socket(zmq.SUB)

# PENTING: zmq.CONFLATE = 1 memastikan laptop HANYA menyimpan frame terakhir.
# Jika ada frame lama di antrean Wi-Fi, otomatis dibuang agar video ANTI-DELAY.
zmq_socket.setsockopt(zmq.CONFLATE, 1)

# Koneksi ke publisher Raspberry Pi
connection_str = f"tcp://{RASPI_IP}:{PORT}"
print(f"[Subscriber] Menghubungkan ke Raspberry Pi di {connection_str}...")
zmq_socket.connect(connection_str)
zmq_socket.setsockopt_string(zmq.SUBSCRIBE, "")

# Siapkan jendela GUI OpenCV di laptop
WINDOW_NAME = "GCS Telemetry & Vision Stream - TUF F15"
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, 960, 720)  # Tampilan jendela diperbesar agar enak dilihat

print("[Subscriber] Menunggu paket video dari Raspberry Pi...")
print("[Petunjuk] Tekan tombol 'q' pada jendela video untuk keluar.")

# Variabel pembantu FPS
prev_time = time.time()
fps = 0

try:
    while True:
        # Terima frame gambar yang dikirim oleh Raspberry Pi
        frame_bytes = zmq_socket.recv()
        
        # Decode byte buffer kembali ke citra OpenCV
        np_arr = np.frombuffer(frame_bytes, dtype=np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is not None:
            # Hitung FPS real-time penerimaan
            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
            prev_time = curr_time

            # Tampilkan indikator FPS lokal di pojok bawah kanan
            cv2.putText(frame, f"RX: {fps:.1f} FPS", (360, 345),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

            # Tampilkan frame ke layar desktop laptop
            cv2.imshow(WINDOW_NAME, frame)

        # Tekan 'q' untuk berhenti
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\n[Subscriber] Program dihentikan.")
finally:
    cv2.destroyAllWindows()
    zmq_socket.close()
    context.term()
    print("[Subscriber] Selesai.")