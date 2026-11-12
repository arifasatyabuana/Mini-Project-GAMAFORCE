import cv2
import numpy as np
import threading
import time
from flask import Flask, Response
from pymavlink import mavutil

# ---------------------------------------------------------
# 1. VARIABEL GLOBAL TELEMETRI
# ---------------------------------------------------------
altitude = 0.0
groundspeed = 0.0
fc_connected = False

# ---------------------------------------------------------
# 2. THREAD PEMBACAAN MAVLINK (FLIGHT CONTROLLER)
# ---------------------------------------------------------
def read_mavlink():
    global altitude, groundspeed, fc_connected
    port = '/dev/ttyACM1'  # Sesuaikan dengan port serial FC Anda
    baudrate = 57600

    print(f"[MAVLink] Connecting to FC at {port}...")
    try:
        master = mavutil.mavlink_connection(port, baud=baudrate)
        master.wait_heartbeat()
        fc_connected = True
        print("[MAVLink] Connected to Flight Controller!")

        while True:
            # Mengambil data VFR_HUD (Groundspeed & Altitude)
            msg = master.recv_match(type='VFR_HUD', blocking=True, timeout=1.0)
            if msg:
                groundspeed = msg.groundspeed
                altitude = msg.alt
            time.sleep(0.01)

    except Exception as e:
        print(f"[MAVLink Error] Connection failed: {e}")
        fc_connected = False

# Jalankan MAVLink di latar belakang (background thread)
mavlink_thread = threading.Thread(target=read_mavlink, daemon=True)
mavlink_thread.start()

# ---------------------------------------------------------
# 3. WEBCAM & STREAMING FLASK (DENGAN DETEKSI & OPTIMASI)
# ---------------------------------------------------------
app = Flask(__name__)
cap = cv2.VideoCapture(0)

# OPTIMASI 1: Set Resolusi Kamera Rendah (480x360 agar streaming lancar)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

def generate_frames():
    global altitude, groundspeed, fc_connected
    
    # OPTIMASI 2: Kompresi JPEG diset ke 50% (Hemat bandwidth Wi-Fi)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50]

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # --- A. DETEKSI EMBER MERAH (HSV FILTER) ---
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Rentang warna merah di HSV
        lower_red1 = np.array([0, 120, 70])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 120, 70])
        upper_red2 = np.array([180, 255, 255])

        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)

        # Morphological filter untuk menghilangkan noise bintik merah
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # Cari kontur & gambar Bounding Box
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) > 800:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, "Red Bucket", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # --- B. OVERLAY TELEMETRI (OSD) ---
        # Bar semi-transparan di bagian atas
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (480, 35), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

        status_color = (0, 255, 0) if fc_connected else (0, 0, 255)
        status_text = "FC: OK" if fc_connected else "FC: OFF"

        # Tampilkan teks telemetri
        cv2.putText(frame, f"ALT: {altitude:.2f}m", (10, 23),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.putText(frame, f"SPD: {groundspeed:.2f}m/s", (160, 23),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.putText(frame, status_text, (380, 23),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 2)

        # --- C. ENCODE KE JPEG & STREAM ---
        ret, buffer = cv2.imencode('.jpg', frame, encode_param)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        # OPTIMASI 3: Pembatasan Frame Rate (Maksimal ~25 FPS agar tidak lag)
        time.sleep(0.04)

@app.route('/')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)