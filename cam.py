import cv2
import numpy as np
import threading
import time
from flask import Flask, Response
from pymavlink import mavutil

# --- KONFIGURASI KAMERA & STREAMING ---
KAMERA_ID = 0
LEBAR = 640
TINGGI = 480
KUALITAS_JPEG = 80
PORT = 5000
LUAS_MINIMAL = 800

# Batas warna merah HSV
MERAH_1_BAWAH = np.array([0, 120, 70])
MERAH_1_ATAS = np.array([2, 255, 255])    # dulu 10. Oranye ikut terdeteksi -> turunkan; bucket merah tidak terdeteksi -> naikkan (maks 5)
MERAH_2_BAWAH = np.array([165, 120, 70])  # dulu 170
MERAH_2_ATAS = np.array([180, 255, 255])

# --- TELEMETRI FLIGHT CONTROLLER ---
altitude = 0.0
groundspeed = 0.0
fc_connected = False

def read_mavlink():
    global altitude, groundspeed, fc_connected
    port = '/dev/ttyACM0'  # Sesuaikan port FC Anda (/dev/ttyACM0 atau /dev/ttyAMA0)
    baudrate = 57600

    print(f"[MAVLink] Mencoba menghubungkan ke FC di {port}...")
    while True:
        try:
            master = mavutil.mavlink_connection(port, baud=baudrate)
            master.wait_heartbeat(timeout=5)
            fc_connected = True
            print("[MAVLink] Terhubung ke Flight Controller!")

            while True:
                msg = master.recv_match(type='VFR_HUD', blocking=True, timeout=1.0)
                if msg:
                    groundspeed = msg.groundspeed
                    altitude = msg.alt
                time.sleep(0.02)
        except Exception as e:
            fc_connected = False
            time.sleep(2)

threading.Thread(target=read_mavlink, daemon=True).start()

# --- DETEKSI HSV ---
def cari_bucket(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, MERAH_1_BAWAH, MERAH_1_ATAS)
    mask2 = cv2.inRange(hsv, MERAH_2_BAWAH, MERAH_2_ATAS)
    mask = cv2.bitwise_or(mask1, mask2)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    kontur, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not kontur:
        return None, mask

    terbesar = max(kontur, key=cv2.contourArea)
    if cv2.contourArea(terbesar) < LUAS_MINIMAL:
        return None, mask

    x, y, w, h = cv2.boundingRect(terbesar)
    return (x, y, w, h), mask

def draw_osd_telemetri(frame):
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (LEBAR, 45), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

    status_color = (0, 255, 0) if fc_connected else (0, 0, 255)
    status_text = "FC: CONNECTED" if fc_connected else "FC: DISCONNECTED"

    cv2.putText(frame, f"ALT: {altitude:.2f} m", (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f"SPEED: {groundspeed:.2f} m/s", (230, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, status_text, (460, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 2)

# --- FLASK STREAMING ---
app = Flask(__name__)
cam = cv2.VideoCapture(KAMERA_ID)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, LEBAR)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, TINGGI)
cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)

def generate_frames(tampilkan_mask=False):
    while True:
        success, frame = cam.read()
        if not success:
            time.sleep(0.01)
            continue

        kotak, mask = cari_bucket(frame)
        if kotak is not None:
            x, y, w, h = kotak
            cx, cy = x + w // 2, y + h // 2
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)
            cv2.putText(frame, f"DROP BUCKET ({cx},{cy})", (x, max(y - 10, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        draw_osd_telemetri(frame)
        output_frame = mask if tampilkan_mask else frame

        ret, buffer = cv2.imencode('.jpg', output_frame, [cv2.IMWRITE_JPEG_QUALITY, KUALITAS_JPEG])
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/')
def index():
    return Response(generate_frames(False), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/mask')
def mask_view():
    return Response(generate_frames(True), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=PORT, threaded=True)