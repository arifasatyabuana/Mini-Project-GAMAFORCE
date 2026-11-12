import cv2
import numpy as np
import threading
import time
import zmq
from pymavlink import mavutil
from ultralytics import YOLO

# ---------------------------------------------------------
# 1. SETUP ZEROMQ PUBLISHER (SENDER)
# ---------------------------------------------------------
context = zmq.Context()
zmq_socket = context.socket(zmq.PUB)
zmq_socket.bind("tcp://*:5555")  # Membuka port 5555 untuk Subscriber

# ---------------------------------------------------------
# 2. MUAT MODEL YOLO
# ---------------------------------------------------------
MODEL_PATH = "best.pt"  # Pastikan best.pt ada di folder yang sama
CONF_THRESHOLD = 0.60   # Ambang batas deteksi 60%

print(f"[YOLO] Loading model {MODEL_PATH}...")
model = YOLO(MODEL_PATH)
print("[YOLO] Model siap digunakan!")

# ---------------------------------------------------------
# 3. TELEMETRI MAVLINK (DENGAN REQUEST DATA STREAM)
# ---------------------------------------------------------
altitude = 0.0
groundspeed = 0.0
fc_connected = False

def read_mavlink():
    global altitude, groundspeed, fc_connected
    port = '/dev/ttyACM1'  # Sesuaikan /dev/ttyACM0 atau /dev/ttyACM1
    baudrate = 57600

    print(f"[MAVLink] Connecting to FC at {port}...")
    try:
        master = mavutil.mavlink_connection(port, baud=baudrate)
        master.wait_heartbeat()
        fc_connected = True
        print("[MAVLink] Connected!")

        # Request data stream otomatis dari FC
        master.mav.request_data_stream_send(
            master.target_system,
            master.target_component,
            mavutil.mavlink.MAV_DATA_STREAM_ALL,
            6,  # 6 Hz
            1   # Aktifkan
        )

        while True:
            msg = master.recv_match(type='VFR_HUD', blocking=True, timeout=1.0)
            if msg:
                groundspeed = msg.groundspeed
                altitude = msg.alt
            else:
                # Pancing ulang jika 1 detik tidak ada data
                master.mav.request_data_stream_send(
                    master.target_system,
                    master.target_component,
                    mavutil.mavlink.MAV_DATA_STREAM_ALL,
                    6,
                    1
                )
            time.sleep(0.01)

    except Exception as e:
        print(f"[MAVLink Error] {e}")
        fc_connected = False

mavlink_thread = threading.Thread(target=read_mavlink, daemon=True)
mavlink_thread.start()

# ---------------------------------------------------------
# 4. KAMERA & INFERENSI YOLO
# ---------------------------------------------------------
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50]  # Kompresi 50%

print("[Sender] Memulai pengiriman stream YOLO ke subscriber...")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # --- INFERENSI YOLO ---
        # imgsz=320 atau 480 agar inferensi di Raspberry Pi 5 kencang (20-30+ FPS)
        results = model.predict(source=frame, conf=CONF_THRESHOLD, imgsz=384, verbose=False)

        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Koordinat Bounding Box
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0])

                # Titik Tengah Ember (Centroid)
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                # Visualisasi
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 4, (255, 0, 0), -1)
                cv2.putText(frame, f"Red Bucket {conf:.2f}", (x1, max(y1 - 8, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # --- OSD TELEMETRI ---
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (480, 35), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

        status_color = (0, 255, 0) if fc_connected else (0, 0, 255)
        status_text = "FC: OK" if fc_connected else "FC: OFF"

        cv2.putText(frame, f"ALT: {altitude:.2f}m", (10, 23),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"SPD: {groundspeed:.2f}m/s", (160, 23),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, status_text, (380, 23),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 2)

        # --- KIRIM FRAME KE SUBSCRIBER VIA ZEROMQ ---
        _, buffer = cv2.imencode('.jpg', frame, encode_param)
        zmq_socket.send(buffer.tobytes())

        # Di Pi 5, YOLO berjalan sangat kencang sehingga sleep kecil cukup untuk kestabilan
        time.sleep(0.01)

except KeyboardInterrupt:
    print("[Sender] Dihentikan.")
finally:
    cap.release()
    zmq_socket.close()
    context.term()