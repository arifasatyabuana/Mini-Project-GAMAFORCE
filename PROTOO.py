import cv2
import numpy as np
import threading
import time
import zmq
from pymavlink import mavutil

# ---------------------------------------------------------
# 1. SETUP ZEROMQ PUBLISHER (OPTIMASI LATENSI)
# ---------------------------------------------------------
context = zmq.Context()
zmq_socket = context.socket(zmq.PUB)

# OPTIMASI LATENSI ZMQ 1: High Water Mark = 1 (Jangan menumpuk antrean frame di RAM)
zmq_socket.setsockopt(zmq.SNDHWM, 1)
zmq_socket.bind("tcp://*:5555")

# ---------------------------------------------------------
# 2. TELEMETRI MAVLINK
# ---------------------------------------------------------
altitude = 0.0
groundspeed = 0.0
fc_connected = False

def read_mavlink():
    global altitude, groundspeed, fc_connected
    port = '/dev/ttyACM1'
    baudrate = 57600

    print(f"[MAVLink] Connecting to FC at {port}...")
    try:
        master = mavutil.mavlink_connection(port, baud=baudrate)
        master.wait_heartbeat()
        fc_connected = True
        print("[MAVLink] Connected!")

        master.mav.request_data_stream_send(
            master.target_system,
            master.target_component,
            mavutil.mavlink.MAV_DATA_STREAM_ALL,
            6, 1
        )

        while True:
            msg = master.recv_match(type='VFR_HUD', blocking=True, timeout=1.0)
            if msg:
                groundspeed = msg.groundspeed
                altitude = msg.alt
            else:
                master.mav.request_data_stream_send(
                    master.target_system,
                    master.target_component,
                    mavutil.mavlink.MAV_DATA_STREAM_ALL,
                    6, 1
                )
            time.sleep(0.01)

    except Exception as e:
        print(f"[MAVLink Error] {e}")
        fc_connected = False

mavlink_thread = threading.Thread(target=read_mavlink, daemon=True)
mavlink_thread.start()

# ---------------------------------------------------------
# 3. KAMERA & PROSESING (OPTIMASI LATENSI)
# ---------------------------------------------------------
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

# OPTIMASI LATENSI KAMERA 1: Batasi internal buffer OpenCV menjadi 1 frame
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# Resolusi hemat bandwidth
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

if not cap.isOpened():
    print("[ERROR] Gagal membuka kamera pada /dev/video1!")
    exit(1)

# OPTIMASI KOMPRESI: Kompresi JPEG 40% (ukuran file lebih kecil = transfer Wi-Fi lebih cepat)
encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 40]

print("[Sender] Memulai pengiriman stream latensi rendah...")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Frame kamera tidak terbaca.")
            break

        # --- DETEKSI EMBER MERAH ---
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, np.array([0, 120, 70]), np.array([2, 255, 255]))
        mask2 = cv2.inRange(hsv, np.array([165, 120, 70]), np.array([180, 255, 255]))
        mask = cv2.bitwise_or(mask1, mask2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) > 800:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, "Red Bucket", (x, y - 10),
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

        # --- KIRIM FRAME KE LAPTOP ---
        _, buffer = cv2.imencode('.jpg', frame, encode_param)
        zmq_socket.send(buffer.tobytes())

        # Hilangkan time.sleep agar loop mengikuti FPS native kamera secara realtime

except KeyboardInterrupt:
    print("\n[Sender] Dihentikan.")
finally:
    cap.release()
    zmq_socket.close()
    context.term()