import cv2
import numpy as np
import threading
import time
import zmq
from pymavlink import mavutil

# ---------------------------------------------------------
# 1. SETUP ZEROMQ PUBLISHER (SENDER)
# ---------------------------------------------------------
context = zmq.Context()
zmq_socket = context.socket(zmq.PUB)
zmq_socket.bind("tcp://*:5555")

# ---------------------------------------------------------
# 2. TELEMETRI MAVLINK
# ---------------------------------------------------------
altitude = 0.0       # VFR_HUD.alt (hasil EKF, lambat)
groundspeed = 0.0
rel_alt = 0.0        # GLOBAL_POSITION_INT.relative_alt (m)
pressure = 0.0       # SCALED_PRESSURE.press_abs (hPa)
baro_alt = 0.0       # ketinggian relatif dihitung dari tekanan (m)
fc_connected = False

press_awal = None    # tekanan referensi saat program mulai
press_samples = []   # kumpulan sampel awal untuk rata-rata referensi


def request_message_interval(master, message_id, hz):
    """Minta FC mengirim pesan tertentu dengan frekuensi hz."""
    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
        0,
        message_id,
        int(1e6 / hz),
        0, 0, 0, 0, 0
    )


def request_all_streams(master):
    # Cara lama (tetap dipakai sebagai cadangan)
    master.mav.request_data_stream_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_DATA_STREAM_ALL,
        6, 1
    )
    # Cara baru: minta pesan spesifik
    request_message_interval(master, mavutil.mavlink.MAVLINK_MSG_ID_VFR_HUD, 10)
    request_message_interval(master, mavutil.mavlink.MAVLINK_MSG_ID_GLOBAL_POSITION_INT, 10)
    request_message_interval(master, mavutil.mavlink.MAVLINK_MSG_ID_SCALED_PRESSURE, 10)


def read_mavlink():
    global altitude, groundspeed, rel_alt, pressure, baro_alt
    global fc_connected, press_awal, press_samples

    port = '/dev/ttyACM1'  # Ganti ke /dev/ttyACM0 jika port FC berubah
    baudrate = 57600

    print(f"[MAVLink] Connecting to FC at {port}...")
    try:
        master = mavutil.mavlink_connection(port, baud=baudrate)
        master.wait_heartbeat()
        fc_connected = True
        print("[MAVLink] Connected!")

        request_all_streams(master)
        last_msg_time = time.time()

        while True:
            msg = master.recv_match(
                type=['VFR_HUD', 'GLOBAL_POSITION_INT', 'SCALED_PRESSURE'],
                blocking=True, timeout=1.0)

            if msg:
                last_msg_time = time.time()
                fc_connected = True
                t = msg.get_type()

                if t == 'VFR_HUD':
                    groundspeed = msg.groundspeed
                    altitude = msg.alt

                elif t == 'GLOBAL_POSITION_INT':
                    rel_alt = msg.relative_alt / 1000.0  # mm -> m

                elif t == 'SCALED_PRESSURE':
                    pressure = msg.press_abs  # hPa

                    # Ambil rata-rata 20 sampel pertama sebagai tekanan referensi
                    if press_awal is None:
                        press_samples.append(pressure)
                        if len(press_samples) >= 20:
                            press_awal = sum(press_samples) / len(press_samples)
                            print(f"[MAVLink] Tekanan referensi: {press_awal:.2f} hPa")
                    else:
                        baro_alt = 44330.0 * (1.0 - (pressure / press_awal) ** 0.1903)

                    print(f"alt={altitude:.2f} rel={rel_alt:.2f} "
                          f"baro={baro_alt:.2f} press={pressure:.2f}hPa")
            else:
                # Tidak ada data selama 1 detik: minta ulang stream
                request_all_streams(master)
                if time.time() - last_msg_time > 3.0:
                    fc_connected = False

            time.sleep(0.01)

    except Exception as e:
        print(f"[MAVLink Error] {e}")
        fc_connected = False


mavlink_thread = threading.Thread(target=read_mavlink, daemon=True)
mavlink_thread.start()

# ---------------------------------------------------------
# 3. KAMERA (INDEKS 1) & PROSESING
# ---------------------------------------------------------
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)  # Menggunakan indeks 1 (SUKSES)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

if not cap.isOpened():
    print("[ERROR] Gagal membuka kamera pada /dev/video1!")
    exit(1)

encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50]

print("[Sender] Memulai pengiriman stream ke subscriber...")

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

        # ALT memakai ketinggian dari barometer (responsif)
        cv2.putText(frame, f"ALT: {baro_alt:.2f}m", (10, 23),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"SPD: {groundspeed:.2f}m/s", (160, 23),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, status_text, (380, 23),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 2)

        # --- KIRIM FRAME KE LAPTOP ---
        _, buffer = cv2.imencode('.jpg', frame, encode_param)
        zmq_socket.send(buffer.tobytes())

        time.sleep(0.03)

except KeyboardInterrupt:
    print("\n[Sender] Dihentikan.")
finally:
    cap.release()
    zmq_socket.close()
    context.term()