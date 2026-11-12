import cv2
import zmq
import numpy as np

# Setup ZeroMQ Subscriber
context = zmq.Context()
zmq_socket = context.socket(zmq.SUB)
zmq_socket.connect("tcp://10.60.69.8:5555")  # IP Raspberry Pi kamu
zmq_socket.setsockopt_string(zmq.SUBSCRIBE, "")

print("[Subscriber] Menghubungkan ke Raspberry Pi di tcp://10.60.69.8:5555...")
cv2.namedWindow("Live Stream", cv2.WINDOW_NORMAL)

while True:
    try:
        # Terima frame dari Raspberry Pi
        frame_bytes = zmq_socket.recv()
        np_arr = np.frombuffer(frame_bytes, dtype=np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is not None:
            cv2.imshow("Live YOLO Stream - TUF F15", frame)

        # Tekan tombol 'q' untuk keluar
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    except Exception as e:
        print(f"Error: {e}")
        continue

cv2.destroyAllWindows()
zmq_socket.close()
context.term()