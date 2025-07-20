# win_cam_sender.py
import cv2, socket, struct, subprocess
PORT = 5005

def wsl_ip():
    try:                                # NAT path
        ip = subprocess.check_output(
            ["wsl", "hostname", "-I"],  # capital “‑I”
            text=True).split()[0]
        return ip
    except Exception:
        return "127.0.0.1"              # mirrored fallback

DEST_IP = wsl_ip()
print(f"Streaming to {DEST_IP}:{PORT}")

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)        # default webcam

while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.resize(frame, (640, 480))       # keep packets < 65 kB
    _, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    data = buf.tobytes()
    if len(data) > 65000:                       # UDP hard‑limit
        continue
    sock.sendto(struct.pack("H", len(data)) + data, (DEST_IP, PORT))