import cv2
import mediapipe as mp

# ========================
# SETUP MEDIAPIPE POSE
# ========================
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    static_image_mode=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ========================
# KAMERA
# ========================
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # flip biar seperti cermin (opsional)
    frame = cv2.flip(frame, 1)

    # convert ke RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    label = "Tidak terdeteksi"

    if results.pose_landmarks:
        # gambar skeleton
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

        # ========================
        # AMBIL LANDMARK PENTING
        # ========================
        lm = results.pose_landmarks.landmark

        left_hip = lm[mp_pose.PoseLandmark.LEFT_HIP]
        left_knee = lm[mp_pose.PoseLandmark.LEFT_KNEE]

        # nilai y makin besar = makin ke bawah
        selisih = left_knee.y - left_hip.y

        # ========================
        # LOGIKA BERDIRI / DUDUK
        # ========================
        if selisih > 0.25:
            label = "BERDIRI"
        else:
            label = "DUDUK"

    # ========================
    # TAMPILKAN TEKS
    # ========================
    cv2.putText(
        frame,
        label,
        (50, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.5,
        (0,255,0),
        3
    )

    cv2.imshow("Deteksi Berdiri/Duduk", frame)

    # tekan Q keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()