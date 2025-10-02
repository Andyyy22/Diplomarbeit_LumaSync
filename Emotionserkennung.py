import cv2
import mediapipe as mp
from fer import FER
from collections import deque, Counter
import time

# -------------------------------
# Mediapipe initialisieren
# -------------------------------
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# -------------------------------
# FER (fertiges Modell)
# -------------------------------
detector = FER(mtcnn=True)  # mtcnn=True nutzt MTCNN zur Gesichtsdetektion

# -------------------------------
# Puffer für Glättung
# -------------------------------
emotion_buffer = deque(maxlen=30)
history = []

# -------------------------------
# Kamera Loop
# -------------------------------
cap = cv2.VideoCapture(1)
last_history_time = time.time()

with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("❌ Kein Bild von der Kamera!")
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(rgb_frame)
        aktuelle_emotion = "kein Gesicht"

        if results.detections:
            for detection in results.detections:
                # Bounding Box
                bbox = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                x = int(bbox.xmin * iw)
                y = int(bbox.ymin * ih)
                w = int(bbox.width * iw)
                h = int(bbox.height * ih)
                face_img = frame[y:y+h, x:x+w]

                # Emotion erkennen
                emotions = detector.detect_emotions(face_img)
                if emotions:
                    emotion, score = max(emotions[0]["emotions"].items(), key=lambda item: item[1])
                    if score < 0.5:
                        emotion = "neutral"
                    aktuelle_emotion = emotion
                    emotion_buffer.append(aktuelle_emotion)

                # Ergebnisse zeichnen
                mp_drawing.draw_detection(frame, detection)
                cv2.putText(frame, aktuelle_emotion, (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

        else:
            cv2.putText(frame, aktuelle_emotion, (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        # Alle 5 Sekunden → History Update
        if time.time() - last_history_time >= 5 and emotion_buffer:
            dominant = Counter(emotion_buffer).most_common(1)[0][0]
            history.append(dominant)
            print(f"[History] Dominante Emotion der letzten 5 Sekunden: {dominant}")
            last_history_time = time.time()

        cv2.imshow("Gesicht + Emotionserkennung", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()

# Finale Prozentwerte
counts = Counter(history)
total = len(history)
print("\n[Emotion Percentages]")
for e, v in counts.items():
    print(f"{e}: {v/total*100:.1f}%")
