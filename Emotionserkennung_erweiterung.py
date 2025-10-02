import cv2
import mediapipe as mp
from fer import FER
from collections import Counter, defaultdict
import time

# Mediapipe initialisieren
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# FER Emotionserkennung
emotion_detector = FER(mtcnn=True)

# Webcam öffnen
cap = cv2.VideoCapture(0)

# Zeitmessung
face_missing_time = 0.0
body_missing_time = 0.0
emotion_times = defaultdict(float)  # kumulierte Zeit pro Emotion

prev_time = time.time()

with mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    refine_face_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as holistic:

    while cap.isOpened():
        current_time = time.time()
        frame_duration = current_time - prev_time
        prev_time = current_time

        ret, frame = cap.read()
        if not ret:
            print("❌ Kein Bild von der Kamera!")
            break

        # Mediapipe braucht RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb_frame)

        # Flags
        face_detected = results.face_landmarks is not None
        body_detected = results.pose_landmarks is not None

        # Fehlende Zeit akkumulieren
        if not face_detected:
            face_missing_time += frame_duration
        if not body_detected:
            body_missing_time += frame_duration

        if face_detected:
            # Emotion mit FER erkennen
            emotions = emotion_detector.detect_emotions(frame)

            if emotions:
                top_emotion, score = max(emotions[0]["emotions"].items(), key=lambda x: x[1])
                print(f"Gesicht erkannt → Emotion: {top_emotion} ({score:.2f})")
                emotion_times[top_emotion] += frame_duration
            else:
                print("Gesicht erkannt, aber keine Emotion erkannt.")

        elif body_detected:
            print("Gesicht nicht erkannt, aber Körper/Kopf sichtbar.")
        else:
            print("Weder Gesicht noch Körper erkannt.")

        # Ergebnisse anzeigen
        annotated_frame = frame.copy()
        if results.face_landmarks:
            mp_drawing.draw_landmarks(
                annotated_frame, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION
            )
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                annotated_frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS
            )
        cv2.imshow("Emotionserkennung mit FER + Mediapipe", annotated_frame)

        # Abbruch mit Taste "q"
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()

# Emotionen nach kumulativer Zeit sortieren
if emotion_times:
    sorted_emotions = sorted(emotion_times.items(), key=lambda x: x[1], reverse=True)
    print("\n--- Emotionen nach kumulativer Zeit ---")
    for idx, (emotion, duration) in enumerate(sorted_emotions, 1):
        print(f"{idx}. {emotion}: {duration:.2f} Sekunden")
else:
    print("\nKeine Emotionen erkannt")

# Dauer ausgeben
print(f"\nGesicht wurde {face_missing_time:.2f} Sekunden lang nicht erkannt.")
print(f"Körper/Kopf wurde {body_missing_time:.2f} Sekunden lang nicht erkannt.")
