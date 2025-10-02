import cv2
import mediapipe as mp

# Mediapipe initialisieren
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Webcam öffnen
cap = cv2.VideoCapture(1)

with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Kein Bild von der Kamera!")
            break

        # Bild nach RGB konvertieren
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Gesichter erkennen
        results = face_detection.process(rgb_frame)

        # Ergebnisse zeichnen
        if results.detections:
            for detection in results.detections:
                mp_drawing.draw_detection(frame, detection)

        # Bild anzeigen
        cv2.imshow("Gesichtserkennung", frame)

        # Abbruch mit Taste "q"
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
