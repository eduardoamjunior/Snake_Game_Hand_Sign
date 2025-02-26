import cv2
import mediapipe as mp

# Carregar o modelo treinado
model = keras.models.load_model("modelo_gestos.h5")

# Inicializar MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# Captura de vídeo
cap = cv2.VideoCapture(0)
scaler = StandardScaler()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            pontos = []
            for lm in hand_landmarks.landmark:
                pontos.extend([lm.x, lm.y, lm.z])

            # Normalizar os dados e prever
            pontos = np.array(pontos).reshape(1, -1)
            pontos = scaler.transform(pontos)  # Normalizar
            prediction = model.predict(pontos)
            gesture_index = np.argmax(prediction)
            gesture_name = [key for key, value in gestures.items() if value == gesture_index][0]

            cv2.putText(frame, f"Gesto: {gesture_name}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Reconhecimento de Gestos", frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
