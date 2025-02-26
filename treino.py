import cv2
import mediapipe as mp
import numpy as np
import csv
import time

# Inicializar MediaPipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# Definir os gestos e arquivos CSV correspondentes
gestures = {
    "cima": "gesto_cima.csv",
    "baixo": "gesto_baixo.csv",
    "esquerda": "gesto_esquerda.csv",
    "direita": "gesto_direita.csv",
    "parar": "gesto_parar.csv"
}

# Criar arquivos CSV e escrever cabeçalhos
for gesto, arquivo in gestures.items():
    with open(arquivo, mode='w', newline='') as f:
        writer = csv.writer(f)
        header = ['x' + str(i) for i in range(21)] + ['y' + str(i) for i in range(21)] + ['z' + str(i) for i in range(21)] + ['label']
        writer.writerow(header)

# Captura de vídeo
cap = cv2.VideoCapture(0)
print("Pressione a tecla correspondente para gravar um gesto:")
print("Cima (C), Baixo (B), Esquerda (E), Direita (D), Parar (P), Pausar (T), Sair (Q)")

coletando = False
pausado = False
gesto_atual = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extrair pontos (x, y, z)
            pontos = []
            for lm in hand_landmarks.landmark:
                pontos.extend([lm.x, lm.y, lm.z])

            # Gravar os dados se estiver coletando e não pausado
            if coletando and not pausado and gesto_atual:
                with open(gestures[gesto_atual], mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(pontos + [gesto_atual])

    # Exibir vídeo
    status_text = f'Gravando: {gesto_atual if coletando else "Nenhum"}'
    if pausado:
        status_text = "PAUSADO"
    cv2.putText(frame, status_text, (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255) if coletando else (255, 255, 255), 2)
    cv2.imshow("Coleta de Gestos", frame)

    # Capturar tecla pressionada
    key = cv2.waitKey(10) & 0xFF
    if key == ord('c'):
        coletando = True
        gesto_atual = "cima"
    elif key == ord('b'):
        coletando = True
        gesto_atual = "baixo"
    elif key == ord('e'):
        coletando = True
        gesto_atual = "esquerda"
    elif key == ord('d'):
        coletando = True
        gesto_atual = "direita"
    elif key == ord('p'):
        coletando = True
        gesto_atual = "parar"
    elif key == ord('t'):
        pausado = not pausado  # Alternar entre pausado e ativo
    elif key == ord('q'):
        break
    elif key == 32:  # Barra de espaço para parar a gravação
        coletando = False
        gesto_atual = None

cap.release()
cv2.destroyAllWindows()
