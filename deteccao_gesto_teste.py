# Atividade realizada por Eduardo Augusto, Jorge Lucas e Igor Gerlach na Faculdade SENAI Felix Guisard 2025

import cv2
import mediapipe as mp

video = cv2.VideoCapture(1)

#configurações do mediapipe
hand = mp.solutions.hands 
Hand = hand.Hands(max_num_hands=1) #Quantidade de mãos
mpDraw = mp.solutions.drawing_utils #Desenha as juntas da mão

while True:
    check,img = video.read()
    imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB) #Conversão da imagem
    results = Hand.process(imgRGB) #Imagem Colorida
    handsPoints = results.multi_hand_landmarks #acha os ponto da mao
    h,w,_ = img.shape
    pontos = [] #array pra guardar cada posição de pontos
    if handsPoints: #Continua a parada mesmo sem ter mão na tela
        for points in handsPoints:
            #print(points)
            mpDraw.draw_landmarks(img, points, hand.HAND_CONNECTIONS)
            for id,cord in enumerate(points.landmark): #pega os ponto e coloca um numer]
                cx,cy = int(cord.x*w), int(cord.y*h)
                #Para ver os pontos da mão na imagem da camera:
                #cv2.putText(img,str(id),(cx,cy+10), cv2.FONT_HERSHEY_SIMPLEX,0.5, (255,0,0), 2)
                pontos.append((cx,cy)) #coordenada pra cada ponto

        #Teste contador de dedos
        dedos = [8,12,16,20] #pontos dos dedos
        contador= 0
        if points:
            for x in dedos:
                if pontos[x][1] < pontos[x-2][1]: #verifica se o valor da ponta do dedo tá menor que o ponto do inicio do dedo
                    contador +=1

        print(contador)

    cv2.imshow("Imagem",img)
    cv2.waitKey(1)

