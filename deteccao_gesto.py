import cv2
import mediapipe as mp

video = cv2.VideoCapture(0)

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
    if handsPoints: #Continua a parada mesmo sem ter mão na tela
        for points in handsPoints:
            #print(points)
            mpDraw.draw_landmarks(img, points, hand.HAND_CONNECTIONS)
            for id,cord in enumerate(points.landmark): #pega os ponto e coloca um numer]
                cx,cy = int(cord.x*w), int(cord.y*h)
                cv2.putText(img,str(id),(cx,cy+10), cv2.FONT_HERSHEY_SIMPLEX,0.5, (255,0,0), 2)


                
    cv2.imshow("Imagem",img)
    cv2.waitKey(1)

