# Jogo da cobrinha com gestos da mão

## Dependências

Python 3.10.x ```www.python.org/downloads/release/python-3100/```

Pygame 
```pip install pygame```

OpenCV
```pip install opencv-python```

MediaPipe
```pip install mediapipe```

Numpy
```pip install numpy```


## Para Jogar

Configuração do caminho da camera, pode variar entre 0,1,2...

Na linha 4 onde está localizado
```video = cv2.VideoCapture(0)```
Altere o valor entre os parenteses até a aplicação abrir

Exemplo do erro:
"File "C:\Users\Test\Desktop\Snake Game Hands\deteccao_gesto_teste.py", line 13, in <module>
    imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB) #Conversão da imagem
cv2.error: OpenCV(4.11.0) D:\a\opencv-python\opencv-python\opencv\modules\imgproc\src\color.cpp:199: error: (-215:Assertion failed) 
!_src.empty() in function 'cv::cvtColor'"

