# proc_de_img
o objetivo da aplicação é fazer o reconhecimento dos objetos
from keras.models import load_model
import cv2
import numpy as np


# Desativa a notação científica para maior clareza
np.set_printoptions(suppress=True)

# Carrega o modelo treinado
modelo = load_model("keras_model.h5", compile=False)

# Carrega os rótulos (nomes das pessoas) do arquivo labels.txt
with open("labels.txt", "r") as arquivo_labels:
    nomes_classes = arquivo_labels.read().splitlines()

# A CÂMERA pode ser 0 ou 1, dependendo da câmera padrão do seu computador
camera = cv2.VideoCapture(1)
#camera = cv2.VideoCapture('http://192.168.24.63:8080/video&#39;)

# Carrega o classificador Haar Cascade para detecção de faces
cascata_face = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    # Captura a imagem da câmera
    ret, imagem = camera.read()

    # Detecta faces na imagem
    cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    faces = cascata_face.detectMultiScale(cinza, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Itera pelas faces detectadas
    for (x, y, w, h) in faces:
        # Extrai a região do rosto da imagem
        regiao_rosto = imagem[y:y + h, x:x + w]

        # Redimensiona a imagem do rosto para o tamanho necessário para o modelo
        regiao_rosto = cv2.resize(regiao_rosto, (224, 224), interpolation=cv2.INTER_AREA)

        # Converte a imagem do rosto em um array numpy e aplica normalização
        array_rosto = np.asarray(regiao_rosto, dtype=np.float32).reshape(1, 224, 224, 3)
        array_rosto = (array_rosto / 127.5) - 1

        # Faz a previsão usando o modelo
        previsao = modelo.predict(array_rosto)
        indice = np.argmax(previsao)
        nome_classe = nomes_classes[indice]
        pontuacao_confianca = previsao[0][indice]

        # Desenha o retângulo na imagem
        cv2.rectangle(imagem, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Retângulo verde
        cv2.putText(imagem, nome_classe, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)  # Nome da pessoa

        # Imprime a previsão e a pontuação de confiança
        print("Pessoa:", nome_classe)
        print("Pontuação de Confiança:", str(np.round(pontuacao_confianca * 100))[:-2], "%")

    # Mostra a imagem na janela
    #cv2.imshow("Imagem da Webcam", imagem)
    redimensionada = cv2.resize(imagem,(600,400))
    cv2.imshow("Tela turma UFOPA", redimensionada) # display frame/image

    # Escuta o teclado para interrupção
    entrada_teclado = cv2.waitKey(1)

    # 27 é o código ASCII para a tecla Esc
    if entrada_teclado == 27:
        break

# Libera a câmera e fecha as janelas
camera.release()
cv2.destroyAllWindows()
