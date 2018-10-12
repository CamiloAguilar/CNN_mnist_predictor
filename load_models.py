

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import cv2
import matplotlib.pyplot as plt
from keras.optimizers import RMSprop

mnist = pd.read_csv('./BASES/mnist_train.csv', header = None, sep = ',')
target = to_categorical(mnist.loc[:, 0])
mnist = mnist.loc[:, 1:]
X_train, X_test, y_train, y_test = train_test_split(mnist, target, test_size=0.2, random_state=42)
Y_train = [np.argmax(x) for x in y_train]

def imprime_num(img):
    return plt.imshow(np.array(img).reshape(28,28), cmap='gray')


## Escalamiento y PCA
#************************************************************************************************************
Sc = StandardScaler()
ScX = Sc.fit(X_train)
ScX_train = ScX.transform(X_train)
ScY_train = ScX.transform(X_test)
pca = PCA(0.7)
principal_fit = pca.fit(ScX_train)
principal_trans = principal_fit.transform(ScX_train)
#principal_test = principal_fit.transform(ScY_train)
#dfprin = pd.DataFrame(principal_trans)
#dfimag = pd.DataFrame(ScX_train)
#df_test = pd.DataFrame(principal_test)
#cant = (len(dfprin.columns)/len(dfimag.columns))*100

## Aplicación del KNN
#************************************************************************************************************
knn = KNeighborsClassifier(n_neighbors = 9)
knn_entrenado = knn.fit(principal_trans, Y_train)

def predict_knn(img, knn_entrenado):
    return knn_entrenado.predict(img)

## Regresión logística
#************************************************************************************************************
logit = LogisticRegression(solver = 'lbfgs')
logit_entrenado = logit.fit(principal_trans, Y_train)

def predict_logit(img, logit_entrenado):
    return logit_entrenado.predict(img)

## Red Neuronal
#************************************************************************************************************
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
import keras

input_shape = (28, 28, 1)
num_classes = 10

# Construímos modelo secuencial
model = Sequential()

# Definimos arquitectura de las capas
model.add(Conv2D(32, kernel_size=(5, 5), strides=(1,1), activation='relu', input_shape = input_shape))
model.add(Conv2D(32, kernel_size=(5, 5), activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

## Compilamos el modelo
optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
model.compile(optimizer = optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

## Cargamos los pesos
model.load_weights('saved_models/CNN_weights_MNIST_3.hdf5')

#************************************************************************************************************
## Realiza transformaciones de pca y escalamiento
def trans(img, ScX, principal_fit):
    img = ScX.transform(img)
    return principal_fit.transform(img)


## Realiza las predicciones
def get_image(img, knn_entrenado, logit, model):
    img2 = trans(img, ScX, principal_fit)
    knn = predict_knn(img2, knn_entrenado)
    logit = predict_logit(img2, logit_entrenado)
    #print(type(img))
    #print(img.shape)
    nnet = np.argmax(model.predict(np.array(img.iloc[0]).reshape(-1, 28, 28, 1)))
    return knn, logit, nnet


## Función para leer una imagen en un archivo
def lee_num(path):
    im = cv2.imread(path)
    blurred = cv2.cvtColor(np.uint8(im), cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(blurred, (5, 5), 0)
    (T, thresh) = cv2.threshold(blurred, 180, 220, cv2.THRESH_BINARY)
    ret, im_th = cv2.threshold(thresh, 100, 255, cv2.THRESH_BINARY_INV)
    im_th = cv2.resize(im_th, (28, 28))
    img = np.reshape(im_th, -1)
    #print("reescalamiento a 1: ", img.shape)
    res = np.transpose(pd.DataFrame(img))
    return res


## Función final para validar un número escrito a mano
def mi_numero(externo = False, path = None, base=None, fila=999, resultados = False):
    if externo:
        img = lee_num(path)
    else: 
        img = np.transpose(pd.DataFrame(base.iloc[fila]))
    
    res_knn, res_logit, nnet = get_image(img, knn_entrenado, logit, model)
    
    ## Escribe los resultados en pantalla
    if resultados:
        print("resultado del knn es :", res_knn)
        print("resultado regresión logística es : ", res_logit)
        print("resultado Red Neuronal es : ", nnet)
        
        #print("el número real es: ", real)
        imprime_num(img)
    
    return res_knn, res_logit, nnet



## Busca números en toda la imagen
#**********************************************************************************************************************


def find_nums(path):
    # Read the input image 
    im = cv2.imread(path)

    # Convert to grayscale and apply Gaussian filtering
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im_gray = cv2.GaussianBlur(im_gray, (5, 5), 0)

    # Threshold the image
    blurred = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(blurred, (5, 5), 0)
    T, thresh = cv2.threshold(blurred, 180, 220, cv2.THRESH_BINARY)

    ret, im_th = cv2.threshold(thresh, 120, 255, cv2.THRESH_BINARY_INV)

    # Find contours in the image
    _, ctrs, hier = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Get rectangles contains each contour
    rects = [cv2.boundingRect(ctr) for ctr in ctrs]

    # For each rectangular region, calculate HOG features and predict
    # the digit using Linear SVM.
    for rect in rects:
        # Draw the rectangles
        cv2.rectangle(im, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3) 
        
        # Make the rectangular region around the digit
        leng = int(rect[3] * 1)
        pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
        pt2 = int(rect[0] + rect[2] // 2 - leng // 2)
        roi = im_th[pt1:pt1+leng, pt2:pt2+leng]
        
        # Resize the image
        if roi.shape[0] == 0 or roi.shape[1] == 0:
            continue
        
        roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
        roi = roi.reshape(-1,28,28,1)
        
        nnet = np.argmax(model.predict(roi))
        #print(nnet)
        cv2.putText(im, str(int(nnet)), (rect[0], rect[1]),
                    cv2.FONT_HERSHEY_DUPLEX, 1.5, (255, 0, 0), 3)

    plt.imshow(im, cmap='gray')
    #plt.imshow(im_th)
    cv2.imwrite('./results/image.png',im)