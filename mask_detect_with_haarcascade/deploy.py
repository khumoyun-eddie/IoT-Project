import cv2 
import numpy as np

names={0: 'No Mask', 1: 'Mask'}

with_mask=np.load('with_mask.npy')
without_mask=np.load('without_mask.npy')
with_mask=with_mask.reshape(200,50*50*3)
without_mask=without_mask.reshape(200,50*50*3)
X=np.r_[with_mask, without_mask]

labels=np.zeros(X.shape[0])
labels[200:]=1.0
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split



from sklearn.decomposition import PCA
pca=PCA(n_components=3)

x_train, x_test, y_train, y_test = train_test_split(X, labels, test_size=0.25)

x_train=pca.fit_transform(x_train)
x_test=pca.transform(x_test)

svm=SVC()
svm.fit(x_train, y_train)

y_pred=svm.predict(x_test)
haar_data=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

capture = cv2.VideoCapture(0)  
data=  []
font=cv2.FONT_HERSHEY_COMPLEX
while True: 
    flag, img = capture.read() 
    if flag: 
        faces = haar_data.detectMultiScale(img) 
        for x,y,w,h in faces: 
            cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,255), 4) 
            face = img[y:y+h, x:x+w, : ]
            face = cv2.resize(face,(50,50))
            face=face.reshape(1,-1)
            face=pca.transform(face)
            pred=svm.predict(face)
            n=names[int(pred)]
            cv2.putText(img, n, (x,y), font, 1, (244,250,250), 2)
            print(n)
            
        cv2.imshow('result',img) 
        #27 - ASCII of Escape 
        if cv2.waitKey(2) == 27 or len(data) >=200: 
            break 
capture. release() 
cv2.destroyAllWindows() 

