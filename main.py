import numpy as np
import cv2
import os


face_cascade= cv2.CascadeClassifier(r"classifier\haarcascade_frontalface_alt2.xml") #Kendim indirdiğim haarcascade classifierımı burada ekliyorum

train_klasor = r"newdatasets\football_stars_train"


def yuzTespiti(img_path):
    img = cv2.imread(img_path)
    
    if img is None:
        print("Resim okunamadi!", img_path)
        return
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #Haar cascade gri görüntüde çalıştığı için resmi griye çeviriyorum!!!
    faces = face_cascade.detectMultiScale(gray,scaleFactor=1.05,minNeighbors=3)

    for(x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w, y+h),(0,255,0),2)

for oyuncu_adi in os.listdir(train_klasor):
    oyuncu_klasoru = os.path.join(train_klasor, oyuncu_adi)
    
    if os.path.isdir(oyuncu_klasoru):
        print(f"\n--- {oyuncu_adi} klasoru isleniyor ---")
        
        for dosya in os.listdir(oyuncu_klasoru): #bu klasordeki
            tam_yol = os.path.join(oyuncu_klasoru, dosya)
            
            if tam_yol.lower().endswith((".jpg", ".jpeg", ".png")):
                yuzTespiti(tam_yol)
print("\n--- Islem tamamlandi ---")