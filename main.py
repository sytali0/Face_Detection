import numpy as np
import cv2
import os
import face_recognition
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

# =========================================================
# GÖRKEM ÖZER
# =========================================================

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade  = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")


train_klasor = r"newdatasets\football_stars_train"
test_klasor  = r"newdatasets\football_stars_test"

cropped_train = r"newdatasets\cropped_faces_train"
cropped_test  = r"newdatasets\cropped_faces_test"


def yuzleri_kirp(giris_klasor, cikti_klasor):
    if not os.path.exists(cikti_klasor):
        os.makedirs(cikti_klasor)

    for oyuncu_adi in os.listdir(giris_klasor):
        oyuncu_klasoru = os.path.join(giris_klasor, oyuncu_adi)
        if not os.path.isdir(oyuncu_klasoru):
            continue

        hedef_yol = os.path.join(cikti_klasor, oyuncu_adi)
        if not os.path.exists(hedef_yol):
            os.makedirs(hedef_yol)

        print(f"--- {oyuncu_adi} işleniyor ---")
        
        for dosya in os.listdir(oyuncu_klasoru):
            tam_yol = os.path.join(oyuncu_klasoru, dosya)
            if not tam_yol.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
                
            img = cv2.imread(tam_yol)
            if img is None: continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Parametreler daha stabil tespit için güncellendi
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

            for (x, y, w, h) in faces:
                face_roi = gray[y:y+h, x:x+w]
                gozler = eye_cascade.detectMultiScale(face_roi)

                # En az 1-2 göz tespiti doğruluğu artırır
                if len(gozler) >= 1:
                    # Yüzü %15 geniş kırp (Padding). 
                    # Bu, face_recognition'ın yüz kenarlarını algılaması için kritiktir.
                    offset = int(w * 0.15)
                    y1, y2 = max(0, y-offset), min(img.shape[0], y+h+offset)
                    x1, x2 = max(0, x-offset), min(img.shape[1], x+w+offset)
                    
                    yuz_kesit = img[y1:y2, x1:x2]
                    cv2.imwrite(os.path.join(hedef_yol, "yuz_" + dosya), yuz_kesit)
                    break

    print(f"--- {cikti_klasor} tamamlandı ---")


yuzleri_kirp(train_klasor, cropped_train)
yuzleri_kirp(test_klasor,  cropped_test)

# =========================================================
#   Seyit Ali Arslan
# =========================================================
def embedding_uret(root_dir):
    X, y = [], []

    for label in os.listdir(root_dir):
        class_dir = os.path.join(root_dir, label)
        if not os.path.isdir(class_dir):
            continue

        for dosya in os.listdir(class_dir):
            path = os.path.join(class_dir, dosya)
            img_bgr = cv2.imread(path)
            if img_bgr is None: continue

            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            
            # Kırpılmış resimde tekrar yüz bul (HOG modeli hızlıdır)
            locs = face_recognition.face_locations(img_rgb, model="hog")
            encs = face_recognition.face_encodings(img_rgb, locs)

            if len(encs) > 0:
                X.append(encs[0])
                y.append(label)

    return np.array(X), np.array(y)


X_train, y_train = embedding_uret(cropped_train)
X_test,  y_test  = embedding_uret(cropped_test)

if len(X_train) == 0 or len(X_test) == 0:
    print("HATA: Yüz embeddingleri oluşturulamadı. Dataset yollarını kontrol edin.")
else:
    # Test setinde, train setinde olmayan bir sınıf varsa temizle
    train_labels = set(y_train)
    mask = np.array([lbl in train_labels for lbl in y_test])
    X_test, y_test = X_test[mask], y_test[mask]

    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc  = le.transform(y_test)

    # SVM Model - Probability=True önemli
    model = SVC(kernel="linear", probability=True)
    model.fit(X_train, y_train_enc)
    
# =========================================================
# Ahmet Kurt
# =========================================================

    pred = model.predict(X_test)
    print(f"\nAccuracy Score: {accuracy_score(y_test_enc, pred):.4f}")
    
    unique_labels = np.unique(y_test_enc)
    target_names = le.inverse_transform(unique_labels)
    print(classification_report(y_test_enc, pred, target_names=target_names))


def tek_fotograftan_tahmin(img_path, threshold=0.20):
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        print("Resim bulunamadı!")
        return

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(img_rgb, model="hog")
    encodings = face_recognition.face_encodings(img_rgb, face_locations)

    if not face_locations:
        print("Yüz bulunamadı!")

    for (top, right, bottom, left), enc in zip(face_locations, encodings):
        probs = model.predict_proba([enc])[0]
        best_idx = np.argmax(probs)
        confidence = probs[best_idx]

        if confidence < threshold:
            name = "Unknown"
        else:
            name = le.inverse_transform([best_idx])[0]

        print(f"Tahmin: {name} | Güven: %{confidence*100:.2f}")

        # Görselleştirme
        cv2.rectangle(img_bgr, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(img_bgr, f"{name} {confidence:.2f}", (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Tahmin Penceresi", img_bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


test_path = r"C:\Users\seyit\Desktop\Face_Detection\newdatasets\football_stars_test\Antony/antony.jpg"
tek_fotograftan_tahmin(test_path)