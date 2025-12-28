Face Recognition with OpenCV & SVM
🇹🇷 Türkçe Açıklama
📌 Proje Tanımı

Bu proje, OpenCV, face_recognition ve Support Vector Machine (SVM) kullanarak yüz tanıma yapan bir Python uygulamasıdır.
Proje, ders kapsamında geliştirilmiş olup yüz algılama, özellik çıkarımı (embedding) ve makine öğrenmesi tabanlı sınıflandırma adımlarını uçtan uca göstermeyi amaçlamaktadır.

🎯 Projenin Amacı

Görüntülerden yüzleri otomatik olarak algılamak

Algılanan yüzleri kırpıp ön işleme tabi tutmak

Yüzlerden embedding (özellik vektörü) çıkarmak

SVM modeli ile yüz tanıma gerçekleştirmek

Model performansını accuracy ve classification report ile değerlendirmek

🛠️ Kullanılan Teknolojiler

Python

OpenCV (cv2) – Yüz ve göz algılama (Haar Cascade)

face_recognition – Yüz embedding çıkarımı (HOG tabanlı)

scikit-learn

SVC (SVM)

LabelEncoder

Accuracy & Classification Report

NumPy

🧠 Sistem Mimarisi ve Çalışma Mantığı
1️⃣ Yüz Kırpma (Preprocessing)

Haar Cascade kullanılarak yüz ve göz algılama yapılır.

En az bir göz tespit edilen yüzler kabul edilir.

Yüz bölgesi %15 padding ile kırpılır.

Kırpılmış yüzler ayrı klasörlere kaydedilir.

newdatasets/
├── football_stars_train
├── football_stars_test
├── cropped_faces_train
└── cropped_faces_test

2️⃣ Embedding (Özellik) Üretimi

Kırpılmış yüzler üzerinde tekrar yüz tespiti yapılır.

HOG tabanlı yüz algılama modeli kullanılır.

Her yüz için 128 boyutlu embedding vektörü çıkarılır.

3️⃣ Model Eğitimi

Etiketler LabelEncoder ile sayısallaştırılır.

Linear kernel kullanan SVM modeli eğitilir.

probability=True ile güven skoru (confidence) hesaplanır.

model = SVC(kernel="linear", probability=True)

4️⃣ Model Değerlendirme

Accuracy Score

Classification Report (precision, recall, f1-score)

5️⃣ Tek Görüntüden Tahmin

Görüntüden yüz algılanır.

SVM modelinden olasılık tahmini alınır.

Belirlenen eşik değerinin (threshold) altındaysa sonuç Unknown olarak işaretlenir.

Sonuç ekranda kutu ve etiket ile gösterilir.

▶️ Kurulum
git clone https://github.com/sytali0/Face_Detection.git
cd Face_Detection
pip install opencv-python face_recognition scikit-learn numpy

▶️ Çalıştırma
python main.py


Eğitim, test, değerlendirme ve tek görsel tahmini aynı dosya içinde yapılmaktadır.

📊 Veri Seti

Futbolculara ait yüz görüntülerinden oluşmaktadır.

Eğitim ve test setleri klasör bazlı ayrılmıştır.

Akademik ve eğitim amaçlı kullanılmıştır.

👥 Katkıda Bulunanlar

Görkem Özer – Yüz kırpma ve ön işleme

Seyit Ali Arslan – Embedding ve model eğitimi

Ahmet Kurt – Model değerlendirme

🔮 Geliştirme Önerileri

CNN / FaceNet tabanlı derin öğrenme modelleri

Gerçek zamanlı kamera tanıma

GUI arayüz

Modelin .pkl olarak kaydedilmesi

Cross-validation ve hiperparametre optimizasyonu

🇬🇧 English Version
📌 Project Description

This project is a Python-based face recognition application developed using OpenCV, face_recognition, and Support Vector Machines (SVM).
It was developed as a course project to demonstrate an end-to-end face recognition pipeline.

🎯 Objectives

Detect faces from images

Preprocess and crop detected faces

Extract facial embeddings

Train an SVM classifier

Evaluate model performance

Perform prediction on a single image

🛠️ Technologies Used

Python

OpenCV (Haar Cascade)

face_recognition (HOG-based embeddings)

scikit-learn (SVM, LabelEncoder)

NumPy

🧠 Workflow

Face & eye detection with Haar Cascades

Face cropping with padding

Embedding extraction

SVM model training

Performance evaluation

Single image prediction with confidence threshold

▶️ Run
python main.py

🎓 Project Type

Academic / Course Project

Computer Vision & Machine Learning
