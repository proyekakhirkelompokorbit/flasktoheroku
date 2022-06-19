import numpy as np
import argparse
import time
import cv2
import os

confthres=0.5
nmsthres=0.1
yolo_path="./"
daftar_harga=[" harga = Rp.6000"," harga = Rp.6500"," harga = Rp.2000"]

def get_labels(labels_path):

    # Memuat label kelas COCO/CLASESS.NAMES model YOLO yang sudah kami latih
    # labelsPath = os.path.sep.join([yolo_path, "yolo_v3/coco.names"])

    lpath=os.path.sep.join([yolo_path, labels_path])
    LABELS = open(lpath).read().strip().split("\n")
    return LABELS

def get_colors(LABELS):

    # Menginisialisasi daftar warna untuk mewakili setiap kemungkinan label kelas

    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),dtype="uint8")
    return COLORS

def get_weights(weights_path):

    # Menyalurkan bobot YOLO dan konfigurasi model

    weightsPath = os.path.sep.join([yolo_path, weights_path])
    return weightsPath

def get_config(config_path):
    configPath = os.path.sep.join([yolo_path, config_path])
    return configPath

def load_model(configpath,weightspath):

    # Memuat detektor objek YOLO kami yang dilatih pada dataset coco/clasess.names (3 kelas)

    print("[INFO] YOLO Sedang mendeteksi dari disk...")
    net = cv2.dnn.readNetFromDarknet(configpath, weightspath)
    return net


def get_predection(image,net,LABELS,COLORS):
    (H, W) = image.shape[:2]

    # Menentukan hanya nama layer *output* yang kita butuhkan dari YOLO

    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # Membuat gumpalan dari gambar input dan kemudian lakukan penerusan
    # Setelah selesai mendeteksi objek YOLO, Kami memberikan kotak pembatas kami dan probabilitas terkait

    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
                                 swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    print(layerOutputs)
    end = time.time()

    # Menampilkan informasi waktu di YOLO

    print("[INFO] YOLO Berhasil Mendeteksi Objek dengan Membutuhkan Waktu {:.6f} seconds".format(end - start))

    # Inisialisasi daftar kotak pembatas yang terdeteksi, kepercayaan, dan ID kelas, masing-masing

    boxes = []
    confidences = []
    classIDs = []

    # Mengulangi setiap output layer
    for output in layerOutputs:
        # Mengulangi setiap deteksi
        for detection in output:
            # Mengekstrak ID kelas dan kepercayaan (yaitu, probabilitas) dari deteksi objek saat ini
            scores = detection[5:]
            # print(scores)
            classID = np.argmax(scores)
            # print(classID)
            confidence = scores[classID]

            # Menyaring prediksi yang lemah dengan memastikan yang terdeteksi
            # Probabilitas lebih besar dari probabilitas minimum

            if confidence > confthres:
                # Skala koordinat kotak pembatas kembali relatif terhadap ukuran gambar
                # Mengingat YOLO sebenarnya mengembalikan pusat (x, y)-koordinat pembatas kotak diikuti dengan lebar dan tinggi kotak
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # Menggunakan koordinat pusat (x, y) untuk menurunkan bagian atas dan sudut kiri kotak pembatas
                
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # Perbarui daftar koordinat kotak pembatas, kepercayaan, dan ID kelas
            
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    # Menerapkan penekanan non-maksima untuk menekan batas yang lemah dan tumpang tindih kotak
    
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, confthres,
                            nmsthres)

    # Merubah warna gambar agar tidak menjadi biru
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Memastikan setidaknya ada satu deteksi
    if len(idxs) > 0:
        # Mengulangi indeks yang kita simpan
        for i in idxs.flatten():
            # Mengekstrak koordinat kotak pembatas
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            # Menggambar persegi panjang kotak pembatas dan beri label pada gambar
            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            #text = "{}: {:.4f} {}".format(LABELS[classIDs[i]], confidences[i], daftar_harga[classIDs[i]])
            text = LABELS[classIDs[i]] + daftar_harga[classIDs[i]] # + str(confidences[i])
            print(boxes)
            print(classIDs)
            print(daftar_harga)
            cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,0.5, color, 2)
    return image

def runModel(image):
    # Memuat gambar input kami dan ambil dimensi spasialnya gambar = cv2.imread(img)
    labelsPath="./coco.names"
    cfgpath="cfg/yolov3.cfg"
    wpath="yolov3.weights"
    Lables=get_labels(labelsPath)
    CFG=get_config(cfgpath)
    Weights=get_weights(wpath)
    nets=load_model(CFG,Weights)
    Colors=get_colors(Lables)
    res=get_predection(image,nets,Lables,Colors)
    return res