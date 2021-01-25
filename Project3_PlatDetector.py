import cv2
import numpy as np

######################
nPlateCascade = cv2.CascadeClassifier("D:\Fauzan Alif\Learn Python\Resources/haarcascade_russian_plate_number.xml")
widthImg = 640
heightImg = 480
######################
#kita ingin mendeteksi plat nomor kendaraan menggunakan cascades
#cascades adalah semacam library untuk mendeteksi object tertentu, yg bisa didapatkan dari opencv sndiri
#cascades berupa file .xml

cap = cv2.VideoCapture(0)
cap.set (3,widthImg)
cap.set(4,heightImg)
cap.set(10,100)

count = 0
minArea= 500

while True:
    success, img = cap.read()

    imgGray = cv2.cvtColor(img ,cv2.COLOR_BGR2GRAY)

    #face cascade.detect multi scale (target gambar, luas besar kotaknya
    # , minNeighbor (dapat dilihat pada kite) )
    platDetector = nPlateCascade.detectMultiScale(imgGray, 1.1,4)
    for (x, y, w, h) in platDetector:
        area = w*h
        if area > minArea:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255,0,0), 3)
            cv2.putText(img, "Plat Kendaraan",(x,y-5), cv2.FONT_HERSHEY_COMPLEX, 2, (255,0,0), 2)
            imgRoi = img[y:y+h, x:x+w]
            
            cv2.imshow("Plat Detector", img)
    
    #sekarang, setelah kita memunculkan plate detection, kita ingin menyimpan hasil deteksi ke dalam folder
    if cv2.waitKey(1) & 0xFF == ord('q'): #bisa jg ord key nya diganti ke huruf s (karena saved)
        #kita ingin menyimpan hasilnya ke folder tertentu
        cv2.imwrite("D:\Fauzan Alif\Learn Python\Resources/Scanned/No_Plate_"+str(count)+".jpg", imgRoi)
        #kita ingin memberikan block warna pada text
        cv2.rectangle(img, (0,200), (640,300), (0,255,0), cv2.FILLED)
        #kita ingin memberikan kalimat notif klo udah disave
        cv2.putText(img, "Plat Numbernya telah disave", (150,265), cv2.FONT_HERSHEY_COMPLEX, 2, (255,0,0), 2)
        
        cv2.imshow("Plat telah disave",img)
        cv2.waitKey(500)
        count+=1

cv2.destroyAllWindows()