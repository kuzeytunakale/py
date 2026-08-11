import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2

model = tf.keras.models.load_model("cnn_modelim.keras")

canvas = np.zeros((280, 280), dtype="uint8")

def ciz(event, x, y, flags, param):
    if flags & cv2.EVENT_FLAG_LBUTTON:
        cv2.circle(canvas, (x, y), 12, 255, -1)

cv2.namedWindow("Cizim")
cv2.setMouseCallback("Cizim", ciz)

while True:
    cv2.imshow("Cizim", canvas)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):  # 's' tuşuna basınca 28x28 kaydeder
        cv2.imwrite("cizim.png", cv2.resize(canvas, (28, 28)))
        print("28x28 cizim.png kaydedildi!")
        break
    elif key == ord('c'):  # 'c' tuşuna basınca temizler
        canvas[:] = 0

img = image.load_img('cizim.png',target_size=(28,28),color_mode='grayscale')

imgArray = image.img_to_array(img)
imgArray = imgArray.reshape(1,28,28,1)
imgArray = imgArray / 255.0

prediction = model.predict(imgArray)
prediced_digit = np.argmax(prediction)
print(prediced_digit)

ekran = np.zeros((280, 280, 3), dtype="uint8")
cv2.putText(ekran, str(prediced_digit), (100,200),
            cv2.FONT_HERSHEY_SIMPLEX, 6, (0, 255, 0), 10)

cv2.imshow("Tahmin Sonucu", ekran)
cv2.waitKey(0)
cv2.destroyAllWindows()
