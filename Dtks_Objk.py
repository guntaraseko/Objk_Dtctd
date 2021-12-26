import numpy as np
import cv2

seko = cv2.dnn.readNet("weights/yolov3.weights", "cfg/yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = seko.getLayerNames()
output_layers = [layer_names[i - 1] for i in seko.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

gambar = cv2.imread("pic/gambar5.jpg") # tempat ganti gambar sesuai yang ada di folder pic
gambar = cv2.resize(gambar, None, fx=0.4, fy=0.4)
height, width, channels = gambar.shape

blob = cv2.dnn.blobFromImage(gambar, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

seko.setInput(blob)
outs = seko.forward(output_layers)

class_ids = []
confidences = []
boxes = []
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            a = int(detection[2] * width)
            b = int(detection[3] * height)
            c = int(center_x - a / 2)
            d = int(center_y - b / 2)

            boxes.append([c, d, a, b])
            confidences.append(float(confidence))
            class_ids.append(class_id)

indeks = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
print(indeks)
huruf = cv2.FONT_HERSHEY_SIMPLEX

for i in range(len(boxes)):
    if i in indeks:
        c, d, a, b = boxes[i]
        label = str(classes[class_ids[i]])
        color = colors[i]
        color = (0,255,255) # warna kotaknya bisa kita ubah sesuai keinginan kita -
        rectangle_bgr = (0, 255, 255) # dengan kode warna (R, G, B)
        cv2.rectangle(gambar, (c, d), (c + a, d + b), color, 2)
        cv2.putText(gambar, label, (c, d + 30), huruf, 1, color, 2)

cv2.imshow("Image", gambar)
cv2.waitKey(0)
cv2.destroyAllWindows()