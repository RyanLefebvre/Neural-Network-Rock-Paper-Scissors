import cv2
import os
import sys

# Gathers images of rock, paper, scissors hand or none.
# Images are stored in the training_images subdirectory 
# and will be passed to the train_model.py file to train 
# the CNN
try:
    label_name = sys.argv[1]
    num_samples = int(sys.argv[2])
except:
    print("Arguments missing.")
    print(desc)
    exit(-1)
IMG_SAVE_PATH = 'image_data'
IMG_CLASS_PATH = os.path.join(IMG_SAVE_PATH, label_name)
try:
    os.mkdir(IMG_SAVE_PATH)
except FileExistsError:
    pass
try:
    os.mkdir(IMG_CLASS_PATH)
except FileExistsError:
    print("{} directory already exists.".format(IMG_CLASS_PATH))
    print("All images gathered will be saved along with existing items in this folder")
cap = cv2.VideoCapture(0)
cap.set(3,960)
cap.set(4,540)
start = False
count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        continue
    if count == num_samples:
        break
    cv2.rectangle(frame, (75, 75), (325, 325), (255,192,203), 2)
    if start:
        roi = frame[75:325, 75:325]
        save_path = os.path.join(IMG_CLASS_PATH, '{}.jpg'.format(count + 1))
        cv2.imwrite(save_path, roi)
        count += 1
    font = cv2.FONT_HERSHEY_COMPLEX
    cv2.putText(frame, "Collecting {}".format(count),
            (5, 50), font, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow("Collecting images here", frame)
    k = cv2.waitKey(10)
    if k == ord('a'):
        start = not start
    if k == ord('q'):
        break
print("\n{} image(s) saved to {}".format(count, IMG_CLASS_PATH))
cap.release()
cv2.destroyAllWindows()