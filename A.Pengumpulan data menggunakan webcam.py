import os
import cv2
from time import sleep

DATA_DIR = 'C:\\Users\\hp\\Documents\\python\\buku\\pelatihan\\DATA'

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 2
datasheet_size = 100

cap = cv2.VideoCapture(0)

for j in range(number_of_classes):

    class_dir = os.path.join(DATA_DIR, str(j))
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print('Collecting data for class {}'.format(j))

    # Tunggu tekan Q
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        cv2.putText(frame, 'Ready? Press "Q" !',
                    (100, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.3,
                    (0, 255, 0),
                    3)

        cv2.imshow('frame', frame)

        if cv2.waitKey(15) & 0xFF == ord('q'):
            break

    counter = 0
    sleep(1)

    while counter < datasheet_size:
        ret, frame = cap.read()
        if not ret:
            continue

        cv2.imshow('frame', frame)
        cv2.waitKey(10)

        cv2.imwrite(os.path.join(class_dir, '{}.jpg'.format(counter)), frame)
        counter += 1

cap.release()
cv2.destroyAllWindows()