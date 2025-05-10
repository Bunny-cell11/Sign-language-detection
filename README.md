import cv2
import numpy as np
import math

cap = cv2.VideoCapture(0)

while True:
    ret, img = cap.read()
    if not ret:
        print("Failed to capture frame")
        continue
    if img is None:
        print("No frame captured")
        continue

    cv2.rectangle(img, (50, 50), (400, 400), (0, 255, 0), 2)  # Draw rectangle with thickness
    crop_img = img[50:400, 50:400]

    grey = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    value = (35, 35)
    blurred = cv2.GaussianBlur(grey, value, 0)

    _, thresh1 = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cv2.imshow('Thresholded', thresh1)

    try:
        contours, _ = cv2.findContours(thresh1.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        if contours:
            cnt = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(crop_img, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Draw rectangle with thickness

            hull = cv2.convexHull(cnt)
            drawing = np.zeros(crop_img.shape, np.uint8)
            cv2.drawContours(drawing, [cnt], 0, (0, 255, 0), 2)  # Draw contours with thickness
            cv2.drawContours(drawing, [hull], 0, (0, 0, 255), 2)  # Draw hull with thickness

            hull = cv2.convexHull(cnt, returnPoints=False)
            defects = cv2.convexityDefects(cnt, hull)
            count_defects = 0
            cv2.drawContours(thresh1, contours, -1, (0, 255, 0), 3)

            if defects is not None:
                for i in range(defects.shape[0]):
                    s, e, f, d = defects[i, 0]
                    start = tuple(cnt[s][0])
                    end = tuple(cnt[e][0])
                    far = tuple(cnt[f][0])
                    a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
                    b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
                    c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
                    angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 57

                    if angle <= 90:
                        count_defects += 1
                        cv2.circle(crop_img, far, 5, [0, 0, 255], -1)
                        cv2.line(crop_img, start, end, [0, 255, 0], 2)

                if count_defects == 1:
                    cv2.putText(img, "GESTURE ONE", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)
                elif count_defects == 2:
                    str_gesture = "GESTURE TWO"
                    cv2.putText(img, str_gesture, (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    # subprocess.Popen("sudo espeak GESTURE_TWO",shell=True)
                elif count_defects == 3:
                    cv2.putText(img, "GESTURE THREE", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)
                elif count_defects == 4:
                    cv2.putText(img, "GESTURE FOUR", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)
                else:
                    cv2.putText(img, "Hello World!!!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)
    except ValueError:
        print("No hand detected or contour found.")

    cv2.imshow('Gesture', img)
    all_img = np.hstack((drawing, crop_img))
    cv2.imshow('Contours', all_img)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(256, 256, 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(units=150, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(units=6, activation='softmax'))

model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   rotation_range=12.,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   zoom_range=0.15,
                                   horizontal_flip=True)
val_datagen = ImageDataGenerator(rescale=1. / 255)

training_set = train_datagen.flow_from_directory('Internship on AI\\Day 13 - Hand Gesture Recognition using DL\\HandGestureDataset\\train',
                                                 target_size=(256, 256),
                                                 color_mode='grayscale',
                                                 batch_size=8,
                                                 classes=['NONE', 'ONE', 'TWO', 'THREE', 'FOUR', 'FIVE'],
                                                 class_mode='categorical')

val_set = val_datagen.flow_from_directory('Internship on AI\\Day 13 - Hand Gesture Recognition using DL\\HandGestureDataset\\test',
                                          target_size=(256, 256),
                                          color_mode='grayscale',
                                          batch_size=8,
                                          classes=['NONE', 'ONE', 'TWO', 'THREE', 'FOUR', 'FIVE'],
                                          class_mode='categorical')

callback_list = [
    EarlyStopping(monitor='val_loss', patience=10),
    ModelCheckpoint(filepath="model.h5", monitor='val_loss', save_best_only=True, verbose=1)
]

model.fit(training_set,
          steps_per_epoch=training_set.samples // training_set.batch_size,
          epochs=25,
          validation_data=val_set,
          validation_steps=val_set.samples // val_set.batch_size,
          callbacks=callback_list)

from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np
import os

model = load_model('model.h5')
print("Model Loaded Successfully")

def classify(img_file):
    img_name = img_file
    try:
        test_image = image.load_img(img_name, target_size=(256, 256), grayscale=True)
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        result = model.predict(test_image)
        arr = np.array(result[0])
        print(f"Probabilities for {img_name}: {arr}")
        max_prob_index = np.argmax(arr)
        classes = ["NONE", "ONE", "TWO", "THREE", "FOUR", "FIVE"]
        prediction = classes[max_prob_index]
        print(f"Prediction for {img_name}: {prediction}")
    except Exception as e:
        print(f"Error processing {img_name}: {e}")

path = 'Internship on AI/Day 13 - Hand Gesture Recognition using DL/HandGestureDataset/val/TWO'
files = []
# r=root, d=directories, f = files
for r, d, f in os.walk(path):
    for file in f:
        if '.png' in file:
            files.append(os.path.join(r, file))

for f in files:
    classify(f)
    print('\n')# Sign-language-detection
