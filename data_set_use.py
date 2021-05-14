import tensorflow as tf
import cv2
import cv2

filepath = "dataset_face"

classe =['1', '2']

model = tf.keras.models.load_model(
    filepath, custom_objects=None, compile=True, options=None
)





def img_to_predict(frame):
    frame = cv2.resize(frame, (256, 256))
    img_array = tf.keras.preprocessing.image.img_to_array(frame)
    img_array = tf.expand_dims(img_array, 0)
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    print(score)
    score = tf.math.argmax(score)
    print(classe[score])

cv2.namedWindow("preview")
vc = cv2.VideoCapture(0)

if vc.isOpened():  # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False

while rval:
    cv2.imshow("preview", frame)

    rval, frame = vc.read()
    img_to_predict(frame)
    key = cv2.waitKey(20)
    if key == 27:  # exit on ESC
        break
cv2.destroyWindow("preview")