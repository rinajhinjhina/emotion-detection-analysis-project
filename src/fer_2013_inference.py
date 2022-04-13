from fer import FER
import cv2
import pprint
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')

print(gpus)

pp = pprint.PrettyPrinter(indent=4)


img = cv2.imread("data/processed/images/1_0.jpg")
detector = FER()
pp.pprint(detector.detect_emotions(img, ))