import cv2
import tensorflow as tf
import numpy as np

RESIZE_IMAGE_HEIGHT = 80
RESIZE_IMAGE_WIDTH = 160

class Lanes():
  def __init__(self):
      self.recent_fit = []
      self.avg_fit = []


if __name__ == '__main__':
    trained_model = tf.keras.models.load_model('segnet_model_weights.h5')
    img = cv2.imread('./test_data/1494452385593783358/1.jpg')
    lanes = Lanes()

    resized_img = cv2.resize(img, dsize=(RESIZE_IMAGE_HEIGHT, RESIZE_IMAGE_WIDTH))
    resized_img = np.array(resized_img)
    resized_img = resized_img[None,:,:,:]

    # Make prediction with neural network (un-normalize value by multiplying by 255)
    prediction = trained_model.predict(resized_img)[0]
    prediction = np.where(prediction < 0.07, 0, prediction) # use 0.07 as our threshold for the lane prediction
    prediction = np.ceil(prediction)
    prediction = prediction*255.0

    lanes.recent_fit.append(prediction)

    if len(lanes.recent_fit) > 5:
        lanes.recent_fit = lanes.recent_fit[1:]

    # Calculate average detection
    lanes.avg_fit = np.mean(np.array([i for i in lanes.recent_fit]), axis = 0)

    # Generate fake R & B color dimensions, stack with G
    blanks = np.zeros_like(lanes.avg_fit).astype(np.uint8)
    lane_drawn = np.dstack((blanks, lanes.avg_fit, blanks))

    # Re-size to match the original image
    lane_image = cv2.resize(lane_drawn, dsize=(1280, 720))

    # Merge the lane drawing onto the original image
    result = cv2.addWeighted(img, 0.5, lane_image, 0.5, 0, dtype=cv2.CV_32F)


    cv2.imwrite('mask_img.jpg', result)
