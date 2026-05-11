import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import load_img, img_to_array

# تحميل المودال
model = tf.keras.models.load_model("tumor_model.h5")

# أسماء الكلاسات
class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

# image path
img_path = "dataset/test/glioma/Te-gl_5.jpg"  # بدّلها باسم صورة موجودة عندك

# تحميل الصورة
img = load_img(img_path, target_size=(128, 128))
img_array = img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# prediction
prediction = model.predict(img_array)
predicted_class = class_names[np.argmax(prediction)]
confidence = np.max(prediction)

print("Predicted class:", predicted_class)
print("Confidence:", confidence)
print("All probabilities:", prediction)