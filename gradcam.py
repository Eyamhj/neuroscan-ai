import tensorflow as tf
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorflow.keras.utils import load_img, img_to_array

# -----------------------------
# 1) Load saved model
# -----------------------------
loaded_model = tf.keras.models.load_model("tumor_model.h5", compile=False)

# -----------------------------
# 2) Rebuild a fresh functional model and copy weights
#    هذا يحل مشاكل Keras 3 مع model.output / gradients
# -----------------------------
model = tf.keras.models.clone_model(loaded_model)
model(tf.zeros((1, 128, 128, 3)))  # build model
model.set_weights(loaded_model.get_weights())

# -----------------------------
# 3) Parameters
# -----------------------------
img_path = "dataset/test/glioma/Te-gl_10.jpg"
img_size = (128, 128)
class_names = ["glioma", "meningioma", "no_tumor", "pituitary"]
last_conv_layer_name = "conv2d_1"

# -----------------------------
# 4) Load and preprocess image
# -----------------------------
img = load_img(img_path, target_size=img_size)
img_array = img_to_array(img).astype("float32") / 255.0
img_array = np.expand_dims(img_array, axis=0)

# -----------------------------
# 5) Normal prediction
# -----------------------------
preds = model.predict(img_array, verbose=0)
pred_index = int(np.argmax(preds[0]))

print("Predicted class:", class_names[pred_index])
print("Confidence:", float(preds[0][pred_index]))

# -----------------------------
# 6) Build Grad-CAM model
# -----------------------------
last_conv_layer = model.get_layer(last_conv_layer_name)

grad_model = tf.keras.models.Model(
    inputs=model.inputs,
    outputs=[last_conv_layer.output, model.outputs[0]]
)

# -----------------------------
# 7) Compute gradients
# -----------------------------
img_tensor = tf.convert_to_tensor(img_array)

with tf.GradientTape() as tape:
    conv_outputs, predictions = grad_model(img_tensor, training=False)
    loss = predictions[:, pred_index]

grads = tape.gradient(loss, conv_outputs)

if grads is None:
    raise ValueError("Gradients are still None. Stop here and send me this message.")

# Global average pooling on gradients
pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

# Remove batch dimension
conv_outputs = conv_outputs[0]

# Weight feature maps
heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)

# ReLU + normalize
heatmap = tf.maximum(heatmap, 0)
max_val = tf.reduce_max(heatmap)
heatmap = heatmap / (max_val + tf.keras.backend.epsilon())
heatmap = heatmap.numpy()

# -----------------------------
# 8) Overlay on original image
# -----------------------------
img_cv = cv2.imread(img_path)
img_cv = cv2.resize(img_cv, img_size)

heatmap_resized = cv2.resize(heatmap, (img_cv.shape[1], img_cv.shape[0]))
heatmap_uint8 = np.uint8(255 * heatmap_resized)
heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

superimposed_img = cv2.addWeighted(img_cv, 0.6, heatmap_color, 0.4, 0)

# -----------------------------
# 9) Save result instead of showing
# -----------------------------
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
plt.title("Original")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(heatmap, cmap="jet")
plt.title("Heatmap")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
plt.title("Grad-CAM")
plt.axis("off")

plt.tight_layout()
plt.savefig("gradcam_result.png", dpi=200, bbox_inches="tight")
print("Grad-CAM saved as gradcam_result.png")