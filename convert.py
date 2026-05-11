import tensorflow as tf

model = tf.keras.models.load_model("tumor_model.keras", compile=False)
model.save_weights("weights.weights.h5")

print("Weights saved successfully!")