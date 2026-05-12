import tensorflow as tf
from tensorflow.keras import layers, models


IMG_SIZE = (128, 128)
BATCH_SIZE = 32

train_data = tf.keras.preprocessing.image_dataset_from_directory(
    "dataset/train",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical"
)

test_data = tf.keras.preprocessing.image_dataset_from_directory(
    "dataset/test",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical"
)

# normalisation
train_data = train_data.map(lambda x, y: (x/255.0, y))
test_data = test_data.map(lambda x, y: (x/255.0, y))

# model
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Flatten(),

    layers.Dense(128, activation='relu'),
    layers.Dense(4, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# training
history = model.fit(
    train_data,
    epochs=5,
    validation_data=test_data
)

# save model
model.save("tumor_model.keras")

print("Training finished!")
