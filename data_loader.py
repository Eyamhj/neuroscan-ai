import tensorflow as tf

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

def normalize(img, label):
    return img / 255.0, label

train_data = train_data.map(normalize)
test_data = test_data.map(normalize)

print("Dataset loaded successfully")