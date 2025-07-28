import os, numpy as np, tensorflow as tf
from sklearn.metrics import classification_report

# 1) Load model
model = tf.keras.models.load_model("liftoff_image_classifier.h5")

# 2) Point this at your val images folder
val_dir = "data/val"
classes = ["not_liftoff","liftoff"]

y_true, y_pred = [], []
for label_idx, label in enumerate(classes):
    folder = os.path.join(val_dir, label)
    for fname in os.listdir(folder):
        img = tf.keras.preprocessing.image.load_img(
            os.path.join(folder, fname), target_size=(224,224))
        arr = tf.keras.preprocessing.image.img_to_array(img) / 255.0
        prob = model.predict(np.expand_dims(arr,0), verbose=0)[0][0]
        y_true.append(label_idx)
        y_pred.append(int(prob >= 0.5))

print(classification_report(y_true, y_pred, target_names=classes))
