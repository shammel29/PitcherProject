import tensorflow as tf
model = tf.keras.models.load_model("keyframe_model.h5")
tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)

def predict(feat_array):
    with tf.device('/GPU:0'):
        return model(feat_array[None, :], training=False).numpy()[0]
