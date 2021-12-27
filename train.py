from params import *
from model import *
from generator import *
import tensorflow as tf
from metrics import IoU_metric


def train():

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=CHECKPOINT_PATH, save_weights_only=True, verbose=1, save_best_only=True)

    train_generator, val_generator = create_generators2()
    model = micro_unet(input_size=(None, None, None, 1))
    model.fit(x=train_generator, epochs=NBR_EPOCH,
              validation_data=val_generator,
              class_weight=None,shuffle= SHUFFLE_DATA, callbacks=[model_checkpoint_callback])

    return model