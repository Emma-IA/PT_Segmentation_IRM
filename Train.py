from Model import *
from Data import *

def train(): 
    train_generator, val_generator = create_generators()
    model = unet()
    model.fit(train_generator, epochs=NBR_EPOCH, callbacks=None,
    validation_data=val_generator,
    class_weight=None,shuffle= SHUFFLE_DATA)
    return history 