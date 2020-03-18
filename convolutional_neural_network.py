from tensorflow.keras import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras import Model

DROPOUT_RATE = 0.1
FIRST_DENSE_LAYERS_UNITS = 400
CLASSES = 10

self.spec_shape = (40, 80, 1)

self.pitch_filter_shape = (32, 1)
self.tempo_channel_filter_shape = (1, 60)
self.bass_channel_filter_shape = (13, 9)

self.maxpooling_shape = (9, 1)
self.maxpooling_shape_2 = (21, 1)
self.maxpooling_shape_3 = (4, 4)

self.reshape1 = (1, 9, 32)
self.reshape3 = (1, 126, 32)


class Musically_motivated_nn():
    
    def __init__(self):

        # init input shape
        


    def define_3_channel_nn(self):

        inputs1 = Input(shape=self.spec_shape)
        inputs2 = Input(shape=self.spec_shape)
        inputs3 = Input(shape=self.spec_shape)

        # channel 1
        conv2D_1 = Conv2D(9, self.pitch_filter_shape)(inputs1)
        batch_n_1 =  BatchNormalization()(conv2D_1)
        maxpooling2D_n_1 =  MaxPooling2D(self.maxpooling_shape)(batch_n_1)
        channel_1 =  Reshape(self.reshape1)(maxpooling2D_n_1)

        # channel 2
        conv2D_2 = Conv2D(32, self.tempo_channel_filter_shape)(inputs2)
        batch_n_2 = BatchNormalization()(conv2D_2)
        channel_2 = MaxPooling2D(self.maxpooling_shape_2)(batch_n_2)
        
        # channel 3
        conv2D_3 = Conv2D(28, self.bass_channel_filter_shape)(inputs3)
        batch_n_3 = BatchNormalization()(conv2D_3)
        maxpooling2D_n_3= MaxPooling2D(self.maxpooling_shape_3)(batch_n_3)
        channel_3 = Reshape(self.reshape3)(maxpooling2D_n_3)

        # merge all channels
        merged = Concatenate(axis=2)([channel_1, channel_2, channel_3])

        # final part of the nn
        flatten = Flatten()(merged)
        dense1 = Dense(FIRST_DENSE_LAYERS_UNITS)(flatten)
        dropout = Dropout(DROPOUT_RATE)(dense1)
        outputs = Dense(CLASSES)(dropout)

        model = Model(inputs=[inputs1, inputs2, inputs3], outputs=outputs)

        model.summary()

        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        return model



