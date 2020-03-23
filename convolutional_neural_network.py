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
from tensorflow.keras import Sequential


DROPOUT_RATE = 0.1
FIRST_DENSE_LAYERS_UNITS = 400
CLASSES = 10
SOFTMAX = 'softmax'
RELU = 'relu'

spec_shape = (40, 80, 1)

pitch_filter_shape = (32, 1)
tempo_channel_filter_shape = (1, 60)
bass_channel_filter_shape = (13, 9)

maxpooling_shape = (9, 1)
maxpooling_shape_2 = (21, 1)
maxpooling_shape_3 = (4, 4)

reshape1 = (1, 9, 32)
reshape3 = (1, 126, 32)


def compile_baseline_nn():

  model = Sequential([
    Dense(10, activation='relu'),
    Dense(10, activation='relu'),
    Dense(10, activation='softmax')
  ])

  model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])
  return model



def compile_3_channel_nn():

    inputs1 = Input(shape=spec_shape)
    inputs2 = Input(shape=spec_shape)
    inputs3 = Input(shape=spec_shape)

    # channel 1
    conv2D_1 = Conv2D(9, pitch_filter_shape)(inputs1)
    batch_n_1 =  BatchNormalization()(conv2D_1)
    maxpooling2D_n_1 =  MaxPooling2D(maxpooling_shape)(batch_n_1)
    channel_1 =  Reshape(reshape1)(maxpooling2D_n_1)

    # channel 2
    conv2D_2 = Conv2D(32, tempo_channel_filter_shape)(inputs2)
    batch_n_2 = BatchNormalization()(conv2D_2)
    channel_2 = MaxPooling2D(maxpooling_shape_2)(batch_n_2)
    
    # channel 3
    conv2D_3 = Conv2D(28, bass_channel_filter_shape)(inputs3)
    batch_n_3 = BatchNormalization()(conv2D_3)
    maxpooling2D_n_3= MaxPooling2D(maxpooling_shape_3)(batch_n_3)
    channel_3 = Reshape(reshape3)(maxpooling2D_n_3)

    # merge all channels
    merged = Concatenate(axis=2)([channel_1, channel_2, channel_3])

    # final part of the nn
    flatten = Flatten()(merged)
    dense1 = Dense(FIRST_DENSE_LAYERS_UNITS,activation=RELU)(flatten)
    dropout = Dropout(DROPOUT_RATE)(dense1)
    outputs = Dense(CLASSES, activation=SOFTMAX)(dropout)

    model = Model(inputs=[inputs1, inputs2, inputs3], outputs=outputs)

    model.summary()

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model



