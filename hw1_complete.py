#!/usr/bin/env python

# TensorFlow and tf.keras
import tensorflow as tf
import keras
from keras import Input, layers, activations, Sequential


# Helper libraries
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import image


print(f"TensorFlow Version: {tf.__version__}")
print(f"Keras Version: {keras.__version__}")

def get_conv_block(channels=32, kernel_size=3, stride=1, padding='same',
                    activation:str="relu", conv_type='standard'):
    if conv_type == 'depthwise':
        conv_layer = layers.DepthwiseConv2D(kernel_size=kernel_size, strides=stride, padding=padding)
    elif conv_type == 'seperable':
        conv_layer = layers.SeparableConv2D(filters=channels, kernel_size=kernel_size, strides=stride, padding=padding)
    else:
        conv_layer = layers.Conv2D(filters=channels, kernel_size=kernel_size, strides=stride, padding=padding, activation=activation)
    return [conv_layer, layers.BatchNormalization()]


class Model1(keras.Model):
    def __init__(self):
        super(Model1, self).__init__()
        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(128, activation="leaky_relu")
        self.dense2 = layers.Dense(128, activation="leaky_relu")
        self.dense3 = layers.Dense(128, activation="leaky_relu")
        self.head = layers.Dense(10)

    def call(self, inputs):
        x = self.dense1(self.flatten(inputs))
        x2 = self.dense2(x)
        return self.head(self.dense3(x2))

class Model2(keras.Model):
    def __init__(self):
        super().__init__()
        # self.flatten = layers.Flatten()
        self.block1 = get_conv_block(32, 3, 2, "same", "relu")
        self.block2 = get_conv_block(64, 3, 2, "same", "relu")
        self.block3 = []
        for _ in range(4):
            self.block3.extend(get_conv_block(128, 3, 1, "same", "relu"))
        self.flatten = layers.Flatten()
        self.head = layers.Dense(10)

    def call(self, inputs):
        x = inputs
        for sub_block in self.block1:
            x = sub_block(x)
        for sub_block2 in self.block2:
            x = sub_block2(x)
        for sub_block3 in self.block3:
            x = sub_block3(x)
        x = self.flatten(x)
        return self.head(x)


class Model3(tf.keras.Model):
    def __init__(self):
        super().__init__()

        self.block1 = get_conv_block(32, 3, 2, "same", activation="relu", conv_type='seperable')
        self.block2 = get_conv_block(64, 3, 2, "same", activation="relu", conv_type='seperable')

        self.block3 = []
        for _ in range(4):
            self.block3.extend(get_conv_block(128, 3, 1, "same", "relu", conv_type='seperable'))
        self.flatten = layers.Flatten()
        self.head = layers.Dense(10)

    def call(self, inputs):

        x = inputs
        for sub_block in self.block1:
            x = sub_block(x)
        for sub_block2 in self.block2:
            x = sub_block2(x)
        for sub_block3 in self.block3:
            x = sub_block3(x)
        x = self.flatten(x)
        return self.head(x)
##
def create_model_50k(input_shape=(28, 28, 1)):
    inputs = tf.keras.Input(shape=input_shape)
    
    x = inputs
    for block in get_conv_block(32, 3, 2, "same", activation="relu"):
        x = block(x)
    x = layers.MaxPool2D((2, 2))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    for block in get_conv_block(64, 3, 1, "same", activation="relu"):
        x = block(x)
    x = layers.MaxPool2D((2, 2))(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.Flatten()(x)
    outputs = layers.Dense(10)(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='Model50k')
    return model



def build_model1():
  model1 = Model1() # Add code to define model 1.
  model1(tf.zeros((1, 32, 32, 3)))
  model1.summary()
  model1.compile(
      optimizer='adam',
      loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      metrics=['accuracy']
  )

  return model1

def build_model2():
  model2 = Model2() # Add code to define model 2.
  model2(tf.zeros((1, 32, 32, 3)))
  model2.summary()

  wd_rate = 1e-4
  optimizer = keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=wd_rate)
  model2.compile(
      optimizer=optimizer,
      loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      metrics=['accuracy']
  )

  return model2

def build_model3():
  model3 = Model3() # Add code to define model 3.
  model3(tf.zeros((1, 32, 32, 3)))
  model3.summary()
  # wd_rate = 1e-4
  optimizer = keras.optimizers.AdamW(learning_rate=1e-3)
  model3.compile(
      optimizer=optimizer,
      loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      metrics=['accuracy']
  )

  return model3

  ## This one should use the functional API so you can create the residual connections

def build_model50k():
  model_50k = create_model_50k((32, 32, 3)) # Add code to define model 3.
  # model_50k.build()
  model_50k.summary()

  optimizer = keras.optimizers.AdamW(learning_rate=1e-3)
  model_50k.compile(
      optimizer=optimizer,
      loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      metrics=['accuracy']
  )# Add code to define model 1.
  return model_50k

# no training or dataset construction should happen above this line
# also, be careful not to unindent below here, or the code be executed on import
if __name__ == '__main__':

  ########################################
  ## Add code here to Load the CIFAR10 data set
  (train_images, train_labels), (test_images, test_labels) = keras.datasets.cifar10.load_data()

  train_images = train_images.astype("float32") / 255.0
  test_images = test_images.astype("float32") / 255.0
  train_labels = train_labels.squeeze()
  test_labels = test_labels.squeeze()
  ########################################
  ## Build and train model 1
  model1 = build_model1()

  epochs = 30
  for ep in range(epochs):
      print(f"Epoch {ep+1}/{epochs}")
      model1.fit(train_images, train_labels, epochs=1, validation_data=(test_images, test_labels), batch_size=128)

  # compile and train model 1.

  ## Build, compile, and train model 2 (DS Convolutions)
  model2 = build_model2()

  epochs = 30
  for ep in range(epochs):
      print(f"Epoch {ep+1}/{epochs}")
      model2.fit(train_images, train_labels, epochs=1, validation_data=(test_images, test_labels), batch_size=128)

  ### Repeat for model 3 and your best sub-50k params model
  model3 = build_model3()

  epochs = 30
  for ep in range(epochs):
      print(f"Epoch {ep+1}/{epochs}")
      model3.fit(train_images, train_labels, epochs=1, validation_data=(test_images, test_labels), batch_size=128)


  model_50k = build_model50k()
  epochs = 30

  for ep in range(epochs):
      print(f"Epoch {ep+1}/{epochs}")
      model_50k.fit(train_images, train_labels, epochs=1, validation_data=(test_images, test_labels), batch_size=64)

  model_50k.save("./best_model.h5", save_format='tf')