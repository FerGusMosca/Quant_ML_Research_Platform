import tensorflow as tf
import tensorflow.keras.layers as tfl

class ConvolutionalNeuralNetwork():

    #everythin to initiate the model
    def __init__(self):
        pass



    #region Public Methods

    def train_model(self,train_x, train_y, test_x,test_y,arch_file,padding,stride,iterations):
        input_img = tf.keras.Input(shape=(train_x.shape[1],train_x.shape[2],train_x.shape[3]))
        ## CONV2D: 8 filters 4x4, stride of 1, padding 'SAME'
        # Definir la primera capa convolucional
        Z1 = tf.keras.layers.Conv2D(filters=8, kernel_size=(4, 4), strides=(stride, stride), padding='same')(input_img)

        ## RELU
        A1 = tf.keras.layers.ReLU()(Z1)
        ## MAXPOOL: window 8x8, stride 8, padding 'SAME'
        # Aplicar la capa de MaxPooling
        P1 = tf.keras.layers.MaxPooling2D(pool_size=(8, 8), strides=(8, 8), padding='same')(A1)

        ## CONV2D: 16 filters 2x2, stride 1, padding 'SAME'
        Z2 = tf.keras.layers.Conv2D(filters=16, kernel_size=(2, 2), strides=(stride, stride), padding='same')(P1)
        ## RELU
        # A2 = None
        A2 = tf.keras.layers.ReLU()(Z2)

        ## MAXPOOL: window 4x4, stride 4, padding 'SAME'
        P2 = tf.keras.layers.MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(A2)

        ## FLATTEN
        F = tf.keras.layers.Flatten()(P2)

        ## Dense layer
        ## 6 neurons in output layer. Hint: one of the arguments should be "activation='softmax'"
        outputs = tf.keras.layers.Dense(units=1, activation='softmax')(F)
        # YOUR CODE STARTS HERE

        # YOUR CODE ENDS HERE
        model = tf.keras.Model(inputs=input_img, outputs=outputs)

        model.compile(optimizer='adam',
                       loss='categorical_crossentropy',
                       metrics=['accuracy'])

        train_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y)).batch(64)
        test_dataset = tf.data.Dataset.from_tensor_slices((test_x, test_y)).batch(64)


        history = model.fit(train_dataset, epochs=iterations, validation_data=test_dataset)


    #endregion