from tensorflow.keras import layers, Model, Sequential, activations


class ConvBNReLU(layers.Layer):
    def __init__(self, out_channel, kernel_size=3, stride=1, **kwargs):
        super(ConvBNReLU, self).__init__(**kwargs)
        self.conv = layers.Conv2D(filters=out_channel, kernel_size=kernel_size,
                                  strides=stride, padding='SAME', use_bias=False)
        self.bn = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)
        self.activation = layers.ReLU(max_value=6.0)

    def call(self, inputs, training=False):
        x = self.conv(inputs)
        x = self.bn(x, training=training)
        x = self.activation(x)
        return x


class InvertedBottleneck(layers.Layer):
    def __init__(self, out_channels, t, strides, shortcut=False, **kwargs):
        super(InvertedBottleneck, self).__init__(**kwargs)
        self.conv1 = ConvBNReLU(out_channels * t, kernel_size=1)
        self.dw = layers.DepthwiseConv2D(kernel_size=3, strides=strides,
                                         padding="same", use_bias=False)
        self.bn1 = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)
        self.conv2 = layers.Conv2D(out_channels, kernel_size=1, strides=1,
                                   padding="same", use_bias=False)
        self.bn2 = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)
        self.relu6 = layers.ReLU(max_value=6.0)
        self.shortcut = shortcut
        self.add = layers.Add()

    def call(self, inputs, training=False):
        identity = inputs
        x = self.conv1(inputs)
        x = self.dw(x)
        x = self.bn1(x, training=training)
        x = self.relu6(x)
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        if self.shortcut:
            return self.add([identity, x])
        else:
            return x


def MobileNetV2(im_height=224, im_width=224, num_class=1000):
    input_image = layers.Input(shape=(im_height, im_width, 3), dtype="float32")
    x = ConvBNReLU(32, stride=2)(input_image)
    # ---------------------------------------------------------------
    x = InvertedBottleneck(16, t=1, strides=1, shortcut=False)(x)
    # ---------------------------------------------------------------
    x = InvertedBottleneck(24, t=6, strides=2, shortcut=False)(x)
    x = InvertedBottleneck(24, t=6, strides=1, shortcut=True)(x)
    # ---------------------------------------------------------------
    x = InvertedBottleneck(32, t=6, strides=2, shortcut=False)(x)
    x = InvertedBottleneck(32, t=6, strides=1, shortcut=True)(x)
    x = InvertedBottleneck(32, t=6, strides=1, shortcut=True)(x)
    # ---------------------------------------------------------------
    x = InvertedBottleneck(64, t=6, strides=2, shortcut=False)(x)
    x = InvertedBottleneck(64, t=6, strides=1, shortcut=True)(x)
    x = InvertedBottleneck(64, t=6, strides=1, shortcut=True)(x)
    x = InvertedBottleneck(64, t=6, strides=1, shortcut=True)(x)
    # ---------------------------------------------------------------
    x = InvertedBottleneck(96, t=6, strides=1, shortcut=False)(x)
    x = InvertedBottleneck(96, t=6, strides=1, shortcut=True)(x)
    x = InvertedBottleneck(96, t=6, strides=1, shortcut=True)(x)
    # ---------------------------------------------------------------
    x = InvertedBottleneck(160, t=6, strides=2, shortcut=False)(x)
    x = InvertedBottleneck(160, t=6, strides=1, shortcut=True)(x)
    x = InvertedBottleneck(160, t=6, strides=1, shortcut=True)(x)
    # ---------------------------------------------------------------
    x = InvertedBottleneck(320, t=6, strides=1, shortcut=False)(x)
    # ---------------------------------------------------------------
    x = ConvBNReLU(1280, kernel_size=1, name='Conv_1')(x)
    # ---------------------------------------------------------------
    x = layers.AvgPool2D(pool_size=(7, 7), padding="same")(x)
    x = layers.Flatten()(x)
    x = layers.Dense(num_class, name="logits")(x)
    predict = layers.Softmax()(x)

    model = Model(inputs=input_image, outputs=predict)

    return model
