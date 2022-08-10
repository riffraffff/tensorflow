from tensorflow.keras import layers, Model, Sequential, activations
from typing import Union


class Se_Model(layers.Layer):
    def __init__(self, filters, reductions=4, **kwargs):
        super(Se_Model, self).__init__(**kwargs)
        self.avpool = layers.GlobalAvgPool2D()
        # 在全局池化以后shape：[batch, height, width, channel] → [batch, channel]
        self.Reshape = layers.Reshape((1, 1, filters))
        # 但是在丢到卷积里的时候是需要高度和宽度两个纬度的 所以进行reshape
        # fc1
        self.conv1 = layers.Conv2D(filters//reductions, kernel_size=1, strides=1,
                                   padding="same")
        self.relu = layers.ReLU()
        # fc2
        self.conv2 = layers.Conv2D(filters=filters, kernel_size=1, strides=1,
                                   padding="same")

        self.Mult = layers.Multiply()

    def call(self, inputs, **kwargs):
        x = self.avpool(inputs)
        x = self.Reshape(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = activations.hard_sigmoid(x)
        x = self.Mult([inputs, x])

        return x


def correct_pad(input_size: Union[int, tuple], kernel_size: int):
    """Returns a tuple for zero-padding for 2D convolution with downsampling.
    Arguments:
      input_size: Input tensor size.
      kernel_size: An integer or tuple/list of 2 integers.
    Returns:
      A tuple.
    """

    if isinstance(input_size, int):
        input_size = (input_size, input_size)

    kernel_size = (kernel_size, kernel_size)

    adjust = (1 - input_size[0] % 2, 1 - input_size[1] % 2)
    correct = (kernel_size[0] // 2, kernel_size[1] // 2)
    return ((correct[0] - adjust[0], correct[0]),
            (correct[1] - adjust[1], correct[1]))


class Bneck_Model(layers.Layer):
    def __init__(self, filters1, filters2, strides, kernel_size, pre_input, se=False, hs=False, **kwargs):
        super(Bneck_Model, self).__init__(**kwargs)
        self.conv1 = layers.Conv2D(filters=filters1, kernel_size=1, strides=1,
                                   padding="same", use_bias=False)
        self.bn1 = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)
        self.dw = layers.DepthwiseConv2D(kernel_size=kernel_size, strides=strides,
                                         padding="same", use_bias=False)
        self.bn2 = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)
        self.conv2 = layers.Conv2D(filters=filters2, kernel_size=1, strides=1,
                                   padding="same", use_bias=False)
        self.bn3 = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)
        self.Relu = layers.ReLU(max_value=6)
        self.hard_sigmoid = activations.hard_sigmoid
        self.se = se
        self.hs = hs
        self.strides = strides
        self.pre_input = pre_input
        self.filters1 = filters1
        self.filters2 = filters2
        self.kernel_size = kernel_size

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        if self.strides == 2:
            input_size = (x.shape[1], x.shape[2])  # height, width
            x = layers.ZeroPadding2D(padding=correct_pad(input_size, self.kernel_size))(x)
        x = self.dw(x)
        x = self.bn2(x, training=training)
        if self.hs:
            x = self.hard_sigmoid(x)
        else:
            x = self.Relu(x)

        if self.se:
            x = Se_Model(filters=self.filters1)(x)
        x = self.conv2(x)
        x = self.bn3(x, training=training)

        if self.strides == 1 and self.pre_input == self.filters2:
            x = layers.Add()([inputs, x])

        return x


def MobileNetV3_large(im_height=224, im_width=224, num_class=1000):
    input_image = layers.Input(shape=(im_height, im_width, 3), dtype="float32")
    x = layers.Conv2D(filters=16, strides=2, kernel_size=3,
                      padding="same", use_bias=False)(input_image)
    x = layers.BatchNormalization()(x)
    x = activations.hard_sigmoid(x)
    x = Bneck_Model(filters1=16, filters2=16, strides=1, kernel_size=3, pre_input=16, se=False, hs=False)(x)
    x = Bneck_Model(filters1=64, filters2=24, strides=2, kernel_size=3, pre_input=16, se=False, hs=False)(x)
    x = Bneck_Model(filters1=72, filters2=24, strides=1, kernel_size=3, pre_input=24, se=False, hs=False)(x)
    x = Bneck_Model(filters1=72, filters2=40, strides=2, kernel_size=5, pre_input=24, se=True, hs=False)(x)
    x = Bneck_Model(filters1=120, filters2=40, strides=1, kernel_size=5, pre_input=40, se=True, hs=False)(x)
    x = Bneck_Model(filters1=120, filters2=40, strides=1, kernel_size=5, pre_input=40, se=True, hs=False)(x)
    x = Bneck_Model(filters1=240, filters2=80, strides=2, kernel_size=3, pre_input=40, se=False, hs=True)(x)
    x = Bneck_Model(filters1=200, filters2=80, strides=1, kernel_size=3, pre_input=80, se=False, hs=True)(x)
    x = Bneck_Model(filters1=184, filters2=80, strides=1, kernel_size=3, pre_input=80, se=False, hs=True)(x)
    x = Bneck_Model(filters1=184, filters2=80, strides=1, kernel_size=3, pre_input=80, se=False, hs=True)(x)
    x = Bneck_Model(filters1=480, filters2=112, strides=1, kernel_size=3, pre_input=80, se=True, hs=True)(x)
    x = Bneck_Model(filters1=672, filters2=112, strides=1, kernel_size=3, pre_input=112, se=True, hs=True)(x)
    x = Bneck_Model(filters1=672, filters2=160, strides=2, kernel_size=5, pre_input=112, se=True, hs=True)(x)
    x = Bneck_Model(filters1=960, filters2=160, strides=1, kernel_size=5, pre_input=160, se=True, hs=True)(x)
    x = Bneck_Model(filters1=960, filters2=160, strides=1, kernel_size=5, pre_input=160, se=True, hs=True)(x)
    x = layers.Conv2D(filters=960, kernel_size=1, strides=1,
                      padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = activations.hard_sigmoid(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Reshape((1, 1, 960))(x)
    # fc1
    x = layers.Conv2D(filters=1280, kernel_size=1,
                      strides=1, padding="same")(x)
    x = activations.hard_sigmoid(x)
    # fc2
    x = layers.Conv2D(filters=num_class, kernel_size=1,
                      strides=1, padding="same", use_bias=False)(x)
    x = layers.Flatten()(x)
    predict = layers.Softmax()(x)

    model = Model(inputs=input_image, outputs=predict)

    return model
