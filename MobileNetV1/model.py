from tensorflow.keras import layers, Model


def conv_block(inputs, filters, stride, **kwargs):
    conv = layers.Conv2D(filters, kernel_size=1, strides=stride,
                         padding="SAME", use_bias=False)(inputs)
    bn = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(conv)
    relu = layers.ReLU(max_value=6)(bn)
    return relu


def depthise_conv_block(inputs, filters, alpha, stride, depth_multipliter, **kwargs):
    filters = int(filters * alpha)
    conv_dw = layers.DepthwiseConv2D(kernel_size=(3, 3), strides=stride,
                                     padding='same', depth_multiplier=depth_multipliter)(inputs)
    bn1 = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(conv_dw)
    relu6_1 = layers.ReLU(max_value=6)(bn1)
    conv_1x1 = layers.Conv2D(filters, kernel_size=1, strides=stride,
                             padding="SAME", use_bias=False)(relu6_1)
    bn2 = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(conv_1x1)
    relu6_2 = layers.ReLU(max_value=6)(bn2)
    return relu6_2


def MobileNetV1(im_width=224, im_height=224, num_classes=5, **kwargs):
    input_image = layers.Input(shape=(im_height, im_width, 3), dtype="float32")
    # （224，224，3）
    x = conv_block(input_image, filters=32, stride=2)
    # （112，112，32）
    x = depthise_conv_block(x, filters=64, alpha=1, stride=1,
                            depth_multipliter=1, name="conv_dw1")
    # （112，112，64）
    x = depthise_conv_block(x, filters=128, alpha=1, stride=2,
                            depth_multipliter=1, name="conv_dw2")
    # （56，56，128）
    x = depthise_conv_block(x, filters=128, alpha=1, stride=1,
                            depth_multipliter=1, name="conv_dw3")
    # （56，56，128）
    x = depthise_conv_block(x, filters=256, alpha=1,  stride=2,
                            depth_multipliter=1, name="conv_dw4")
    # （28，28，256）
    x = depthise_conv_block(x, filters=256, alpha=1, stride=1,
                            depth_multipliter=1, name="conv_dw5")
    # （28，28，256）
    x = depthise_conv_block(x, filters=256, alpha=1, stride=2,
                            depth_multipliter=1, name="conv_dw6")
    # （14，14，256）
    x = conv_block(x, filters=512, stride=1)
    # （14，14，512）
    x = depthise_conv_block(x, filters=256, alpha=1, stride=1,
                            depth_multipliter=1, name="conv_dw7")
    # -------------------------------------------------------------------------
    x = depthise_conv_block(x, filters=512, alpha=1, stride=1,
                            depth_multipliter=1, name="conv_dw_1")
    x = depthise_conv_block(x, filters=512, alpha=1, stride=1,
                            depth_multipliter=1, name="conv_dw_2")
    x = depthise_conv_block(x, filters=512, alpha=1, stride=1,
                            depth_multipliter=1, name="conv_dw_3")
    x = depthise_conv_block(x, filters=512, alpha=1, stride=1,
                            depth_multipliter=1, name="conv_dw_4")
    x = depthise_conv_block(x, filters=512, alpha=1, stride=1,
                            depth_multipliter=1, name="conv_dw_5")
    # --------------------------------------------------------------------------
    # (14,14,512)
    x = depthise_conv_block(x, filters=512, alpha=1, stride=2,
                            depth_multipliter=1, name="conv_dw7")
    # (7,7,512)
    x = conv_block(x, filters=1024, stride=1)
    # (7,7,1024)
    x = depthise_conv_block(x, filters=1024, alpha=1, stride=1,
                            depth_multipliter=1, name="conv_dw8")
    # (7,7,1024)
    x = conv_block(x, filters=1024, stride=1, name="conv9")
    x = layers.GlobalAvgPool2D()(x)
    x = layers.Dense(num_classes, name="logits")(x)
    x = layers.Dropout(rate=0.2)(x)
    predict = layers.Softmax()(x)

    model = Model(inputs=input_image, outputs=predict)

    return model

