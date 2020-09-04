"""
Collection of models used by the system.
"""

import tensorflow as tf
from config import IMG_SIZE, DESCRIPTOR_SIZE

# mlp = tf.keras.models.Sequential([
#     tf.keras.layers.Flatten(input_shape=(28, 28)),
#     tf.keras.layers.Dense(128, activation='relu'),
#     tf.keras.layers.Dropout(0.2),
#     tf.keras.layers.Dense(128, activation='relu'),
#     tf.keras.layers.Dropout(0.2),
#     tf.keras.layers.Dense(1)
# ])

embedding_mlp = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(128,)),
    tf.keras.layers.Dense(270, activation=tf.nn.softmax, kernel_regularizer='l2')
])

cnn_classifier = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation=tf.nn.relu),
    tf.keras.layers.MaxPooling2D((2, 2), strides=2),
    tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation=tf.nn.relu),
    tf.keras.layers.MaxPooling2D((2, 2), strides=2),
    tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation=tf.nn.relu),
    tf.keras.layers.MaxPooling2D((2, 2), strides=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation=tf.nn.relu),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(270, activation=tf.nn.softmax)
])


def define_direction_detector():
    input = tf.keras.layers.Input((416, 416, 3))
    vgg16_layer = tf.keras.applications.InceptionV3(
        include_top=False,
        weights='imagenet',
        input_shape=(416, 416, 3)
    )
    vgg16_layer.trainable = False
    vgg16_batch = vgg16_layer(input)
    model = tf.keras.layers.GlobalAveragePooling2D()(vgg16_batch)
    model = tf.keras.layers.Dense(32, activation='relu')(model)
    model = tf.keras.layers.Dense(1, activation='sigmoid')(model)
    model = tf.keras.Model(inputs=input, outputs=model)
    return model


def define_region_detector():
    input = tf.keras.layers.Input((416, 416, 3))
    vgg16_layer = tf.keras.applications.VGG16(
        include_top=False,
        weights=None,
        input_shape=(416, 416, 3)
    )
    vgg16_layer.trainable = False
    vgg16_batch = vgg16_layer(input)
    model = tf.keras.layers.Flatten()(vgg16_batch)
    model = tf.keras.layers.Dense(128, activation='relu')(model)
    model = tf.keras.layers.Dropout(0.05)(model)
    model = tf.keras.layers.Dense(64, activation='relu')(model)
    model = tf.keras.layers.Dropout(0.05)(model)
    model = tf.keras.layers.Dense(32, activation='relu')(model)
    model = tf.keras.layers.Dense(4)(model)
    model = tf.keras.Model(inputs=input, outputs=model)
    return model


def define_vgg16_classifier():
    input = tf.keras.layers.Input((IMG_SIZE, IMG_SIZE, 3))
    vgg16_layer = tf.keras.applications.VGG16(
        include_top=False,
        weights='imagenet',
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    vgg16_layer.trainable = False
    vgg16_batch = vgg16_layer(input)
    model = tf.keras.layers.GlobalAveragePooling2D()(vgg16_batch)
    #model = tf.keras.layers.Dense(64, activation='relu')(model)
    model = tf.keras.layers.Dropout(0.05)(model)
    model = tf.keras.layers.Dense(366, activation=tf.nn.softmax)(model)
    model = tf.keras.Model(inputs=input, outputs=model)
    return model


def define_siamese_network(contrastive=True):
    input_shape = (IMG_SIZE, IMG_SIZE, 3)
    left_input = tf.keras.layers.Input(input_shape, name="left_input")
    right_input = tf.keras.layers.Input(input_shape, name="right_input")

    # feature_extractor = tf.keras.models.Sequential([
    #     tf.keras.layers.Conv2D(16, 3, padding='same', activation=tf.nn.relu),
    #     tf.keras.layers.BatchNormalization(),
    #     tf.keras.layers.MaxPooling2D(),
    #     tf.keras.layers.Conv2D(32, 3, padding='same', activation=tf.nn.relu),
    #     tf.keras.layers.BatchNormalization(),
    #     tf.keras.layers.MaxPooling2D(),
    #     tf.keras.layers.Conv2D(32, 3, padding='same', activation=tf.nn.relu),
    #     tf.keras.layers.BatchNormalization(),
    #     tf.keras.layers.MaxPooling2D(),
    # ], name="feature_extractor")

    inception_layer = tf.keras.applications.InceptionV3(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape
    )
    inception_layer.trainable = False

    cnn_part = tf.keras.models.Sequential([
        inception_layer,
        # feature_extractor,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(DESCRIPTOR_SIZE, activation=tf.nn.sigmoid), #activation=tf.nn.sigmoid),
        #tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))
    ], name="cnn_part")

    encoded_l = cnn_part(left_input)
    encoded_r = cnn_part(right_input)

    if contrastive:
        L1_layer = tf.keras.layers.Lambda(lambda tensors: tf.linalg.norm(tf.math.abs(tensors[0] - tensors[1]), axis=1))
        L1_distance = L1_layer([encoded_l, encoded_r])

        siamese_network = tf.keras.models.Model(inputs=[left_input, right_input], outputs=[L1_distance])
    else:
        L1_layer = tf.keras.layers.Lambda(lambda tensors: tf.math.abs(tensors[0] - tensors[1]))
        L1_distance = L1_layer([encoded_l, encoded_r])

        prediction = tf.keras.layers.Dense(1, activation='sigmoid', name="prediction_layer")(L1_distance)
        siamese_network = tf.keras.models.Model(inputs=[left_input, right_input], outputs=[prediction])

    embedding_network = tf.keras.models.Model(inputs=[left_input], outputs=[encoded_l], name="embedding_network")

    return siamese_network, embedding_network


def define_siamese_triplet_network():
    input_shape = (IMG_SIZE, IMG_SIZE, 3)
    anchor_input = tf.keras.layers.Input(input_shape, name="anchor_input")
    positive_input = tf.keras.layers.Input(input_shape, name="positive_input")
    negative_input = tf.keras.layers.Input(input_shape, name="negative_input")

    inception_layer = tf.keras.applications.InceptionV3(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape
    )
    inception_layer.trainable = False

    cnn_part = tf.keras.models.Sequential([
        inception_layer,
        tf.keras.layers.GlobalAveragePooling2D(),
        # tf.keras.layers.Dense(128, activation=tf.nn.relu),
        tf.keras.layers.Dense(DESCRIPTOR_SIZE, activation=None),
        tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))
    ], name="cnn_part")

    encoded_anc = cnn_part(anchor_input)
    encoded_pos = cnn_part(positive_input)
    encoded_neg = cnn_part(negative_input)

    stacked = tf.keras.layers.concatenate([encoded_anc, encoded_pos, encoded_neg])

    siamese_triplet_network = tf.keras.models.Model(inputs=[anchor_input, positive_input, negative_input],
                                                    outputs=[stacked],
                                                    name="siamese_triplet_network")
    embedding_network = tf.keras.models.Model(inputs=[anchor_input], outputs=[encoded_anc], name="embedding_network")

    return siamese_triplet_network, embedding_network


def define_ohnm_triplet_network():
    input_shape = (IMG_SIZE, IMG_SIZE, 3)

    feature_extractor = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, 3, padding='same', activation=tf.nn.relu),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, padding='same', activation=tf.nn.relu),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D()
    ], name="feature_extractor")

    inception_layer = tf.keras.applications.InceptionV3(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape,
        #layers=tf.keras.layers
    )
    inception_layer.trainable = False

    vgg16_layer = tf.keras.applications.VGG16(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape
    )
    vgg16_layer.trainable = False

    #model = tf.keras.layers.GlobalAveragePooling2D()(inception_layer.layers[196].output)
    #model = tf.keras.layers.GlobalAveragePooling2D()(inception_layer.layers[132].output)
    model = tf.keras.layers.GlobalAveragePooling2D()(inception_layer.output)
    #model = tf.keras.layers.Flatten()(inception_layer.output)
    #model = tf.keras.layers.Dropout(0.05)(model)
    #model = tf.keras.layers.Dense(128, activation=tf.nn.relu)(model)
    #model = tf.keras.layers.Dropout(0.05)(model)
    #model = tf.keras.layers.Dense(256, activation=tf.nn.relu)(model)
    #model = tf.keras.layers.Dropout(0.2)(model)
    model = tf.keras.layers.Dense(DESCRIPTOR_SIZE, activation=None)(model)
    model = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(model)
    triplet_network = tf.keras.Model(inception_layer.input, model)

    # triplet_network = tf.keras.models.Sequential([
    #     inception_layer,
    #     #vgg16_layer,
    #     #tf.keras.layers.Input(input_shape),
    #     #feature_extractor,
    #     # tf.keras.layers.Conv2D(2048, 3, padding='same', activation=tf.nn.relu),
    #     # tf.keras.layers.BatchNormalization(),
    #     # tf.keras.layers.MaxPooling2D(),
    #     tf.keras.layers.GlobalAveragePooling2D(),
    #     # tf.keras.layers.Dense(128, activation=tf.nn.relu),
    #     tf.keras.layers.Dropout(0.05),
    #     tf.keras.layers.Dense(128, activation=None),
    #     tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))
    # ], name="cnn_part")

    return triplet_network


region_detector = define_region_detector()
vgg16_classifier = define_vgg16_classifier()
direction_detector = define_direction_detector()

siamese_network_contrastive, embedding_network_contrastive = define_siamese_network(contrastive=True)
siamese_network_binary, embedding_network_binary = define_siamese_network(contrastive=False)
siamese_network_triplet, embedding_network_triplet = define_siamese_triplet_network()

triplet_network_ohnm = define_ohnm_triplet_network()

