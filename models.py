"""
Collection of models used by the system.
"""

import tensorflow as tf
from config import IMG_SIZE, DESCRIPTOR_SIZE, IMG_SHAPE


def define_direction_detector():
    input = tf.keras.layers.Input((IMG_SIZE, IMG_SIZE, 3))
    vgg16_layer = tf.keras.applications.InceptionV3(
        include_top=False,
        weights='imagenet',
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    vgg16_layer.trainable = False
    vgg16_batch = vgg16_layer(input)
    model = tf.keras.layers.GlobalAveragePooling2D()(vgg16_batch)
    model = tf.keras.layers.Dense(32, activation='relu')(model)
    model = tf.keras.layers.Dense(1, activation='sigmoid')(model)
    model = tf.keras.Model(inputs=input, outputs=model)
    return model


def define_region_detector():
    input = tf.keras.layers.Input(IMG_SHAPE)
    vgg16_layer = tf.keras.applications.InceptionV3(
        include_top=False,
        weights=None,
        input_shape=IMG_SHAPE
    )
    vgg16_layer.trainable = False
    vgg16_batch = vgg16_layer(input)
    model = tf.keras.layers.GlobalAveragePooling2D()(vgg16_batch)
    model = tf.keras.layers.Dropout(0.5)(model)
    model = tf.keras.layers.Dense(2048, activation=tf.nn.relu)(model)
    model = tf.keras.layers.Dense(4, activation=tf.nn.sigmoid)(model)
    model = tf.keras.Model(inputs=input, outputs=model)
    return model


def define_cnn_classifier():
    inception_layer = tf.keras.applications.InceptionV3(
        include_top=False,
        weights='imagenet',
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    inception_layer.trainable = False
    model = tf.keras.layers.GlobalAveragePooling2D()(inception_layer.layers[132].output)
    model = tf.keras.layers.Dense(1005, activation=tf.nn.softmax)(model)
    model = tf.keras.Model(inputs=inception_layer.input, outputs=model)
    return model


def define_siamese_network(loss='contrastive'):
    input_shape = (IMG_SIZE, IMG_SIZE, 3)
    left_input = tf.keras.layers.Input(input_shape, name="left_input")
    right_input = tf.keras.layers.Input(input_shape, name="right_input")

    inception_layer = tf.keras.applications.InceptionV3(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape
    )
    inception_layer.trainable = False
    inception_model = tf.keras.Model(inception_layer.input, inception_layer.layers[132].output)

    cnn_part = tf.keras.models.Sequential()
    cnn_part.add(inception_model)
    cnn_part.add(tf.keras.layers.GlobalAveragePooling2D())
    cnn_part.add(tf.keras.layers.Dense(DESCRIPTOR_SIZE, activation=None))
    cnn_part.add(tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1)))

    encoded_l = cnn_part(left_input)
    encoded_r = cnn_part(right_input)

    if loss=='contrastive':
        L1_layer = tf.keras.layers.Lambda(lambda tensors: tf.linalg.norm(tf.math.abs(tensors[0] - tensors[1]), axis=1))
        L1_distance = L1_layer([encoded_l, encoded_r])

        siamese_network = tf.keras.models.Model(inputs=[left_input, right_input], outputs=[L1_distance])
    elif loss=='binary':
        L1_layer = tf.keras.layers.Lambda(lambda tensors: tf.math.abs(tensors[0] - tensors[1]))
        L1_distance = L1_layer([encoded_l, encoded_r])

        prediction = tf.keras.layers.Dense(1, activation='sigmoid', name="prediction_layer")(L1_distance)
        siamese_network = tf.keras.models.Model(inputs=[left_input, right_input], outputs=[prediction])
    elif loss=='ntxent':
        stacked = tf.keras.layers.concatenate([encoded_l, encoded_r])
        siamese_network = tf.keras.models.Model(inputs=[left_input, right_input], outputs=[stacked])

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

    inception_layer = tf.keras.applications.InceptionV3(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape
    )
    inception_layer.trainable = False

    model = tf.keras.layers.GlobalAveragePooling2D()(inception_layer.layers[132].output)
    #model = tf.keras.layers.GlobalAveragePooling2D()(inception_layer.layers[196].output)
    #model = tf.keras.layers.Dropout(0.05)(model)
    model = tf.keras.layers.Dense(DESCRIPTOR_SIZE, activation=None)(model)
    model = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(model)
    triplet_network = tf.keras.Model(inception_layer.input, model)

    return triplet_network


region_detector = define_region_detector()
cnn_classifier = define_cnn_classifier()
direction_detector = define_direction_detector()

siamese_network_contrastive, embedding_network_contrastive = define_siamese_network(loss='contrastive')
siamese_network_binary, embedding_network_binary = define_siamese_network(loss='binary')
siamese_network_ntxent, embedding_network_ntxent = define_siamese_network(loss='ntxent')
siamese_network_triplet, embedding_network_triplet = define_siamese_triplet_network()

triplet_network_ohnm = define_ohnm_triplet_network()