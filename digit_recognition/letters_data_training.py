import tensorflow as tf
from emnist import extract_training_samples, extract_test_samples

def training_data():

    X_train, y_train = extract_training_samples('letters')
    X_test, y_test = extract_test_samples('letters')

    X_train = tf.keras.utils.normalize(X_train, axis=1)
    X_test = tf.keras.utils.normalize(X_test, axis=1)

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(units=27, activation=tf.nn.softmax))

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=3)

    model.save('handwritten.model')

    model = tf.keras.models.load_model('handwritten.model')
    return model