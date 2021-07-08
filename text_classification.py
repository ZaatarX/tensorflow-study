import tensorflow as tf
import tensorflow.keras as k
import tensorflow_hub as hub
import tensorflow_datasets as tfds
import os

# Load compressed models from tensorflow_hub
os.environ["TFHUB_MODEL_LOAD_FORMAT"] = "COMPRESSED"

(train_examples, train_labels), (validation_examples, validation_labels), (test_examples, test_labels) = \
    tfds.as_numpy(
    tfds.load(name="imdb_reviews",
              split=('train[:60%]', 'train[60%:]', 'test'),
              batch_size=-1,
              as_supervised=True))

print("Training entries: {}, validation entries: {}, test entries: {}".format(
    len(train_examples), len(validation_examples), len(test_examples)))


model = k.Sequential()
# add embedding layer
model.add(k.layers.Dropout(rate=0.3))
model.add(k.layers.Dense(64, activation='relu'))
model.add(k.layers.Dense(16, activation='relu'))
model.add(k.layers.Dense(1))

model.summary()

model.compile(optimizer="adam",
              loss=tf.losses.BinaryCrossentropy(from_logits=True),
              metrics=[tf.metrics.BinaryAccuracy(threshold=0.3, name='accuracy')])

history = model.fit(validation_examples, validation_labels, epochs=20,
                    batch_size=8, validation_data=(train_examples, train_labels), verbose=1)

results = model.evaluate(test_examples, test_labels)

print(results)
