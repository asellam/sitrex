from tqdm import tqdm
import tensorflow as tf

def transformer(input_shape, embed_dim, num_heads, ff_dim, rate=0.1):
    x = tf.keras.layers.Input(input_shape)
    a = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)(x, x)
    b = tf.keras.layers.Dropout(rate)(a)
    c = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + b)
    d = tf.keras.layers.Dense(ff_dim, activation="relu")(c)
    e = tf.keras.layers.Dense(embed_dim)(d)
    f = tf.keras.layers.Dropout(rate)(e)
    y = tf.keras.layers.LayerNormalization(epsilon=1e-6)(c + f)
    return tf.keras.models.Model(x, y)

def positionEmbedding(input_shape, embed_dim, maxlen):
    x = tf.keras.layers.Input(input_shape)
    p = tf.range(0, maxlen)
    y = tf.keras.layers.Dense(embed_dim)(x) + tf.keras.layers.Embedding(input_dim=maxlen, output_dim=embed_dim)(p)
    return tf.keras.models.Model(x, y)

# Function to build the Angle Usefulness model using Transformer blocks
def angle_usefulness_model(maxlen, module, embed_dim=64, num_heads=4, ff_dim=64, lr=1e-3):
    inputs = tf.keras.layers.Input(shape=(maxlen, 23))
    module = module.lower()
    if module == 'transformer':
        projection = positionEmbedding((maxlen, 23), maxlen=maxlen, embed_dim=embed_dim)(inputs)
    else:
        projection = tf.keras.layers.Dense(embed_dim)(inputs)
    if module == 'gru':
        x = tf.keras.layers.GRU(embed_dim, return_sequences=True)(projection)
    elif module == 'lstm':
        x = tf.keras.layers.LSTM(embed_dim, return_sequences=True)(projection)
    else:
        x = transformer((maxlen, embed_dim), embed_dim, num_heads, ff_dim)(projection)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    outputs = tf.keras.layers.Dense(23, activation="sigmoid")(x)
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss='binary_crossentropy', metrics=["binary_accuracy", "Precision", "Recall"])
    return model

# Function to build the Siamese model using Transformer blocks
def angle_similarity_model(maxlen, module, embed_dim=64, num_heads=4, ff_dim=64, lr=1e-3):
    input_l = tf.keras.layers.Input(shape=(maxlen, 23))
    input_r = tf.keras.layers.Input(shape=(maxlen, 23))
    module = module.lower()
    if module == 'transformer':
        projection = positionEmbedding((maxlen, 23), maxlen=maxlen, embed_dim=embed_dim)
    else:
        projection = tf.keras.layers.Dense(embed_dim)
    if module == 'gru':
        seq_block = tf.keras.layers.GRU(embed_dim, return_sequences=True)
    elif module == 'lstm':
        seq_block = tf.keras.layers.LSTM(embed_dim, return_sequences=True)
    else:
        seq_block = transformer((maxlen, embed_dim), embed_dim, num_heads, ff_dim)
    embedding_l = seq_block(projection(input_l))
    embedding_r = seq_block(projection(input_r))
    embedding_l = tf.keras.layers.GlobalAveragePooling1D()(embedding_l)
    embedding_r = tf.keras.layers.GlobalAveragePooling1D()(embedding_r)
    dense = tf.keras.layers.Dense(32, activation="relu")
    embedding_l = dense(embedding_l)
    embedding_r = dense(embedding_r)
    combined = tf.keras.layers.Lambda(function=lambda x: tf.abs(x[0]-x[1]), output_shape=lambda s: s[0])([embedding_l, embedding_r])
    outputs = tf.keras.layers.Dense(23, activation="sigmoid")(combined)
    model = tf.keras.models.Model(inputs=[input_l, input_r], outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss='binary_crossentropy', metrics=["binary_accuracy", "Precision", "Recall"])
    return model

class TQDMProgressBar(tf.keras.callbacks.Callback):
    def __init__(self, epochs):
        super().__init__()
        self.epochs = epochs
        self.epoch_bar = None

    def on_train_begin(self, logs=None):
        # Initialize tqdm bar
        self.epoch_bar = tqdm(total=self.epochs, desc="Training Progress", unit="epoch")

    def on_epoch_end(self, epoch, logs=None):
        # Update after each epoch
        self.epoch_bar.update(1)
        # Optionally show metrics in postfix
        if logs:
            self.epoch_bar.set_postfix({k: f"{v:.4f}" for k, v in logs.items()})

    def on_train_end(self, logs=None):
        # Close the progress bar
        self.epoch_bar.close()
