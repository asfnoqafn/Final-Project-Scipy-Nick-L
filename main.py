import sys
from utils import *


def main(input_len, output_len, lr, filename):
    """Do all the needed steps for time series prediction."""
    tf.random.set_seed(42)
    np.random.seed(42)

    # Loading and preprocessing of the data
    data, date = load_data(input_len, output_len, filename)
    train_df, test, train_min, train_max = preprocess(
        data, input_len, output_len)
    train_feature, train_label, val_feature, val_label = window(
        train_df, input_len, output_len)

    # Creation of model
    # Single LSTM layer followed by 2 Dense-Layers with the second
    # having the required output shape
    model = Sequential()
    model.add(LSTM(64, activation='relu', input_shape=(input_len, 1)))
    model.add(Dense(32))
    model.add(Dense(output_len))
    model.compile(
        loss='mae',
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=lr))

    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    model.fit(train_feature, train_label, epochs=20, verbose=1,
              validation_data=(val_feature, val_label),
              callbacks=[callback], shuffle=True)

    # Prediction datashape needs to be (1,inputlen) therefore we need
    # only the first inputlen datapoints from the test dataset
    prediction = model.predict(np.reshape(test[:input_len], (1, input_len)))

    # Denormalization of predictiondata
    prediction = (prediction) * (train_max - train_min) + train_min

    plot(prediction, data[-150:], date[-150:], output_len)


if __name__ == "__main__":
    FILENAME = "Dax.csv"
    INPUT_LEN = 20
    OUTPUT_LEN = 5
    LEARNING_RATE = 0.001

    # Parse command line arguments
    if len(sys.argv) > 2:
        INPUT_LEN = int(sys.argv[1])
        OUTPUT_LEN = int(sys.argv[2])
    if len(sys.argv) > 3:
        LEARNING_RATE = float(sys.argv[3])
    if len(sys.argv) > 4:
        FILENAME = sys.argv[4]

    print("Inputlength: " + str(INPUT_LEN))
    print("Outputlength: " + str(OUTPUT_LEN))
    print("Learning rate: " + str(LEARNING_RATE))
    print("Filename: " + FILENAME)

    main(INPUT_LEN, OUTPUT_LEN, LEARNING_RATE, FILENAME)
