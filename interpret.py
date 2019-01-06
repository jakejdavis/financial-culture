import quandl
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pickle
import os.path

def normalise_windows(window_data, tag):
    if os.path.exists("scalers/%s.p" % tag):
        print("[Interpret][Normalise Windows] Loading scaler %s" % (tag))
        scaler = pickle.load(open("scalers/%s.p" % tag, "rb"))
    else:
        print("[Interpret][Normalise Windows] Creating scaler %s" % (tag))
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler = scaler.fit(window_data)

    print("[Interpret][Normalise Windows] Normalising %s" % (tag))
    normalised_data = scaler.transform(window_data)

    if not os.path.exists("scalers/%s.p" % tag):
        print("[Interpret][Normalise Windows] Saving scaler %s" % tag)
        pickle.dump(scaler, open("scalers/%s.p" % tag, "wb"))

    return normalised_data


def denormalise_windows(window_data, tag):
    print("[Interpret][Denormalise Windows] Loading scaler %s" % tag)
    scaler = pickle.load(open("scalers/%s.p" % tag, "rb"))
    print("[Interpret][Denormalise Windows] Denormalising window data")
    denormalised_data = scaler.inverse_transform(window_data)

    return denormalised_data

def abstract_matrix(matrix):
    abstracted_matrix = []
    initial_date = int(matrix[0]["Date"])
    for data_form in matrix:
        abstracted_matrix.append([int(data_form["Date"])-initial_date, data_form["Close"]])
    return np.array(abstracted_matrix)

def values_from_tag(tag, seq_len, normalise_window):
    # Get stocks
    data = quandl.get(tag, authtoken="hqUKYvKzjJxm23narJ4x")["Close"]
    

    sequence_length = seq_len + 1
    result = []
    for index in range(len(data) - sequence_length):
        result.append(data[index: index + sequence_length])
    
    if normalise_window:
        result = normalise_windows(result, tag)

    result = np.array(result)

    row = round(0.9 * result.shape[0])
    train = result[:int(row), :]
    np.random.shuffle(train)
    x_train = train[:, :-1]
    y_train = train[:, -1]
    x_test = result[int(row):, :-1]
    y_test = result[int(row):, -1]

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))  

    return [x_train, y_train, x_test, y_test]


if __name__ == "__main__":
    values = values_from_tag('EOD/DIS', 50, True)
    print(values)