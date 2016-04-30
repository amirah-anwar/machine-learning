from network import Network
import numpy as np

def read_data(file, split_ratio = .8):
    # Data is in the format feature1 feature2 .. featureN label(+1, -1)
    data = []
    labels = []
    with open(file) as fi:
        for lines in fi:
            lines = lines.strip()
            elements = lines.split()
            features = []
            for item in elements[:-1]:
                features.append(float(item))
            data.append(features)
            if elements[-1][0] == '+':
                labels.append(1)
            else:
                labels.append(0)

    #data = np.matrix(data)
    #labels = np.array(labels)
    data = np.array(data)
    labels = np.array(labels)
    print('The data shape is ',data.shape)
    print('The label shape is', labels.shape)
    assert data.shape[0] == labels.shape[0]
    indices = np.random.permutation(data.shape[0])
    data = data[indices]
    labels = labels[indices]
    train_indices = int(data.shape[0] * split_ratio)
    train_data = data[:train_indices]
    train_label = labels[:train_indices]

    test_data = data[train_indices :]
    test_label = labels[train_indices:]
    assert train_data.shape[0] + test_data.shape[0] == data.shape[0]
    return (train_data, train_label), (test_data, test_label)
train, test = read_data('nnsvm-data.txt', .60)
sample_network = Network([2,6, 2])
sample_network.train(train, test, .20, 10)
