import json
from config import learning_rate, threshold

weights = []
bias = []

def initialize():
    #init a 3d array of weights:[[to-x-connected 2d weights], [to-o-connected 2d weights]]
    for k in range(2):
        weights.append([])
        for i in range(5):
            weights[k].append([])
            for j in range(5):
                weights[k][i].append(0)     
        bias.append(0) #connected to x
        bias.append(0) #connected to o

def activation(Yin):
    if Yin > threshold:
        return 1
    elif -threshold <= Yin <= threshold:
        return 0
    else:
        return -1

def encode_label(label):
    #target vector is [x, o]
    # [1, -1] for x and [-1, 1] for o
    label = label.lower()
    if label == 'x':
        return [1, -1]
    elif label == 'o':
        return [-1, 1]
    else:
        raise ValueError('please select a valid label among x or o') 

def decode_label(estimation):
    if estimation == [1, -1]:
        return 'X'
    elif estimation == [-1, 1]:
        return 'O'
    else:
        return 'NONE'

def train():

    file = None
    
    try:

        initialize()

        file = open('dataset.txt', 'r')
        data_set = file.readline()
        file.close()

        data_set = json.loads(data_set)

        #run perceptron algorithm:
        changed = False
        first_pass = True
        epochs = 0

        while changed or first_pass:
            first_pass = False
            changed = False
            for data in data_set:
                for k in range(2):
                    Yin = 0
                    for i in range(5):
                        for j in range(5):
                            Yin += weights[k][i][j]*data['features'][i][j]
                    Yin += bias[k]
                    if activation(Yin) != encode_label(data['label'])[k]:
                        #has error so update weights:
                        changed = True
                        for i in range(5):
                            for j in range(5):
                                weights[k][i][j] += learning_rate * data['features'][i][j] * encode_label(data['label'])[k]
                        bias[k] += learning_rate * encode_label(data['label'])[k]
            epochs += 1

        print(f'training successful through {epochs} epochs!')
        print('weights:\n', weights)

    except ValueError as err:
        if err:
            print(err)
        else:
            print('An unExpected error occured!')

    finally:
        if file:
            file.close()


def test(test_data):
    try:
        
        estimation = [0, 0]

        for k in range(2):
            Yin = 0
            for i in range(5):
                for j in range(5):
                    Yin += test_data[i][j]*weights[k][i][j]
            Yin += bias[k]
            print(f"k = {k}, Yin = {Yin}")
            estimation[k] = activation(Yin)

        return decode_label(estimation)


    except:
        print("An Unexpected error occured!")
        