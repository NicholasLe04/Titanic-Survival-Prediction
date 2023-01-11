from nn import *
import csv
import numpy as np

input_array = [[],[],[],[],[],[],[]]
output_array = []


with open('train.csv', 'r') as csv_file:
    data = csv.reader(csv_file)

    next(data)


    for row in data:
        output_array.append([float(row[1])])
        input_array[0].append(float(row[2]))
        input_array[1].append(0) if row[4] == "male" else input_array[1].append(1)
        input_array[2].append(20) if row[5] == "" else input_array[2].append(float(row[5]))
        input_array[3].append(float(row[6]))
        input_array[4].append(float(row[7]))
        input_array[5].append(float(row[9]))
        input_array[6].append(13) if row[11] == '' else input_array[6].append(ord(row[11])-64)


training_inputs = np.array(input_array)

training_outputs = np.array(output_array).T

## 7 Inputs -> class, sex, age, sibsp, parch, fare, embarked

nn_architecture = [
    {"input_dim": 7, "output_dim": 10, "activation": "elu"},   
    {"input_dim": 10, "output_dim": 10, "activation": "elu"},
    {"input_dim": 10, "output_dim": 10, "activation": "elu"},
    {"input_dim": 10, "output_dim": 5, "activation": "elu"},
    {"input_dim": 5, "output_dim": 1, "activation": "tanh"},
]




nn = NeuralNetwork(nn_architecture, 10)

nn.train(training_inputs, training_outputs, 25000, 0.005)

while True:
    test = []

    test.append([float(input("\nWhich class was this passenger? (1/2/3)   "))])
    test.append([0]) if input("Was this passenger male or female?    ") == 'male' else test.append([1])
    test.append([float(input("How old was this passenger?   "))])
    test.append([float(input("How many siblings or spouses did this passenger have on board?    "))])
    test.append([float(input("How many parents or children did this passenger have on board?    "))])
    test.append([float(input("How much was this passenger's fare?    "))])
    test.append([float(input("Where did this passenger depart from? (Cherbourg = 3, Queenstown = 17, Southampton = 19)   "))])

    print("\nProbability of Survival: " + str(round(nn.predict(np.asarray(test)), 4) * 100) + "%\n")

    if (input("Type 'STOP' to stop or anything else to continue:    ") == 'STOP'):
        break