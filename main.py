from nn import *
import csv
import numpy as np

input_array = []
output_array = []


with open('train.csv', 'r') as csv_file:
    data = csv.reader(csv_file)

    next(data)

    pclass=[]
    sex=[]
    age=[]
    sibsp=[]
    parch=[]
    fare=[]
    embarked=[]

    for row in data:
        output_array.append([float(row[1])])

        pclass.append(float(row[2]))
        sex.append(0) if row[4] == "male" else sex.append(1)
        age.append(30) if row[5] == "" else age.append(float(row[5]))
        sibsp.append(float(row[6]))
        parch.append(float(row[7]))
        fare.append(float(row[9]))
        embarked.append(0) if row[11] == '' else embarked.append(ord(row[11])-64)

    input_array.append(pclass)
    input_array.append(sex)
    input_array.append(age)
    input_array.append(sibsp)
    input_array.append(parch)
    input_array.append(fare)
    input_array.append(embarked)

X = np.array(input_array)

Y = np.array(output_array).T

## 7 Inputs -> class, sex, age, sibsp, parch, fare, embarked

nn_architecture = [
    {"input_dim": 7, "output_dim": 10, "activation": "relu"},   
    {"input_dim": 10, "output_dim": 10, "activation": "relu"},
    {"input_dim": 10, "output_dim": 10, "activation": "relu"},
    {"input_dim": 10, "output_dim": 5, "activation": "relu"},
    {"input_dim": 5, "output_dim": 1, "activation": "sigmoid"},
]


test = []
    

nn = NeuralNetwork(nn_architecture)

nn.train(X, Y, 10000, 0.01)

test.append([float(input("Which class was this passenger? (1/2/3)   "))])
test.append([float(input("Was this passenger male or female? (male = 0, female = 1)    "))])
test.append([float(input("How old was this passenger?   "))])
test.append([float(input("How many siblings or spouses did this passenger have on board?    "))])
test.append([float(input("How many parents or children did this passenger have on board?    "))])
test.append([float(input("How much was this passengers ticket?    "))])
test.append([float(input("Where did this passenger depart from? (C = 3, Q = 17, S = 19)   "))])

print(nn.predict(np.asarray(test))[0][0])