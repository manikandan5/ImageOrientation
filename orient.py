#Authors : Manikandan Murugesan, Debasis Dwivedy

#Report
"""
    We have successfully implemented the KNN algorithm and Neural Networks algorithm.
    KNN algorithm though takes a lot of time to run gives us approximately 69% accuracy when learned from the whole test date
    While our Neural Networks gives us an accuracy of 40%
    For our best algorithm, we have converted the RGB vectors to monochrome vectors using formulas and then iterated on them, so that the time is lowered considerably.
    As far as KNN is concerned, we have observed that the accuracy doesn't vary that much for the values of K.
    Currently for Best, we are generating the test data every time by reading from the file and processing it.
    For improvement, we could just write it into the file the first time and then read it from the file later on.
    More details in the report
"""
import sys
import numpy as np
import math
import random
import time

#Intialisation Data
trainData = {}
trainOri = {}
testOri = {}
testData = {}
correctOri = {}
tempData = {}
deviationData = {}

#Checking for Inputs
if len(sys.argv) != 5:
    print "Usage: python orient.py train_file.txt test_file.txt mode value"
    sys.exit()

#Reading the inputs
trainFile = sys.argv[1]
testFile = sys.argv[2]
mode = sys.argv[3]
k = int(sys.argv[4])
hidden_count = int(sys.argv[4])

#Function to add two matrices
def matrix_add(A,B):
    Z = []
    for i in range(len(A)):
        row = []
        for j in range(len(A[i])):
            row.append(A[i][j] + B[i][j])
        Z.append(row)
    return Z

#Reading file from Training Data for KNN
def ReadTrainData(fName):

    file = open(fName, 'r')

    for line in file:
        imgData = line.split()
        imgFile = imgData[0]
        trainOri[imgFile] = imgData[1]
        imgVec = [int(i) for i in imgData[2:194]]
        trainData[imgFile+"-"+trainOri[imgFile]] = np.array(imgVec)

#Reading file from Test Data for KNN
def ReadTestData(fName):

    file = open(fName, 'r')

    for line in file:
        imgData = line.split()
        imgFile = imgData[0]
        correctOri[imgFile] = imgData[1]
        imgVec = [int(i) for i in imgData[2:194]]
        testData[imgFile] = np.array(imgVec)
        tempData[imgFile] = 10000000

#KNN Algorithm run
def KNNRun():
    i = 0
    for testImg in testData:
        i=i+1
        print "Processing Image "+testImg
        for trainImg in trainOri:

            for ori in ['0','90','180','270']:

                vectorDifference = trainData[trainImg+"-"+ori] - testData[testImg]

                vectorDifference = [item*item for item in vectorDifference]

                deviation = np.sum(vectorDifference)

                deviation = math.sqrt(deviation)

                deviationData[trainImg+"-"+ori] = deviation
    
    #Learning from all the orientations of the image
        count = 0
        orientationCount = {0:0, 90:0, 180:0, 270:0 }
        for w in sorted(deviationData, key=deviationData.get, reverse=False):
            if count < k:
                if(w.split("-")[1]=="0"):
                    orientationCount[0] = orientationCount[0] + 1
                elif(w.split("-")[1]=="90"):
                    orientationCount[90] = orientationCount[90] + 1
                elif(w.split("-")[1]=="180"):
                    orientationCount[180] = orientationCount[180] + 1
                elif(w.split("-")[1]=="270"):
                    orientationCount[270] = orientationCount[270] + 1

                count = count+1
        testOri[testImg] = sorted(orientationCount, key =orientationCount.get, reverse=True)[0]

#Calculating Confusion Matrix for KNN
def ConfusionMatrixCalculation():
    confusion_matrix1=[[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]

    for row in testOri:


        if int(correctOri[row])==0:
            if int(testOri[row])==0 :
                predicted_matrix=[[1,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]
            elif testOri[row]==90:
                predicted_matrix=[[0,1,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]
            elif testOri[row]==180:
                predicted_matrix=[[0,0,1,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]
            elif testOri[row]==270:
                predicted_matrix=[[0,0,0,1],[0,0,0,0],[0,0,0,0],[0,0,0,0]]
            else:
                predicted_matrix=[[1,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]
        elif int(correctOri[row])==90:
            if int(testOri[row])==0 :
                predicted_matrix=[[0,0,0,0],[1,0,0,0],[0,0,0,0],[0,0,0,0]]
            elif int(testOri[row])==90:
                predicted_matrix=[[0,0,0,0],[0,1,0,0],[0,0,0,0],[0,0,0,0]]
            elif int(testOri[row])==180:
                predicted_matrix=[[0,0,0,0],[0,0,1,0],[0,0,0,0],[0,0,0,0]]
            elif int(testOri[row])==270:
                predicted_matrix=[[0,0,0,0],[0,0,0,1],[0,0,0,0],[0,0,0,0]]
            else:
                predicted_matrix=[[0,0,0,0],[0,1,0,0],[0,0,0,0],[0,0,0,0]]
        elif int(correctOri[row])==180:
            if int(testOri[row])==0 :
                predicted_matrix=[[0,0,0,0],[0,0,0,0],[1,0,0,0],[0,0,0,0]]
            elif int(testOri[row])==90:
                predicted_matrix=[[0,0,0,0],[0,0,0,0],[0,1,0,0],[0,0,0,0]]
            elif int(testOri[row])==180:
                predicted_matrix=[[0,0,0,0],[0,0,0,0],[0,0,1,0],[0,0,0,0]]
            elif int(testOri[row])==270:
                predicted_matrix=[[0,0,0,0],[0,0,0,0],[0,0,0,1],[0,0,0,0]]
            else:
                predicted_matrix=[[0,0,0,0],[0,0,0,0],[0,0,1,0],[0,0,0,0]]
        elif int(correctOri[row])==270:
            if int(testOri[row])==0 :
                predicted_matrix=[[0,0,0,0],[0,0,0,0],[0,0,0,0],[1,0,0,0]]
            elif int(testOri[row])==90:
                predicted_matrix=[[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,1,0,0]]
            elif int(testOri[row])==180:
                predicted_matrix=[[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,1,0]]
            elif int(testOri[row])==270:
                predicted_matrix=[[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,1]]
            else:
                predicted_matrix=[[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,1]]

        confusion_matrix1=matrix_add(confusion_matrix1,predicted_matrix)
    return confusion_matrix1

#Calculating the accuracy for KNN
def AccuracyCalculation():
    correct = 0
    count = 0
    for item in testOri:
        if int(testOri[item])==int(correctOri[item]):
            correct = correct + 1
        count = count + 1
    print "Accuracy - ",
    print (float(correct)/count)*100
    print "Confusion Matrix : "
    print "Values\t\t0\t\t90\t\t180\t\t270 "
    print "-"*40
    confusion_matrix1 = ConfusionMatrixCalculation()
    orientation = ["0","90","180","270"]
    i=0
    for row in confusion_matrix1:
        print orientation[i]+"\t\t\t",
        for col in row:
            print col,
            print "\t\t",
        print
        i=i+1

#Writing output to the file
def FileWrite():
    file = open("knn output.txt",'w')
    for image in testOri:
        line = image+" "+str(testOri[image])+"\n"
        file.write(line)
    file.close()

#KNN Main call
def KNN():
    ReadTrainData(trainFile)
    ReadTestData(testFile)
    KNNRun()
    AccuracyCalculation()
    FileWrite()

#Code for Neural Networks
class NeuralNetwork:
    LEARNING_RATE = 0.5

    def __init__(self, num_inputs, num_hidden, num_outputs, hidden_layer_weights = None, hidden_layer_bias = None, output_layer_weights = None, output_layer_bias = None):
        self.num_inputs = num_inputs

        self.hidden_layer = NeuronLayer(num_hidden, hidden_layer_bias)
        self.output_layer = NeuronLayer(num_outputs, output_layer_bias)

        self.init_weights_from_inputs_to_hidden_layer_neurons(hidden_layer_weights)
        self.init_weights_from_hidden_layer_neurons_to_output_layer_neurons(output_layer_weights)

    def init_weights_from_inputs_to_hidden_layer_neurons(self, hidden_layer_weights):
        weight_num = 0
        for h in range(len(self.hidden_layer.neurons)):
            for i in range(self.num_inputs):
                if not hidden_layer_weights:
                    self.hidden_layer.neurons[h].weights.append(random.random())
                else:
                    self.hidden_layer.neurons[h].weights.append(hidden_layer_weights[weight_num])
                weight_num += 1

    def init_weights_from_hidden_layer_neurons_to_output_layer_neurons(self, output_layer_weights):
        weight_num = 0
        for o in range(len(self.output_layer.neurons)):
            for h in range(len(self.hidden_layer.neurons)):
                if not output_layer_weights:
                    self.output_layer.neurons[o].weights.append(random.random())
                else:
                    self.output_layer.neurons[o].weights.append(output_layer_weights[weight_num])
                weight_num += 1

    def inspect(self):
        print('------')
        print('* Inputs: {}'.format(self.num_inputs))
        print('------')
        print('Hidden Layer')
        self.hidden_layer.inspect()
        print('------')
        print('* Output Layer')
        self.output_layer.inspect()
        print('------')

    def feed_forward(self, inputs):
        hidden_layer_outputs = self.hidden_layer.feed_forward(inputs)
        return self.output_layer.feed_forward(hidden_layer_outputs)

    def train(self, training_inputs, training_outputs):
        self.feed_forward(training_inputs)
        pd_errors_wrt_output_neuron_total_net_input = [0] * len(self.output_layer.neurons)

        for o in range(len(self.output_layer.neurons)):
            pd_errors_wrt_output_neuron_total_net_input[o] = self.output_layer.neurons[o].calculate_pd_error_wrt_total_net_input(training_outputs[o])
        pd_errors_wrt_hidden_neuron_total_net_input = [0] * len(self.hidden_layer.neurons)

        for h in range(len(self.hidden_layer.neurons)):
            d_error_wrt_hidden_neuron_output = 0
            for o in range(len(self.output_layer.neurons)):
                d_error_wrt_hidden_neuron_output += pd_errors_wrt_output_neuron_total_net_input[o] * self.output_layer.neurons[o].weights[h]
            pd_errors_wrt_hidden_neuron_total_net_input[h] = d_error_wrt_hidden_neuron_output * self.hidden_layer.neurons[h].calculate_pd_total_net_input_wrt_input()

        for o in range(len(self.output_layer.neurons)):
            for w_ho in range(len(self.output_layer.neurons[o].weights)):
                pd_error_wrt_weight = pd_errors_wrt_output_neuron_total_net_input[o] * self.output_layer.neurons[o].calculate_pd_total_net_input_wrt_weight(w_ho)
                self.output_layer.neurons[o].weights[w_ho] -= self.LEARNING_RATE * pd_error_wrt_weight

        for h in range(len(self.hidden_layer.neurons)):
            for w_ih in range(len(self.hidden_layer.neurons[h].weights)):
                pd_error_wrt_weight = pd_errors_wrt_hidden_neuron_total_net_input[h] * self.hidden_layer.neurons[h].calculate_pd_total_net_input_wrt_weight(w_ih)
                self.hidden_layer.neurons[h].weights[w_ih] -= self.LEARNING_RATE * pd_error_wrt_weight

    def calculate_total_error(self, training_sets):
        total_error = 0
        for t in range(len(training_sets)):
            training_inputs, training_outputs = training_sets[t]
            self.feed_forward(training_inputs)
            for o in range(len(training_outputs)):
                total_error += self.output_layer.neurons[o].calculate_error(training_outputs[o])
        return total_error

class NeuronLayer:
    def __init__(self, num_neurons, bias):
        self.bias = bias if bias else random.random()
        self.neurons = []
        for i in range(num_neurons):
            self.neurons.append(Neuron(self.bias))

    def inspect(self):
        layer_weights=[]
        for n in range(len(self.neurons)):
            neuron_weights=[]
            for w in range(len(self.neurons[n].weights)):
                neuron_weights.append(self.neurons[n].weights[w])
            layer_weights.append(neuron_weights)
        return layer_weights

    def feed_forward(self, inputs):
        outputs = []
        for neuron in self.neurons:
            outputs.append(neuron.calculate_output(inputs))
        return outputs

    def get_outputs(self):
        outputs = []
        for neuron in self.neurons:
            outputs.append(neuron.output)
        return outputs

class Neuron:
    def __init__(self, bias):
        self.bias = bias
        self.weights = []

    def calculate_output(self, inputs):
        self.inputs = inputs
        self.output = self.squash(self.calculate_total_net_input())
        return self.output

    def calculate_total_net_input(self):
        total = 0
        for i in range(len(self.inputs)):
            total += self.inputs[i] * self.weights[i]
        return total + self.bias

    def squash(self, total_net_input):
        return 1 / (1 + math.exp(-total_net_input))

    def calculate_pd_error_wrt_total_net_input(self, target_output):
        return self.calculate_pd_error_wrt_output(target_output) * self.calculate_pd_total_net_input_wrt_input();

    def calculate_error(self, target_output):
        return 0.5 * (target_output - self.output) ** 2

    def calculate_pd_error_wrt_output(self, target_output):
        return -(target_output - self.output)

    def calculate_pd_total_net_input_wrt_input(self):
        return self.output * (1 - self.output)

    def calculate_pd_total_net_input_wrt_weight(self, index):
        return self.inputs[index]

def readinputfile(name):
    input_file_matrix=[]
    with open(name, 'r') as f:
        for line in f:
            line_matrix=[]
            output=[]
            color_combination=[]
            words = line.split()
            line_matrix.append(words.pop(0))
            output=words.pop(0)
            if output=='0' :
                output=[1,0,0,0]
            elif output=='90':
                output=[0,1,0,0]
            elif output=='180':
                output=[0,0,1,0]
            elif output=='270':
                output=[0,0,0,1]
            else:
                output=[1,0,0,0]

            line_matrix.append(output)
            color_list=[int(numeric_string) for numeric_string in words]
            for i in range(0,len(color_list)-2,3):
                r=+ color_list[i]
                g=+ color_list[i+1]
                b=+color_list[i+2]
            color_combination.append(r/64)
            color_combination.append(g/64)
            color_combination.append(b/64)
            line_matrix.append(color_combination)
            input_file_matrix.append(line_matrix)
    return input_file_matrix

def readoutputfile(name):
    input_file_matrix=[]
    with open(name, 'r') as f:
        for line in f:
            line_matrix=[]
            output=[]
            color_combination=[]
            words = line.split()
            line_matrix.append(words.pop(0))
            line_matrix.append(words.pop(0))
            color_list=[int(numeric_string) for numeric_string in words]
            for i in range(0,len(color_list)-2,3):
                r=+ color_list[i]
                g=+ color_list[i+1]
                b=+color_list[i+2]
            color_combination.append(r/64)
            color_combination.append(g/64)
            color_combination.append(b/64)
            line_matrix.append(color_combination)
            input_file_matrix.append(line_matrix)
    return input_file_matrix

def file_write(solution):
        file = open("nnet output.txt", "w")
        for row in solution:
            file.write(''.join(str(e) for e in row))
            file.write("\n")
        file.close()

def Best():
    file = open(testFile, 'r')
    vec = []
    for line in file:
        imgData = line.split()
        imgFile = imgData[0]
        correctOri[imgFile] = imgData[1]
        imgVec = [int(i) for i in imgData[2:194]]
        for values in range(len(imgVec)/3):
            r=imgVec[0]
            imgVec.remove(imgVec[0])
            g=imgVec[0]
            imgVec.remove(imgVec[0])
            b=imgVec[0]
            imgVec.remove(imgVec[0])
            mono = (0.2125 * r) + (0.7154 * g) + (0.0721 * b)
            vec.append(mono)
        testData[imgFile] = np.array(vec)
        tempData[imgFile] = 10000000
    k=3
    KNN()
    return

def NNET():
    ##############  The Main Function#####################################################

    start=time.time()
    hidden_count=5
    success_rate=0
    confusion_matrix=[[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]
    hidden_layer_weights=[round(random.random(),9) for _ in range(0, 3*hidden_count)]
    output_layer_weights=[round(random.random(),9) for _ in range(0, 4*hidden_count)]
    nn = NeuralNetwork(3, hidden_count, 4,
                               hidden_layer_weights=hidden_layer_weights,
                               hidden_layer_bias=0,
                               output_layer_weights=output_layer_weights,
                               output_layer_bias=0)

    ###########Training The Network##########################
    input_file_matrix=readinputfile(trainFile)
    for i in range(100):
        for row in input_file_matrix:
            nn.train(row[2], row[1])

    ######## Testing The Network#####################################

    input_file_matrix=readoutputfile(testFile)
    output_file=[]
    for row in input_file_matrix:
        output_per_input=[]
        outputs=nn.feed_forward(row[2])
        m = max(outputs)
        index=[i for i, j in enumerate(outputs) if j == m]
        if index[0]==0 :
            output=0
        elif index[0]==1:
            output=90
        elif index[0]==2:
            output=180
        elif index[0]==3:
            output=270
        else:
            output=0

        output_per_input.append(row[0])
        output_per_input.append(' ')
        output_per_input.append(output)
        if int(row[1])==0:
            if output==0 :
                predicted_matrix=[[1,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]
            elif output==90:
                predicted_matrix=[[0,1,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]
            elif output==180:
                predicted_matrix=[[0,0,1,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]
            elif output==270:
                predicted_matrix=[[0,0,0,1],[0,0,0,0],[0,0,0,0],[0,0,0,0]]
            else:
                predicted_matrix=[[1,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]
        elif int(row[1])==90:
            if output==0 :
                predicted_matrix=[[0,0,0,0],[1,0,0,0],[0,0,0,0],[0,0,0,0]]
            elif output==90:
                predicted_matrix=[[0,0,0,0],[0,1,0,0],[0,0,0,0],[0,0,0,0]]
            elif output==180:
                predicted_matrix=[[0,0,0,0],[0,0,1,0],[0,0,0,0],[0,0,0,0]]
            elif output==270:
                predicted_matrix=[[0,0,0,0],[0,0,0,1],[0,0,0,0],[0,0,0,0]]
            else:
                predicted_matrix=[[0,0,0,0],[0,1,0,0],[0,0,0,0],[0,0,0,0]]
        elif int(row[1])==180:
            if output==0 :
                predicted_matrix=[[0,0,0,0],[0,0,0,0],[1,0,0,0],[0,0,0,0]]
            elif output==90:
                predicted_matrix=[[0,0,0,0],[0,0,0,0],[0,1,0,0],[0,0,0,0]]
            elif output==180:
                predicted_matrix=[[0,0,0,0],[0,0,0,0],[0,0,1,0],[0,0,0,0]]
            elif output==270:
                predicted_matrix=[[0,0,0,0],[0,0,0,0],[0,0,0,1],[0,0,0,0]]
            else:
                predicted_matrix=[[0,0,0,0],[0,0,0,0],[0,0,1,0],[0,0,0,0]]
        elif int(row[1])==270:
            if output==0 :
                predicted_matrix=[[0,0,0,0],[0,0,0,0],[0,0,0,0],[1,0,0,0]]
            elif output==90:
                predicted_matrix=[[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,1,0,0]]
            elif output==180:
                predicted_matrix=[[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,1,0]]
            elif output==270:
                predicted_matrix=[[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,1]]
            else:
                predicted_matrix=[[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,1]]

        confusion_matrix=matrix_add(confusion_matrix,predicted_matrix)
        if output==int(row[1]):
            success_rate=success_rate+1
        output_file.append(output_per_input)

    #print success_rate
    #print len(output_file)
    accuracy=float(success_rate)/float(len(output_file))
    print "Accuracy Percentage is:",accuracy*100

    ######  Matrix   Output   Without Using Nympy#########################
    print "Without using Nympy Matrix display:-"
    print confusion_matrix

    ######  Matrix   Output   Using Nympy#########################
    print "With using Nympy Matrix display:-"
    Confusion_Matrix = np.array(confusion_matrix)
    print Confusion_Matrix

    file_write(output_file)
    end=time.time()
    print end-start



if mode == "best":
    Best()
elif mode == "knn":
    KNN()
elif mode == "nnet":
    NNET()
