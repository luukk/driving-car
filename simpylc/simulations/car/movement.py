from numpy import exp
from typing import List
from torch import nn, LongTensor, FloatTensor, ones
from numpy.random import randint
from numpy.random import rand
import random as rnd
import lidar_pilot as lidarPilot
import subprocess as sub
import time as tm
import threading
import signal
import os


class NeuralNet:
    def __init__(self):
        self.NeuralNet = None

        self.initializeNetwork()

    def initializeNetwork(self):
        input_size = 2
        hidden_sizes = [2, 2]
        output_size = 2# Build a feed-forward network
        model = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            nn.Tanh(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.Tanh(),
            nn.Linear(hidden_sizes[1], output_size),
            nn.Tanh())

        self.NeuralNet = model


    def forward(self, distance, angle):
        x_data = FloatTensor([[distance, angle]])
        predict = self.NeuralNet(x_data)
        na = predict.detach().numpy()

        return self.sigmoid(na[0][0]), na[0][1] #velocity, angle. adjust weights every iteration. randomly mutate weights

    def getWeights(self):
        weights = []
        for name, param in self.NeuralNet.named_parameters():
            weights.append(param.data)

        return weights
            # print(param.data)

    def changeWeight(self, given_index, tensor):
        for index, (name, param) in enumerate(self.NeuralNet.named_parameters()):
            if index == given_index:
                # print(param.data)
                param.data = tensor

    def sigmoid(self, x):
        return 1 / (1 + exp(-x))


class Citizen:
    net = None
    distanceTraveled = 0
    timeTraveled = 0
    hitObstacle = False

    def __init__(self, net: NeuralNet):
        self.net = net
    
    def run(self):
        self.net.initializeNetwork()
        # w = self.net.getWeights()
        # ssl = []
        # for x in w:
        #     ssl.append("-".join(x))
        # nk = " ".join(ssl)
        proc = sub.Popen(['python3 runner.py'], stdout=sub.PIPE, shell=True, preexec_fn=os.setsid)
        tm.sleep(0.1)
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)



class GeneticAlgo:
    popsize = 10
    population = []
    gensize = 0
    crossover_rate = 0
    
    def __init__(self, gensize: int, crossover: int, population: List[NeuralNet]):
        self.popsize = len(population)
        self.gensize = gensize
        self.crossover_rate = crossover
        self.population = [Citizen(x) for x in population]
    
    def run(self, max_gens: int) -> None:
        self.runGeneration(self.population)
        # current_best = 0, self.objective(self.population[0])
        # best, bestEval = 10000, 0
        # for x in range(max_gens):
        #     scores = [self.objective(x) for x in self.population] 
        #     for s in scores:
        #         if s > current_best:
        #             current_best = s
        # return current_best

    def runGeneration(self, gen):
        work = []
        cs = 10
        for x in gen:
            work.append(threading.Thread(target=x.run()))
        currWork = work[:cs]
        
        for t in range(0,len(gen)/cs):
            for c in currWork:
                c.start()
            
            tm.sleep(5)
            for s in currWork:
                s.join()
            currWork = work[:cs*t]
    def objective(self, citizen: Citizen) -> int:
        score = citizen.run()
        return score
    
    def crossover(self, p1, p2) -> List:
        c1, c2 = p1.copy(), p2.copy()
        if rand() < self.crossover_rate:
            pt = randint(1, len(p1)-2)
            c1 = p1[:pt] + p2[pt:]
            c2 = p2[:pt] + p1[pt:]
        return [c1, c2]
        pass

    def mut(self, item) -> None:
        pass
    
    def select(self, population: List, k=3) -> None:
        selection = randint(len(population))
        for n in randint(0, len(population), k-1):
            if scores[n] < scores[selection]:
                selection = n
        return population[selection]


# if __name__ == '__main__':
#     popsize = 200
#     genLength = 20
#     crossoverRate = 0.9

#     initialPop = [NeuralNet() for x in range(0,popsize)]
#     ga = GeneticAlgo(genLength, crossoverRate, initialPop)

#     net = NeuralNet()
#     res = net.forward(1,2)
#     print(net.getWeights())
#     print("---------------")
#     values = ones(2)
#     net.changeWeight(2, values)

#     print(net.getWeights())
