from numpy import exp
from torch import nn, LongTensor, FloatTensor, ones

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

        return self.sigmoid(na[0][0]), na[0][1]

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

if __name__ == '__main__':
    net = NeuralNet()
    res = net.forward(1,2)
    print(net.getWeights())
    print("---------------")
    values = ones(2)
    net.changeWeight(2, values)

    print(net.getWeights())
