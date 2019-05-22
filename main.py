from numpy import exp, array, random, dot


class NeuralNetwork():
    def __init__(self):
        # Semeia o gerador de números aleatórios, por isso gera os mesmos números
        # toda vez que o programa roda.
        random.seed(1)

        # Nós modelamos um único neurônio, com 3 conexões de entrada e 1 conexão de saída.
        # Atribuímos pesos aleatórios a uma matriz 3 x 1, com valores no intervalo de -1 a 1
        # e significa 0.
        self.synaptic_weights = 2 * random.random((3, 1)) - 1

    # A função sigmóide, que descreve uma curva em forma de S.
    # Nós passamos a soma ponderada das entradas através desta função para
    # normalize-os entre 0 e 1.
    def __sigmoid(self, x):
        return 1 / (1 + exp(-x))

    # A derivada da função Sigmoide.
    # Este é o gradiente da curva sigmóide.
    # Isso indica como estamos confiantes sobre o peso existente.
    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    # Nós treinamos a rede neural através de um processo de tentativa e erro.
    # Ajustando os pesos sinápticos a cada vez.
    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
        for iteration in range(number_of_training_iterations):
            # Passe o conjunto de treinamento através da nossa rede neural (um único neurônio).
            output = self.think(training_set_inputs)

            # Calcule o erro (a diferença entre a saída desejada
            # e a saída prevista).
            error = training_set_outputs - output

            # Multiplique o erro pela entrada e novamente pelo gradiente da curva Sigmoide.
            # Isso significa que pesos menos confiantes são ajustados mais.
            # Isso significa que as entradas, que são zero, não causam alterações nos pesos.
            adjustment = dot(training_set_inputs.T, error * self.__sigmoid_derivative(output))

            # Ajuste os pesos.
            self.synaptic_weights += adjustment

    # A rede neural pensa.
    def think(self, inputs):
        # Passe as entradas através da nossa rede neural (nosso único neurônio).
        return self.__sigmoid(dot(inputs, self.synaptic_weights))


if __name__ == "__main__":

    # Inicialize uma rede neural de um único neurônio.
    neural_network = NeuralNetwork()

    print ("Pesos sinápticos iniciais aleatórios: ")
    print (neural_network.synaptic_weights)

    # O conjunto de treinamento. Nós temos 4 exemplos, cada um consistindo em 3 valores de entrada
    # e 1 valor de saída.
    training_set_inputs = array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
    training_set_outputs = array([[0, 1, 1, 0]]).T

    # Treine a rede neural usando um conjunto de treinamento.
    # Faça isso 10.000 vezes e faça pequenos ajustes a cada vez.
    neural_network.train(training_set_inputs, training_set_outputs, 10000)

    print ("Novos pesos sinápticos depois do treinamento: ")
    print (neural_network.synaptic_weights)

    # Teste a rede neural com uma nova situação.
    print ("Considerando nova situação [1, 0, 0] -> ?: ")
    print (neural_network.think(array([1, 0, 0])))
