import pandas 
#implementando o algoritmo perceptrom. 
#weights: quanto impacto cada dado deve ter no processo de aprendizagem.
#row: linhas da tabela de treinamento ou teste.
#l_rate: learing rate, usado para limitar o quanto cada nodo é corrigido toda vez que ele [nodo] é atualizado
#epoch: numero de vezes que deve "estudar" a base de teste enquanto atualiza o peso
#

def predict(row, weights):
    activation = weights[0]
    for i in range(len(row)-1):
        activation += weights[i + 1] * row[i]
    return 1.0 if activation >= 0.0 else 0.0


#treinando os pesos dos inputs 
def train_weights(tab_train, l_rate, n_epoch):
    weights = [0.0 for i in range(len(tab_train[0]))] #criando uma lista vazia com a mesma quantidade de colunas de tab_train
    for epoch in range(n_epoch):
        sum_error = 0.0
        for row in tab_train:
            prediction = predict(row, weights)             
            error = row[-1] - prediction #indo para a ultima posicao do array
            sum_error += error**2 
            weights[0] = weights[0] + l_rate * error
            for i in range(len(row)-1):
                weights[i + 1] = weights[i + 1] + l_rate * error * row[i]
    return weights   

#formata o output
def perceptron(train, test, l_rate, n_epoch):
    predictions = list()
    weights = train_weights(train, l_rate, n_epoch)
    print('==========Tabela de respostas==========')
    for row in test:
        prediction = predict(row, weights)
        print('entrada01=%d, entrada02=%d, resposta_verdadeira=%d, agoritmo_resposta=%d' % (row[1], row[2], row[-1], prediction))
        predictions.append(prediction)
    print('=======================================')
    return(predictions)

#aplicando o perceptron para AND
datasetTreinoAND = [[0, 0, 0],  
                    [1, 0, 0],  
                    [0, 1, 0],  
                    [1, 1, 1]] 

datasetAND  = [[0, 1, 0],  
               [1, 1, 1],  
               [1, 0, 0],  
               [0, 0, 0],  
               [1, 1, 1],  
               [0, 1, 0],  
               [1, 0, 0],  
               [0, 1, 0],  
               [0, 1, 0],  
               [1, 1, 1]]

datasetTreinoOR = [[0, 0, 0],  
                   [1, 0, 1],  
                   [0, 1, 1],  
                   [1, 1, 1]]  

datasetOr = [[0, 1, 1],  
             [0, 0, 0],  
             [1, 0, 1],  
             [0, 0, 0],  
             [1, 1, 1],  
             [0, 1, 1],  
             [1, 0, 1],  
             [0, 1, 1],  
             [0, 1, 1],  
             [0, 0, 0]]

# XOR
datasetTreinoXOR = [[0, 0, 0],  
                    [1, 0, 1],  
                    [0, 1, 1],  
                    [1, 1, 0]]

datasetXor = [[0, 1, 1],  
              [0, 0, 0],  
              [1, 0, 1],  
              [0, 0, 0],  
              [1, 1, 0],  
              [0, 1, 1],  
              [1, 0, 1],  
              [0, 1, 1],  
              [0, 1, 1],  
              [0, 0, 0]]

l_rate = 0.1
n_epoch = 5
resposta = perceptron(datasetTreinoAND, datasetAND, l_rate, n_epoch) 

