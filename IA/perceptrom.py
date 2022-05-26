#implementando o algoritmo perceptrom. 
#weights: quanto impacto cada dado deve ter no processo de aprendizagem.
#row: linhas da tabela de treinamento ou teste.
#l_rate: learing rate, usado para limitar o quanto cada nodo é corrigido toda vez que ele [nodo] é atualizado
#epoch: numero de vezes que deve "estudar" a base de teste enquanto atualiza o peso

#
def predict(row, weights):
    print("--------------------------------------------------------------------------------------------------------------------")
    activation = weights[0]
    print(len(row)-1)
    for i in range(len(row)-1):
        print("=====================")
        print("i = ",i)
        print("row[i] = ", row[i])
        print("=====================")
        activation += weights[i + 1] * row[i]
    return 1.0 if activation >= 0.0 else 0.0


#treinando os pesos dos inputs 
def train_weights(tab_train, l_rate, n_epoch)
    weights = [0.0 for i in range(len(tab_train[0]))] #criando uma lista vazia com a mesma quantidade de colunas de tab_train
    for epoch in range(n_epoch):
        sum_error = 0.0
        for row in tab_train:
            prediction = predict(row, weights)             
                       #indo para a ultima posicao do array
            error = row[-1] - prediction
            sum_error += error**2
            weights[0] = weights[0] + l_rate * error
            for i in range(len(row)-1):
                weights[i + 1] = weights[i + 1] + l_rate * error * row[i]
        print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))
    return weights   



# testes          W1          W2
dataset =   [[2.7810836,2.550537003,0],
            [1.465489372,2.362125076,0],
            [3.396561688,4.400293529,0],
            [1.38807019,1.850220317,0],
            [3.06407232,3.005305973,0],
            [7.627531214,2.759262235,1],
            [5.332441248,2.088626775,1],
            [6.922596716,1.77106367,1],
            [8.675418651,-0.242068655,1],
            [7.673756466,3.508563011,1]]
          #bias         X1                     X2 
weights = [-0.1, 0.20653640140000007, -0.23418117710000003]
for row in dataset:
    prediction = predict(row, weights)
    print("Expected=%d, Predicted=%d" % (row[-1], prediction))
