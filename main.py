import tensorflow as tf
import numpy as np
from PIL import Image


#Importando o dataset 
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#Dados do dataset
treino = mnist.train.num_examples #Imagens para treino
validacao =  mnist.validation.num_examples #Imagens para validacao
teste = mnist.test.num_examples #Imagens para teste

entrada = 784 #Camada de entrada, são imagens de 28x28 pixels
camada1Esc = 512 #Camada oculta 1
camada2Esc = 256 #Camada oculta 2
camada3Esc = 128 #Camada oculta 3
numeroSaida = 10 #Camada de saida


#Learningrate, representa o quanto os parâmetros
#serão ajustados em cada etapa do processo de aprendizado
learning_rate = 1e-4

#Numero de iteracoes
numeroIteracoes = 1000

#lote se refere a quantos exemplos de treinamento 
#estamos usando em cada etapa
lote = 128

#A variável fora representa um 
#limiar no qual eliminamos algumas unidades aleatoriamente.
fora = 0.5


#Para X usamos um formato [None, 784], onde None representa qualquer 
#quantidade, pois estaremos alimentando em um número indefinido de 
#imagens de 784 pixels. O formato de Y é [None, 10] pois iremos usá-lo
#para um número indefinido de saídas de rótulo, com 10 classes possíveis. 
X = tf.placeholder("float", [None, entrada])
Y = tf.placeholder("float", [None, numeroSaida])
guardar = tf.placeholder(tf.float32) 


#Aqui são definidos os pesos
pesos = {
    'w1': tf.Variable(tf.truncated_normal([entrada, camada1Esc], stddev=0.1)),
    'w2': tf.Variable(tf.truncated_normal([camada1Esc, camada2Esc], stddev=0.1)),
    'w3': tf.Variable(tf.truncated_normal([camada2Esc, camada3Esc], stddev=0.1)),
    'out': tf.Variable(tf.truncated_normal([camada3Esc, numeroSaida], stddev=0.1)),
}

# bias ou tendência, usamos um pequeno valor constante para garantir que os tensores
# se ativem nos estágios iniciais e, portanto, contribuam para a propagação.
bias = {
    'b1': tf.Variable(tf.constant(0.1, shape=[camada1Esc])),
    'b2': tf.Variable(tf.constant(0.1, shape=[camada2Esc])),
    'b3': tf.Variable(tf.constant(0.1, shape=[camada3Esc])),
    'out': tf.Variable(tf.constant(0.1, shape=[numeroSaida]))
}

#Cada camada oculta executará a multiplicação da matriz nas saídas da camada anterior e os pesos da camada
#atual e adicionará o bias a esses valores. Na última camada oculta, 
#aplicaremos uma operação de eliminação usando nosso valor keep_prob de 0.5.
camada1 = tf.add(tf.matmul(X, pesos['w1']), bias['b1'])
camada2 = tf.add(tf.matmul(camada1, pesos['w2']), bias['b2'])
camada3 = tf.add(tf.matmul(camada2, pesos['w3']), bias['b3'])
camaadaSolta = tf.nn.dropout(camada3, guardar)
camadaSaida = tf.matmul(camada3, pesos['out']) + bias['out']

#Entropia cruzada, também conhecida como log-loss, que quantifica a diferença entre duas distribuições de probabilidade
entropia = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=camadaSaida))
treinoP = tf.train.AdamOptimizer(1e-4).minimize(entropia)

predicaoCorreta = tf.equal(tf.argmax(camadaSaida, 1), tf.argmax(Y, 1))
acuracia = tf.reduce_mean(tf.cast(predicaoCorreta, tf.float32))


init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

#Exibindo acuracia
for i in range(numeroIteracoes):
    batch_x, batch_y = mnist.train.next_batch(lote)
    sess.run(treinoP, feed_dict={X: batch_x, Y: batch_y, guardar:fora})

    if i%100==0:
        minibatch_loss, minibatch_accuracy = sess.run([entropia, acuracia], feed_dict={X: batch_x, Y: batch_y, guardar:1.0})
        print("Iteração", str(i), "\t| Perdido =", str(minibatch_loss), "\t| Val =", str(minibatch_accuracy))




testeAcuracia = sess.run(acuracia, feed_dict={X: mnist.test.images, Y: mnist.test.labels, guardar:1.0})
print("\Valor aproximado do teste:", testeAcuracia)

img = np.invert(Image.open("oito.png").convert('L')).ravel()

prediction = sess.run(tf.argmax(camadaSaida,1), feed_dict={X: [img]})

print ("A imagem é", np.squeeze(prediction))


