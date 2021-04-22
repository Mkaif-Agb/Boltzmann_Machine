import numpy as np 
import matplotlib.pyplot as plt 
import tensorflow as tf 
from utils import getKaggleMNIST
from sklearn.utils import shuffle
from autoencoder_tensorflow import DNN


class RBM(object):
    def __init__(self, D, M, id):
        self.D = D
        self.M = M 
        self.id = id
        self.build(D, M)
    
    def set_session(self, session):
        self.session = session
        
    def free_energy(self, V):
        b = tf.reshape(self.b, (self.D, 1)) # To convert into 2d
        first_term = -tf.matmul(V, b)
        first_term = tf.reshape(first_term, (-1)) # To convert into scalar
        
        second_term = -tf.reduce_sum(
            tf.nn.softplus(tf.matmul(V, self.W) + self.c)
        )
        return first_term + second_term
    
    def forward_hidden(self, X):
        return tf.nn.sigmoid(tf.matmul(X, self.W) + self.c)
    
    def forward_logits(self, X):
        Z = self.forward_hidden(X)
        return tf.matmul(Z, tf.transpose(self.W)) + self.b
    
    def forward_output(self, X):
        return tf.nn.sigmoid(self.forward_logits(X))
    
    def transform(self, X):
        return self.session.run(self.p_h_given_v, feed_dict={
            self.X_in : X
        })
    
    def build(self, D, M):
        self.W = tf.Variable(tf.random_normal(shape=(D,M)) * np.sqrt(2.0/M))
        self.c = tf.Variable(np.zeros(M).astype(np.float32))
        self.b = tf.Variable(np.zeros(D).astype(np.float32))
        
        self.X_in = tf.placeholder(tf.float32, shape=(None, D))
        V = self.X_in
        p_h_given_v = tf.nn.sigmoid(tf.matmul(V, self.W) + self.c)
        self.p_h_given_v = p_h_given_v
        
        r = tf.random_uniform(shape=tf.shape(p_h_given_v))
        H = tf.to_float(r < p_h_given_v)
        
        p_v_given_h = tf.nn.sigmoid(tf.matmul(H, tf.transpose(self.W)) + self.b)
        r = tf.random_uniform(shape=tf.shape(p_v_given_h))
        X_sample = tf.to_float(r < p_v_given_h)
        
        
        objective = tf.reduce_mean(self.free_energy(self.X_in)) - tf.reduce_mean(self.free_energy(X_sample))
        self.train_op = tf.train.AdamOptimizer(1e-2).minimize(objective)
        logits = self.forward_logits(self.X_in)
        self.cost = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=logits,
                labels=self.X_in
            )
        )
    
    def fit(self, X, epochs=1, batch_size=100, show_fig=False):
        N, D = X.shape
        n_batches = N // batch_size
        costs = []
        print("Rbm is Training")
        for i in range(epochs):
            print('Epoch {}'.format(i))
            X = shuffle(X)
            for j in range(n_batches):
                batch = X[j*batch_size:(j*batch_size+ batch_size)]
                _, c = self.session.run((self.train_op, self.cost), feed_dict={
                    self.X_in:batch
                })
                costs.append(c)
        if show_fig:
            plt.plot(costs)
            plt.show()
            

def main():
    X_train, y_train, X_test, y_test = getKaggleMNIST()
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    _, D = X_train.shape
    K = len(set(y_train))
    dnn = DNN(D, [1000, 750, 500], K, UnsupervisedModel=RBM)
    init_op = tf.global_variables_initializer()
    with tf.Session() as session:
        session.run(init_op)
        dnn.set_session(session)
        dnn.fit(X_train, y_train, X_test, y_test, pretrain=True, epochs=10)
    
    
if __name__ == '__main__':
    main()            
        
        
        