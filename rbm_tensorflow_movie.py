import numpy as np 
import tensorflow as tf 
import matplotlib.pyplot as plt 
from sklearn.utils import shuffle

import pandas as pd 
from scipy.sparse import load_npz, save_npz, lil_matrix, csr_matrix
from datetime import datetime

# One hot encoding 

def one_hot_encoder(X, K):
    N, D = X.shape
    Y = np.zeros((N, D, K))
    for n, d in zip(*X.nonzero()):
        # 0.5...5 --> 1..10 --> 0..9
        k = int(X[n,d]*2 -1)
        Y[n,d,k] = 1
    return Y 

def one_hot_mask(X, K):
    N, D = X.shape
    Y = np.zeros((N, D, K))
    # if X[n,d] == 0, there's a missing rating
    # so the mask should be all zeros
    # else, it should be all ones
    # [0,0,0,1,0] * [0,0,0,0,0] = [0,0,0,0,0]
    for n, d in zip(*X.nonzero()):
        Y[n,d,:] = 1
    return Y

one_to_ten = np.arange(10)
def convert_probs_to_ratings(probs):
    return probs.dot(one_to_ten)/2


def dot1(V, W):
    # V = (N x D x K)
    # W = (D x K x M)
    return tf.tensordot(V, W, axes=[[1,2],[0,1]])
    # returns N x M 
    
def dot2(H, W):
    # H = (N x M)
    # W = (D x K x M)
    return tf.tensordot(H, W, axes=[[1],[2]])
    # returns (N x D x K)
    
    
class RBM(object):
    def __init__(self, D, M, K): # input, hidden, output
        self.D = D 
        self.M = M 
        self.K = K 
        self.build(D, M, K)

    
    def free_energy(self, V):
        first_term = -tf.reduce_sum(dot1(V, self.b))
        second_term = -tf.reduce_sum(
            tf.nn.softplus(dot1(V, self.W) + self.c)
            , axis=1
        )

        return first_term + second_term
    
    def forward_hidden(self, X):
        return tf.nn.sigmoid(dot1(X,self.W) + self.c)
    
    def forward_logits(self,X):
        Z = self.forward_hidden(X)
        return dot2(Z, self.W) + self.b
    
    def forward_output(self, X):
        return tf.nn.softmax(self.forward_logits(X))
    
    def transform(self, X):
        return self.session.run(self.p_h_given_v, feed_dict={
            self.X_in:X
        })
    
    def get_visible(self, X):
        return self.session.run(self.output_visible, feed_dict={
            self.X_in:X
        })

    def build(self, D, M, K):
        # Variables
        self.W = tf.Variable(tf.random_normal(shape=(D,K,M)) * np.sqrt(2.0/M))
        self.b = tf.Variable(np.zeros((D, K)).astype(np.float32))
        self.c = tf.Variable(np.zeros(M).astype(np.float32))
        
        # Data 
        self.X_in = tf.placeholder(tf.float32, shape=(None, D, K))
        self.mask = tf.placeholder(tf.float32, shape=(None, D, K))
        
        V = self.X_in
        p_h_given_v = tf.nn.sigmoid(dot1(V, self.W) + self.c)
        self.p_h_given_v = p_h_given_v 
        r = tf.random_uniform(shape=tf.shape(self.p_h_given_v))
        H = tf.to_float(r < p_h_given_v)
        
        logits = dot2(H, self.W) + self.b
        cdist = tf.distributions.Categorical(logits=logits)
        X_sample = cdist.sample()
        X_sample = tf.one_hot(X_sample, depth=K)
        X_sample = X_sample * self.mask 
        
        objective = tf.reduce_mean(self.free_energy(self.X_in)) + tf.reduce_mean(self.free_energy(X_sample))
        self.train_op = tf.train.AdamOptimizer(0.001).minimize(objective)
        logits = self.forward_logits(self.X_in)
        self.cost = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                logits=logits,
                labels=self.X_in
            )
        )
        
        self.output_visible = self.forward_output(self.X_in)
        
        initop = tf.global_variables_initializer()
        self.session = tf.Session()
        self.session.run(initop)
        
            
    def fit(self, X, mask, X_test, mask_test, epochs=10, batch_size=256, show_fig=True):
        
        N, D = X.shape 
        n_batches = N // batch_size
        costs = []
        test_costs = []
        for i in range(epochs):
            t0 = datetime.now()
            print('Epoch : {}'.format(i+1))
            X, mask, X_test, mask_test = shuffle(X, mask, X_test, mask_test)
            for j in range(n_batches):
               x = X[j*batch_size:j*batch_size + batch_size].toarray()
               m = mask[j*batch_size:j*batch_size + batch_size].toarray()
               batch_one_hot = one_hot_encoder(x, self.K)
               m = one_hot_mask(m, self.K)
               
               _, c = self.session.run((self.train_op, self.cost),
                                       feed_dict={
                                           self.X_in:batch_one_hot,
                                           self.mask:m
                                       })
               
               if j % 100 == 0:
                   print('{} / {} Cost {}'.format(j, n_batches, c))
            print("Duration ", (datetime.now()-t0))
            
            t1 = datetime.now()
            sse = 0 
            test_sse = 0
            n = 0 
            test_n = 0
            for j in range(n_batches):
                x = X[j*batch_size:(j*batch_size+batch_size)].toarray()
                m = mask_test[j*batch_size:(j*batch_size+batch_size)].toarray()
                
                xoh = one_hot_encoder(x, self.K)
                probs = self.get_visible(xoh)
                xhat = convert_probs_to_ratings(probs)
                sse += (m * (xhat - x)* (xhat - x)).sum()
                n += m.sum()
                
                xt = X_test[j*batch_size:j*batch_size+batch_size].toarray()
                mt = mask_test[j*batch_size:j*batch_size+batch_size].toarray()
                
                test_sse += (mt*(xhat-xt)*(xhat-xt)).sum()
                test_n += mt.sum()
            c = sse/n
            ct = test_sse/test_n
            print('Train MSE: ', c )
            print('Test MSE: ', ct)
            print('Cost Duration {}', (datetime.now() - t1))
            costs.append(c)
            test_costs.append(ct)
        if show_fig:
            plt.plot(costs, label='Train MSE')
            plt.plot(test_costs, label='Test MSE')
            plt.legend()
            plt.show()
        
def main():
    A = load_npz('Lazy/unsupervised/Atrain.npz')    
    A_test = load_npz('Lazy/unsupervised/Atest.npz')
    mask = (A > 0) * 1.0
    print(mask)
    mask_test = (A_test > 0) * 1.0
    N, M = A.shape 
    rbm = RBM(M, 50, 10)
    rbm.fit(A, mask, A_test, mask_test)

if __name__ == '__main__':
    main()
