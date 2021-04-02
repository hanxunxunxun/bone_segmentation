"""
Using a precomputed kernel matrix, compute test scores of GP regression
"""
import scipy.io as sio
import sys
import pickle_utils as pu
import numpy as np
import scipy
import sklearn.metrics
import time
import pandas as pd
import os
from save_kernels import create_kern, compute_big_K, mnist_1hot_all

from absl import app as absl_app
from absl import flags
import tensorflow as tf

def cut_training(N_train, N_vali, Kxx, Kxvx, Kv_diag, Kxtx, Kt_diag, X, Y, Xv, Yv, Xt, Yt):
    """
    If you computed a kernel matrix larger than the size of the training set
    you want to use, this is useful!
    For example: the default TensorFlow MNIST training set is 55k examples, but
    the paper "Deep Neural Networks as Gaussian Processes" by Jaehon Lee et al.
    (https://arxiv.org/abs/1711.00165) used only 50k, and we had to make a fair
    comparison.
    """
    return (Kxx[:N_train, :N_train],
            np.concatenate([Kxx[N_train:, :N_train], Kxvx[:, :N_train]], axis=0)[:N_vali, :],
            (None if Kv_diag is None else
             np.concatenate([np.diag(Kxx[N_train:, N_train:]), Kv_diag], axis=0)[:N_vali]),
            Kxtx[:, :N_train],
            Kt_diag,
            X[:N_train], Y[:N_train],
            np.concatenate([X[N_train:], Xv], axis=0)[:N_vali, :],
            np.concatenate([Y[N_train:], Yv], axis=0)[:N_vali, :],
            Xt, Yt)


def main(_):
    start_time = time.time()
    FLAGS = flags.FLAGS
    seed = FLAGS.seed
    path = FLAGS.path
    
#    seed=1
#    path="Traffic_results"

    # noise = float(sys.argv[2])
    params_file = os.path.join(path, "params_{:02d}.pkl.gz".format(seed))
    gram_file = os.path.join(path, "gram_{:02d}.npy".format(seed))
    Kxvx_file = os.path.join(path, "Kxvx_{:02d}.npy".format(seed))
    Kxtx_file = os.path.join(path, "Kxtx_{:02d}.npy".format(seed))
    Kv_diag_file = os.path.join(path, "Kv_diag_{:02d}.npy".format(seed))
    Kt_diag_file = os.path.join(path, "Kt_diag_{:02d}.npy".format(seed))
    csv_file = os.path.join(path, FLAGS.csv_dir, "{:02d}.csv".format(seed))

    params = pu.load(params_file)
    print("Kernel params:", params)

    # Kxtx = np.load(Kxtx_file)
    # if Kxtx.shape[1] != 10000:
    #     params['test_error'] = -1.
    #     params['validation_error'] = -1.
    #     params['training_error'] = -1.
    #     params['time'] = -1.
    #     pd.DataFrame(data=params, index=pd.Index([0])).to_csv(csv_file)
    #     print("Test is wrong size for seed", seed)
    #     sys.exit(1)

    print("Loading data and kernels")
    Kxx, Kxvx, _, Kxtx, _, X, Y, Xv, Yv, Xt, Yt = cut_training(
        FLAGS.N_train, FLAGS.N_vali,
        np.load(gram_file),
        np.load(Kxvx_file), None,
        np.load(Kxtx_file), None,
        *mnist_1hot_all())
#    
#    Kxx, Kxvx, _, Kxtx, _, X, Y, Xv, Yv, Xt, Yt = cut_training(
#        26774, 1646,
#        np.load(gram_file),
#        np.load(Kxvx_file), None,
#        np.load(Kxtx_file), None,
#        *mnist_1hot_all())

    # center labels
    Kxx = Kxx.astype(np.float64)
#    Kxx[1,5] = Kxx[1,5]+0.1    
    #Y[Y == 0.] = -1

    print("Solving system")
    print("Size of Kxx, Y, Kxvx, Yv, Kxtx, Yt:", Kxx.shape, Y.shape, Kxvx.shape, Yv.shape, Kxtx.shape, Yt.shape)
    
    print("Ill_conditioned matrix regularisation")
    bias = 0.1;
    A = Kxx;
    medA = np.transpose(A)@A
    defb = np.transpose(A)@Y
    defA = medA + bias * np.eye(np.shape(medA)[0], dtype = 'float64')
    
    K_inv_y = scipy.linalg.solve(defA, defb, overwrite_a=True, overwrite_b=False, check_finite=False, assume_a='pos')
    
#    K_inv_y = scipy.linalg.solve(Kxx, Y, overwrite_a=True, overwrite_b=False, check_finite=False, assume_a='pos')
   
    #K_inv_y = Kxx @ Y
    #print("Saving K_inv_y...")
    #np.save(sys.argv[1].replace('gram_', 'K_inv_y_'), K_inv_y)
    #ys.exit(0)

#    def print_error(K_xt_x, Ytv, dit, key):
#        print("Running for", key)
#        Y_pred = K_xt_x @ K_inv_y
#        t = sklearn.metrics.accuracy_score(
#            np.argmax(Ytv, 1), np.argmax(Y_pred, 1))
#        dit[key] = (1-t)*100
  
    # RMSE calculation       
    def print_error(K_xt_x, Ytv, dit, key):
        print("Running for", key)
        Y_pred = K_xt_x @ K_inv_y
        np.save(os.path.join('Prediction','Pre_'+key+".npy"),Y_pred)
        np.save(os.path.join('Truth',key+".npy"),Ytv)
#        pre=Y_pred
        t = np.square((np.mat(Ytv)-np.mat(Y_pred)))
        dit[key+'_error'] = np.mean(np.sqrt(np.sum(t,1)/np.size(t,1)))
#        dit[key] = np.mean(np.sqrt(np.sum(t,0)/np.size(t,0)))
#        dit[key] = tf.sqrt(tf.losses.mean_squared_error(labels=Ytv, predictions=Y_pred))
#    pretrain=np.array(np.zeros(shape=[800,36,36,1]))
#    prevel=np.array(np.zeros(shape=[500,36,36,1]))
#    pret=np.array(np.zeros(shape=[300,36,36,1]))
    print_error(Kxx, Y, params, "training")#,pretrain)
    print_error(Kxvx, Yv, params, "validation")#,prevel)
    print_error(Kxtx, Yt, params, "test")#,pret)
#    np.save("testR.npy",pret)
    params['time'] = time.time() - start_time
    pd.DataFrame(data=params, index=pd.Index([0])).to_csv(csv_file)

if __name__ == '__main__':
    f = flags
    f.DEFINE_integer('seed', None, 'random seed')
#    f.DEFINE_integer('N_train', 50000, 'number of training data points')
#    f.DEFINE_integer('N_vali', 10000, 'number of validation data points')
    f.DEFINE_integer('N_train', 26774, 'number of training data points')
    f.DEFINE_integer('N_vali', 1646, 'number of validation data points')
    f.DEFINE_string('path', None,
                    "the path to the precomputed kernel matrices")
    f.DEFINE_string('csv_dir', 'dfs',
                    "directory to save CSVs with results, under `FLAGS.path`")
    absl_app.run(main)