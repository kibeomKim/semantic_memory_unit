import os
os.environ["OMP_NUM_THREADS"] = "1"
import torch
import torch.multiprocessing as mp
import torch.optim as optim

import argparse
import logging
import time

from train import run_sim
from shared_optim import SharedRMSprop, SharedAdam
from models import A3C_LSTM_GA
from eval import test

targets = ['bedroom', 'kitchen', 'bathroom', 'dining_room', 'living_room']

class Params():
    def __init__(self):
        self.n_process = 5
        self.max_episode = 300000
        self.gamma = 0.95
        self.entropy_coef = 0.1
        self.gpu_ids_train = [1,2]
        self.gpu_ids_test = [0]
        self.lr = 1e-3
        self.tau = 1.0
        self.seed = 1
        self.value_loss_coef = 1.0
        self.amsgrad = True
        self.num_steps = 30
        self.hardness = 0.6
        self.width = 120
        self.height = 90
        self.n_eval = 1000
        self.n_test = 2000
        self.house_id = -1   #if -1, multi_env
        self.max_steps = 100
        self.semantic_mode = True  #if false, RGB mode on
        self.log_file = 'training_1119_kl_em_cnn_lstm_initialize'
        self.weight_dir = './train_1119_kl_em_cnn_lstm_initialize/'
        self.weight_decay = 0 #0.00005   #

def main():
    params = Params()

    mp.set_start_method('spawn')
    count = mp.Value('i', 0)
    best_acc = mp.Value('d', 0.0)
    lock = mp.Lock()

    shared_model = A3C_LSTM_GA()
    shared_model = shared_model.share_memory()

    shared_optimizer = SharedAdam(shared_model.parameters(), lr=params.lr, amsgrad=params.amsgrad, weight_decay=params.weight_decay)
    shared_optimizer.share_memory()
    #run_sim(0, params, shared_model, None,  count, lock)
    #test(params, shared_model, count, lock, best_acc)

    processes = []

    train_process = 0
    test_process = 0

    for rank in range(params.n_process):

        p = mp.Process(target=test, args=(test_process, params, shared_model, count, lock, best_acc, ))
        p.start()
        processes.append(p)
        test_process += 1

        for i in range(2):
            p = mp.Process(target=run_sim, args=(train_process, params, shared_model, shared_optimizer, count, lock, ))
            p.start()
            processes.append(p)
        train_process += 1

    for p in processes:
        p.join()


def run_test(type):
    params = Params()
    #'./weight_one_hot/model3.ckpt' is best
    if type == 1:
        load_model = torch.load('./weight_0810/model46301.ckpt', map_location=lambda storage, loc: storage.cuda(params.gpu_ids_test[0]))
    else:
        load_model = None
    test(0, params, load_model, None, None, None, evaluation=False)


if __name__ == "__main__":
    main()
    #run_test(1)  # 1 = trained model load, else = random