import pickle
from itertools import islice

import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils import pretty_print, functions
from utils.symbolic_network import SymbolicNet
from utils.regularization import L12Smooth
from inspect import signature
import time
import argparse
from cmath import inf




# Standard deviation of random distribution for weight initializations.

init_sd_first = 0.01
init_sd_last = 0.01
init_sd_middle = 0.05


"""Generates datasets."""
def generate_data(type):
    input_x = []
    input_y = []
    filepath = 'dataset\\{0}_{1}.txt'.format(problem_name,type)
    with open(filepath, "r") as f:
        lines = f.readlines()
        numberOfLines = len(lines)  # read first line
        firstline = (lines[0].strip()).split('\t')
        sample_num = int(firstline[0])
        x_dim = int(firstline[1])
        if (numberOfLines != sample_num + 1):
            print("read file error: the number of samples is wrong!!!")
            exit(1)
        # Read all the remaining data
        for line in islice(lines, 1, None):
            line = line.strip()
            # formLine = line.split(',')
            formLine = line.split('\t')
            input_x.append(formLine[0:-1])
            input_y.append((formLine[-1]))
    # Converts the read data type from str to float
    input_x= np.array(input_x, dtype=np.float32)
    input_x = np.array(input_x, dtype=np.float32).reshape(sample_num, x_dim)
    input_y = np.array(input_y, dtype=np.float32).reshape(sample_num,1)
    x = torch.from_numpy(input_x)
    y = torch.from_numpy(input_y)
    return x, y


# save the results directory (results_dir) and the hyper-parameters. ""
class Benchmark:
    def __init__(self, results_dir, n_layers=2, reg_weight=5e-3, learning_rate=1e-2,
                 n_epochs1=10000, n_epochs2=10000):

        # Set the number and type of hidden layer neurons.
        if (dim == 1):  # When the ADF dimension is 1
            self.activation_funcs = [
                *[functions.Constant()] * 2,
                *[functions.Identity()] * 2,
                *[functions.Square()] * 2,
                *[functions.Sin()] * 2,
                *[functions.Cos()] * 2,
                *[functions.ADF()] * 4,
            ]
        else:
            self.activation_funcs = [
                *[functions.Constant()] * 2,
                *[functions.Identity()] * 2,
                *[functions.Square()] * 2,
                *[functions.Sin()] * 2,
                *[functions.Cos()] * 2,
                *[functions.ADF2()] * 4,
            ]
        self.n_layers = n_layers  # Number of hidden layers
        self.reg_weight = reg_weight  # Regularization weight
        self.learning_rate = learning_rate
        self.summary_step = 1000  # Number of iterations at which to print to screen
        self.n_epochs1 = n_epochs1
        self.n_epochs2 = n_epochs2

        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        self.results_dir = results_dir

        # Save hyperparameters to file
        result = {
            "learning_rate": self.learning_rate,
            "summary_step": self.summary_step,
            "n_epochs1": self.n_epochs1,
            "n_epochs2": self.n_epochs2,
            "activation_funcs_name": [func.name for func in self.activation_funcs],
            "n_layers": self.n_layers,
            "reg_weight": self.reg_weight,
        }
        with open(os.path.join(self.results_dir, 'params.pickle'), "wb+") as f:
            pickle.dump(result, f)

    def benchmark(self, Var_dim, func_name, trials):

        # Create a new sub-directory just for the specific function
        func_dir = os.path.join(self.results_dir, func_name)
        if not os.path.exists(func_dir):
            os.makedirs(func_dir)

        # Train network!
        expr_list, error_test_list = self.train(Var_dim, func_name, trials, func_dir)


        error_expr_sorted = sorted(zip(error_test_list, expr_list))  # List of (error, expr)
        error_test_sorted = [x for x, _ in error_expr_sorted]  # Separating out the errors
        expr_list_sorted = [x for _, x in error_expr_sorted]  # Separating out the expr

        fi = open(os.path.join(func_dir, 'eq_summary.txt'), 'a')
        for i in range(trials):
            fi.write("[%f]\t\t%s\n\n" % (error_test_sorted[i], str(expr_list_sorted[i])))
        fi.close()
        return expr_list, error_test_list


    def train(self, Var_dim, func_name='', trials=1, func_dir='results/test'):
        """Train the network to find a given function"""

        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda:0" if use_cuda else "cpu")
        # print("Use cuda:", use_cuda, "Device:", device)


        x, y = generate_data(type='train')
        data, target = x.to(device), y.to(device)

        x_test, y_test=generate_data(type='test')

        test_data, test_target = x_test.to(device), y_test.to(device)

        # Setting up the symbolic regression network
        x_dim=Var_dim  #Number of input arguments to the function

        width = len(self.activation_funcs)

        n_double = functions.count_double(self.activation_funcs)

        # Arrays to keep track of various quantities as a function of epoch
        loss_list = []  # Total loss (MSE + regularization)
        error_list = []  # MSE
        reg_list = []  # Regularization
        error_test_list = []  # Test error

        error_test_final = []
        eq_list = []

        for trial in range(trials):
            print("Training on function " + func_name + " Trial " + str(trial + 1) + " out of " + str(trials))

            # reinitialize for each trial
            net = SymbolicNet(self.n_layers,
                              funcs=self.activation_funcs,
                              initial_weights=[
                                  # kind of a hack for truncated normal
                                  torch.fmod(torch.normal(0, init_sd_first, size=(x_dim, width + n_double)), 2),
                                  torch.fmod(torch.normal(0, init_sd_middle, size=(width, width + n_double)), 2),
                                  torch.fmod(torch.normal(0, init_sd_middle, size=(width, width + n_double)), 2),
                                  torch.fmod(torch.normal(0, init_sd_last, size=(width, 1)), 2)
                              ]).to(device)


            loss_val = np.nan
            while np.isnan(loss_val):
                # training restarts if gradients blow up
                criterion = nn.MSELoss()
                optimizer = optim.RMSprop(net.parameters(),
                                          lr=self.learning_rate * 10,
                                          alpha=0.9,  # smoothing constant
                                          eps=1e-10,
                                          momentum=0.0,
                                          centered=False)

                # adaptive learning rate
                lmbda = lambda epoch: 0.1
                scheduler = optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lmbda)


                t0 = time.time()

                # First stage of training
                for epoch in range(self.n_epochs1):
                    optimizer.zero_grad()  # zero the parameter gradients
                    outputs = net(data)  # forward pass
                    regularization = L12Smooth()     #regularization
                    mse_loss = criterion(outputs, target)

                    reg_loss = regularization(net.get_weights_tensor())
                    loss = mse_loss + self.reg_weight * reg_loss
                    loss.backward()

                    optimizer.step()

                    if epoch % self.summary_step == 0:
                        error_val = mse_loss.item()
                        reg_val = reg_loss.item()
                        loss_val = loss.item()
                        error_list.append(error_val)
                        reg_list.append(reg_val)
                        loss_list.append(loss_val)

                        with torch.no_grad():  # test error
                            test_outputs = net(test_data)
                            test_loss = F.mse_loss(test_outputs, test_target)
                            error_test_val = test_loss.item()
                            error_test_list.append(error_test_val)

                        if np.isnan(loss_val) or loss_val > 1000:  # If loss goes to NaN, restart training
                            loss_val= float(inf)
                            error_test_val= float(inf)
                            break

                        mean_true = torch.mean(test_target)
                        r_squared = 1 - torch.sum((test_target - test_outputs) ** 2) / torch.sum(
                            (test_target - mean_true) ** 2)

                        # fi = open(os.path.join(func_dir, 'loss.txt'), 'a')
                        # fi.write("%d\t%f\t\t%f\t\t%f\n" % (epoch,loss_val, error_test_val,r_squared))
                        # fi.close()
                        print("Epoch: %d\tTotal training loss: %f\tTest error: %f\tr_squared: %f" % (epoch, loss_val, error_test_val, r_squared))



                    if epoch == 2000:
                        scheduler.step()  # lr /= 10

                scheduler.step()  # lr /= 10 again

                # Second stage of training
                for epoch in range(self.n_epochs2):
                    optimizer.zero_grad()  # zero the parameter gradients
                    outputs = net(data)
                    regularization = L12Smooth()
                    mse_loss = criterion(outputs, target)
                    reg_loss = regularization(net.get_weights_tensor())
                    loss = mse_loss + self.reg_weight * reg_loss
                    loss.backward()
                    optimizer.step()

                    if epoch % self.summary_step == 0:
                        error_val = mse_loss.item()
                        reg_val = reg_loss.item()
                        loss_val = loss.item()
                        error_list.append(error_val)
                        reg_list.append(reg_val)
                        loss_list.append(loss_val)

                        with torch.no_grad():  # test error
                            test_outputs = net(test_data)
                            test_loss = F.mse_loss(test_outputs, test_target)
                            error_test_val = test_loss.item()
                            error_test_list.append(error_test_val)


                        if np.isnan(loss_val) or loss_val > 1000:  # If loss goes to NaN, restart training
                            loss_val= float(inf)
                            error_test_val= float(inf)
                            break

                        "Calculate R square"
                        mean_true = torch.mean(test_target)
                        r_squared = 1 - torch.sum((test_target - test_outputs) ** 2) / torch.sum((test_target - mean_true) ** 2)



                        print("Epoch: %d\tTotal training loss: %f\tTest error: %f\tr_squared: %f" % (epoch, loss_val, error_test_val, r_squared))



                t1 = time.time()




            tot_time = t1 - t0
            print("Total time: %f" % (tot_time))
            fi = open(os.path.join(func_dir, 'Time.txt'), 'a')
            fi.write("%f\t\n" % (tot_time))
            fi.close()


            # Print the expressions
            with torch.no_grad():
                weights = net.get_weights()
                expr = pretty_print.network(weights, self.activation_funcs, var_names[:x_dim])
                print("The expression is:")
                print(expr)

            # Save results
            trial_file = os.path.join(func_dir, 'trial%d.pickle' % trial)
            results = {
                "weights": weights,
                "loss_list": loss_list,
                "error_list": error_list,
                "reg_list": reg_list,
                "error_test": error_test_list,
                "expr": expr,
                "runtime": tot_time
            }
            with open(trial_file, "wb+") as f:
                pickle.dump(results, f)

            error_test_final.append(error_test_list[-1])

            eq_list.append(expr)

        return eq_list, error_test_final




def eql(Var_dim, func_name, trials, Dim,Var_names):

    global dim
    dim =Dim

    global problem_name
    problem_name=func_name

    global var_names
    var_names=Var_names

    kwargs={"results_dir": "results/benchmark/test",
            "n_layers": 2,
            "reg_weight": 0.005,
            "learning_rate": 1e-2,
            "n_epochs1": 10000,
            "n_epochs2": 10000}

    if not os.path.exists(kwargs['results_dir']):
        os.makedirs(kwargs['results_dir'])
    meta = open(os.path.join(kwargs['results_dir'], 'args.txt'), 'a')
    import json
    meta.write(json.dumps(kwargs))
    meta.close()

    bench = Benchmark(**kwargs)

    expr_list, error_test_list = bench.benchmark(Var_dim, func_name, trials)
    return expr_list, error_test_list
