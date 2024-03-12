from inspect import signature
import numpy as np
import torch
import os

func = lambda x,y:   np.sin(x)+np.sin(y**2)
#x**4 + x**3 + x**2 + x    x**5 - 2*x**3 + x   x**6 + x**5 + x**4 + x**3 + x**2 + x  np.sin(x**2)*np.cos(x)-1
# x**4 - x**3 + 0.5*y**2 - y    (x-3)*(y-3)+2*np.sin(x-4)*(y-4)  np.log(x+1)+np.log(x^2+1)

func_name = "sin(x)+sin(y^2)"
#x^4+x^3+x^2+x     x^5-2x^3+x     x^6 + x^5 + x^4 + x^3 + x^2 + x  sin(x^2)cos(x)-1
# x^4 - x^3 + 0.5y^2 - y   (x-3)(y-3)+2sin(x-4)(y-4)


N_TRAIN = 1000  # Size of training dataset
DOMAIN = (-2, 2)  # Domain of dataset - range
N_TEST = 100  # Size of test dataset
DOMAIN_TEST = (-1, 2)  # Domain of test dataset - should be larger than training domain to test extrapolation

"""Generates datasets."""
def generate_data(func, N, range_min, range_max):
    x_dim = len(signature(func).parameters)  # Number of inputs to the function, or, dimensionality of x
    x = (range_max - range_min) * torch.rand([N, x_dim]) + range_min
    y = torch.tensor([[func(*x_i)] for x_i in x])
    return x, y



x, y = generate_data(func, N_TRAIN,range_min=DOMAIN[0], range_max=DOMAIN[1])
x=np.array(x)
y=np.array(y)
print(x)
print(y)

x_test, y_test = generate_data(func, N_TEST, range_min=DOMAIN_TEST[0], range_max=DOMAIN_TEST[1])

func_dir='dataset'
fi = open(os.path.join(func_dir, 'sin(x)+sin(y^2)_train.txt'), 'a')
for i in range(N_TRAIN):
    fi.write("%f\t%f\t%f\n" % (x[i][0], x[i][1],y[i]))
    # fi.write("%f\t%f\n" % (x[i], y[i]))
fi.close()


func_dir='dataset'
fi = open(os.path.join(func_dir, 'sin(x)+sin(y^2)_test.txt'), 'a')
for i in range (N_TEST):
    fi.write("%f\t%f\t%f\n" % (x_test[i][0], x_test[i][1],y_test[i]))
    # fi.write("%f\t%f\n" % (x_test[i], y_test[i]))
fi.close()