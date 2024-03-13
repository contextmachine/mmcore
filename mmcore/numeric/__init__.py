from enum import Enum
from mmcore.numeric.routines import *
import numpy as np


class PDMethods(str, Enum):
    central = "central"
    forward = 'forward'
    backward = 'backward'


def deriv(f, method: PDMethods = PDMethods.central, h=0.01):
    '''Compute the difference formula for f'(t) with step size h.

    Parameters
    ----------
    f : function
        Vectorized function of one variable

    method : string
        Difference formula: 'forward', 'backward' or 'central'
    h : number
        Step size in difference formula

    Returns
    -------
    lambda t:
        Difference formula:
            central: f(a+h) - f(a-h))/2h
            forward: f(a+h) - f(a))/h
            backward: f(a) - f(a-h))/h
    '''
    if method == 'central':
        return lambda t: (f(t + h) - f(t - h)) / (2 * h)
    elif method == 'forward':
        return lambda t: (f(t + h) - f(t)) / h
    elif method == 'backward':
        return lambda t: (f(t) - f(t - h)) / h
    else:
        raise ValueError("Method must be 'central', 'forward' or 'backward'.")


def _ns(dx, dy):
    return np.sqrt((dx ** 2) + (dy ** 2))


def offset_curve_2d(c, d):
    df = deriv(c)

    def wrap(t):
        x, y = c(t)
        dx, dy = df(t)
        ox = x + (d * dy / _ns(dx, dy))
        oy = y - (d * dx / _ns(dx, dy))
        return [ox, oy]

    wrap.__name__ = c.__name__ + f"_normal_{d}"
    return wrap


def simulated_annealing(objective, bounds, n_iterations, step_size, temp, record=False):
    """
    # objective function
    def objective(x):
        return x[0]**2.0
    # seed the pseudorandom number generator
    seed(1)
    # define range for input
    bounds = asarray([[-5.0, 5.0]])
    # define the total iterations
    n_iterations = 1000
    # define the maximum step size
    step_size = 0.1
    # initial temperature
    temp = 10
    # perform the simulated annealing search
    best, score, scores = simulated_annealing(objective, bounds, n_iterations, step_size, temp)
    print('Done!')
    print('f(%s) = %f' % (best, score))
    # line plot of best scores
    #pyplot.plot(scores, '.-')
    ##pyplot.xlabel('Improvement Number')
    #pyplot.ylabel('Evaluation f(x)')
    #pyplot.show()
    Parameters
    ----------
    objective
    bounds
    n_iterations
    step_size
    temp

    Returns
    -------

    """

    best = bounds[:, 0] + np.random.random(len(bounds)) * (bounds[:, 1] - bounds[:, 0])
    # evaluate the initial point
    best_eval = objective(best)
    # current working solution
    curr, curr_eval = best, best_eval
    if record:
        scores = np.zeros([n_iterations, len(bounds)])
    # run the algorithm
    for i in range(n_iterations):
        # take a step
        candidate = curr + np.random.randn(len(bounds)) * step_size
        # evaluate candidate point
        candidate_eval = objective(candidate)
        # check for new best solution
        if candidate_eval < best_eval:
            # store new best point
            best, best_eval = candidate, candidate_eval
            # keep track of scores
            if record:
                scores[i] = best_eval
            # report progress
            print('>%d f(%s) = %.5f' % (i, best, best_eval))
        # difference between candidate and current point evaluation
        diff = candidate_eval - curr_eval

        # calculate temperature for current epoch
        t = temp / float(i + 1)
        # calculate metropolis acceptance criterion
        metropolis = np.exp(-diff / t)
        # check if we should keep the new point
        if diff < 0 or np.random.random() < metropolis:
            # store the new current point
            curr, curr_eval = candidate, candidate_eval

    return best, best_eval


# Initialize Adam moments
def initialize_adam(dims=2):
    s = np.zeros(dims)
    v = np.zeros(dims)
    return s, v


# Update parameters using Adam
def update_parameters_with_adam(x, grads, s, v, t, learning_rate=0.01, average_factor=0.9, average_square_factor=0.999,
                                epsilon=1e-8):
    s = average_factor * s + (1.0 - average_factor) * grads
    v = average_square_factor * v + (1.0 - average_square_factor) * grads ** 2
    s_hat = s / (1.0 - average_factor ** (t + 1))
    v_hat = v / (1.0 - average_square_factor ** (t + 1))
    x = x - learning_rate * s_hat / (np.sqrt(v_hat) + epsilon)
    return x, s, v


# Adam optimization algorithm
def adam(objective, derivative, bounds, n_iterations=1000, step_size=0.01, average_factor=0.8,
         average_square_factor=0.999, eps=1e-8):
    """

    :param objective: The objective function to be minimized
    :type objective: function
    :param derivative: The derivative function of the objective function
    :type derivative: function
    :param bounds: The bounds of the search space for each dimension
    :type bounds: numpy array
    :param n_iterations: The number of iterations for the optimization
    :type n_iterations: int
    :param step_size: The step size or learning rate of the Adam algorithm
    :type step_size: float
    :param average_factor: The decay rate for the moving average of the gradient
    :type average_factor: float
    :param average_square_factor: The decay rate for the moving average of the squared gradient
    :type average_square_factor: float
    :param eps: The small constant added to avoid division by zero
    :type eps: float
    :return: The optimized solution and the score of the objective function
    :rtype: list

    Usage Example
    -------------
>>> # Objective function
>>> def objective(x):
...     return x ** 2.0
>>> # Derivative of the objective function
>>> def objective_derivative(x):
...     return 2.0 * x
>>> # Set the random seed
>>> np.random.seed(1)

>>> # Define the range for input
>>> bounds = np.array([[-1.0, 1.0], [-1.0, 1.0]])

>>> # Define the total number of iterations
>>> n_iterations = 60
>>> # Set the step size
>>> step_size = 0.02

>>> # Set the factor for average gradient
>>> average_factor = 0.8

>>> # Set the factor for average squared gradient
>>> average_square_factor = 0.999

>>> # Perform the gradient descent search with Adam
>>> best, score = adam(objective, derivative,
>>>                 bounds, n_iterations, step_size,
>>>                 average_factor, average_square_factor)
>>>

    """
    # Generate an initial point
    x = bounds[:, 0] + np.random.rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])
    score = objective(x)

    # Initialize Adam moments
    s, v = initialize_adam(len(x))

    # Run the gradient descent updates
    for t in range(n_iterations):
        # Calculate gradient g(t)
        g = derivative(x)

        # Update parameters using Adam
        x, s, v = update_parameters_with_adam(x, g, s, v, t, step_size, average_factor, average_square_factor, eps)

        # Evaluate candidate point
        score = objective(x)

        # Report progress  # print('>%d f(%s) = %.5f' % (t, x, score))

    return x, score


from mmcore.geom.vec import norm
from mmcore.geom.pde import PDE
from collections import namedtuple

AdamOutput = namedtuple("AdamOutput", ['x', 'f'])


class AdamOutputOptions(str, Enum):
    x = 'x'
    f = 'f'
    all = ''


class Adam:
    def __init__(self, n_iterations=1000, step_size=0.01, average_factor=0.8,
                 average_square_factor=0.999, eps=1e-8):
        self.n_iterations = n_iterations
        self.step_size = step_size
        self.average_factor = average_factor
        self.average_square_factor = average_square_factor
        self.eps = eps

    def __call__(self, fun, bounds=np.array([0., 1.]), derivative=None, output=AdamOutputOptions.all, **kwargs):
        props = {**self.__dict__, **kwargs}

        derivative = PDE(fun) if derivative is None else derivative

        out = AdamOutput(*adam(fun, derivative, bounds=bounds, **props))

        if output:
            return getattr(out, output)
        return out



class AdamCurvesIntersection:
    def __init__(self, bounds=np.array([[0., 1.], [0., 1.]]), n_iterations=1000, step_size=0.01, average_factor=0.8,
                 average_square_factor=0.999, eps=1e-8):
        self.bounds = bounds
        self.n_iterations = n_iterations
        self.step_size = step_size
        self.average_factor = average_factor
        self.average_square_factor = average_square_factor
        self.eps = eps

    def __call__(self, curve_a, curve_b, **kwargs):
        self.__dict__ |= kwargs

        def objective(x):
            return norm(curve_a(x[0]) - curve_b(x[1]))

        pa, pb = PDE(curve_a), PDE(curve_b)

        def der(x):
            pa(x[0]), pb([x[1]])

        res, score = adam(objective, PDE(objective), bounds=self.bounds, n_iterations=self.n_iterations,
                          step_size=self.step_size, average_factor=self.average_factor,
                          average_square_factor=self.average_square_factor, eps=self.eps
                          )
        return res
