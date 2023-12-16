from enum import Enum

import numpy as np


class PDMethods(str, Enum):
    central = "central"
    forward = 'forward'
    backward = 'backward'


def derivative(f, method: PDMethods = PDMethods.central, h=0.01):
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
    df = derivative(c)

    def wrap(t):
        x, y = c(t)
        dx, dy = df(t)
        ox = x + (d * dy / _ns(dx, dy))
        oy = y - (d * dx / _ns(dx, dy))
        return [ox, oy]

    wrap.__name__ = c.__name__ + f"_normal_{d}"
    return wrap


"""    {\displaystyle x_{d}(t)=x(t)+{\frac {d\;y'(t)}{\sqrt {x'(t)^{2}+y'(t)^{2}}}}}
    y d ( t ) = y ( t ) − d x ′ ( t ) x ′ ( t ) 2 + y ′ ( t ) 2
    . {\displaystyle y_{d}(t)=y(t)-{\frac {d\;x'(t)}{\sqrt {x'(t)^{2}+y'(t)^{2}}}}\ .}"""
# simulated annealing search of a one-dimensional objective function


# simulated annealing algorithm
def simulated_annealing(objective, bounds, n_iterations, step_size, temp):
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
    return best, best_eval, scores
