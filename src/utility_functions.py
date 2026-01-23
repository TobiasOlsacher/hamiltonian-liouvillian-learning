
"""
This module contains utility functions for working with data and memory.
"""
import sys
from types import ModuleType, FunctionType
from gc import get_referents
import numpy as np
import scipy as sp
import time as timer
from icecream import ic

def get_total_size(obj, seen=None):
    """Recursively find the total size of an object."""
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    seen.add(obj_id)
    size = sys.getsizeof(obj)
    if isinstance(obj, dict):
        size += sum(get_total_size(v, seen) for v in obj.values())
        size += sum(get_total_size(k, seen) for k in obj.keys())
    elif hasattr(obj, '__dict__'):
        size += get_total_size(vars(obj), seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum(get_total_size(i, seen) for i in obj)
    return size

def deep_get_size(obj, seen=None):
    """Recursively calculates the total memory size of an object."""
    if seen is None:
        seen = set()
    obj_id = id(obj)
    # Avoid infinite recursion by skipping already seen objects
    if obj_id in seen:
        return 0
    seen.add(obj_id)
    # Get the shallow size of the object
    size = sys.getsizeof(obj)
    # If the object has attributes, process them
    if isinstance(obj, dict):
        size += sum(deep_get_size(k, seen) + deep_get_size(v, seen) for k, v in obj.items())
    elif hasattr(obj, '__dict__'):
        size += deep_get_size(vars(obj), seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum(deep_get_size(i, seen) for i in obj)
    return size

def get_size(obj):
    """sum size of object & members."""
    # Custom objects know their class.
    # Function objects seem to know way too much, including modules.
    # Exclude modules as well.
    BLACKLIST = type, ModuleType, FunctionType
    if isinstance(obj, BLACKLIST):
        raise TypeError('getsize() does not take argument of type: '+ str(type(obj)))
    seen_ids = set()
    size = 0
    objects = [obj]
    while objects:
        need_referents = []
        for obj in objects:
            if not isinstance(obj, BLACKLIST) and id(obj) not in seen_ids:
                seen_ids.add(id(obj))
                size += sys.getsizeof(obj)
                need_referents.append(obj)
        objects = get_referents(*need_referents)
    return size

# TODO: parallelize brute optimizer!!!
def minimize(cost,x0,**kwargs):
    ''' 
    Solver for minimizing a cost function,
    given an initial guess x0. 
    args:
        - cost (function)
            cost function to be minimized
        - x0 (np.array)
            initial guess
    kwargs:
        - method (str) [default: 'lsq']
            method for minimization
            options: 'lsq' (least squares), 'brute' (brute force), 'nelder' (Nelder-Mead)
        - bounds (list of tuples)
            bounds for the solution
            bounds is a list of tuples where the
            first element is the lower bound
            and the second element is the upper bound
        - jacobian (function)
            jacobian of the cost function
            returns matrix of shape (m,n) with m=len(cost) and n=len(x0)
            jacobian[i,j] = d cost[i] / d x0[j]
        - total_time (float)
            total time in seconds available for optimization
        - max_nfev (int)
            maximal number of function evaluations
        - eps (float)
            parameter for 'direct' method
        - num_cpus (int) [default: 1]
            number of cpus to use for parallelization
    returns:
        - solution (np.array)
            solution of the nonlinear equation
        - optimal_cost (np.array)    
            value of the cost function at the solution
        - landscape_grid (np.array)
            grid of the landscape of the cost function
        - landscape_vals (np.array)
            values of the cost function at the grid points
    '''
    method = kwargs.get('method','lsq')
    bounds = kwargs.get('bounds',None) 
    jacobian = kwargs.get('jacobian','2-point')
    total_time = kwargs.get('total_time',None)
    max_nfev = kwargs.get('max_nfev',None)  #(default: 100*number_of_variables for lsq)
    num_cpus = kwargs.get('num_cpus',1)
    if num_cpus>1:
        raise ValueError('num_cpus>1 not yet implemented.')
    # make x0 real
    # x0 = np.real(x0)
    # set max number of function evaluations
    if total_time is not None:
        start = timer.time()
        cost(x0)
        end = timer.time()
        evaluation_time = end-start
        if max_nfev is None:
            max_nfev = int(total_time/evaluation_time) 
        else:
            max_nfev = min(max_nfev, int(total_time/evaluation_time))
    # # choose 'brute' if only one variable
    # if method=='lsq' and len(x0)==1 and bounds is not None:
    #     method = 'brute'
    #     if max_nfev is None:
    #         max_nfev = 20
    # get bounds for lsq method
    if bounds is None:
        bounds_lsq = (-np.inf, np.inf)
    else:
        bounds_lsq = [[bound[0] for bound in bounds], [bound[1] for bound in bounds]]
    ### run optimizer ###
    if method=='lsq':
        res = sp.optimize.least_squares(cost, x0, bounds=bounds_lsq, jac=jacobian, max_nfev=max_nfev)
        solution = res.x
    elif method=='nelder':
        res = sp.optimize.minimize(cost, x0, method='Nelder-Mead', bounds=bounds, jac=jacobian, options={'maxiter':max_nfev})
        solution = res.x
    elif method=='direct':
        eps = kwargs.get('eps',1e-4)
        if max_nfev is None:
            max_nfev = 1000
        # maxfun is maximal number of function evaluations [default:1000*ndims]
        # maxiter is maximal number of iterations [default:1000]
        # eps is used to decide which hypercubes to further divide, the smaller the more local the search [default:1e-4]
        # res = sp.optimize.direct(cost, bounds=bounds, maxfun=max_nfev, len_tol=0.001, vol_tol=0, eps=eps)
        res = sp.optimize.direct(cost, bounds=bounds, maxfun=max_nfev, eps=eps, maxiter=max_nfev)
        solution = res.x
        # print('stopping criterion',res.message)
    elif method=='brute':
        if bounds is None:
            raise ValueError('Bounds must be specified for brute force optimization.')
        if max_nfev is None:
            raise ValueError('max_nfev or total_time must be specified for brute force optimization.')
        def cost_brute(x):
            return np.linalg.norm(cost(x))
        ranges = bounds 
        Ns = int(max_nfev**(1/len(x0)))
        res = sp.optimize.brute(cost_brute, ranges=ranges, Ns=Ns, full_output=True, finish=None, workers=num_cpus)
        solution = res[0]
        if not isinstance(solution,np.ndarray):
            solution = np.array([solution])
        landscape_grid = res[2]
        landscape_vals = res[3]
        # # refine solution using 'lsq' method
        # res = sp.optimize.least_squares(cost, solution, bounds=bounds_lsq, jac=jacobian) #, max_nfev=max_nfev)
        # solution = res.x
    # evaluate optimal cost
    optimal_cost = cost(solution)
    # # check if solution is sufficiently different from initial guess
    # dist = np.linalg.norm(solution-x0)/np.linalg.norm(x0)
    # if dist<1e-3:
    #     print('Warning: solution is close to initial guess, dist={}.'.format(dist))
    if method != 'brute':
        landscape_grid = None
        landscape_vals = None
    return solution, optimal_cost, landscape_grid, landscape_vals

def integrate(x,y,**kwargs):
    '''
    Integrate a function y(x) using given method.
    args:
        - x (np.array)
            x values
        - y (np.array)
            y values
    kwargs:
        - method (str) [default: 'trapz']
            method for integration
        - fct (function) [default: None]
            function to be integrated
    returns:
        - integral (float)
            integral of y(x) over x
    '''
    # get kwargs
    method = kwargs.get('method','trapz')
    fct = kwargs.get('fct',None)
    # check if x and y have the same length
    if len(x)!=len(y):
        raise ValueError('x and y have different lengths.')
    # check if x is sorted
    if not np.all(x[:-1]<=x[1:]):
        raise ValueError('x is not sorted.')
    # calculate integral
    if method=='trapz':
        integral = np.trapz(y,x=x)
        neval = len(y)
    elif method=='simpson':
        integral = sp.integrate.simpson(y,x=x)
        neval = len(y)
    elif method=='simpson38':
        integral = simpson38(x,y)
        neval = len(y)
    elif method=='romb':
        integral = sp.integrate.romb(y,dx=x[1]-x[0])
        neval = len(y)
    elif method=='romberg':
        integral = sp.integrate.romberg(fct, x[0], x[-1], show=True)
        neval = 1
    elif method=='quad':
        res = sp.integrate.quad(fct, x[0], x[-1], limit=len(x)-1, full_output=1)
        integral = res[0]
        neval = res[2]['neval']
    elif method=='gaussian':
        integral = sp.integrate.quadrature(fct, x[0], x[-1])[0]
        neval = 1
    else:
        raise ValueError('Unknown method {}.'.format(method))
    return integral, neval

def simpson38(x,y):
    ''' Simpson38 method for integration. '''
    # calculating step size
    dt = x[1] - x[0]
    n = len(y)
    # Finding sum 
    integration = y[0] + y[-1] #f(x0) + f(xn)
    for i in range(1, n, 3):
        integration += 3 * y[i] #f(a + i * h)
    for i in range(3, n-1, 3):
        integration += 3 * y[i] #f(a + i * h)
    for i in range(2, n-2, 3):
        integration += 2 * y[i] #f(a + i * h)
    # Finding final integration value
    integration = integration * 3 * dt / 8
    return integration

def estimate_derivative(values, times, **kwargs):
    '''
    Estimate the derivative of a function given
    its values at certain times.
    args:
        - values (np.array)
            values of the function at certain times
        - times (np.array)
            times at which the function is evaluated
        - type 
    kwargs:
        - order (int) [default: 1]
            order of the error of the 2-point derivative
    returns:
        - derivative (float)
            derivative of the function at time times[0]
    '''
    order = kwargs.get('order',1)
    # estimate derivative
    # check if values and times have the same length
    if len(values)!=len(times):
        raise ValueError('values and times have different lengths.')
    # check if all times are different
    if len(times)!=len(set(times)):
        raise ValueError('times must be unique.')
    if len(times)==2:
        # estimate derivative using forward difference
        derivative = (values[1]-values[0])/(times[1]-times[0])
    elif len(times)==3:
        # estimate derivatives using forward difference
        deriv1 = (values[1]-values[0])/(times[1]-times[0])
        deriv2 = (values[2]-values[0])/(times[2]-times[0])
        # perform Richardson extrapolation
        derivative = deriv1 + (deriv1-deriv2)/(((times[2]-times[0])/(times[1]-times[0]))**order - 1)
        ### below for derivative at t=dt instead of t=0 ###
        # # estimate derivatives using forward difference
        # deriv1 = (values[2]-values[1])/(times[2]-times[1])
        # deriv2 = (values[2]-values[0])/(times[2]-times[0])
        # # perform Richardson extrapolation
        # derivative = deriv1 + (deriv1-deriv2)/(((times[2]-times[0])/(times[1]-times[0]))**order - 1)
    # check if len(times) is power of 2 +1
    elif np.log2(len(times)-1)%1==0:
        # order = int(np.log2(len(times)-1))
        # estimate derivatives using forward difference
        derivs = [(values[inx]-values[0])/(times[inx]-times[0]) for inx in range(1,len(values))]
        timesteps = [times[inx]-times[0] for inx in range(1,len(times))]
        # perform iterative Richardson extrapolation
        while len(derivs)!=1:
            npairs = int(np.log2(len(times)-1))
            val_pairs = np.array_split(derivs,npairs)
            dt_pairs = np.array_split(timesteps,npairs)
            derivs_new = []
            timesteps_new = []
            for pairinx, val_pair in enumerate(val_pairs):
                dt_pair = dt_pairs[pairinx]
                Rextrapolation = Richardson_extrapolation(val_pair[0], val_pair[1], dt_pair[0], dt_pair[1], order)
                derivs_new.append(Rextrapolation)
                timesteps_new.append(np.max(dt_pair))
            order = order+1
            derivs = derivs_new
            timesteps = timesteps_new
        derivative = derivs[0]
    else:
        raise ValueError('len(times) must be 2 or 3.')
    # print('derivative',derivative)
    return derivative
    
def Richardson_extrapolation(deriv1, deriv2, dt1, dt2, order):
    '''
    Perform Richardson extrapolation on two derivatives.
    args:
        - deriv1 (float)
            first derivative
        - deriv2 (float)
            second derivative
        - dt1 (float)
            first time step
        - dt2 (float)   
            second time step
        - order (int)
            order of the derivatives (same for both derivatives)
    '''
    Rextrapolation = deriv1 + (deriv1-deriv2)/((dt2/dt1)**order - 1)
    return Rextrapolation

