import numpy as np
def richardson_center (f, z, h, nsteps, args=()) :
    """Evaluate the first derivative of a function at z, that is f'(z),
    using Richardson extrapolation and center differencing.

    Returned is the full table of approximations, Fij for j <= i. The
    values of Fij for j > i are set to zero.  The final value F[-1,-1]
    should be the most accurate estimate.

    Parameters
    ----------
    f : function
        Python function returning a number.  This is the function for which
        we are estimating the derivative.
    z : number
        Value at which to evaluate the derivative.
    h : number
        Initial stepsize.
    nsteps : integer
        Number of steps to perform.
    args : tuple, optional
        extra arguments to pass to the function, f.
    """
    # Extra check to allow for args=(1) to be handled properly.  This is a
    # technical detail that you do not need to worry about.
    if not isinstance(args, (tuple, list, np.ndarray)) :
        args = (args,)
    # Create a zero filled table
    F = np.zeros ((nsteps,nsteps))
    # First column of F is the center differencing estimate
    harr = h / 2.**np.arange(nsteps)
    F[:,0] = (f(z+harr,*args) - f(z-harr,*args)) / (2.*harr)
    # Now iterate, unfortunately we do need one loop.  We could probably
    # get rid of the inner loop but the algorithm is a little easier to
    # understand if we do not.
    fact = 1.0
    for i in xrange(1,nsteps) :
        fact *= 0.25
        for j in xrange(1,i+1) :
            F[i,j] = F[i-1,j-1] - (F[i-1,j-1] - F[i,j-1])/ (1-fact)
    return F
