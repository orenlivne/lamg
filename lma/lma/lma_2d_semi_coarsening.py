'''
============================================================
Local mode analysis of a two-level method for the isotropic
2D Laplacian on the unit square with periodic BC.
Gauss-Seidel relaxation (lexicographic ordering), linear
interpolation and Galerkin coarsening.  

Created on Apr 28, 2014
@author: Oren Livne <livne@uchicago.edu>
============================================================
'''
import numpy as np
from numpy.matrixlib.defmatrix import matrix_power
from numpy.linalg.linalg import inv

I = np.sqrt(-1 + 0j)
PI = np.pi

def harmonics((t1, t2)):
    '''Return the vector of harmonics of the scaled frequency (t1,t2).'''
    return ((t1, t2), (t1 + PI, t2))

def apply_to_harmonics(f, t):
    '''Apply f element-wise to the list of harmonics of the scaled frequency t.'''
    return map(f, harmonics(t))  

def s((t1, t2)):
    '''Gauss-Seidel relaxation symbol, lexicographic ordering.'''
    return (np.exp(I * t1) + np.exp(I * t2)) / (4 - np.exp(-I * t1) - np.exp(-I * t2))

def S(t):
    '''Matrix symbol of Gauss-Seidel. Lex ordering does not couple frequencies.'''
    return np.diag(apply_to_harmonics(s, t))

def a((t1, t2)):
    '''Symbol of the fine-level operator h^2*A^h(t).'''
    return 2 * (2 - np.cos(t1) * np.cos(t2))

def r((t1, t2)):
    '''Linear full-weighting symbol into a semi-coarsening in x.'''
    return 0.5 * (1 + np.cos(t1))
    #return 0.5 * (1 + np.exp(I * t1)); # First-order interpolation
    
def R(t):
    '''Matrix symbol of the restriction operator.'''
    return np.matrix(apply_to_harmonics(r, t))

def P(t):
    '''Matrix symbol of the interpolation operator.'''
    return R(t).transpose()

def A(t):
    '''Matrix symbol of the fine-level operator h^2*A^h(t).'''
    return np.diag(apply_to_harmonics(a, t))

def Ac(t):
    '''Symbol of the Galerkin coarse-level operator.'''
    return R(t) * A(t) * P(t)

def M(t, nu):
    '''Two-level method''s symbol with nu pre-relaxations per cycle.'''
    return (np.eye(2) - P(t) * inv(Ac(t)) * R(t) * A(t)) * matrix_power(S(t), nu) 

def mu(t, nu):
    '''Asymptotic Amplification factor of frequency t in the two-level Cycle(0,3).'''
    return max(np.abs(lam) for lam in np.linalg.eig(M(t, nu))[0])

def acf(nu, n=100):
    '''Return the Asymptotic Convergence Factor (ACF) of the two-level method. Use a
    mesh of nxn points in scaled frequency space.''' 
    return max(mu((t1, t2), nu) for t1 in np.linspace(-PI, PI, n + 1) for t2 in np.linspace(-PI, PI, n + 1))

if __name__ == '__main__':
    for nu in xrange(1, 5):
        print 'nu = %d, ACF = %.3f' % (nu, acf(nu))
