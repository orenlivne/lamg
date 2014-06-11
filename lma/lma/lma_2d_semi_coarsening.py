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

def apply_to_harmonics(f, t):
    '''Apply f element-wise to the list of harmonics of the scaled frequency t.'''
    return map(f, harmonics(t))  

def harmonics(self, (t1, t2)):
    '''Return the vector of harmonics of the scaled frequency (t1,t2).'''
    return ((t1, t2), (t1 + PI, t2))
    
class Grid2dSemiCoarseningLma(object):
    def __init__(self):
        self.coarse_operator = 'galerkin'
        self.interpolation = 'linear' 
        
    def s(self, (t1, t2)):
        '''Gauss-Seidel relaxation symbol, lexicographic ordering.'''
        return (np.exp(I * t1) + np.exp(I * t2)) / (4 - np.exp(-I * t1) - np.exp(-I * t2))
    
    def a(self, (t1, t2)):
        '''Symbol of the fine-level operator h^2*A^h(t).'''
        return 2 * (2 - np.cos(t1) - np.cos(t2))
    
    def r2(self, (t1, t2)):
        '''Linear full-weighting symbol into a semi-coarsening in x.'''
        return 0.5 * (1 + np.cos(t1))
        # return 0.5 * (1 + np.exp(-I * t1))  # transpose of first-order interpolation
    
    def r(self, (t1, t2)):
        '''Linear full-weighting symbol into a semi-coarsening in x.'''
        # return 0.5 * (1 + np.cos(t1))
        return 0.5 * (1 + np.exp(-I * t1))  # transpose of first-order interpolation
    
    def Ac(self, t):
        '''Symbol of the Galerkin coarse-level operator.'''
        R = np.matrix(apply_to_harmonics(self.r2, t))
        P = R.conjugate().transpose()
        return R * self.A(t) * P
        # return R(t) * A(t) * P(t)
    
    def S(self, t):
        '''Matrix symbol of Gauss-Seidel. Lex ordering does not couple frequencies.'''
        return np.diag(apply_to_harmonics(self.s, t))
        
    def R(self, t):
        '''Matrix symbol of the restriction operator.'''
        return np.matrix(apply_to_harmonics(self.r, t))
    
    def P(self, t):
        '''Matrix symbol of the interpolation operator.'''
        return self.R(t).conjugate().transpose()
    
    def A(self, t):
        '''Matrix symbol of the fine-level operator h^2*A^h(t).'''
        return np.diag(apply_to_harmonics(self.a, t))
    
    def M(self, t, nu):
        '''Two-level method''s symbol with nu pre-relaxations per cycle.'''
        return (np.eye(2) - self.P(t) * inv(self.Ac(t)) * self.R(t) * self.A(t)) * matrix_power(self.S(t), nu) 
    
    def mu(self, t, nu):
        '''Asymptotic Amplification factor of frequency t in the two-level Cycle(0,3).'''
        return max(np.abs(lam) for lam in np.linalg.eig(self.M(t, nu))[0])
    
    def acf(self, nu, n=100):
        '''Return the Asymptotic Convergence Factor (ACF) of the two-level method. Use a
        mesh of nxn points in scaled frequency space.'''
    #     t = (PI, 0)
    #     print 'S'
    #     print S(t)
    #     print 'A'
    #     print A(t)
    #     print 'Ac'
    #     print Ac(t)
    #     print 'P'
    #     print P(t)
    #     print 'R'
    #     print R(t)
    #     print 'CGC'
    #     print M(t, 0)
    #     print 'M'
    #     print M(t, nu)
    #     print 'mu', mu(t, nu)
    #     for t1 in np.linspace(-PI, PI, n + 1):
    #         for t2 in np.linspace(-PI, PI, n + 1):
    #             print t1, t2, mu((t1, t2), nu)
        eps = 1e-5
        return max(self.mu((t1, t2), nu) for t1 in np.linspace(-PI + eps, PI + eps, n + 1) for t2 in np.linspace(-PI + eps, PI + eps, n + 1))

if __name__ == '__main__':
    for nu in xrange(1, 5):
        print 'nu = %d, ACF = %.3f' % (nu, acf(nu, n=64))
