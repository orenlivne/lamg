'''
============================================================
Local mode analysis of a two-level method for the isotropic
1D Laplacian on the unit square with periodic BC.
Gauss-Seidel relaxation (lexicographic ordering),
semi-coarsening in the x-direction, constant/linear
interpolation and Galerkin/direct discretization coarse
operator.  

Created on Apr 28, 2014
@author: Oren Livne <livne@uchicago.edu>
============================================================
'''
import itertools as it
import numpy as np
from numpy.matrixlib.defmatrix import matrix_power
# from numpy.linalg.linalg import inv
#import cProfile
#import pstats

I = np.sqrt(-1 + 0j)
PI = np.pi

'''Cycle ingredients for semi-coarsening in the x-direction of the 2-D anisotropic Laplacian
operator a*uxx + c*uyy.'''
class Grid2dXSemiCoarsening(object):
    def __init__(self, interpolation='linear', coarse_operator='galerkin'):
        self.coarse_operator = coarse_operator
        
        if interpolation == 'constant':
            self.r = self.r1
        elif interpolation == 'linear':
            self.r = self.r2
        else:
            raise ValueError('Unsupported interpolation type ''%s''' % (interpolation,))

        if coarse_operator == 'galerkin':
            self.ac = None
        elif coarse_operator == 'direct':
            self.ac = self.ac_fd_5point
        else:
            raise ValueError('Unsupported coarse operator type ''%s''' % (coarse_operator,))
        
    def harmonics(self, t):
        '''Return the vector of harmonics of the scaled frequency t.'''
        return (t, t + PI)
        
    def s(self, t):
        '''Gauss-Seidel relaxation symbol, lexicographic ordering.'''
        return np.exp(I * t) / (2 - np.exp(-I * t))
    
    def a(self, t):
        '''Symbol of the fine-level operator h^2*A^h(t).'''
        return 2 * (1 - np.cos(t))
    
    def r1(self, t):
        '''Linear full-weighting symbol into a semi-coarsening in x.'''
        # return 0.5 * (1 + np.cos(t))
        return 0.5 * (1 + np.exp(-I * t))  # transpose of first-order interpolation
    
    def r2(self, t):
        '''Linear full-weighting symbol into a semi-coarsening in x.'''
        return 0.5 * (1 + np.cos(t))
        # return 0.5 * (1 + np.exp(-I * t))  # transpose of first-order interpolation
    
    def ac_fd_5point(self, t):
        '''Symbol of direct FD discretization on the coarse grid.'''
        return np.matrix(0.5 * (1 - np.cos(2 * t)))

'''A generic local mode analysis runner. Accepts the two-level cycle's building blocks and
calculates its convergence factor.'''
class LocalModeAnalyzer(object):
    eps = 1e-6
    
    def __init__(self, builder):
        self._builder = builder 
        if self._builder.coarse_operator == 'galerkin':
            self.ac = self.ac_galerkin
        else:
            self.ac = self._builder.ac
    
    def apply_to_harmonics(self, f, t):
        '''Apply f element-wise to the list of harmonics of the scaled frequency t.'''
        return map(f, self._builder.harmonics(t))  

    def S(self, t):
        '''Matrix symbol of Gauss-Seidel. Lex ordering does not couple frequencies.'''
        return np.diag(self.apply_to_harmonics(self._builder.s, t))
        
    def R(self, t):
        '''Matrix symbol of the restriction operator.'''
        return np.matrix(self.apply_to_harmonics(self._builder.r, t))
    
    def P(self, t):
        '''Matrix symbol of the interpolation operator.'''
        return self.R(t).conjugate().transpose()
    
    def A(self, t):
        '''Matrix symbol of the fine-level operator h^2*A^h(t).'''
        return np.diag(self.apply_to_harmonics(self._builder.a, t))
    
    def ac_galerkin(self, t):
        '''Symbol of the Galerkin coarse-level operator.'''
        return self.R(t) * self.A(t) * self.P(t)

    def M(self, t, nu):
        '''Two-level method''s symbol with nu pre-relaxations per cycle.'''
        return (np.eye(2) - self.P(t) * (1. / (self.ac(t))) * self.R(t) * self.A(t)) * matrix_power(self.S(t), nu) 
    
    def mu(self, t, nu):
        '''Asymptotic Amplification factor of frequency t in the two-level Cycle(0,3).'''
        return max(np.abs(lam) for lam in np.linalg.eig(self.M(t, nu))[0])
    
    def acf(self, nu, n=100):
        '''Return the Asymptotic Convergence Factor (ACF) of the two-level method. Use a
        mesh of nxn points in scaled frequency space.'''
#        t = (PI + LocalModeAnalyzer.eps, LocalModeAnalyzer.eps)
#         print 'S'
#         print self.S(t)
#         print 'A'
#         print self.A(t)
#         print 'Ac'
#         print self.ac(t)
#         print 'P'
#         print self.P(t)
#         print 'R'
#         print self.R(t)
#         print 'CGC'
#         print self.M(t, 0)
#         print 'M'
#         print self.M(t, nu)
#         print 'mu', self.mu(t, nu)
#         for t in np.linspace(-PI + LocalModeAnalyzer.eps, PI + LocalModeAnalyzer.eps, n + 1):
#             for t2 in np.linspace(-PI + LocalModeAnalyzer.eps, PI + LocalModeAnalyzer.eps, n + 1):
#                 print t, t2, self.mu(t, nu)
        return max(self.mu(t, nu)
                   for t in np.linspace(-PI + LocalModeAnalyzer.eps, PI + LocalModeAnalyzer.eps, n + 1))

if __name__ == '__main__':
    # Profile the slow part of the LMA run.
#     lma = LocalModeAnalyzer(Grid2dXSemiCoarsening(coarse_operator='direct', interpolation='constant'))
#     def lma_run():
#         return lma.acf(3, n=64)
#     cProfile.run('lma_run()', 'lma_run')
#     s = pstats.Stats('lma_run')
#     s.strip_dirs().sort_stats('cumulative').print_stats()

    for interpolation, coarse_operator in it.product(['constant', 'linear'], ['galerkin', 'direct']):
    # lma = LocalModeAnalyzer(Grid2dXSemiCoarsening(coarse_operator='galerkin', interpolation='linear'))
    # lma = LocalModeAnalyzer(Grid2dXSemiCoarsening(coarse_operator='galerkin', interpolation='constant'))
    # lma = LocalModeAnalyzer(Grid2dXSemiCoarsening(coarse_operator='direct', interpolation='constant'))
        print 'interpolation', interpolation, 'coarse_operator', coarse_operator
        lma = LocalModeAnalyzer(Grid2dXSemiCoarsening(coarse_operator=coarse_operator, interpolation=interpolation))
        for nu in xrange(1, 6):  # [3]:  # xrange(1, 5):
            print 'nu = %d, ACF = %.3f' % (nu, lma.acf(nu, n=128))
