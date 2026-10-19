#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 24 12:44:48 2025

@author: s2719899
"""
import numpy as np
from numpy import linalg as LA

# sampling uniformly from an ellipsoid is easiest by first sampling from a unit ball, then stretching, rotating, and shifting the point to match the ellipsoid.
def SampleUnitNBall(dim = 3,num = 1):
    '''
    uniformly sample a N-dimensional unit UnitBall
    Reference:
      Efficiently sampling vectors and coordinates from the n-sphere and n-ball
      http://compneuro.uwaterloo.ca/files/publications/voelker.2017.pdf
    Input:
        num - no. of samples
        dim - dimensions
    Output:
        uniformly sampled points within N-dimensional unit ball
    '''
    #Sample on a unit N+1 sphere
    u = np.random.normal(0, 1, (num, dim + 2))
    '''
    If you take a point uniformly distributed on the surface of a unit sphere in Rdim+2Rdim+2, 
    and project it down by keeping only its first dim components,
    then that projected point is uniformly distributed inside the dim-ball.
    '''
    norm = LA.norm(u, axis = -1,keepdims = True) #2-norm
    u = u/norm
    #The first N coordinates are uniform in a unit N ball
    if num == 1: return u[0,:dim]
    return u[:,:dim]

class EllipsoidSampler:
    '''
    uniformly sample within a N-dimensional Ellipsoid
    Reference:
      Informed RRT*: Optimal Sampling-based Path Planning Focused via Direct Sampling
      of an Admissible Ellipsoidal Heuristic https://arxiv.org/pdf/1404.2334.pdf
    '''
    def __init__(self,center,axes = [],rot = []):
        '''
        Input:
            center -  centre of the N-dimensional ellipsoid in the N-dimensional
            axes -  axes length across each dimension in ellipsoid frame
            rot - rotation matrix from ellipsoid frame to world frame
        Output:
            uniformly sampled points within the hyperellipsoid
        '''
        self.dim = center.shape[0]
        self.center = center
        self.rot = rot
        if len(rot) == 0: self.rot = np.eye(self.dim)
        if len(axes) == 0: axes = [1]*self.dim
        self.L = np.diag(axes)

    def sample(self,num = 1):
        xball = SampleUnitNBall(self.dim,num)
        #Transform points in UnitBall to ellipsoid
        xellip = (self.rot@self.L@xball.T).T + self.center
        return xellip

n = 5
center = np.random.randn(n) * 10 # Return from the standard normal distribution, lie in roughly [−30,  30][−30,30]
log_lengths = np.random.uniform(-2, 2, size=n)
axes = np.exp(log_lengths) # n positive numbers with exponentially unifrom, [0.14,7.4], extreme axis ratios are common

Q, _ = np.linalg.qr(np.random.randn(n, n))
if np.linalg.det(Q) < 0: Q[:, 0] *= -1
rot = Q

sampler = EllipsoidSampler(center=center, axes=axes, rot=rot)
x_star = sampler.sample()



G = [[ 0.6,  1.4],         # random Gaussian entries
     [-0.8, 0.3]]

Q, R = np.linalg.qr(G)               # Q ≈ [[-0.83, -0.56],
                           #        [ 0.56, -0.83]]
np.linalg.det(Q) = -1                # mirror + 146° rotation
Q[:,0] *= -1               # flip first column
det(Q) = +1                # now a pure 146° rotation
