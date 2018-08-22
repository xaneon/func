#!/usr/local/bin/env python
# -*- coding: utf-8 -*-

"""
Monte Carlo simulations and statistics.

.. module:: monte.py
.. moduleauthor:: Bonne Habekost <b.habekost1@ncl.ac.uk>
.. modulemodified:: April 27, 2015
"""

import setpath
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

F = setpath.setpath('y')


def create_vectors(n_vector):
    """
    This method is creating one independent one random vector, another
    random vector, an completely independent vector and a
    completely dependent vector.

    :param n_vector: Length of the vector
    :type : Integer
    """
    a1 = np.random.randn(n_vector)
    a2 = np.random.randn(n_vector)
    a3 = np.random.rand(n_vector)
    a4 = a1
    return a1, a2, a3, a4


def get_stats(v1, v2, num_sim=None, alpha=None, showfigue=None):
    """
    Calculates the statistics using a Monte Carlo permutation approach.
    The H0 is that both vectors are coming form the same distribution
    or population (H1 vectors are coming from different populations).

    :param v1: Vector 1
    :type v1: List
    :param v2: Vector 2
    :type v2: List
    :param num_sim: Number of simulations / permutations.
    :type num_sim: Integer
    :param alpha: Significance level
    :type alpha: Integer
    :param showfigue: Show the distribution
    :type showfigue: Boolean
    :return: H=0 for no difference, H=1 for difference.
    :return: p-value of the likelihood `v1` and `v2` same distribution.
    """
    if(num_sim is None):
        num_sim = 1000
    if(alpha is None):
        alpha = .05
    v1 = np.array(v1)
    v2 = np.array(v2)
    m1 = np.nanmean(v1)
    m2 = np.nanmean(v2)
    n1 = len(v1)
    n2 = len(v2)
    dobs = m1 - m2  # observed mean difference
    dsim = list()  # difference distribution
    for i in xrange(0, num_sim):
        idx1 = np.random.randint(n1, size=n1)
        idx2 = np.random.randint(n2, size=n2)
        curr_m1 = np.mean(v1[idx1])
        curr_m2 = np.mean(v2[idx2])
        curr_d = curr_m1 - curr_m2
        dsim.append(curr_d)
    mdsim = np.mean(dsim, axis=0)
    sdsim = np.std(dsim, axis=0)
    dsim_counts, dsim_edges = np.histogram(dsim, bins=100)
    zdobs = (dobs - mdsim) / sdsim
    z1 = (m1 - mdsim) / sdsim
    z2 = (m2 - mdsim) / sdsim
    zdsim = (dsim - mdsim) / sdsim
    zdsim_counts, zdsim_edges = np.histogram(zdsim, bins=100)
    zdsim_edges = zdsim_edges[:-1]
    zdsim_counts_norm = zdsim_counts / float(np.max(zdsim_counts))
    dobs = (mdsim - np.abs(dobs)) - mdsim
    z3 = z1 - z2
    p3 = stats.norm.sf(np.abs(z3))*2  # two sided
    if(zdobs <= 0.05):
        p3 = p3
    else:
        p3 = 1 - p3
    if(p3 <= alpha):
        H = 1
    else:
        H = 0
    if(showfigue is None):
        pass
    else:
        plt.figure(1)
        plt.step(zdsim_edges, zdsim_counts_norm)
        plt.hold(True)
        plt.vlines(zdobs, 0, 1, colors='r')
        plt.vlines(np.mean(zdsim), 0, 1, colors='k')
    return H, p3

if __name__ == '__main__':
    n_vector = 1000  # Length of a vector
    a, b, c, d = create_vectors(n_vector)  # Create example vectors
    H_ab, p_ab = get_stats(a, b, num_sim=10000, alpha=0.05)
    print H_ab, p_ab
