#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from scipy.optimize import root_scalar
from scipy.stats import bernoulli


def approx_eql(x, y, tol=1e-10):
    return np.abs(x-y) <= tol


def approx_lss(x, y, tol=1e-10):
    return x <= y + tol


def approx_gtr(x, y, tol=1e-10):
    return x >= y - tol


def get_root_max(f, tol=1e-10):
    """Get right end point, within tolerance, of interval on which f = 0"""
    x_min, x_max = 0.0, 1.0
    f_min, f_max = f(x_min), f(x_max)

    if f_min > 0.0:
        return x_min
    if f_max <= 0.0:
        return x_max

    while x_max - x_min > tol:
        x_mid = (x_min + x_max) / 2
        f_mid = f(x_mid)
        if f_mid <= 0.0:
            x_min = x_mid
            f_min = f_mid
        else:
            x_max = x_mid
            f_max = f_mid

    return x_min


def get_root_min(f, tol=1e-10):
    """Get left end point, within tolerance, of interval on which f = 0"""
    x_min, x_max = 0.0, 1.0
    f_min, f_max = f(x_min), f(x_max)

    if f_min >= 0.0:
        return x_min
    if f_max < 0.0:
        return x_max

    while x_max - x_min > tol:
        x_mid = (x_min + x_max) / 2
        f_mid = f(x_mid)
        if f_mid >= 0.0:
            x_max = x_mid
            f_max = f_mid
        else:
            x_min = x_mid
            f_min = f_mid

    return x_max


def get_root_interval(f, tol=1e-10):
    """Get interval on which f = 0"""
    x_max = get_root_max(f, tol)
    x_min = get_root_min(f, tol)

    # If the root is unique, the derivative is not.
    # Return the maximum of the differential (left derivative).
    if np.abs(x_max - x_min) < 2 * tol:
        x_root = (x_min + x_max) / 2
        f_min = f(x_root - tol)
        return (x_root, x_root), f_min

    # If the derivative is unique, the root is not.
    # Return interval and unique derivative.
    else:
        f_root = f((x_min + x_max) / 2)
        return (x_min, x_max), f_root


def get_psi(gamma, drho_left):
    """randomize such that drho_left * (1-psi) = gamma, 
       where drho_left denotes the left derivative of rho.
    """
    if gamma == 0.0 or drho_left > 0.0:
        return 1.0
    else:
        return 1 - gamma / (gamma - drho_left)


def get_pmf_from_bounds(p_min, p_max):
    """Pick a pmf from the feasible band"""
    sum_min = np.sum(p_min)
    if approx_eql(sum_min, 1.0):
        return p_min / np.sum(p_min)

    sum_max = np.sum(p_max)
    if approx_eql(sum_max, 1.0):
        return p_max / np.sum(p_max)

    a = (1 - sum_min) / (sum_max - sum_min)

    return (1 - a) * p_min + a * p_max


def get_x_next(x, n):
    """Update sample avfter having observed n"""
    x_next = list(x).copy()
    x_next[n] += 1
    return tuple(x_next)


class PWLFun:
    """Piecewise linear function on unit interval"""

    def __init__(self, dc_vec, df_vec, f0):
        """Arguments:
            - dc_vec: vector of discontinuities of the derivative
            - df_vec: vector of derivative values between dicontinuities
            - f0:     value of functoin a 0
        """
        self.dc = dc_vec
        self.f0 = f0
        self.df_vec = df_vec

    def f(self, z):
        """Evvaluate function at z \in [0, 1]"""
        f_val = self.f0
        if z <= self.dc[0]:
            f_val += z * self.df_vec[0]
        else:
            f_val += self.dc[0] * self.df_vec[0]
            for i in range(1, self.dc.size):
                if z >= self.dc[i]:
                    f_val += self.df_vec[i] * (self.dc[i] - self.dc[i - 1])
                else:
                    f_val += self.df_vec[i] * (z - self.dc[i - 1])
                    break
        return f_val

    def df(self, z):
        """Evvaluate derivative at z \in [0, 1]"""
        df_gtr = self.df_vec[self.dc >= z]
        if df_gtr.size > 0:
            return df_gtr[0]
        else:
            return 0.0


class Test:
    """Main class, implements the nonparametric Kiefer--Weiss test"""

    def __init__(self, horizon, p1, p2, cost, verbose=False, dz_min=1e-10):
        """Arguments:
            - horizon: horizon of the test (sclar)
            - p1, p2:  pmfs under H1, H2 (np.array)
            - cost:    two cost coefficients \lambda (scalar or np.array)
            - verbose: print log and progress
            - dz_min:  minimum segment size of rho
        """
        self.horizon = horizon
        self.N = p1.size
        self.p1 = p1
        self.p2 = p2
        self.verbose = verbose
        self.dz_min = dz_min
        self.rho_dict = {}

        if np.isscalar(cost):
            self.cost = [cost, cost]
        else:
            self.cost = cost

    def pmf(self, H, x):
        """pmf under H evaluated at x"""
        if H == 1:
            return np.prod(self.p1 ** x)
        elif H == 2:
            return np.prod(self.p2 ** x)
        else:
            return None

    def g(self, x):
        """cost for stopping given x"""
        cost_H1 = self.cost[0] * self.pmf(1, x)
        cost_H2 = self.cost[1] * self.pmf(2, x)
        return np.minimum(cost_H1, cost_H2)

    def d(self, z0, x):
        """cost for continuing given x"""
        q, psi, gamma, d = self._look_ahead(z0, x)
        return d

    def rho(self, z0, x):
        """optimal cost given z0 and x"""
        rho = self.get_rho(x)
        return rho.f(z0)

    def gamma(self, z0, x):
        """Vector of expected remaining sample sizes after observing the next sample"""
        q, psi, gamma, d = self._look_ahead(z0, x)
        return gamma

    def psi(self, z0, x):
        """Vector of stopping probabilities after observing the next sample"""
        q, psi, gamma, d = self._look_ahead(z0, x)
        return psi

    def q(self, z0, x):
        """least favorable distributions"""
        q, psi, gamma, d = self._look_ahead(z0, x)
        return q

    def setup(self):
        """calculate all rho functions"""
        self.get_rho(self.N * (0,))

    def get_rho(self, x):
        """get function rho(z0) for sample x"""

        # stop if horizon is reached
        if np.sum(x) == self.horizon:
            rho = PWLFun(np.array([1.0]), np.array([0.0]), self.g(x))
            return rho

        # use stored function if it has been calculated already
        elif x in self.rho_dict:
            return self.rho_dict[x]

        # otherwise calculate rho recursively
        else:
            if self.verbose:
                print("assembling rho: " + str(x))

            # cost difference for continuing/stopping (increasing in z)
            # z0_max denotes the root of this function
            def cost_diff(z0):
                return z0 + self.d(z0, x) - self.g(x)

            # stop if it is cheaper for all z0 -> rho = g
            if cost_diff(0.0) >= 0.0:
                self.rho_dict[x] = PWLFun(np.array([1.0]), np.array([0.0]), self.g(x))
                return self.rho_dict[x]

            # catch continue-for-all-z0 case
            elif cost_diff(1.0) <= 0.0:
                z0_max = 1.0

            # otherwise determine z0 via root finding
            else:
                res = root_scalar(cost_diff, method="bisect", bracket=(0, 1))
                z0_max = res.root

            # calculate rho from scratch
            rho_params = self._get_rho_params(z0_max, x)
            self.rho_dict[x] = PWLFun(*rho_params)
            return self.rho_dict[x]

    def _get_rho_params(self, z0_max, x):
        """get segments and slopes of piecewise linear rho"""

        # rho(0)
        f0 = np.minimum(self.d(0.0, x), self.g(x))

        # derivative of rho calcuated via a look ahead
        def gamma_func(z0):
            if z0 > z0_max:
                return 0.0
            else:
                return 1 + self.gamma(z0, x)

        # Get vector discontinuities iteratively finding the intersections of
        # drho with a constant function. Since dhro is integer valued, its
        # intersection with a constant c+0.5 occurs at a discontinuity.
        df_list = []
        dc_list = []
        dc = 0.0
        drho_min = gamma_func(1.0)
        while True:
            # get derivative just right of dc
            drho = gamma_func(dc + self.dz_min)
            if drho <= drho_min:
                df_list.append(drho_min)
                dc_list.append(1.0)
                break

            def dc_func(z0):
                return gamma_func(z0) - (drho - 0.5)

            res = root_scalar(dc_func, method="bisect", bracket=(0, 1))
            dc = res.root
            assert dc <= z0_max

            dc_list.append(dc)
            df_list.append(drho)

        return np.array(dc_list), np.array(df_list), f0

    def _look_ahead(self, z0, x):
        """Get LFDs, stopping rules, expected remaining sample size and cost 
           for continuing by solving the maximization in (18)
        """
        q_min = np.zeros(self.N)
        q_max = np.zeros(self.N)
        psi = np.zeros(self.N)
        optimal = False

        # Iterate over c = 0, 1, 2,... until LFDs can be found that satisfy
        # the optimality conditions in (33) and (34)
        for c in range(self.horizon):
            for n in range(self.N):
                x_next = get_x_next(x, n)
                rho_next = self.get_rho(x_next)

                if rho_next.df(z0) > c:
                    feasible = False
                    break
                else:

                    def c_diff(qn):
                        return c - rho_next.df(z0 * qn)

                    root, f_val = get_root_interval(c_diff)

                    q_min[n], q_max[n] = root
                    psi[n] = get_psi(c, f_val)

                    feasible = True

            sum_min = np.sum(q_min)
            sum_max = np.sum(q_max)
            if feasible and approx_lss(sum_min, 1.0) and approx_gtr(sum_max, 1.0):
                optimal = True
                break

        assert(optimal)
        gamma = c
        q = get_pmf_from_bounds(q_min, q_max)

        # calculate d aacording to (18)
        d = 0
        for n in range(self.N):
            x_next = get_x_next(x, n)
            rho_next = self.get_rho(x_next)
            d += rho_next.f(z0 * q[n])

        return q, psi, gamma, d

    def simulate(self, p=(None,), verbose=False):
        """Simulate test under the LFDs or a user defined pmf p. 
           Set 'verbose=True' to see the progress."""
        z0 = 1.0
        x = self.N * (0,)

        if verbose:
            print("\nStarting test:")
            print("\nsample = " + str(x) + "\n")

        for n in range(self.horizon):
            q, psi, gamma, d = self._look_ahead(z0, x)

            if verbose:
                print("q     = " + str(q))
                print("psi   = " + str(psi))
                print("gamma = " + str(gamma))

            if any(p):
                s = np.random.multinomial(1, p)
            else:
                s = np.random.multinomial(1, q)

            x = tuple(x + s)
            idx = np.nonzero(s)[0].item()
            z0 *= q[idx]

            if verbose:
                print("\nsample = " + str(x) + "\n")

            if bernoulli.rvs(psi[idx]) or np.sum(x) == self.horizon:
                break
            
        if self.cost[0] * self.pmf(1, x) > self.cost[1] * self.pmf(2, x):
            H = 1
        else:
            H = 2    
        
        if verbose:
            print("Stopped with decision for H" + str(H));

        return H
