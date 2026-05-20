import numpy as np
import numpy.typing as npt

from scipy.interpolate import CubicSpline
from scipy.optimize import minimize_scalar
from scipy.stats import bernoulli, multinomial, norm
from scipy.stats import _continuous_distns, _discrete_distns
from tqdm import tqdm

from typing import Callable, cast, Literal, overload, Protocol

rho_type = Callable[[float | npt.NDArray[np.float64]], float | npt.NDArray[np.float64]]


class RhoSpline(Protocol):
    @overload
    def __call__(self, log_z: float) -> float: ...
    @overload
    def __call__(self, log_z: np.ndarray) -> np.ndarray: ...


class HasRvs(Protocol):
    def rvs(self, size: int | tuple[int, ...] | None = None) -> np.ndarray: ...


@overload
def g(log_z: float, base: float = np.e) -> float: ...


@overload
def g(log_z: npt.NDArray[np.float64], base: float = np.e) -> npt.NDArray[np.float64]: ...


def g(log_z: float | npt.NDArray[np.float64], base: float = np.e) -> float | npt.NDArray[np.float64]:
    """
    Cost of stopping at given log-likelihood ratio:
    min(1, z) = min(1, base ** log_z)
    """
    if np.isscalar(log_z):
        log_z_scalar = cast(float, log_z)
        if log_z_scalar >= 0:
            return 1.0
        else:
            return float(base) ** float(log_z_scalar)
    else:
        log_z_array = cast(np.ndarray, log_z)
        cost = np.ones_like(log_z_array, dtype=float)
        is_negative = log_z_array < 0
        cost[is_negative] = float(base) ** log_z_array[is_negative]
        return cost


def linear_interpolation(x: float, x_grid: np.ndarray) -> tuple[np.signedinteger, np.signedinteger, float, float]:
    """
    Linearly interpolate x on x_grid.

    Returns (indices, weights) such that

        x ≈ weights[0] * x_grid[indices[0]] + weights[1] * x_grid[indices[1]].
    """
    idx_right = np.searchsorted(x_grid, x, side="right")
    idx_right = np.clip(idx_right, 1, len(x_grid) - 1)
    idx_left = idx_right - 1

    dx = x_grid[idx_right] - x_grid[idx_left]
    weight_right = (x - x_grid[idx_left]) / dx
    weight_left = 1 - weight_right

    return idx_left, idx_right, weight_left, weight_right


def get_rho_spline(log_z_grid: npt.NDArray[np.float64], rho_grid: npt.NDArray[np.float64]) -> RhoSpline:
    """
    Fit a cubic spline to rho on log_z_grid and return it as a callable.

    The returned function evaluates the spline at a given log-likelihood ratio,
    clipped to [0, 1] and with flat extrapolation: 0 below the grid and 1 above.

    Parameters
    ----------
    log_z_grid : np.ndarray
        Monotonically increasing grid of log-likelihood ratio values.
    rho_grid : np.ndarray
        Values of rho at each grid point.

    Returns
    -------
    RhoSpline
        A callable that accepts a float or array of log-z values and returns
        the interpolated rho, clipped to [0, 1].
    """
    spline = CubicSpline(log_z_grid, rho_grid, extrapolate=True)

    @overload
    def rho_spline(log_z: float) -> float: ...

    @overload
    def rho_spline(log_z: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]: ...

    def rho_spline(log_z: float | npt.NDArray[np.float64]) -> float | npt.NDArray[np.float64]:
        log_z = np.asarray(log_z)
        rho_val = np.where(
            log_z < log_z_grid[0], 0.0, np.where(log_z > log_z_grid[-1], 1.0, np.clip(spline(log_z), 0.0, 1.0))
        )
        return float(rho_val) if rho_val.ndim == 0 else rho_val

    return rho_spline


class NPKWTDiscreteLimited:
    """
    Nonparametric Kiefer-Weiss Test (NPKWT) between two discrete distributions
    with limited number of randomization uses.
    """

    def __init__(self, k: int, p0: npt.NDArray[np.float64], p1: npt.NDArray[np.float64], log_base: float = np.e):
        """
        Parameters
        ----------
        k       : Maximum number of randomization uses
        p0      : PMF of distribution under null hypothesis
        p1      : PMF of distribution under alternative hypothesis
        log_base: base used for log-likelihood ratio
        """
        self.k = k

        self.p0 = p0
        self.p1 = p1
        self.base = log_base
        self.llr = np.emath.logn(self.base, p1 / p0)

        self.mu0 = p0 @ self.llr
        self.mu1 = p1 @ self.llr
        self.var0 = p0 @ (self.llr - self.mu0) ** 2
        self.var1 = p1 @ (self.llr - self.mu1) ** 2

        self.log_z_grid = np.empty(0)
        self.log_z_min = np.zeros(k + 1)
        self.log_z_max = np.zeros(k + 1)

        self.c_grid = np.empty(0)
        self.c_ppui = 0
        self.c_max = 0

        self.cost_stop = np.empty(0)
        self.rho_vectors = np.empty((k + 1, 0, 0))
        self.rho_splines: list[list[RhoSpline]] = [[] for _ in range(k + 1)]

        self.b_star_vectors = np.empty((0, 0))
        self.b_linear_pos = np.zeros((k + 1, 2))
        self.b_linear_neg = np.zeros((k + 1, 2))

        self.initialized = False

    def run(
        self,
        log_z: float,
        c: float | int,
        p: npt.NDArray[np.float64],
        out_of_range: Literal["extrapolate", "stop"] = "extrapolate",
    ) -> tuple[int, int, int]:
        """
        Run an NPKWT between p0 and p1.

        Parameters
        ----------
        log_z       : Log of cost function weight
        c           : Maximum expected sample size
        p           : PMF of true distribution
        out_of_range: Behavior when log_z falls outside the provided grid
                      extrapolate: Use linear approximation of m_start
                      stop: Stop test

        Returns
        -------
        d     : Accepted hypothesis [0,1]
        t     : Number of samples
        n_rand: Number of randomization uses
        """
        if not self.initialized:
            print("Test uninitialized, call `setup` first.")
            return -1, 0, 0

        k, t = self.k, 0
        while True:
            if k == 0:
                if c < 1:
                    break
                else:
                    b_opt = c
            else:
                if self.log_z_min[k] <= log_z <= self.log_z_max[k]:
                    b_opt = np.maximum(c, self.b_star(k, log_z))
                else:
                    if out_of_range == "stop":
                        break
                    elif out_of_range == "extrapolate":
                        b_opt = np.maximum(c, self.b_star_extrapolated(k, log_z))
                    else:
                        raise ValueError(f"Unknown out_of_range value: {out_of_range!r}")

            if b_opt > c:
                k -= 1
                if bernoulli.rvs(1 - c / b_opt):
                    break

            log_z += (self.llr @ multinomial.rvs(1, p, size=())).item()
            c = b_opt - 1
            t += 1

        d = int(bernoulli.rvs(0.5)) if log_z == 0 else int(log_z > 0)
        return d, t, self.k - k

    def simulate(
        self,
        log_z: float,
        c: float | int,
        p: npt.NDArray[np.float64],
        runs: int,
        out_of_range: Literal["extrapolate", "stop"] = "extrapolate",
    ) -> npt.NDArray[np.integer]:
        """
        Simulate `runs` number of NPKWTs with given parameters.
        See `run` method for parameter details.
        """
        outcomes = [self.run(log_z, c, p, out_of_range) for _ in tqdm(range(runs))]
        return np.array(list(zip(*outcomes)))

    def rho(self, k: int, log_z_grid: npt.ArrayLike, c_grid: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """
        Evaluate rho_k(log_z, c).
        """
        log_z_grid = np.atleast_1d(log_z_grid)
        c_grid = np.atleast_1d(c_grid)
        
        rho_vals = np.zeros((c_grid.size, log_z_grid.size))
        for row, c in enumerate(c_grid):
            if not 0 <= c <= self.c_max:
                print(f"rho_{k}(•, c) only defined for 0 <= c <= {self.c_max}")
                rho_vals[row] = log_z_grid.size * [None]
            else:
                idx_left, idx_right, weight_left, weight_right = linear_interpolation(c, self.c_grid)
                rho_left = self.rho_splines[k][idx_left](log_z_grid)
                rho_right = self.rho_splines[k][idx_right](log_z_grid)
                rho_vals[row] = weight_left * rho_left + weight_right * rho_right
        return np.squeeze(rho_vals)

    def b_star(self, k: int, log_z: float) -> float:
        """
        Evaluate b_k^*(log_z).
        """
        if k == 0:
            return 1
        
        b_grid = self.c_grid[self.c_ppui :]
        cost_cont = np.array(
            [self.rho_splines[k - 1][self.c_idx(b) - self.c_ppui](log_z + self.llr) @ self.p0 for b in b_grid]
        )
        gain = (g(log_z, self.base) - cost_cont) / b_grid
        return 1 + gain.argmax() / self.c_ppui

    def initialize(self, log_z_grid: npt.NDArray[np.float64], c_ppui: int) -> None:
        """
        Initialize NPKWT with given log_z and c grids. See `setup` method for more information.
        """
        self.log_z_grid = log_z_grid
        
        c_max_pos = np.ceil(self.b_star_ub(log_z_grid[-1])).astype(int)
        c_max_neg = np.ceil(self.b_star_ub(log_z_grid[0])).astype(int)
        self.c_max = np.maximum(c_max_pos, c_max_neg)
        self.c_ppui = c_ppui
        self.c_grid = np.arange(0, self.c_max, 1 / self.c_ppui)
        self.b_star_vectors = np.ones((self.k + 1, self.log_z_grid.size))

        self.cost_stop = g(self.log_z_grid, self.base)

        self.rho_vectors = np.empty((self.k + 1, self.c_grid.size, self.log_z_grid.size))
        for idx, c in enumerate(self.c_grid):
            if c < 1:
                self.rho_vectors[0][idx] = self.cost_stop
            else:
                self.rho_vectors[0][idx] = (
                    self.rho_splines[0][idx - self.c_ppui](self.log_z_grid[:, None] + self.llr[None, :]) @ self.p0
                )
            self.rho_splines[0].append(get_rho_spline(self.log_z_grid, self.rho_vectors[0][idx]))

    def setup(self, log_z_grid: npt.NDArray[np.float64], c_ppui: int, verbose=True) -> None:
        """
        Solve NPKWT on given z and c grids.

        Parameters
        ----------
        log_z_grid: z grid on which rho_k is evaluated
        c_ppui    : "point per unit interval" used to construct the c grid.
                    For example, c_ppui = 10 results in a grid spacing of 0.1.
        """
        self.initialize(log_z_grid, c_ppui)
        for k in range(self.k):
            if verbose:
                print(f"Solving k = {k + 1}")
            self.iterate_rho(k + 1)
        self.initialized = True

    def rho0_approx(self, log_z: float | npt.NDArray[np.float64], c: float) -> float | npt.NDArray[np.float64]:
        """
        Normal approximation of rho_0(log_z, c).
        """
        return norm.cdf((log_z + c * self.mu0) / np.sqrt(self.var0 * c)) + (float(self.base) ** log_z) * norm.cdf(
            -(log_z + c * self.mu1) / np.sqrt(self.var1 * c)
        )

    def iterate_rho(self, k: int) -> None:
        """
        Calculate rho_k from rho_{k + 1} on given log_z and c grid.
        """
        self.b_star_vectors[k] = self.get_b_star_vec(k)
        self.rho_vectors[k] = np.empty_like(self.rho_vectors[0])
        self.rho_splines[k] = []

        self.log_z_min[k] = self.log_z_grid[
            np.flatnonzero(self.log_z_grid <= 0)[np.argmax(self.b_star_vectors[k][self.log_z_grid <= 0])]
        ]
        self.log_z_max[k] = self.log_z_grid[
            np.flatnonzero(self.log_z_grid >= 0)[np.argmax(self.b_star_vectors[k][self.log_z_grid >= 0])]
        ]

        for c_idx, c in enumerate(self.c_grid):
            b_opt = np.maximum(c, self.b_star_vectors[k])
            c_over_b = c / b_opt
            cost_cont_rand = np.array(
                [
                    self.rho_splines[k - 1][self.c_idx(b) - self.c_ppui](log_z + self.llr) @ self.p0
                    for log_z, b in zip(self.log_z_grid, b_opt)
                ]
            )
            cost_rand = (1 - c_over_b) * self.cost_stop + c_over_b * cost_cont_rand
            if c == 0:
                self.rho_vectors[k][c_idx] = self.cost_stop
            elif c < 1:
                self.rho_vectors[k][c_idx] = cost_rand
            else:
                cost_detm = (
                    self.rho_splines[k][c_idx - self.c_ppui](self.log_z_grid[:, None] + self.llr[None, :]) @ self.p0
                )
                self.rho_vectors[k][c_idx] = np.minimum(cost_detm, cost_rand)
            self.rho_splines[k].append(get_rho_spline(self.log_z_grid, self.rho_vectors[k][c_idx]))

    def b_star_ub(self, log_z : float) -> int:
        """
        Approximate upper bound on b_k^*(z).
        """
        res = minimize_scalar(lambda b: (self.rho0_approx(log_z, b) - g(log_z, self.base)) / b, bounds=(1, 1e4))
        return res.x

    def b_star_lb(self, log_z : float) -> int:
        """
        Approximate lower bound on b_k^*(z).
        """
        return np.max([- log_z / self.mu1 - self.var1 / self.mu1 ** 2, 1, - log_z / self.mu0 - self.var0 / self.mu0 ** 2])

    def c_idx(self, b: float | int) -> int:
        """
        Map real-valued c to closest point in grid.
        """
        return int(np.round(self.c_ppui * b))

    def get_b_star_vec(self, k: int) -> npt.NDArray[np.float64]:
        """
        Evaluate b_k^* on log_z grid.
        """
        b_grid = self.c_grid[self.c_ppui :]
        one_over_b = 1 / b_grid
        cost_cont_rand = np.array(
            [
                self.rho_splines[k - 1][self.c_idx(b) - self.c_ppui](self.log_z_grid[:, None] + self.llr[None, :])
                @ self.p0
                for b in b_grid
            ]
        )
        cost = (1 - one_over_b) * self.cost_stop[:, None] + one_over_b * cost_cont_rand.T
        return 1 + cost.argmin(axis=1) / self.c_ppui

    def b_star_extrapolated(self, k: int, log_z: float) -> int:
        """
        Use linear extrapolation to approximate b_k^* outside the given log-z grid.
        """
        if np.all(self.b_linear_pos[k] == 0):
            mask = np.logical_and(self.log_z_grid <= self.log_z_max[k], self.log_z_grid >= 0)
            log_z_pos = self.log_z_grid[mask]
            log_z_pos = log_z_pos[log_z_pos.size // 10 : -log_z_pos.size // 10]
            b_star_pos = self.b_star_vectors[k][mask]
            b_star_pos = b_star_pos[b_star_pos.size // 10 : -b_star_pos.size // 10]
            self.b_linear_pos[k] = np.polyfit(log_z_pos, b_star_pos, 1)
        if np.all(self.b_linear_neg[k] == 0):
            mask = np.logical_and(self.log_z_grid >= self.log_z_min[k], self.log_z_grid <= 0)
            log_z_neg = self.log_z_grid[mask]
            log_z_neg = log_z_neg[log_z_neg.size // 10 : -log_z_neg.size // 10]
            b_star_neg = self.b_star_vectors[k][mask]
            b_star_neg = b_star_neg[b_star_neg.size // 10 : -b_star_neg.size // 10]
            self.b_linear_neg[k] = np.polyfit(log_z_neg, b_star_neg, 1)

        if log_z > 0:
            return np.maximum(1, np.round(self.b_linear_pos[k, 0] * log_z + self.b_linear_pos[k, 1]))
        if log_z < 0:
            return np.maximum(1, np.round(self.b_linear_neg[k, 0] * log_z + self.b_linear_neg[k, 1]))
        return 1


class NPKWTDiscrete:
    """
    Nonparametric Kiefer-Weiss Test (NPKWT) between two discrete distributions.
    """

    def __init__(self, p0: npt.NDArray[np.float64], p1: npt.NDArray[np.float64], log_base: float = np.e):
        """
        Parameters
        ----------
        p0      : PMF of distribution under null hypothesis
        p1      : PMF of distribution under alternative hypothesis
        log_base: base used for log-likelihood ratio
        """
        self.p0 = p0
        self.p1 = p1
        self.base = log_base
        self.llr = np.emath.logn(self.base, p1 / p0)

        self.mu0 = p0 @ self.llr
        self.mu1 = p1 @ self.llr
        self.var0 = p0 @ (self.llr - self.mu0) ** 2
        self.var1 = p1 @ (self.llr - self.mu1) ** 2

        self.log_z_grid = np.empty(0, dtype=float)
        self.log_z_min = 0
        self.log_z_max = 0

        self.n_max = 0

        self.cost_stop = np.empty(0)
        self.rho_vectors = np.empty((0, 0))
        self.rho_splines: list[RhoSpline] = []

        self.m_star_vector = np.empty((0))
        self.m_linear_pos = np.zeros(2)
        self.m_linear_neg = np.zeros(2)

        self.initialized = False

    def run(
        self,
        log_z: float,
        c: float | int,
        p: npt.NDArray[np.float64],
        out_of_range: Literal["extrapolate", "stop"] = "extrapolate",
    ) -> tuple[int, int, int]:
        """
        Run an NPKWT between p0 and p1.

        Parameters
        ----------
        log_z       : Log of cost function weight
        c           : Maximum expected sample size
        p           : PMF of true distribution
        out_of_range: Behavior when log_z falls outside the provided grid
                      extrapolate: Use linear approximation of m_start
                      stop: Stop test

        Returns
        -------
        d     : Accepted hypothesis [0,1]
        t     : Number of samples
        n_rand: Number of randomization uses
        """
        if not self.initialized:
            print("Test uninitialized, call `setup` first.")
            return -1, 0, 0

        n_rand, t = 0, 0
        while True:
            if self.log_z_min <= log_z <= self.log_z_max:
                m_opt = np.maximum(c, self.m_star(log_z))
            else:
                if out_of_range == "stop":
                    break
                elif out_of_range == "extrapolate":
                    m_opt = np.maximum(c, self.m_star_extrapolated(log_z))
                else:
                    raise ValueError(f"Unknown out_of_range value: {out_of_range!r}")

            if m_opt > c:
                n_rand += 1
                if bernoulli.rvs(1 - c / m_opt):
                    break

            log_z += (self.llr @ multinomial.rvs(1, p, size=())).item()
            c = m_opt - 1
            t += 1

        d = int(bernoulli.rvs(0.5)) if log_z == 0 else int(log_z > 0)
        return d, t, n_rand

    def simulate(
        self,
        log_z: float,
        c: float | int,
        p: npt.NDArray[np.float64],
        runs: int,
        out_of_range: Literal["extrapolate", "stop"] = "extrapolate",
    ) -> npt.NDArray[np.integer]:
        """
        Simulate `runs` number of NPKWTs with given parameters.
        See `run` method for parameter details.
        """
        outcomes = [self.run(log_z, c, p, out_of_range) for _ in tqdm(range(runs))]
        return np.array(list(zip(*outcomes)))

    def rho(self, log_z_grid: npt.ArrayLike, c_grid: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """
        Evaluate rho(log_z, c).
        """
        log_z_grid = np.atleast_1d(log_z_grid)
        c_grid = np.atleast_1d(c_grid)
        
        rho_vals = np.zeros((c_grid.size, log_z_grid.size))
        for row, c in enumerate(c_grid):
            if not 0 <= c <= self.n_max:
                print(f"rho(•, c) only defined for 0 <= c <= {self.n_max}")
                rho_vals[row] = log_z_grid.size * [None]
            else:
                idx_left, idx_right, weight_left, weight_right = linear_interpolation(c, np.arange(self.n_max + 1))
                rho_left = self.rho_splines[idx_left](log_z_grid)
                rho_right = self.rho_splines[idx_right](log_z_grid)
                rho_vals[row] = weight_left * rho_left + weight_right * rho_right
        return np.squeeze(rho_vals)

    def m_star(self, log_z: float) -> np.signedinteger:
        """
        Evaluate m^*(log_z).
        """
        m_grid = np.arange(1, self.n_max + 1)
        cost_cont = np.array([self.rho_splines[m - 1](log_z + self.llr) @ self.p0 for m in m_grid])
        gain = (g(log_z, self.base) - cost_cont) / m_grid
        return 1 + gain.argmax()

    def initialize(self, log_z_grid: npt.NDArray[np.float64]) -> None:
        """
        Initialize NPKWT with given log_z and c grids. See `setup` method for more information.
        """
        self.log_z_grid = log_z_grid

        n_max_pos = np.ceil(self.m_star_ub(self.log_z_grid[-1])).astype(int)
        n_max_neg = np.ceil(self.m_star_ub(self.log_z_grid[0])).astype(int)
        self.n_max = np.maximum(n_max_pos, n_max_neg)
        self.m_star_vector = np.ones(self.log_z_grid.size)

        self.cost_stop = g(self.log_z_grid, self.base)

        self.rho_vectors = np.empty((self.n_max + 1, self.log_z_grid.size))
        for n in range(self.n_max + 1):
            if n == 0:
                self.rho_vectors[n] = self.cost_stop
            else:
                self.rho_vectors[n] = self.rho_splines[n - 1](self.log_z_grid[:, None] + self.llr[None, :]) @ self.p0
            self.rho_splines.append(get_rho_spline(self.log_z_grid, self.rho_vectors[n]))
        self.m_star_vector = self.get_m_star_vec()

    def setup(self, log_z_grid: npt.NDArray[np.float64], tol: float = 1e-6, verbose=True) -> None:
        """
        Solve NPKWT on given z and c grids.

        Parameters
        ----------
        log_z_grid: z grid on which rho_k is evaluated
        tol       : Tolerance for fixed-point iteration. Convergence is reached when max norm of
                    difference drops below tol.
        """
        self.initialize(log_z_grid)
        n_it, diff = 0, np.inf
        while diff > tol:
            diff = self.iterate_rho()
            n_it += 1
            if verbose:
                print(f"Iteration {n_it}: diff = {diff}")
        self.initialized = True

    def rho0_approx(self, log_z: float | npt.NDArray[np.float64], c: float | int) -> float | npt.NDArray[np.float64]:
        """
        Normal approximation of rho_0(log_z, c).
        """
        return norm.cdf((log_z + c * self.mu0) / np.sqrt(self.var0 * c)) + (float(self.base) ** log_z) * norm.cdf(
            -(log_z + c * self.mu1) / np.sqrt(self.var1 * c)
        )

    def iterate_rho(self) -> float:
        """
        Calculate rho_k from rho_{k + 1} on given log_z and c grid.
        """
        rho_vectors_new = np.empty_like(self.rho_vectors)
        rho_splines_new: list[RhoSpline] = []

        self.log_z_min = self.log_z_grid[
            np.flatnonzero(self.log_z_grid <= 0)[np.argmax(self.m_star_vector[self.log_z_grid <= 0])]
        ].item()
        self.log_z_max = self.log_z_grid[
            np.flatnonzero(self.log_z_grid >= 0)[np.argmax(self.m_star_vector[self.log_z_grid >= 0])]
        ].item()

        for n in range(self.n_max + 1):
            if n == 0:
                rho_vectors_new[n] = self.cost_stop
            else:
                m_opt = np.maximum(n, self.m_star_vector)
                n_over_m = n / m_opt
                cost_cont_rand = np.array(
                    [self.rho_splines[m - 1](log_z + self.llr) @ self.p0 for log_z, m in zip(self.log_z_grid, m_opt)]
                )
                cost_rand = (1 - n_over_m) * self.cost_stop + n_over_m * cost_cont_rand
                cost_detm = rho_splines_new[n - 1](self.log_z_grid[:, None] + self.llr[None, :]) @ self.p0
                rho_vectors_new[n] = np.minimum(cost_detm, cost_rand)
            rho_splines_new.append(get_rho_spline(self.log_z_grid, rho_vectors_new[n]))

        diff = np.max(np.abs(self.rho_vectors - rho_vectors_new))

        self.rho_vectors = rho_vectors_new
        self.rho_splines = rho_splines_new
        self.m_star_vector = self.get_m_star_vec()

        return diff

    def m_star_ub(self, log_z : float) -> int:
        """
        Approximate upper bound on m^*(z).
        """
        res = minimize_scalar(lambda b: (self.rho0_approx(log_z, b) - g(log_z, self.base)) / b, bounds=(1, 1e4))
        return res.x

    def m_star_lb(self, log_z : float) -> int:
        """
        Approximate lower bound on m^*(z).
        """
        return np.max([- log_z / self.mu1 - self.var1 / self.mu1 ** 2, 1, - log_z / self.mu0 - self.var0 / self.mu0 ** 2])

    def get_m_star_vec(self):
        """
        Evaluate m^* on log_z grid.
        """
        m_grid = np.arange(1, self.n_max + 1)
        one_over_m = 1 / m_grid
        cost_cont_rand = np.array(
            [self.rho_splines[m - 1](self.log_z_grid[:, None] + self.llr[None, :]) @ self.p0 for m in m_grid]
        )
        cost = (1 - one_over_m) * self.cost_stop[:, None] + one_over_m * cost_cont_rand.T
        return 1 + cost.argmin(axis=1)

    def m_star_extrapolated(self, log_z: float) -> int:
        """
        Use linear extrapolation to approximate m^* outside the given log_z grid.
        """
        if np.all(self.m_linear_pos == 0):
            mask = np.logical_and(self.log_z_grid <= self.log_z_max, self.log_z_grid >= 0)
            log_z_pos = self.log_z_grid[mask]
            log_z_pos = log_z_pos[log_z_pos.size // 10 : -log_z_pos.size // 10]
            m_star_pos = self.m_star_vector[mask]
            m_star_pos = m_star_pos[m_star_pos.size // 10 : -m_star_pos.size // 10]
            self.m_linear_pos = np.polyfit(log_z_pos, m_star_pos, 1)
        if np.all(self.m_linear_neg == 0):
            mask = np.logical_and(self.log_z_grid >= self.log_z_min, self.log_z_grid <= 0)
            log_z_neg = self.log_z_grid[mask]
            log_z_neg = log_z_neg[log_z_neg.size // 10 : -log_z_neg.size // 10]
            m_star_neg = self.m_star_vector[mask]
            m_star_neg = m_star_neg[m_star_neg.size // 10 : -m_star_neg.size // 10]
            self.m_linear_neg = np.polyfit(log_z_neg, m_star_neg, 1)

        if log_z > 0:
            return np.maximum(1, np.round(self.m_linear_pos[0] * log_z + self.m_linear_pos[1]))
        if log_z < 0:
            return np.maximum(1, np.round(self.m_linear_neg[0] * log_z + self.m_linear_neg[1]))
        return 1


class NPKWTNormalLimited(NPKWTDiscreteLimited):
    """
    Nonparametric Kiefer-Weiss Test (NPKWT) for a shift in the mean of a
    standard normal distribution with limited number of randomization uses.

    The test is implemented via an NPKWT between two discrete distributions,
    where the latter are chosen such that they implement a Gauss-Hermite quadrature
    rule.
    """

    def __init__(self, k: int, mu: float, n_gh: int = 32):
        """
        Parameters
        ----------
        k   : Maximum number of randomization uses
        mu  : Mean of Gaussian under H1 (mu = 0 under H0)
        n_gh: Number of Gauss-Hermite quadrature nodes (default 32).
        """

        self.mu = mu

        gh_t, gh_w = np.polynomial.hermite.hermgauss(n_gh)
        self.gh_x = mu * (np.sqrt(2) * gh_t - mu / 2)
        self.gh_w = gh_w / np.sqrt(np.pi)

        # The GH weights gh_w form a valid discrete PMF (positive, sum to 1),
        # representing P0 = N(0,1). Under P1 = N(mu,1) the likelihood ratio at
        # node x_k is exp(x_k * mu - mu^2/2) = exp(gh_x_k), so
        # p1_k = p0_k * exp(gh_x_k) = gh_w_k * exp(gh_x_k) is also a valid PMF.
        super().__init__(k, self.gh_w, self.gh_w * np.exp(self.gh_x))

    def run(
        self,
        log_z: float,
        c: float | int,
        P: HasRvs,  # type: ignore[override]
        out_of_range: Literal["extrapolate", "stop"] = "extrapolate",
    ) -> tuple[int, int, int]:  
        """
        Run an NPKWT between p0 and p1.

        Parameters
        ----------
        log_z       : Log of cost function weight
        c           : Maximum expected sample size
        P           : True distribution
        out_of_range: Behavior when log_z falls outside the provided grid
                      extrapolate: Use linear approximation of m_start
                      stop: Stop test

        Returns
        -------
        d     : Accepted hypothesis [0,1]
        t     : Number of samples
        n_rand: Number of randomization uses
        """
        if not self.initialized:
            print("Test uninitialized, call `setup` first.")
            return -1, 0, 0

        k, t = self.k, 0
        while True:
            if k == 0:
                if c < 1:
                    break
                else:
                    b_opt = c
            else:
                if self.log_z_min[k] <= log_z <= self.log_z_max[k]:
                    b_opt = np.maximum(c, self.b_star(k, log_z))
                else:
                    if out_of_range == "stop":
                        break
                    elif out_of_range == "extrapolate":
                        b_opt = np.maximum(c, self.b_star_extrapolated(k, log_z))
                    else:
                        raise ValueError(f"Unknown out_of_range value: {out_of_range!r}")

            if b_opt > c:
                k -= 1
                if bernoulli.rvs(1 - c / b_opt):
                    break

            log_z += self.mu * (float(P.rvs()) - self.mu / 2)
            c = b_opt - 1
            t += 1

        d = int(bernoulli.rvs(0.5)) if log_z == 0 else int(log_z > 0)
        return d, t, self.k - k

    def simulate(
        self,
        log_z: float,
        c: float | int,
        P: HasRvs,  # type: ignore[override]
        runs: int,
        out_of_range: Literal["extrapolate", "stop"] = "extrapolate",
    ) -> npt.NDArray[np.integer]:
        """
        Simulate `runs` number of NPKWTs with given parameters.
        See `run` method for parameter details.
        """
        outcomes = [self.run(log_z, c, P, out_of_range) for _ in tqdm(range(runs))]
        return np.array(list(zip(*outcomes)))

    def initialize(self, log_z_grid: npt.NDArray[np.float64], c_ppui: int) -> None:
        """
        Initialize NPKWT with given log_z and c grids. See `setup` method for more information.
        """
        self.log_z_grid = log_z_grid
        self.c_ppui = c_ppui
        c_max_pos = np.ceil(self.b_star_ub(log_z_grid[-1])).astype(int)
        c_max_neg = np.ceil(self.b_star_ub(log_z_grid[0])).astype(int)
        self.c_max = np.maximum(c_max_pos, c_max_neg)
        self.c_grid = np.arange(0, self.c_max, 1 / self.c_ppui)
        self.b_star_vectors = np.ones((self.k + 1, self.log_z_grid.size))

        self.cost_stop = g(self.log_z_grid, self.base)

        self.rho_vectors = np.empty((self.k + 1, self.c_grid.size, self.log_z_grid.size))
        for idx, c in enumerate(self.c_grid):
            if c < 1:
                self.rho_vectors[0][idx] = self.cost_stop
            else:
                self.rho_vectors[0][idx] = self.rho0_approx(self.log_z_grid, np.floor(c))
            self.rho_splines[0].append(get_rho_spline(self.log_z_grid, self.rho_vectors[0][idx]))


class NPKWTNormal(NPKWTDiscrete):
    """
    Nonparametric Kiefer-Weiss Test (NPKWT) for a shift in the mean of a
    standard normal distribution.

    The test is implemented via an NPKWT between two discrete distributions,
    where the latter are chosen such that they implement a Gauss-Hermite quadrature
    rule.
    """

    def __init__(self, mu: float, n_gh: int = 32):

        self.mu = mu

        gh_t, gh_w = np.polynomial.hermite.hermgauss(n_gh)
        self.gh_x = mu * (np.sqrt(2) * gh_t - mu / 2)
        self.gh_w = gh_w / np.sqrt(np.pi)

        # See NPKWTNormalLimited for the derivation of p0 and p1.
        super().__init__(self.gh_w, self.gh_w * np.exp(self.gh_x))

        self.mu0 = - mu ** 2 / 2
        self.mu1 = mu ** 2 / 2
        self.var0 = mu ** 2
        self.var1 = mu ** 2

    def run(
        self,
        log_z: float,
        c: float | int,
        P: HasRvs,  # type: ignore[override]
        out_of_range: Literal["extrapolate", "stop"] = "extrapolate",
    ) -> tuple[int, int, int]:
        """
        Run an NPKWT between p0 and p1.

        Parameters
        ----------
        log_z       : Log of cost function weight
        c           : Maximum expected sample size
        p           : True distribution
        out_of_range: Behavior when log_z falls outside the provided grid
                      extrapolate: Use linear approximation of m_start
                      stop: Stop test

        Returns
        -------
        d     : Accepted hypothesis [0,1]
        t     : Number of samples
        n_rand: Number of randomization uses
        """
        if not self.initialized:
            print("Test uninitialized, call `setup` first.")
            return -1, 0, 0

        n_rand, t = 0, 0
        while True:
            if self.log_z_min <= log_z <= self.log_z_max:
                m_opt = np.maximum(c, self.m_star(log_z))
            else:
                if out_of_range == "stop":
                    break
                elif out_of_range == "extrapolate":
                    m_opt = np.maximum(c, self.m_star_extrapolated(log_z))
                else:
                    raise ValueError(f"Unknown out_of_range value: {out_of_range!r}")

            if m_opt > c:
                n_rand += 1
                if bernoulli.rvs(1 - c / m_opt):
                    break

            log_z += self.mu * (float(P.rvs()) - self.mu / 2)
            c = m_opt - 1
            t += 1

        d = int(bernoulli.rvs(0.5)) if log_z == 0 else int(log_z > 0)
        return d, t, n_rand

    def simulate(
        self,
        log_z: float,
        c: float | int,
        P: HasRvs,  # type: ignore[override]
        runs: int,
        out_of_range: Literal["extrapolate", "stop"] = "extrapolate",
    ) -> npt.NDArray[np.integer]:
        """
        Simulate `runs` number of NPKWTs with given parameters.
        See `run` method for parameter details.
        """
        outcomes = [self.run(log_z, c, P, out_of_range) for _ in tqdm(range(runs))]
        return np.array(list(zip(*outcomes)))

    def initialize(self, log_z_grid: npt.NDArray[np.float64]) -> None:
        """
        Initialize NPKWT with given log_z and c grids. See `setup` method for more information.
        """
        self.log_z_grid = log_z_grid
        n_max_pos = np.ceil(self.m_star_ub(log_z_grid[-1])).astype(int)
        n_max_neg = np.ceil(self.m_star_ub(log_z_grid[0])).astype(int)
        self.n_max = np.maximum(n_max_pos, n_max_neg)
        self.m_star_vector = np.ones(self.log_z_grid.size)

        self.cost_stop = g(self.log_z_grid, self.base)

        self.rho_vectors = np.empty((self.n_max + 1, self.log_z_grid.size))
        for n in range(self.n_max + 1):
            if n == 0:
                self.rho_vectors[n] = self.cost_stop
            else:
                self.rho_vectors[n] = self.rho0_approx(self.log_z_grid, n)
            self.rho_splines.append(get_rho_spline(self.log_z_grid, self.rho_vectors[n]))
        self.m_star_vector = self.get_m_star_vec()

