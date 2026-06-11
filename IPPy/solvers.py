import math
import time
import warnings

try:
    import numba as nb
except ImportError:
    class _NumbaShim:
        @staticmethod
        def njit(*args, **kwargs):
            if args and callable(args[0]) and len(args) == 1 and not kwargs:
                return args[0]

            def decorator(func):
                return func

            return decorator

    nb = _NumbaShim()
    warnings.warn(
        "numba is not installed. IPPy solvers will run without JIT acceleration.",
        RuntimeWarning,
    )

import numpy as np
import torch

from . import operators
from .utilities import metrics


##################################
# Multibatch decorator
##################################
def on_batch(call_method):
    def wrapper(self, input_tensor, x_true, starting_point, *args, **kwargs):
        if input_tensor.shape[0] > 1:  # Check if batch size N > 1
            outputs = []
            for i in range(input_tensor.shape[0]):
                if x_true is None:
                    x_true = [None] * input_tensor.shape[0]

                if starting_point is None:
                    starting_point = [None] * input_tensor.shape[0]

                single_output, single_info = call_method(
                    self,
                    input_tensor[i : i + 1],
                    x_true=x_true[i : i + 1],
                    starting_point=starting_point[i : i + 1],
                    *args,
                    **kwargs,
                )

                outputs.append(single_output)

                # Handle infos
                if i == 0:
                    info = single_info
                    info["iterations"] = (info["iterations"],)
                else:
                    for k in info.keys():
                        if isinstance(info[k], torch.Tensor):
                            info[k] = torch.cat((info[k], single_info[k]), dim=1)
                        elif isinstance(info[k], tuple):
                            info[k] = info[k] + (single_info[k],)

            return torch.cat(outputs, dim=0), info
        else:
            return call_method(
                self,
                input_tensor,
                x_true=x_true,
                starting_point=starting_point,
                *args,
                **kwargs,
            )

    return wrapper


##################################
# DIRECT SOLVERS
##################################
class FBP:
    def __init__(self, K: operators.Operator):
        self.K = K

        self.nx, self.ny = K.nx, K.ny
        self.mx, self.my = K.mx, K.my

    @on_batch
    def __call__(
        self,
        y_delta: torch.Tensor,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        return self.K.FBP(y_delta), {"iterations": 1}


class Identity:
    def __init__(self, K: operators.Operator):
        self.K = K

        self.nx, self.ny = K.nx, K.ny
        self.mx, self.my = K.mx, K.my

    @on_batch
    def __call__(
        self,
        y_delta: torch.Tensor,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        return y_delta, {"iterations": 1}


##################################
# CP-TpV Constrained
##################################
class ChambollePockTpVConstrained:
    def __init__(self, K: operators.Operator):
        self.K = K
        self.nx, self.ny = K.nx, K.ny
        self.mx, self.my = K.mx, K.my

        # Initialize Gradient operator
        self.grad = operators.Gradient((self.nx, self.ny))

    @on_batch
    def __call__(
        self,
        y_delta: torch.Tensor,
        epsilon: float,
        lmbda: float,
        x_true: torch.Tensor | None = None,
        starting_point: torch.Tensor | None = None,
        eta: float = 2e-3,
        maxiter: int = 100,
        p: int = 1,
        verbose: bool = False,
        *args,
        **kwargs,
    ):
        # Compute the approximation to || K ||_2
        nu = math.sqrt(
            self.power_method(self.K, num_iterations=10)
            / self.power_method(self.grad, num_iterations=10)
        )

        Gamma = math.sqrt(
            self.power_method_dual_operator(self.K, self.grad, num_iterations=10)
        )

        # Compute the parameters given Gamma
        tau = 1 / Gamma
        sigma = 1 / Gamma
        theta = 1

        # Iteration counter
        k = 0

        # Initialization
        if starting_point is None or starting_point == [None]:
            x = torch.zeros((1, 1, self.nx, self.ny))
        else:
            x = starting_point
        y = torch.zeros((1, 1, self.mx, self.my))
        w = torch.zeros((1, 2, self.nx, self.ny))

        xx = x

        # Initialize infos
        info = dict()
        info["residues"] = torch.zeros((maxiter + 1, 1))
        info["obj"] = torch.zeros((maxiter + 1, 1))
        info["RE"] = torch.zeros((maxiter + 1, 1))
        info["RMSE"] = torch.zeros((maxiter + 1, 1))
        info["PSNR"] = torch.zeros((maxiter + 1, 1))
        info["SSIM"] = torch.zeros((maxiter + 1, 1))
        info["iterations"] = 0

        # Stopping conditions
        start_time = time.time()
        con = True
        while con and (k < maxiter):
            # Update y
            yy = y + sigma * (self.K(xx) - y_delta)
            y = max(torch.norm(yy) - (sigma * epsilon), 0) * yy / torch.norm(yy)

            # Compute the magnitude of the gradient
            grad_x = self.grad(xx)
            grad_mag = torch.square(grad_x[:, 0:1]) + torch.square(grad_x[:, 1:2])

            # Compute the reweighting factor
            W = (torch.sqrt(eta**2 + grad_mag) / eta) ** (p - 1)
            WW = torch.cat((W, W), dim=1)

            # Update w
            x_grad = self.grad(xx)
            ww = w + sigma * x_grad

            abs_ww = torch.square(ww[:, 0:1]) + torch.square(ww[:, 1:2])
            abs_ww = torch.cat((abs_ww, abs_ww), dim=1)

            lmbda_vec_over_nu = lmbda * WW / nu
            w = lmbda_vec_over_nu * ww / torch.maximum(lmbda_vec_over_nu, abs_ww)

            # Save the value of x
            xtmp = x

            # Update x
            x = xtmp - tau * (self.K.T(y) + nu * self.grad.T(w))

            # Project x to (x>0)
            x[x < 0] = 0

            # Acceleration step
            xx = x + theta * (x - xtmp)

            # Compute relative error
            if x_true is not None:
                info["RE"][k] = metrics.RE(x, x_true)
                info["PSNR"][k] = metrics.PSNR(x, x_true)
                info["RMSE"][k] = metrics.RMSE(x, x_true)
                info["SSIM"][k] = metrics.SSIM(x, x_true)

            # Compute the magnitude of the gradient of the actual iterate
            grad_x = self.grad(x)
            grad_mag = torch.sqrt(
                torch.square(grad_x[:, 0:1]) + torch.square(grad_x[:, 1:2])
            )

            # Compute the value of TpV by reweighting
            ftpv = torch.sum(torch.abs(W * grad_mag))
            res = torch.norm(self.K(x) - y_delta, 2) ** 2

            # Save the values into info
            info["residues"][k] = res
            info["obj"][k] = 0.5 * res + lmbda * ftpv

            # Stopping criteria
            c = math.sqrt(res) / (torch.max(y_delta) * math.sqrt(self.mx * self.my))
            d_abs = torch.norm(x.flatten() - xtmp.flatten())

            if (c >= 9e-6) and (c <= 1.1e-5):
                con = False

            if d_abs < 1e-3 * (1 + torch.norm(xtmp.flatten())):
                con = False

            # Update k
            k = k + 1
            if verbose:
                # Measure time
                total_time = time.time() - start_time

                # Convert elapsed time to hours, minutes, and seconds
                hours, rem = divmod(total_time, 3600)
                minutes, seconds = divmod(rem, 60)

                # Format using an f-string with %H:%M:%S style
                formatted_time = f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"

                print(
                    f"({formatted_time}) Iteration {k}/{maxiter} -> RE: {info['RE'][k-1, 0]:0.4f}, SSIM: {info['SSIM'][k-1, 0]:0.4f}."
                )

        # Save number of iterations in info and truncate
        info["residues"] = info["residues"][:k]
        info["obj"] = info["obj"][:k]
        info["RE"] = info["RE"][:k]
        info["RMSE"] = info["RMSE"][:k]
        info["PSNR"] = info["PSNR"][:k]
        info["SSIM"] = info["SSIM"][:k]
        info["iterations"] = k
        return x, info

    def power_method(self, K, num_iterations: int):
        b_k = torch.rand((1, 1, K.nx, K.ny))

        for _ in range(num_iterations):
            # calculate the matrix-by-vector product Ab
            b_k1 = K.T(K(b_k))

            # calculate the norm
            b_k1_norm = torch.norm(b_k1.flatten())

            # re normalize the vector
            b_k = b_k1 / b_k1_norm

        return b_k1_norm

    def power_method_dual_operator(self, K, D, num_iterations: int):
        b_k = torch.rand((1, 1, K.nx, K.ny))

        for _ in range(num_iterations):
            # calculate the matrix-by-vector product Ab
            b_k1 = K.T(K(b_k)) + D.T(D(b_k))

            # calculate the norm
            b_k1_norm = torch.norm(b_k1.flatten())

            # re normalize the vector
            b_k = b_k1 / b_k1_norm

        return b_k1_norm

    def compute_obj_value(self, x, y, lmbda, p, eta):
        # Compute the value of the objective function TpV by reweighting
        grad_x = self.grad(x)  # (1, 2, nx, ny)
        grad_mag = torch.square(grad_x[:, 0:1]) + torch.square(grad_x[:, 1:2])
        W = (torch.sqrt(eta**2 + grad_mag) / eta) ** (p - 1)

        ftpv = torch.sum(torch.abs(W * torch.sqrt(grad_mag)))
        return 0.5 * torch.norm(self.K(x) - y, 2) ** 2 + lmbda * ftpv


##################################
# CP-TpV Unconstrained
##################################
class ChambollePockTpVUnconstrained:
    def __init__(self, K: operators.Operator):
        self.K = K
        self.nx, self.ny = K.nx, K.ny
        self.mx, self.my = K.mx, K.my

        # Initializ Gradient operator
        self.grad = operators.Gradient((self.nx, self.ny))

    @on_batch
    def __call__(
        self,
        y_delta: torch.Tensor,
        lmbda: float,
        x_true: torch.Tensor | None = None,
        starting_point: torch.Tensor | None = None,
        eta: float = 2e-3,
        maxiter: int = 100,
        p: int = 1,
        verbose: bool = False,
        *args,
        **kwargs,
    ):
        """
        Chambolle-Pock algorithm for the minimization of the objective function
            ||K*x - d||_2^2 + Lambda*TpV(x)
        by reweighting

        K : projection operator
        Lambda : weight of the TV penalization (the higher Lambda, the more sparse is the solution)
        L : norm of the operator [P, Lambda*grad] (see power_method)
        maxit : number of iterations
        """
        # Compute the approximation to || K ||_2
        nu = math.sqrt(
            self.power_method(self.K, num_iterations=10)
            / self.power_method(self.grad, num_iterations=10)
        )

        Gamma = math.sqrt(
            self.power_method_dual_operator(self.K, self.grad, num_iterations=10)
        )

        # Compute the parameters given Gamma
        tau = 1 / Gamma
        sigma = 1 / Gamma
        theta = 1

        # Iteration counter
        k = 0

        # Initialization
        if starting_point is None or starting_point == [None]:
            x = torch.zeros((1, 1, self.nx, self.ny))
        else:
            x = starting_point
        y = torch.zeros((1, 1, self.mx, self.my))
        w = torch.zeros((1, 2, self.nx, self.ny))

        xx = x

        # Initialize infos
        info = dict()
        info["residues"] = torch.zeros((maxiter + 1, 1))
        info["obj"] = torch.zeros((maxiter + 1, 1))
        info["RE"] = torch.zeros((maxiter + 1, 1))
        info["RMSE"] = torch.zeros((maxiter + 1, 1))
        info["PSNR"] = torch.zeros((maxiter + 1, 1))
        info["SSIM"] = torch.zeros((maxiter + 1, 1))
        info["iterations"] = 0

        # Stopping conditions
        start_time = time.time()
        con = True
        while con and (k < maxiter):
            # Update y
            y = (y + sigma * (self.K(xx) - y_delta)) / (1 + lmbda * sigma)

            # Compute the magnitude of the gradient
            grad_x = self.grad(xx)
            grad_mag = torch.square(grad_x[:, 0:1]) + torch.square(grad_x[:, 1:2])

            # Compute the reweighting factor
            W = (torch.sqrt(eta**2 + grad_mag) / eta) ** (p - 1)
            WW = torch.cat((W, W), dim=1)

            # Update w
            x_grad = self.grad(xx)
            ww = w + sigma * x_grad

            abs_ww = torch.square(ww[:, 0:1]) + torch.square(ww[:, 1:2])
            abs_ww = torch.cat((abs_ww, abs_ww), dim=1)

            lmbda_vec_over_nu = lmbda * WW / nu
            w = lmbda_vec_over_nu * ww / torch.maximum(lmbda_vec_over_nu, abs_ww)

            # Save the value of x
            xtmp = x

            # Update x
            x = xtmp - tau * (self.K.T(y) + nu * self.grad.T(w))

            # Project x to (x>0)
            x[x < 0] = 0

            # Acceleration step
            xx = x + theta * (x - xtmp)

            # Compute relative error
            if x_true is not None:
                info["RE"][k] = metrics.RE(x, x_true)
                info["PSNR"][k] = metrics.PSNR(x, x_true)
                info["RMSE"][k] = metrics.RMSE(x, x_true)
                info["SSIM"][k] = metrics.SSIM(x, x_true)

            # Compute the magnitude of the gradient of the actual iterate
            grad_x = self.grad(x)
            grad_mag = torch.sqrt(
                torch.square(grad_x[:, 0:1]) + torch.square(grad_x[:, 1:2])
            )

            # Compute the value of TpV by reweighting
            ftpv = torch.sum(torch.abs(W * grad_mag))
            res = torch.norm(self.K(x) - y_delta, 2) ** 2

            # Save the values into info
            info["residues"][k] = res
            info["obj"][k] = 0.5 * res + lmbda * ftpv

            # Stopping criteria
            c = math.sqrt(res) / (torch.max(y_delta) * math.sqrt(self.mx * self.my))
            d_abs = torch.norm(x.flatten() - xtmp.flatten())

            if (c >= 9e-6) and (c <= 1.1e-5):
                con = False

            if d_abs < 1e-3 * (1 + torch.norm(xtmp.flatten())):
                con = False

            # Update k
            k = k + 1
            if verbose:
                # Measure time
                total_time = time.time() - start_time

                # Convert elapsed time to hours, minutes, and seconds
                hours, rem = divmod(total_time, 3600)
                minutes, seconds = divmod(rem, 60)

                # Format using an f-string with %H:%M:%S style
                formatted_time = f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"

                print(
                    f"({formatted_time}) Iteration {k}/{maxiter} -> RE: {info['RE'][k-1, 0]:0.4f}, SSIM: {info['SSIM'][k-1, 0]:0.4f}."
                )

        # Save number of iterations in info and truncate
        info["residues"] = info["residues"][:k]
        info["obj"] = info["obj"][:k]
        info["RE"] = info["RE"][:k]
        info["RMSE"] = info["RMSE"][:k]
        info["PSNR"] = info["PSNR"][:k]
        info["SSIM"] = info["SSIM"][:k]
        info["iterations"] = k
        return x, info

    def power_method(self, K, num_iterations: int):
        b_k = torch.rand((1, 1, K.nx, K.ny))

        for _ in range(num_iterations):
            # calculate the matrix-by-vector product Ab
            b_k1 = K.T(K(b_k))

            # calculate the norm
            b_k1_norm = torch.norm(b_k1.flatten())

            # re normalize the vector
            b_k = b_k1 / b_k1_norm

        return b_k1_norm

    def power_method_dual_operator(self, K, D, num_iterations: int):
        b_k = torch.rand((1, 1, K.nx, K.ny))

        for _ in range(num_iterations):
            # calculate the matrix-by-vector product Ab
            b_k1 = K.T(K(b_k)) + D.T(D(b_k))

            # calculate the norm
            b_k1_norm = torch.norm(b_k1.flatten())

            # re normalize the vector
            b_k = b_k1 / b_k1_norm

        return b_k1_norm

    def compute_obj_value(self, x, y, lmbda, p, eta):
        # Compute the value of the objective function TpV by reweighting
        grad_x = self.grad(x)  # (1, 2, nx, ny)
        grad_mag = torch.square(grad_x[:, 0:1]) + torch.square(grad_x[:, 1:2])
        W = (torch.sqrt(eta**2 + grad_mag) / eta) ** (p - 1)

        ftpv = torch.sum(torch.abs(W * torch.sqrt(grad_mag)))
        return 0.5 * torch.norm(self.K(x) - y, 2) ** 2 + lmbda * ftpv


##################################
# CGLS
##################################
class CGLS:
    def __init__(self, K: operators.Operator):
        self.K = K
        self.nx, self.ny = K.nx, K.ny
        self.mx, self.my = K.mx, K.my

    @on_batch
    def __call__(
        self,
        y_delta: torch.Tensor,
        x_true: torch.Tensor | None = None,
        starting_point: torch.Tensor | None = None,
        maxiter: int = 100,
        tolf: float = 1e-6,
        tolx: float = 1e-6,
        verbose: bool = False,
        *args,
        **kwargs,
    ):
        # Initialization
        if starting_point is None or starting_point == [None]:
            x = torch.zeros((1, 1, self.nx, self.ny))
        else:
            x = starting_point
        d = y_delta
        r0 = self.K.T(y_delta)
        p = r0
        t = self.K(p)
        r = r0
        k = 0

        # Initialize infos
        info = dict()
        info["residues"] = torch.zeros((maxiter + 1, 1))
        info["RE"] = torch.zeros((maxiter + 1, 1))
        info["RMSE"] = torch.zeros((maxiter + 1, 1))
        info["PSNR"] = torch.zeros((maxiter + 1, 1))
        info["SSIM"] = torch.zeros((maxiter + 1, 1))
        info["iterations"] = 0

        # Stopping condition
        start_time = time.time()
        con = True
        while con and (k < maxiter):
            x0 = x

            # Update cycle
            alpha = torch.norm(r0.flatten()) ** 2 / torch.norm(t.flatten()) ** 2
            x = x0 + alpha * p
            d = d - alpha * t
            r = self.K.T(d)
            beta = torch.norm(r.flatten()) ** 2 / torch.norm(r0.flatten()) ** 2
            p = r + beta * p
            t = self.K(p)
            r0 = r

            # Compute relative error
            if x_true is not None:
                info["RE"][k] = metrics.RE(x, x_true)
                info["PSNR"][k] = metrics.PSNR(x, x_true)
                info["RMSE"][k] = metrics.RMSE(x, x_true)
                info["SSIM"][k] = metrics.SSIM(x, x_true)

            # Save the values into info
            info["residues"][k] = torch.norm(r.flatten()) ** 2

            # Stopping criteria
            d_abs = torch.norm(x.flatten() - x0.flatten())

            if d_abs < tolx * (1 + torch.norm(x0.flatten())):
                con = False

            if torch.norm(r.flatten()) < tolf:
                con = False

            # Update k
            k = k + 1
            if verbose:
                # Measure time
                total_time = time.time() - start_time

                # Convert elapsed time to hours, minutes, and seconds
                hours, rem = divmod(total_time, 3600)
                minutes, seconds = divmod(rem, 60)

                # Format using an f-string with %H:%M:%S style
                formatted_time = f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"

                print(
                    f"({formatted_time}) Iteration {k}/{maxiter} -> RE: {info['RE'][k-1, 0]:0.4f}, SSIM: {info['SSIM'][k-1, 0]:0.4f}."
                )

        # Save number of iterations in info and truncate
        info["residues"] = info["residues"][:k]
        info["RE"] = info["RE"][:k]
        info["RMSE"] = info["RMSE"][:k]
        info["PSNR"] = info["PSNR"][:k]
        info["SSIM"] = info["SSIM"][:k]
        info["iterations"] = k
        return x, info


##################################
# SGP TV
##################################
class SGP:
    def __init__(self, K: operators.Operator):
        self.K = K
        self.nx, self.ny = K.nx, K.ny
        self.mx, self.my = K.mx, K.my

        # Initialize Gradient operator
        self.grad = operators.Gradient((self.nx, self.ny))

    @on_batch
    def __call__(
        self,
        y_delta: torch.Tensor,
        lmbda: float,
        x_true: torch.Tensor | None = None,
        starting_point: torch.Tensor | None = None,
        m_bb: int = 3,
        tau_bb: float = 0.5,
        tol_grad: float = 1e-6,
        tol_x: float = 1e-6,
        maxiter: int = 100,
        alpha: float = 1,
        verbose: bool = False,
        *args,
        **kwargs,
    ):

        # Initialization
        if starting_point is None or starting_point == [None]:
            x = torch.zeros((1, 1, self.nx, self.ny))
        else:
            x = starting_point
        self.y_delta = y_delta
        self.lmbda = lmbda

        # SGP additional parameters
        self.alpha = alpha
        self.m_bb = m_bb
        self.tau_bb = tau_bb
        self.alpha_bb2_vec = torch.tensor([self.alpha] * self.m_bb)

        u_ls = self.U_LS()
        v_ls = self.V_LS(x)
        u_tv = self.U_TV(x)
        v_tv = self.V_TV(x)

        grad = v_ls - u_ls + self.lmbda * (v_tv - u_tv)

        rho = math.sqrt(1 + 1e15)
        s = self.compute_scaling(x, v_ls, v_tv, rho)

        k = 0

        # Initialize infos
        info = dict()
        info["obj"] = torch.zeros((maxiter + 1, 1))
        info["grad_norm"] = torch.zeros((maxiter + 1, 1))
        info["RE"] = torch.zeros((maxiter + 1, 1))
        info["RMSE"] = torch.zeros((maxiter + 1, 1))
        info["PSNR"] = torch.zeros((maxiter + 1, 1))
        info["SSIM"] = torch.zeros((maxiter + 1, 1))
        info["iterations"] = 0

        # Assign initial values to info
        info["obj"][0] = self.f(x)
        info["grad_norm"][0] = torch.norm(grad.flatten())

        if x_true is not None:
            info["RE"][0] = metrics.RE(x, x_true)
            info["PSNR"][0] = metrics.PSNR(x, x_true)
            info["RMSE"][0] = metrics.RMSE(x, x_true)
            info["SSIM"][0] = metrics.SSIM(x, x_true)

        # Stopping conditions
        start_time = time.time()
        con = True
        while con and (k < maxiter):
            desc_direction = self.Proj_positive(x - self.alpha * s * grad) - x
            step_length = self.backtracking(x, self.f, grad, desc_direction)

            x0 = x

            # Update x
            x = x + step_length * desc_direction

            # Update info
            if x_true is not None:
                info["obj"][k] = self.f(x)
                info["grad_norm"][k] = torch.norm(grad.flatten())
                info["RE"][k] = metrics.RE(x, x_true)
                info["PSNR"][k] = metrics.PSNR(x, x_true)
                info["RMSE"][k] = metrics.RMSE(x, x_true)
                info["SSIM"][k] = metrics.SSIM(x, x_true)

            # Stopping criteria
            if info["grad_norm"][k] < tol_grad * info["grad_norm"][0]:
                con = False

            d_abs = torch.norm(x.flatten() - x0.flatten())
            if d_abs < tol_x:
                con = False

            # Update k
            k = k + 1

            # Setup for next step
            if con:
                # Update v_ls, u_tv, v_tv
                v_ls = self.V_LS(x)
                u_tv = self.U_TV(x)
                v_tv = self.V_TV(x)
                grad_0 = grad
                grad = v_ls - u_ls + self.lmbda * (v_tv - u_tv)

                rho = math.sqrt(1 + 1e15 / (k**2.1))
                s = self.compute_scaling(x, v_ls, v_tv, rho)
                self.alpha = self.bb(s, x0, x, grad_0, grad, self.alpha)

            if verbose:
                # Measure time
                total_time = time.time() - start_time

                # Convert elapsed time to hours, minutes, and seconds
                hours, rem = divmod(total_time, 3600)
                minutes, seconds = divmod(rem, 60)

                # Format using an f-string with %H:%M:%S style
                formatted_time = f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"

                print(
                    f"({formatted_time}) Iteration {k}/{maxiter} -> RE: {info['RE'][k-1, 0]:0.4f}, SSIM: {info['SSIM'][k-1, 0]:0.4f}."
                )

        # Save number of iterations in info and truncate
        info["obj"] = info["obj"][:k]
        info["grad_norm"] = info["grad_norm"][:k]
        info["RE"] = info["RE"][:k]
        info["RMSE"] = info["RMSE"][:k]
        info["PSNR"] = info["PSNR"][:k]
        info["SSIM"] = info["SSIM"][:k]
        info["iterations"] = k
        return x, info

    def f(self, x):
        res = self.K(x) - self.y_delta
        tv = self.TV(x)

        return 0.5 * torch.norm(res.flatten()) ** 2 + self.lmbda * tv

    def grad_f(self, x):  # TODO
        grad = self.K.T(self.K(x) - self.y_delta)
        return grad + self.lmbda * x

    def V_LS(self, x):
        return self.K.T(self.K(x))

    def U_LS(self):
        return self.K.T(self.y_delta)

    def phi(self, t, beta=1e-3):
        return 2 * torch.sqrt(t + beta**2)

    def dphi(self, t, beta=1e-3):
        return _dphi(t, beta)

    def delta(self, x):
        Dx = self.grad(x).numpy()
        return _delta(Dx)

    def V_TV(self, x, beta=1e-3):  # Zero padding
        return torch.tensor(_V_TV(self.delta(x), _dphi, x.numpy(), beta))

    def U_TV(self, x, beta=1e-3):  # Zero padding
        return torch.tensor(_U_TV(self.delta(x), _dphi, x.numpy(), beta))

    def TV(self, x, beta=1e-3):
        Dx = self.grad(x)

        return torch.sum(
            torch.sqrt(torch.square(Dx[:, 0:1]) + torch.square(Dx[:, 1:2]) + beta**2)
        )

    def compute_scaling(self, x, V_LS, V_TV, rho):
        return torch.tensor(
            _compute_scaling(self.lmbda, x.numpy(), V_LS.numpy(), V_TV.numpy(), rho)
        )

    def backtracking(self, x, f, grad, d):
        alpha = 1
        rho = 0.8
        c1 = 0.25

        fx = f(x)
        while f(x + alpha * d) > fx + alpha * c1 * torch.sum(grad * d):
            alpha = alpha * rho

        return alpha

    def bb(self, s, x0, x, grad_0, grad, alpha_old):
        alpha_min = 1e-10
        alpha_max = 1e5

        s_k = x - x0
        z_k = grad - grad_0

        Dz = z_k * s

        alpha_bb1_denom = torch.sum(s_k * z_k / s)
        alpha_bb2_num = torch.sum(s_k * Dz)

        if alpha_bb1_denom <= 0:
            alpha_bb1 = min(10 * alpha_old, alpha_max)
        else:
            alpha_bb1 = torch.norm((s_k / s).flatten()) ** 2 / alpha_bb1_denom
            alpha_bb1 = max(min(alpha_bb1, alpha_max), alpha_min)

        if alpha_bb2_num <= 0:
            alpha_bb2 = min(10 * alpha_old, alpha_max)
        else:
            alpha_bb2 = alpha_bb2_num / (torch.norm(Dz.flatten()) ** 2)
            alpha_bb2 = max(min(alpha_bb2, alpha_max), alpha_min)

        self.alpha_bb2_vec = torch.cat((self.alpha_bb2_vec, torch.tensor([alpha_bb2])))

        if alpha_bb2 / alpha_bb1 < self.tau_bb:
            alpha = torch.min(self.alpha_bb2_vec[-self.m_bb - 1 :])
            self.tau_bb = self.tau_bb * 0.9
        else:
            alpha = alpha_bb1
            self.tau_bb = self.tau_bb * 1.1

        return alpha

    def Proj_positive(self, x):
        return torch.tensor(_Proj_positive(x.numpy()))


"""
Jit wrappers
"""


@nb.njit()  # To avoid numba errors, the computation here is happens within numpy arrays
def _V_TV(D, dphi, x, beta=1e-3):  # Zero padding
    _, _, nx, ny = x.shape

    V_tv = np.zeros((1, 1, nx, ny), dtype=np.float32)

    V_tv[0, 0, 0, 0] = (
        2 * dphi(D[0, 0, 0, 0]) + dphi(x[0, 0, 0, 0]) + dphi(x[0, 0, 0, 0])
    ) * x[0, 0, 0, 0]
    for j in range(1, ny):
        V_tv[0, 0, 0, j] = (
            2 * dphi(D[0, 0, 0, j]) + dphi(x[0, 0, 0, j]) + dphi(D[0, 0, 0, j - 1])
        ) * x[0, 0, 0, j]
    for i in range(1, ny):
        V_tv[0, 0, i, 0] = (
            2 * dphi(D[0, 0, i, 0]) + dphi(D[0, 0, i - 1, 0]) + dphi(x[0, 0, i, 0])
        ) * x[0, 0, i, 0]

    for i in range(1, nx):
        for j in range(1, ny):
            V_tv[0, 0, i, j] = (
                2 * dphi(D[0, 0, i, j])
                + dphi(D[0, 0, i - 1, j])
                + dphi(D[0, 0, i, j - 1])
            ) * x[0, 0, i, j]

    return V_tv


@nb.njit(parallel=True)
def _U_TV(D, dphi, x, beta=1e-3):  # Zero padding
    _, _, nx, ny = x.shape

    U_tv = np.empty((1, 1, nx, ny), dtype=np.float32)

    U_tv[0, 0, 0, 0] = dphi(D[0, 0, 0, 0]) * (x[0, 0, 1, 0] + x[0, 0, 0, 1])
    for j in range(1, ny - 1):
        U_tv[0, 0, 0, j] = (
            dphi(D[0, 0, 0, j]) * (x[0, 0, 1, j] + x[0, 0, 0, j + 1])
            + dphi(D[0, 0, 0, j - 1]) * x[0, 0, 0, j - 1]
        )
    U_tv[0, 0, 0, ny - 1] = (
        dphi(D[0, 0, 0, ny - 1]) * x[0, 0, 1, ny - 1]
        + dphi(D[0, 0, 0, ny - 2]) * x[0, 0, 0, ny - 2]
    )
    for i in range(1, nx - 1):
        U_tv[0, 0, i, 0] = (
            dphi(D[0, 0, i, 0]) * (x[0, 0, i + 1, 0] + x[0, 0, i, 1])
            + dphi(D[0, 0, i - 1, 0]) * x[0, 0, i - 1, 0]
        )
    U_tv[0, 0, nx - 1, 0] = (
        dphi(D[0, 0, nx - 1, 0]) * x[0, 0, nx - 1, 1]
        + dphi(D[0, 0, nx - 2, 0]) * x[0, 0, nx - 2, 0]
    )

    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            U_tv[0, 0, i, j] = (
                dphi(D[0, 0, i, j]) * (x[0, 0, i + 1, j] + x[0, 0, i, j + 1])
                + dphi(D[0, 0, i - 1, j]) * x[0, 0, i - 1, j]
                + dphi(D[0, 0, i, j - 1]) * x[0, 0, i, j - 1]
            )

    U_tv[0, 0, 1:, ny - 1] = 0
    U_tv[0, 0, nx - 1, 1:] = 0

    return U_tv


@nb.njit()
def _compute_scaling(lmbda, x, V_LS, V_TV, rho):
    _, _, nx, ny = x.shape

    V = V_LS + lmbda * V_TV

    d = np.empty((1, 1, nx, ny), dtype=np.float32)
    for i in range(nx):
        for j in range(ny):
            if V[0, 0, i, j] < 1e-4:
                V[0, 0, i, j] = 1e-4

            if rho == 0:
                d[0, 0, i, j] = x[0, 0, i, j] / V[0, 0, i, j]
            else:
                d[0, 0, i, j] = min(rho, max(1 / rho, x[0, 0, i, j] / V[0, 0, i, j]))
    return d


@nb.njit()
def _Proj_positive(x):
    _, _, nx, ny = x.shape
    for i in range(nx):
        for j in range(ny):
            x[0, 0, i, j] = max(0, x[0, 0, i, j])
    return x


@nb.njit(fastmath=True)
def _dphi(t, beta=1e-3):
    return 1 / np.sqrt(t + beta**2)


@nb.njit(fastmath=True)
def _delta(Dx):
    D_h = Dx[:, 0:1]
    D_v = Dx[:, 1:2]
    return np.square(D_h) + np.square(D_v)
