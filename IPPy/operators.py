import numpy as np

import math
import torch
import torch.nn.functional as F

import warnings  # To warn if falling back to CPU

# Try importing CuPy (silently — warnings only surface when CTProjector is used)
try:
    import cupy

    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

# Try importing ASTRA (silently — warnings only surface when CTProjector is used)
try:
    import astra

    ASTRA_AVAILABLE = True
except ImportError:
    ASTRA_AVAILABLE = False


class OperatorFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, op, x):
        """Forward pass: applies op._matvec to each (c, h, w)"""
        device = x.device
        ctx.op = op  # Store the operator for backward pass

        # Initialize output tensor
        batch_size = x.shape[0]
        y = []

        # Apply the operator to each sample in the batch (over the batch dimension)
        for i in range(batch_size):
            y_i = op._matvec(x[i].unsqueeze(0))  # Apply to each (c, h, w) tensor
            y.append(y_i)

        # Stack the results back into a batch
        y = torch.cat(y, dim=0)
        ctx.save_for_backward(x)  # Save input for gradient computation
        return y.to(device)

    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass: applies op._adjoint to each (c, h, w)"""
        op = ctx.op
        device = grad_output.device

        # Initialize gradient input tensor
        batch_size = grad_output.shape[0]
        grad_input = []

        # Apply the adjoint operator to each element in the batch
        for i in range(batch_size):
            grad_i = op._adjoint(
                grad_output[i].unsqueeze(0)
            )  # Apply adjoint to each (c, h, w)
            grad_input.append(grad_i)

        # Stack the gradients back into a batch
        grad_input = torch.cat(grad_input, dim=0)

        return None, grad_input.to(device)  # No gradient for `op`, only `x`


class Operator:
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Applies operator using PyTorch autograd wrapper"""
        return OperatorFunction.apply(self, x)

    def __matmul__(self, x: torch.Tensor) -> torch.Tensor:
        """Matrix-vector multiplication"""
        return self.__call__(x)

    def T(self, y: torch.Tensor) -> torch.Tensor:
        """Transpose operator (adjoint)"""
        device = y.device
        # Apply adjoint to the batch
        return self._adjoint(y).to(device).requires_grad_(True)

    def _matvec(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the operator to a single (c, h, w) tensor"""
        raise NotImplementedError

    def _adjoint(self, y: torch.Tensor) -> torch.Tensor:
        """Apply the adjoint operator to a single (c, h, w) tensor"""
        raise NotImplementedError


class CTProjector(Operator):
    r"""
    Implements a CTProjector operator using the ASTRA toolbox.

    Automatically attempts to use CUDA-accelerated projectors and CuPy for
    GPU tensors if CUDA is available and CuPy is installed. Falls back to
    CPU-based projectors and NumPy operations otherwise, or if the input
    tensor is on the CPU.

    Ensures the output tensor is on the same device as the input tensor.

    Parameters:
        img_shape (tuple[int]): Shape of the input image (nx, ny).
        angles (np.array): Array of projection angles in radians.
        det_size (int | None): Number of detector pixels. Defaults to 2*max(nx, ny).
        geometry (str): Type of geometry ('parallel' or 'fanflat').
        force_cpu (bool): If True, forces CPU usage even if CUDA is available. Default: False.
        # Optional Fanflat parameters (can be added if needed)
        # source_origin (float): Distance source to origin (for fanflat).
        # origin_det (float): Distance origin to detector (for fanflat).
    """

    def __init__(
        self,
        img_shape: tuple[int],
        angles: np.array,
        det_size: int | None = None,
        geometry: str = "parallel",
        source_origin: float = 1800.0,
        origin_det: float = 500.0,
        force_cpu: bool = False,
    ) -> None:
        super().__init__()

        if not ASTRA_AVAILABLE:
            raise ImportError(
                "ASTRA toolbox is required for CTProjector but not found. "
                "Install it via conda: conda install -c astra-toolbox astra-toolbox"
            )
        if not CUPY_AVAILABLE:
            warnings.warn(
                "CuPy not found. GPU acceleration for ASTRA via CuPy will be disabled."
            )

        # Input setup
        self.nx, self.ny = img_shape
        self.geometry = geometry
        self.angles = angles
        self.n_angles = len(angles)

        self.source_origin = source_origin
        self.origin_det = origin_det

        # Projector setup
        if det_size is None:
            # Ensure reasonable default, maybe link to image diagonal?
            diag = np.sqrt(self.nx**2 + self.ny**2)
            self.det_size = 2 * int(np.ceil(diag / 2.0))  # Often needs padding
        else:
            self.det_size = det_size

        # Set sinogram shape
        self.mx, self.my = self.n_angles, self.det_size

        # Determine if GPU should be used
        self.use_gpu = torch.cuda.is_available() and CUPY_AVAILABLE and not force_cpu
        if force_cpu and torch.cuda.is_available():
            warnings.warn("CUDA is available but CTProjector is forced to use CPU.")
        elif not torch.cuda.is_available() and not force_cpu:
            print("CUDA not available. CTProjector will use CPU.")
        elif not CUPY_AVAILABLE and torch.cuda.is_available() and not force_cpu:
            warnings.warn(
                "CUDA available but CuPy not found. CTProjector limited to CPU operations for ASTRA data transfer."
            )
            # Force CPU mode if CuPy isn't there for GPU data handling
            self.use_gpu = False

        # Define projector based on availability
        self.proj, self.proj_id, self.vol_geom, self.proj_geom = (
            self._get_astra_projection_operator()
        )
        if self.proj is None:
            # Handle case where projector creation failed in the method
            raise RuntimeError("Failed to create ASTRA projector.")

        # Store shape info if OpTomo provides it
        try:
            # OpTomo shape might be (output_flat_dim, input_flat_dim)
            self.astra_shape = self.proj.shape
        except AttributeError:
            self.astra_shape = None  # Or calculate from mx*my, nx*ny

        # Determine reconstruction algorithm based on GPU availability
        if self.use_gpu:
            # Check if FBP_CUDA is actually available in ASTRA install
            # This requires testing or a more robust check
            self.fbp_algorithm = "FBP_CUDA"  # Or other CUDA FBP variants if needed
        else:
            self.fbp_algorithm = "FBP"  # Standard CPU FBP

        print(
            f"CTProjector initialized. Geometry: {self.geometry}. Using GPU: {self.use_gpu}. FBP Algorithm: {self.fbp_algorithm}"
        )

    def _get_astra_projection_operator(self):
        """Creates ASTRA geometries and the projector object."""
        vol_geom = astra.create_vol_geom(self.nx, self.ny)

        # Determine the base projector type based on GPU availability
        gpu_projector_available = (
            self.use_gpu
        )  # True if CUDA available, CuPy installed, not forced CPU

        if self.geometry == "parallel":
            proj_geom = astra.create_proj_geom(
                "parallel", 1.0, self.det_size, self.angles
            )
            # Use 'cuda' if possible for GPU, otherwise 'linear' for CPU
            if gpu_projector_available:
                projector_type = "cuda"
            else:
                projector_type = "linear"

        elif self.geometry == "fanflat":
            # Fanflat often requires GPU for reasonable speed
            proj_geom = astra.create_proj_geom(
                "fanflat",
                1.0,
                self.det_size,
                self.angles,
                self.source_origin,
                self.origin_det,
            )
            if gpu_projector_available:
                projector_type = "cuda"
            else:
                # Use a valid CPU fan-beam projector type like 'strip'
                projector_type = "line_fanflat"
                warnings.warn(
                    f"Using CPU projector type '{projector_type}' for fanflat geometry. This might be very slow."
                )

        else:
            print(f"Geometry '{self.geometry}' is not supported.")
            # Return None values to indicate failure
            return None, None, None, None

        # Create projector
        try:
            print(
                f"Attempting to create ASTRA projector type: '{projector_type}' for '{self.geometry}' geometry..."
            )
            proj_id = astra.create_projector(projector_type, proj_geom, vol_geom)
            proj = astra.OpTomo(proj_id)
            print(f"Successfully created ASTRA projector type: '{projector_type}'")
            # If we intended GPU but ended up with a CPU type due to fallback logic below, update self.use_gpu
            if gpu_projector_available and projector_type != "cuda":
                warnings.warn(
                    f"Projector creation resulted in CPU type '{projector_type}' despite GPU request. Adjusting."
                )
                self.use_gpu = False
                self.fbp_algorithm = "FBP"  # Ensure FBP algo is also CPU
            return proj, proj_id, vol_geom, proj_geom

        except Exception as e:
            print(f"Error creating ASTRA projector type '{projector_type}': {e}")
            # Fallback attempt (e.g., try CPU 'linear' for parallel if 'cuda' failed, or CPU 'strip' for fanflat if 'cuda' failed)
            fallback_tried = False
            if (
                gpu_projector_available
            ):  # Only try fallback if GPU was initially attempted
                if self.geometry == "parallel" and projector_type != "linear":
                    fallback_type = "linear"
                elif self.geometry == "fanflat" and projector_type != "strip":
                    fallback_type = "strip"
                else:
                    fallback_type = None  # No fallback specified for this case

                if fallback_type:
                    warnings.warn(
                        f"Falling back to CPU projector '{fallback_type}' due to error."
                    )
                    try:
                        proj_id = astra.create_projector(
                            fallback_type, proj_geom, vol_geom
                        )
                        proj = astra.OpTomo(proj_id)
                        self.use_gpu = False  # Update flag as we fell back to CPU
                        self.fbp_algorithm = "FBP"  # Update FBP algo too
                        print(
                            f"Successfully created ASTRA projector type: '{fallback_type}' (Fallback)"
                        )
                        fallback_tried = True
                        return proj, proj_id, vol_geom, proj_geom
                    except Exception as e2:
                        print(f"Fallback projector creation failed: {e2}")

            # If initial creation and fallback failed
            print("Failed to create any suitable ASTRA projector.")
            return None, None, None, None

    def _matvec(self, x: torch.Tensor) -> torch.Tensor:
        """Applies the projection operator (forward pass), handling device."""
        input_device = x.device
        input_dtype = x.dtype
        batch_size = x.shape[0]

        # Decide path: Use GPU only if requested, available, AND input is on GPU
        run_on_gpu = self.use_gpu and x.is_cuda

        if run_on_gpu:
            # --- GPU Path (via CuPy) ---
            try:
                x_shape_original = x.shape  # (N, C, nx, ny)
                # ASTRA usually works with flat 2D or 3D, need C=1?
                if x.shape[1] != 1:
                    warnings.warn(
                        f"Input tensor has {x.shape[1]} channels, expected 1. Using first channel."
                    )
                    x = x[:, 0:1, ...]  # Take first channel, keep dim

                # Reshape for ASTRA (N, nx*ny) or just (nx, ny) if OpTomo handles loop?
                # OpTomo likely expects (nx, ny) based on original code.
                # We must loop outside if OpTomo doesn't handle batch.
                # Let's assume the OperatorFunction loop handles batching.
                # Input x here is (1, 1, nx, ny)
                x_flat_shape = (self.nx, self.ny)
                x_gpu_flat = x.reshape(x_flat_shape)  # Shape (nx, ny)

                x_cp = cupy.asarray(x_gpu_flat)  # No CPU transfer

                # Perform ASTRA operation
                y_cp = self.proj @ x_cp  # Assumes OpTomo accepts CuPy

                # Convert result back to PyTorch GPU tensor
                y = torch.as_tensor(y_cp, device=input_device)  # No CPU transfer

                # Reshape back to PyTorch standard (1, 1, mx, my)
                y = y.reshape((1, 1, self.mx, self.my))  # Add back batch/channel

            except Exception as e:
                warnings.warn(f"GPU _matvec failed: {e}. Falling back to CPU path.")
                run_on_gpu = False  # Force CPU path on failure

        # --- CPU Path (NumPy) ---
        # Executes if run_on_gpu is False (due to config, availability, input device, or GPU error)
        if not run_on_gpu:
            x_np = x.reshape(self.nx, self.ny).cpu().numpy()  # Ensure CPU and NumPy
            y_np = self.proj @ x_np

            # Convert back to PyTorch tensor on the original device
            y = torch.tensor(
                y_np.reshape((1, 1, self.mx, self.my)),  # Add back batch/channel
                device=input_device,
                dtype=input_dtype,
            )

        # The OperatorFunction loop will concatenate the batch results
        return y

    def _adjoint(self, y: torch.Tensor) -> torch.Tensor:
        """Applies the backprojection operator (adjoint), handling device."""
        input_device = y.device
        input_dtype = y.dtype
        batch_size = y.shape[0]

        # Decide path: Use GPU only if requested, available, AND input is on GPU
        run_on_gpu = self.use_gpu and y.is_cuda

        if run_on_gpu:
            # --- GPU Path (via CuPy) ---
            try:
                y_shape_original = y.shape  # (N, C, mx, my)
                if y.shape[1] != 1:
                    warnings.warn(
                        f"Input tensor has {y.shape[1]} channels, expected 1. Using first channel."
                    )
                    y = y[:, 0:1, ...]

                # Reshape for ASTRA (mx, my) assuming OperatorFunction loop
                y_flat_shape = (self.mx, self.my)
                y_gpu_flat = y.reshape(y_flat_shape)

                y_cp = cupy.asarray(y_gpu_flat)  # No CPU transfer

                # Perform ASTRA adjoint operation
                x_cp = self.proj.T @ y_cp  # Assumes OpTomo.T accepts CuPy

                # Convert result back to PyTorch GPU tensor
                x = torch.as_tensor(x_cp, device=input_device)  # No CPU transfer

                # Reshape back to PyTorch standard (1, 1, nx, ny)
                x = x.reshape((1, 1, self.nx, self.ny))

            except Exception as e:
                warnings.warn(f"GPU _adjoint failed: {e}. Falling back to CPU path.")
                run_on_gpu = False  # Force CPU path on failure

        # --- CPU Path (NumPy) ---
        if not run_on_gpu:
            y_np = (
                y.reshape(self.mx, self.my).detach().cpu().numpy()
            )  # Ensure CPU and NumPy
            x_np = self.proj.T @ y_np

            # Convert back to PyTorch tensor on the original device
            x = torch.tensor(
                x_np.reshape((1, 1, self.nx, self.ny)),  # Add back batch/channel
                device=input_device,
                dtype=input_dtype,
            )

        # The OperatorFunction loop will concatenate the batch results
        return x

    # --- FBP Method ---
    def FBP(self, y: torch.Tensor) -> torch.Tensor:
        """Performs Filtered Back-Projection reconstruction."""
        input_device = y.device
        input_dtype = y.dtype
        batch_size = y.shape[0]

        results = []
        for i in range(batch_size):
            y_slice = y[i : i + 1]  # Shape (1, C, mx, my)
            slice_device = y_slice.device  # Device of this specific slice

            # Decide path for this slice
            run_on_gpu = self.use_gpu and slice_device.type == "cuda"
            current_fbp_algo = self.fbp_algorithm

            # Adjust FBP algorithm if we determined GPU use but slice is on CPU
            if current_fbp_algo == "FBP_CUDA" and slice_device.type != "cuda":
                warnings.warn(
                    "FBP_CUDA requested but input slice is on CPU. Using 'FBP'."
                )
                current_fbp_algo = "FBP"
            elif (
                current_fbp_algo == "FBP"
                and slice_device.type == "cuda"
                and self.use_gpu
            ):
                # If default is CPU but slice is GPU and we allow GPU, maybe try CUDA?
                # Let's stick to the initialized self.fbp_algorithm primarily.
                pass

            x_recon_np = None
            if run_on_gpu:
                # --- GPU FBP Path ---
                try:
                    if y_slice.shape[1] != 1:
                        warnings.warn(
                            f"FBP input slice has {y_slice.shape[1]} channels, expected 1. Using first."
                        )
                        y_slice = y_slice[:, 0:1, ...]

                    y_slice_flat = y_slice.reshape(self.mx, self.my)
                    y_cp = cupy.asarray(y_slice_flat)

                    # Perform ASTRA reconstruction
                    # NOTE: Check if reconstruct method *actually* takes CuPy arrays.
                    # It might still expect NumPy, requiring cupy.asnumpy(y_cp).
                    # Let's assume it might need NumPy for now based on common practice
                    y_np_from_gpu = cupy.asnumpy(y_cp)
                    x_recon_np = self.proj.reconstruct(current_fbp_algo, y_np_from_gpu)

                    # Alternative: If reconstruct *does* take CuPy:
                    # x_recon_cp = self.proj.reconstruct(current_fbp_algo, y_cp)
                    # x_recon_np = cupy.asnumpy(x_recon_cp)

                except Exception as e:
                    warnings.warn(
                        f"GPU FBP failed for slice {i}: {e}. Falling back to CPU."
                    )
                    run_on_gpu = False  # Force CPU path for this slice
                    if current_fbp_algo == "FBP_CUDA":
                        current_fbp_algo = "FBP"

            # --- CPU FBP Path ---
            if not run_on_gpu:
                y_slice_np = (
                    y_slice.reshape(self.mx, self.my).cpu().numpy()
                )  # Ensure CPU, NumPy
                try:
                    x_recon_np = self.proj.reconstruct(current_fbp_algo, y_slice_np)
                except Exception as e:
                    print(f"CPU FBP failed for slice {i}: {e}")
                    # Create a zero tensor as fallback? Or raise error?
                    x_recon_np = np.zeros((self.nx, self.ny), dtype=y_slice_np.dtype)

            # Convert result back to PyTorch tensor on the original slice device
            x_tmp = torch.tensor(
                x_recon_np.reshape((1, 1, self.nx, self.ny)),  # Add back batch/channel
                device=slice_device,  # Place on the device of the input slice y[i]
                dtype=input_dtype,
            )
            results.append(x_tmp)

        # Stack results from all slices
        return torch.cat(results, dim=0).to(
            input_device
        )  # Ensure final tensor is on input y's device

    # Cleanup method (optional but good practice for ASTRA)
    def __del__(self):
        try:
            if hasattr(self, "proj_id") and self.proj_id:
                astra.projector.delete(self.proj_id)
            if hasattr(self, "vol_geom") and self.vol_geom:
                # Geometry deletion might not be necessary or available via API
                pass
            if hasattr(self, "proj_geom") and self.proj_geom:
                # Geometry deletion might not be necessary or available via API
                pass
        except Exception as e:
            # Might fail if ASTRA is already unloaded during shutdown
            # print(f"Ignoring error during ASTRA object cleanup: {e}")
            pass


class Blurring(Operator):
    def __init__(
        self,
        img_shape: tuple[int],
        kernel: torch.Tensor = None,
        kernel_type: str = None,
        kernel_size: int = 3,
        kernel_variance: float = 1.0,
        motion_angle: float = 0.0,
    ):
        """
        Blurring operator using convolution.

        Parameters:
        - kernel (torch.Tensor, optional): Custom kernel for convolution. If `kernel_type` is provided, this is ignored.
        - kernel_type (str, optional): Type of kernel to use. Supports 'gaussian' and 'motion'.
        - kernel_size (int, optional): Size of the kernel (for 'gaussian' and 'motion'). Must be an odd integer.
        - kernel_variance (float, optional): Variance of the Gaussian kernel (only used if kernel_type='gaussian').
        - motion_angle (float, optional): Angle of motion blur in degrees (only used if kernel_type='motion').
        """
        super().__init__()

        # Shape setup
        self.nx, self.ny = img_shape
        self.mx, self.my = img_shape

        if kernel_type is not None:
            if kernel_type == "gaussian":
                if kernel_size % 2 == 0:
                    raise ValueError("Kernel size must be an odd integer.")
                self.kernel = self._generate_gaussian_kernel(
                    kernel_size, kernel_variance
                )
            elif kernel_type == "motion":
                if kernel_size % 2 == 0:
                    raise ValueError("Kernel size must be an odd integer.")
                self.kernel = self._generate_motion_kernel(kernel_size, motion_angle)
            else:
                raise ValueError("kernel_type must be either 'gaussian' or 'motion'")
        elif kernel is None:
            raise ValueError("Either `kernel` or `kernel_type` must be provided.")
        else:
            self.kernel = kernel

        # Ensure kernel is a 4D tensor with shape (out_channels, in_channels, k, k)
        if len(self.kernel.shape) == 2:
            self.kernel = self.kernel.unsqueeze(0)

        if len(self.kernel.shape) == 3:
            # Meaning only the batch dimension in missing
            self.kernel = self.kernel.unsqueeze(0)

    def _generate_gaussian_kernel(
        self, kernel_size: int, kernel_variance: float
    ) -> torch.Tensor:
        """
        Generates a Gaussian kernel with the given size and variance.
        """
        ax = torch.arange(kernel_size) - kernel_size // 2
        xx, yy = torch.meshgrid(ax, ax, indexing="ij")
        kernel = torch.exp(-(xx**2 + yy**2) / (2 * kernel_variance))
        kernel /= kernel.sum()  # Normalize kernel to sum to 1
        return kernel

    def _generate_motion_kernel(self, kernel_size: int, angle: float) -> torch.Tensor:
        """
        Generates a motion blur kernel that blurs in a linear direction.
        """
        kernel = torch.zeros((kernel_size, kernel_size), dtype=torch.float32)
        center = kernel_size // 2
        angle = math.radians(angle)

        # Compute motion blur direction
        dx, dy = math.cos(angle), math.sin(angle)
        for i in range(kernel_size):
            x = int(center + (i - center) * dx)
            y = int(center + (i - center) * dy)
            if 0 <= x < kernel_size and 0 <= y < kernel_size:
                kernel[y, x] = 1.0

        kernel /= kernel.sum()  # Normalize to keep intensity unchanged
        return kernel

    def _matvec(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the blurring operator (forward convolution).
        """
        blurred = F.conv2d(
            x, self.kernel.to(x.device), padding="same"
        )  # Apply convolution
        return blurred

    def _adjoint(self, y: torch.Tensor) -> torch.Tensor:
        """
        Applies the adjoint operator, which in this case is also a convolution
        with a flipped kernel (assuming symmetric kernels like Gaussian).
        """
        flipped_kernel = torch.flip(self.kernel, dims=[2, 3])  # Flip spatial dimensions
        adjoint_result = F.conv2d(y, flipped_kernel.to(y.device), padding="same")
        return adjoint_result


class DownScaling(Operator):
    def __init__(
        self,
        img_shape: tuple[int],
        downscale_factor: int,
        mode: str = "avg",
    ):
        """
        Initializes the DownScaling operator.

        Parameters:
        - downscale_factor (int): The factor by which the input is downscaled.
        - mode (str): The type of downscaling, either "avg" (average pooling) or "naive" (removes odd indices).
        """
        super().__init__()

        # Shape setup
        self.nx, self.ny = img_shape
        self.mx, self.my = self.nx // downscale_factor, self.ny // downscale_factor

        self.downscale_factor = downscale_factor
        if mode not in ["avg", "naive"]:
            raise ValueError("mode must be either 'avg' or 'naive'")
        self.mode = mode

    def _matvec(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the downscaling operator.
        """
        factor = self.downscale_factor
        if self.mode == "avg":
            y = F.avg_pool2d(x, factor, stride=factor)
        elif self.mode == "naive":
            y = x[..., ::factor, ::factor]  # Take every 'factor'-th element
        return y

    def _adjoint(self, y: torch.Tensor) -> torch.Tensor:
        """
        Applies the transposed operator (true adjoint).
        """
        factor = self.downscale_factor
        if self.mode == "avg":
            # Spread values uniformly back
            y_upsampled = F.interpolate(y, scale_factor=factor, mode="nearest")
            x_out = y_upsampled / (factor**2)  # Normalize to preserve energy
        elif self.mode == "naive":
            # Create a zero-filled tensor of the original size
            N, C, H, W = y.shape
            H_up, W_up = H * factor, W * factor
            x_up = torch.zeros((N, C, H_up, W_up), device=y.device, dtype=y.dtype)
            x_up[..., ::factor, ::factor] = y  # Insert values at correct locations
            x_out = x_up
        return x_out


class Gradient(Operator):
    r"""
    Implements the Gradient operator, acting on standardized Pytorch tensors of shape (N, 1, nx, ny) and returning a tensor of
    shape (N, 2, nx, ny), where the first channel contains horizontal derivatives, while the second channel contains vertical
    derivatives.
    """

    def __init__(self, img_shape: tuple[int]) -> None:
        super().__init__()

        self.nx, self.ny = img_shape
        self.mx, self.my = img_shape

    def _matvec(self, x: torch.Tensor) -> torch.Tensor:
        N, c, nx, ny = x.shape
        D_h = torch.diff(x, n=1, dim=2, prepend=torch.zeros((N, c, 1, ny), device=x.device, dtype=x.dtype))
        D_v = torch.diff(x, n=1, dim=3, prepend=torch.zeros((N, c, nx, 1), device=x.device, dtype=x.dtype))

        return torch.cat((D_h, D_v), dim=1)

    def _adjoint(self, y: torch.Tensor) -> torch.Tensor:
        # y has shape (N, 2, nx, ny): channel 0 = horizontal diffs, channel 1 = vertical diffs.
        # The adjoint of forward-difference-with-zero-prepend is the negative divergence:
        #   D_h^T z[i] = z[i] - z[i+1]  (with z[nx] = 0)
        #   D_v^T z[j] = z[j] - z[j+1]  (with z[ny] = 0)
        N, _, nx, ny = y.shape
        y_h = y[:, 0:1, :, :]
        y_v = y[:, 1:2, :, :]

        D_h_T = -torch.diff(y_h, n=1, dim=2, append=torch.zeros((N, 1, 1, ny), device=y.device, dtype=y.dtype))
        D_v_T = -torch.diff(y_v, n=1, dim=3, append=torch.zeros((N, 1, nx, 1), device=y.device, dtype=y.dtype))

        return D_h_T + D_v_T
