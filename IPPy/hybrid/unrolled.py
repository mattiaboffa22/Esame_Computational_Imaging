# unrolled.py
import torch
import torch.nn as nn

# Import the Operator base class and potentially specific operators if needed for type hinting
from ..operators import Operator


class CNNBlock(nn.Module):
    """
    A simple CNN block: Conv -> Activation -> Conv.
    Maintains spatial resolution using padding='same'.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: int = 32,
        kernel_size: int = 3,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, mid_channels, kernel_size, padding="same", bias=True
        )
        self.activation = nn.LeakyReLU(0.1)  # Or nn.ReLU()
        self.conv2 = nn.Conv2d(
            mid_channels, out_channels, kernel_size, padding="same", bias=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        return x


class LearnedPrimalDual(nn.Module):
    r"""
    Implementation of the Learned Primal-Dual (LPD) reconstruction algorithm.

    This network unrolls a fixed number of primal-dual iterations, where the update
    steps are replaced by learned CNNs.

    Based on the formulation by Adler & Öktem.

    Initialization:
        x_0 = A.T(y)  (or zeros)
        p_0 = 0

    Iteration k:
        # Dual Update Block
        Ax = A(x_k)
        dual_input = torch.cat([p_k, Ax, y], dim=1) # Concatenate along channel dim
        delta_p = DualNet_k(dual_input)
        p_{k+1} = p_k + delta_p # Or p_{k+1} = DualNet_k(dual_input) for direct mapping

        # Primal Update Block
        ATp = A.T(p_{k+1})
        primal_input = torch.cat([x_k, ATp], dim=1) # Concatenate along channel dim
        delta_x = PrimalNet_k(primal_input)
        x_{k+1} = x_k + delta_x # Or x_{k+1} = PrimalNet_k(primal_input) for direct mapping

    Parameters:
        operator_A (Operator): An instance of the Operator class (or its subclass)
                               from operators.py, representing the forward operator A.
        num_iterations (int): The number of unrolled iterations.
        primal_channels (int): Number of channels expected for the primal variable x
                               (usually 1 for grayscale images).
        dual_channels (int): Number of channels in the measurement/dual space y/p.
                             This depends on the output shape of operator_A.
        num_cnn_features (int): Number of features used in the intermediate layers
                                of the CNN blocks.
    """

    def __init__(
        self,
        operator_A: Operator,
        num_iterations: int = 10,
        primal_channels: int = 1,  # e.g., 1 for grayscale image x
        dual_channels: int = 1,  # Will be inferred from operator if possible, adjust if needed
        num_cnn_features: int = 32,
        learn_initialization: bool = False,
    ):  # Option to learn A.T operation
        super().__init__()

        self.operator_A = operator_A
        # Access the adjoint method defined in the Operator class
        self.operator_AT = operator_A.T
        self.num_iterations = num_iterations
        self.primal_channels = primal_channels

        # --- Infer dual channels ---
        # Attempt to infer dual channels from the operator's properties if possible
        # This requires operator_A to have attributes like mx, my or an output shape hint
        # Placeholder: Create a dummy input for shape inference
        try:
            # Assuming operator works with dummy input of correct spatial size
            # Use operator properties if available (like CTProjector's mx, my)
            if hasattr(operator_A, "mx") and hasattr(operator_A, "my"):
                # Example: CTProjector might output (N, 1, mx, my)
                # Warning: This assumes the operator adds a channel dimension if needed
                # Or use a more robust shape inference:
                dummy_primal = torch.zeros(
                    1, primal_channels, operator_A.nx, operator_A.ny
                )
                with torch.no_grad():
                    dummy_dual = operator_A(dummy_primal)
                self.dual_channels = dummy_dual.shape[1]
                self.dual_shape_h = dummy_dual.shape[2]
                self.dual_shape_w = dummy_dual.shape[3]
                print(
                    f"Inferred dual channels: {self.dual_channels}, H: {self.dual_shape_h}, W: {self.dual_shape_w}"
                )
            else:
                # Fallback if shape info isn't directly on operator
                self.dual_channels = dual_channels  # Use provided value
                print(
                    f"Warning: Could not infer dual channels, using provided value: {self.dual_channels}"
                )
                # Need dual height/width for initialization
                # This part is tricky without operator info, assume same as primal for now?
                # Or require dual_height, dual_width as arguments?
                # Let's assume y passed to forward() will have the correct shape
                self.dual_shape_h = None
                self.dual_shape_w = None

        except Exception as e:
            print(
                f"Warning: Error during dual channel inference: {e}. Using provided value: {dual_channels}"
            )
            self.dual_channels = dual_channels
            self.dual_shape_h = None
            self.dual_shape_w = None

        # --- Learnable Components ---
        self.primal_nets = nn.ModuleList()
        self.dual_nets = nn.ModuleList()

        # Define the networks for each iteration
        # Input channels depend on concatenation strategy
        # DualNet input: p_k (dual_ch) + Ax_k (dual_ch) + y (dual_ch)
        dual_net_in_channels = self.dual_channels * 3
        # PrimalNet input: x_k (primal_ch) + ATp_{k+1} (primal_ch)
        primal_net_in_channels = self.primal_channels * 2

        for _ in range(num_iterations):
            # Note: Output channels should match the variable being updated (p or x)
            self.dual_nets.append(
                CNNBlock(
                    dual_net_in_channels,
                    self.dual_channels,
                    mid_channels=num_cnn_features,
                )
            )
            self.primal_nets.append(
                CNNBlock(
                    primal_net_in_channels,
                    self.primal_channels,
                    mid_channels=num_cnn_features,
                )
            )

        # --- Optional: Learnable Initialization ---
        # Instead of fixed A.T(y), learn an operation mapping y -> x_0
        self.learn_initialization = learn_initialization
        if self.learn_initialization:
            # Simple Conv layer mapping y channels to primal channels
            # Kernel size 1x1 acts like a learned linear combination across channels
            # followed by pixel-wise processing if needed.
            # Might need up/downsampling if y and x shapes differ significantly beyond channels
            # This assumes spatial dims allow direct mapping or operator handles it.
            # A more complex network could be used here.
            self.initial_primal_net = nn.Conv2d(
                self.dual_channels, self.primal_channels, kernel_size=1
            )
        else:
            self.initial_primal_net = None

    def forward(
        self, y: torch.Tensor, x_init: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Applies the unrolled network to the measurement y.

        Parameters:
            y (torch.Tensor): The measurement tensor, shape (N, dual_channels, H_dual, W_dual).
            x_init (torch.Tensor, optional): An initial guess for x. If None, calculated
                                             using A.T(y) or learned initialization.
                                             Shape (N, primal_channels, H_primal, W_primal).

        Returns:
            torch.Tensor: The reconstructed primal variable x, shape (N, primal_channels, H_primal, W_primal).
        """
        # Get batch size, device
        N = y.shape[0]
        device = y.device

        # --- Initialization ---
        # Initialize primal variable x
        if x_init is not None:
            x = x_init.to(device)
        elif self.learn_initialization:
            # Check if spatial dims match or need adjustment
            # This simple version assumes initial_primal_net handles it or operator_AT does.
            # A more robust way might involve explicit up/downsampling if y/x sizes differ.
            try:
                x = self.initial_primal_net(
                    self.operator_AT(y)
                )  # Apply learned op after adjoint
            except Exception as e:
                print(f"Warning: Learned initialization failed ({e}). Falling back.")
                # Fallback: Apply adjoint T directly
                # Ensure T handles batches and device placement correctly
                x = self.operator_AT(y)
        else:
            # Default initialization: Apply adjoint T directly
            # Ensure T handles batches and device placement correctly
            x = self.operator_AT(y)

        # Ensure x has the correct number of primal channels if needed (e.g., if A.T output doesn't match)
        # This might happen if A.T returns more/fewer channels than expected primal_channels
        if x.shape[1] != self.primal_channels:
            print(
                f"Warning: Initial x channels ({x.shape[1]}) != primal_channels ({self.primal_channels}). Adjusting..."
            )
            # Example adjustment (e.g., taking first channel, or using a conv1x1) - requires care!
            if x.shape[1] > self.primal_channels:
                x = x[:, : self.primal_channels, ...]  # Take first channel(s)
            else:  # Pad with zeros or replicate - problematic
                # Better: Ensure operator_AT output shape is compatible or use a learnable layer
                raise ValueError(
                    f"Initial x channels {x.shape[1]} < required {self.primal_channels}. Cannot proceed."
                )

        # Initialize dual variable p
        # Use shape inferred during __init__ or from y if not inferred
        dual_h = self.dual_shape_h if self.dual_shape_h is not None else y.shape[2]
        dual_w = self.dual_shape_w if self.dual_shape_w is not None else y.shape[3]
        p = torch.zeros(
            N, self.dual_channels, dual_h, dual_w, device=device, dtype=y.dtype
        )

        # --- Unrolled Iterations ---
        for k in range(self.num_iterations):
            # -- Dual Update --
            # Apply forward operator A
            Ax = self.operator_A(x)  # Shape: (N, dual_channels, H_dual, W_dual)

            # Check shapes before concatenation
            if Ax.shape != p.shape or Ax.shape != y.shape:
                raise ValueError(
                    f"Shape mismatch in dual update iteration {k}: "
                    f"p: {p.shape}, Ax: {Ax.shape}, y: {y.shape}"
                )

            # Concatenate along the channel dimension (dim=1)
            dual_input = torch.cat((p, Ax, y), dim=1)

            # Apply k-th dual network
            # Option 1: Learn the update step delta_p
            delta_p = self.dual_nets[k](dual_input)
            p = p + delta_p
            # Option 2: Learn the direct mapping to p_{k+1}
            # p = self.dual_nets[k](dual_input)

            # -- Primal Update --
            # Apply adjoint operator A.T
            ATp = self.operator_AT(p)  # Shape: (N, primal_channels, H_primal, W_primal)

            # Check shapes before concatenation
            if ATp.shape != x.shape:
                raise ValueError(
                    f"Shape mismatch in primal update iteration {k}: "
                    f"x: {x.shape}, ATp: {ATp.shape}"
                )

            # Concatenate along the channel dimension (dim=1)
            primal_input = torch.cat((x, ATp), dim=1)

            # Apply k-th primal network
            # Option 1: Learn the update step delta_x
            delta_x = self.primal_nets[k](primal_input)
            x = x + delta_x
            # Option 2: Learn the direct mapping to x_{k+1}
            # x = self.primal_nets[k](primal_input)

        return x
