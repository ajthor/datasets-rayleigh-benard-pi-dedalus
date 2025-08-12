"""
Dedalus script simulating 2D horizontally-periodic Rayleigh-Benard convection.
This script demonstrates solving a 2D Cartesian initial value problem and produces
dataset samples using PyTorch IterableDataset for operator learning.

The Rayleigh-Benard system consists of:
- Buoyancy field b(x,z,t): thermal field driving convection
- Velocity field u(x,z,t): fluid motion (2D vector)
- Pressure field p(x,z,t): enforces incompressibility

For operator learning, this dataset provides:
- Input: Initial buoyancy and velocity fields
- Output: Temporal evolution of buoyancy, velocity, and vorticity
- Parameters: Rayleigh and Prandtl numbers, perturbation scales
"""

import numpy as np
from torch.utils.data import IterableDataset
import dedalus.public as d3
import logging

logger = logging.getLogger(__name__)


class RayleighBenardDataset(IterableDataset):
    def __init__(
        self,
        Lx=4,
        Lz=1,
        Nx=256,
        Nz=64,
        Rayleigh=2e6,
        Prandtl=1,
        dealias=3 / 2,
        stop_sim_time=50,
        save_interval=0.25,
        perturbation_scale_range=(1e-4, 1e-2),
        timestepper=d3.RK222,
        max_timestep=0.125,
        dtype=np.float64,
    ):
        """
        Dataset for Rayleigh-Benard convection simulations.

        This dataset generates samples for operator learning where:
        - Input: Initial buoyancy and velocity fields with random perturbations
        - Output: Temporal evolution of buoyancy, pressure, velocity, and vorticity fields

        Args:
            Lx: Domain width
            Lz: Domain height
            Nx: Number of grid points in x-direction
            Nz: Number of grid points in z-direction
            Rayleigh: Rayleigh number (controls convection strength)
            Prandtl: Prandtl number (momentum/thermal diffusivity ratio)
            dealias: Dealiasing factor for spectral accuracy
            stop_sim_time: Total simulation time
            save_interval: Time interval for saving snapshots
            perturbation_scale_range: (min, max) range for random perturbation amplitude
            timestepper: Dedalus timestepper (e.g., d3.RK222)
            max_timestep: Maximum allowed timestep
            dtype: Data type for computations
        """
        super().__init__()
        self.Lx = Lx
        self.Lz = Lz
        self.Nx = Nx
        self.Nz = Nz
        self.Rayleigh = Rayleigh
        self.Prandtl = Prandtl
        self.dealias = dealias
        self.stop_sim_time = stop_sim_time
        self.save_interval = save_interval
        self.perturbation_scale_range = perturbation_scale_range
        self.timestepper = timestepper
        self.max_timestep = max_timestep
        self.dtype = dtype

        # Physical parameters (non-dimensionalized)
        self.kappa = (Rayleigh * Prandtl) ** (-1 / 2)  # Thermal diffusivity
        self.nu = (Rayleigh / Prandtl) ** (-1 / 2)  # Kinematic viscosity

        # Setup Dedalus coordinate system
        self.coords = d3.CartesianCoordinates("x", "z")
        self.dist = d3.Distributor(self.coords, dtype=dtype)
        self.xbasis = d3.RealFourier(
            self.coords["x"], size=Nx, bounds=(0, Lx), dealias=dealias
        )
        self.zbasis = d3.ChebyshevT(
            self.coords["z"], size=Nz, bounds=(0, Lz), dealias=dealias
        )

        # Get coordinate grids
        self.x, self.z = self.dist.local_grids(self.xbasis, self.zbasis)

        # Setup unit vectors and lift operator for boundary conditions
        self.ex, self.ez = self.coords.unit_vector_fields(self.dist)
        self.lift_basis = self.zbasis.derivative_basis(1)
        self.lift = lambda A: d3.Lift(A, self.lift_basis, -1)

    def __iter__(self):
        """Generate infinite samples from the dataset."""
        while True:
            # Sample-varying parameters for Gaussian bump at lower boundary
            bump_center = np.random.uniform(
                0.2 * self.Lx, 0.8 * self.Lx
            )  # Random x position
            bump_width = np.random.uniform(0.1 * self.Lx, 0.3 * self.Lx)  # Random width
            bump_amplitude = np.random.uniform(0.01, 0.05)  # Random amplitude

            # Fixed perturbation scale for deterministic background noise
            perturbation_scale = 1e-3

            # Solve and yield result
            yield self.solve(perturbation_scale, bump_center, bump_width, bump_amplitude)

    def solve(self, perturbation_scale, bump_center, bump_width, bump_amplitude):
        """
        Solve the Rayleigh-Benard equations with random initial conditions and Gaussian bump boundary.

        Args:
            perturbation_scale: Amplitude of random perturbations
            bump_center: X position of Gaussian bump center
            bump_width: Width of Gaussian bump
            bump_amplitude: Amplitude of Gaussian bump

        Returns:
            Dictionary with flattened solution data for parquet compatibility
        """
        # Create fields for this solve
        p = self.dist.Field(name="p", bases=(self.xbasis, self.zbasis))
        b = self.dist.Field(name="b", bases=(self.xbasis, self.zbasis))
        u = self.dist.VectorField(
            self.coords, name="u", bases=(self.xbasis, self.zbasis)
        )

        # Tau fields for boundary conditions
        tau_p = self.dist.Field(name="tau_p")
        tau_b1 = self.dist.Field(name="tau_b1", bases=self.xbasis)
        tau_b2 = self.dist.Field(name="tau_b2", bases=self.xbasis)
        tau_u1 = self.dist.VectorField(self.coords, name="tau_u1", bases=self.xbasis)
        tau_u2 = self.dist.VectorField(self.coords, name="tau_u2", bases=self.xbasis)

        # Initial conditions: linear background + deterministic perturbations
        b.fill_random("g", distribution="normal", scale=perturbation_scale, seed=42)
        b["g"] *= self.z * (self.Lz - self.z)  # Damp noise at walls
        b["g"] += self.Lz - self.z  # Add linear background
        # Note: Gaussian bump is handled by boundary condition, not initial condition

        # Store initial conditions
        b_initial = np.copy(b["g"])
        p_initial = np.copy(p["g"])  # Pressure starts at zero
        u_initial = np.array([u["g"][0].copy(), u["g"][1].copy()])
        vorticity_initial = np.copy((-d3.div(d3.skew(u))).evaluate()["g"])

        # Setup variables for namespace
        kappa = self.kappa
        nu = self.nu
        Lz = self.Lz
        x, z = self.x, self.z
        ex, ez = self.ex, self.ez
        lift = self.lift

        # Create Gaussian bump function for boundary condition
        x_1d = self.x[:, 0:1]
        gaussian_bump = bump_amplitude * np.exp(
            -((x_1d - bump_center) ** 2) / (2 * bump_width**2)
        )
        bump_field = self.dist.Field(name="bump_field", bases=self.xbasis)
        bump_field["g"] = gaussian_bump

        # First-order gradient operations
        grad_u = d3.grad(u) + ez * lift(tau_u1)
        grad_b = d3.grad(b) + ez * lift(tau_b1)

        # Setup the initial value problem
        problem = d3.IVP(
            [p, b, u, tau_p, tau_b1, tau_b2, tau_u1, tau_u2], namespace=locals()
        )
        problem.add_equation("trace(grad_u) + tau_p = 0")  # Incompressibility
        problem.add_equation(
            "dt(b) - kappa*div(grad_b) + lift(tau_b2) = - u@grad(b)"
        )  # Heat equation
        problem.add_equation(
            "dt(u) - nu*div(grad_u) + grad(p) - b*ez + lift(tau_u2) = - u@grad(u)"
        )  # Momentum

        # Boundary conditions
        problem.add_equation("b(z=0) = Lz + bump_field")  # Hot bottom wall with Gaussian bump
        problem.add_equation("u(z=0) = 0")  # No-slip bottom wall
        problem.add_equation("b(z=Lz) = 0")  # Cold top wall
        problem.add_equation("u(z=Lz) = 0")  # No-slip top wall
        problem.add_equation("integ(p) = 0")  # Pressure gauge

        # Build solver
        solver = problem.build_solver(self.timestepper)
        solver.stop_sim_time = self.stop_sim_time

        # Storage for solution trajectories
        buoyancy_list = []
        vorticity_list = []
        velocity_list = []
        pressure_list = []
        time_list = []

        # CFL condition
        CFL = d3.CFL(
            solver,
            initial_dt=self.max_timestep,
            cadence=10,
            safety=0.5,
            threshold=0.05,
            max_change=1.5,
            min_change=0.5,
            max_dt=self.max_timestep,
        )
        CFL.add_velocity(u)

        # Flow properties
        flow = d3.GlobalFlowProperty(solver, cadence=10)
        flow.add_property(np.sqrt(u @ u) / nu, name="Re")

        # Initial state - store fields
        b.change_scales(1)
        u.change_scales(1)
        p.change_scales(1)
        vorticity_data = -d3.div(d3.skew(u)).evaluate()["g"]

        buoyancy_list.append(np.copy(b["g"]))
        velocity_list.append(np.copy(u["g"]))
        pressure_list.append(np.copy(p["g"]))
        vorticity_list.append(np.copy(vorticity_data))
        time_list.append(solver.sim_time)

        # Main solution loop
        save_counter = 0
        try:
            logger.info(
                f"Starting Rayleigh-Benard solve (perturbation_scale={perturbation_scale:.1e})"
            )
            while solver.proceed:
                timestep = CFL.compute_timestep()
                solver.step(timestep)
                save_counter += 1

                # Save at specified intervals
                if save_counter * timestep >= self.save_interval:
                    save_counter = 0
                    
                    # Store solution data
                    b.change_scales(1)
                    u.change_scales(1)
                    p.change_scales(1)
                    vorticity_data = -d3.div(d3.skew(u)).evaluate()["g"]

                    buoyancy_list.append(np.copy(b["g"]))
                    velocity_list.append(np.copy(u["g"]))
                    pressure_list.append(np.copy(p["g"]))
                    vorticity_list.append(np.copy(vorticity_data))
                    time_list.append(solver.sim_time)

                if (solver.iteration - 1) % 50 == 0:
                    max_Re = flow.max("Re")
                    logger.info(
                        f"Iteration={solver.iteration}, Time={solver.sim_time:.3f}, dt={timestep:.1e}, max(Re)={max_Re:.1f}"
                    )
        except Exception as e:
            logger.error(f"Exception raised during simulation: {e}")
            raise
        finally:
            solver.log_stats()

        # Convert lists to arrays
        buoyancy_trajectory = np.array(buoyancy_list)
        velocity_trajectory = np.array(velocity_list)
        pressure_trajectory = np.array(pressure_list)
        vorticity_trajectory = np.array(vorticity_list)
        time_trajectory = np.array(time_list)

        # Create coordinate meshgrids and format data for parquet compatibility
        X, Z = np.meshgrid(self.x.ravel(), self.z.ravel(), indexing="ij")
        
        # Combine coordinates into single array
        spatial_coords = np.column_stack((X.ravel(), Z.ravel()))

        # Return flattened arrays suitable for parquet serialization
        return {
            # Spatial coordinates (N, 2)
            "spatial_coordinates": spatial_coords,
            # Initial conditions (flattened)
            "buoyancy_initial": b_initial.ravel(),
            "pressure_initial": p_initial.ravel(),
            "velocity_x_initial": u_initial[0].ravel(),
            "velocity_z_initial": u_initial[1].ravel(),
            "vorticity_initial": vorticity_initial.ravel(),
            # Trajectories (time_steps, spatial_points_flattened)
            "buoyancy_trajectory": buoyancy_trajectory.reshape(
                len(time_trajectory), -1
            ),
            "pressure_trajectory": pressure_trajectory.reshape(
                len(time_trajectory), -1
            ),
            "velocity_x_trajectory": velocity_trajectory[:, 0].reshape(
                len(time_trajectory), -1
            ),
            "velocity_z_trajectory": velocity_trajectory[:, 1].reshape(
                len(time_trajectory), -1
            ),
            "vorticity_trajectory": vorticity_trajectory.reshape(
                len(time_trajectory), -1
            ),
            # Time coordinates
            "time_coordinates": time_trajectory,
            # Parameters and metadata
            "rayleigh_number": self.Rayleigh,
            "prandtl_number": self.Prandtl,
            "perturbation_scale": perturbation_scale,
            "bump_center": bump_center,
            "bump_width": bump_width,
            "bump_amplitude": bump_amplitude,
            "grid_shape_x": self.Nx,
            "grid_shape_z": self.Nz,
            "domain_size_x": self.Lx,
            "domain_size_z": self.Lz,
        }
