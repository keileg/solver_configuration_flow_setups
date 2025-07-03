import numpy as np
import porepy as pp
import FTHM_Solver
import scipy.stats as stats
from pprint import pprint
from porepy.applications.md_grids.domains import nd_cube_domain

from porepy.examples.flow_benchmark_2d_case_4 import (
    FlowBenchmark2dCase4Model,
    solid_constants as solid_constants_2d,
)


class StochasticGeometry:
    def set_domain(self) -> None:
        """Set domain of the problem.

        Defaults to a 2d unit square domain.
        Override this method to define a geometry with a different domain.

        """
        self._domain = nd_cube_domain(
            self.params["nd"], self.units.convert_units(1.0, "m")
        )

    def set_fractures(self):
        # Spatial dimension.
        nd = self.params["nd"]
        assert nd in [2, 3], "Only 2D and 3D geometries are supported."

        np.random.seed(self.params.get("seed", 42))

        num_fractures = self.params.get("num_fractures", 10)

        # Generate random fracture orientations. For now, we assume uniform distribution
        # of orientations in the range [0, pi]. We may change this later.
        orientation_distr = stats.uniform(loc=0, scale=np.pi)
        orientation = orientation_distr.rvs(size=num_fractures)

        # Incline from the xy-plane in 3D (EK believes this is what is implemented, but
        # it is not a big problem if there is a slight inconsistency in the logic here).
        # For now, we assume uniform distribution
        incline_distr = stats.uniform(loc=0, scale=np.pi / 2)
        incline = incline_distr.rvs(size=num_fractures)

        # Represent fracture lengths as a log-normal distribution. The parameters set
        # here are quite random.
        length_distr = stats.lognorm(s=1, scale=0.5)
        lengths = length_distr.rvs(size=num_fractures)

        # Generate random fracture centers. For now, we assume uniform distribution
        # of positions in the range [0, 1] in both x and y directions.
        position_distr = stats.uniform(loc=0, scale=1)
        positions_x = position_distr.rvs(size=num_fractures)
        positions_y = position_distr.rvs(size=num_fractures)

        if nd == 3:
            positions_z = position_distr.rvs(size=num_fractures)

        # Create the endpoints of the fractures based on the lengths and orientations.
        fractures = []

        for i in range(num_fractures):
            cx = positions_x[i]
            cy = positions_y[i]

            # Length of the fracture (in 2D) or size of the fracture square (in 3D).
            length = lengths[i]
            # Orientation of the fracture (angle in the xy-plane in 2D or around the
            # z-axis in 3D).
            theta = orientation[i]

            x0 = 0.5 * length
            y0 = 0
            x1 = -0.5 * length
            y1 = 0

            if nd == 2:
                p0 = np.array([x0, y0]).reshape((2, 1))
                p1 = np.array([x1, y1]).reshape((2, 1))

                pc = np.array([cx, cy]).reshape((2, 1))

                R = np.array(
                    [
                        [np.cos(theta), -np.sin(theta)],
                        [np.sin(theta), np.cos(theta)],
                    ]
                )
                p0_rot = pc + R @ p0
                p1_rot = pc + R @ p1

                fractures.append(pp.LineFracture(np.hstack([p0_rot, p1_rot])))
            else:
                z0 = -0.5 * length
                z1 = 0.5 * length
                p0 = np.array([x0, y0, z0]).reshape((3, 1))
                p1 = np.array([x0, y0, z1]).reshape((3, 1))
                p2 = np.array([x1, y1, z1]).reshape((3, 1))
                p3 = np.array([x1, y1, z0]).reshape((3, 1))

                # Rotate first around the z-axis, then around the x-axis.
                R_0 = np.array(
                    [
                        [np.cos(theta), -np.sin(theta), 0],
                        [np.sin(theta), np.cos(theta), 0],
                        [0, 0, 1],
                    ]
                )
                R_1 = np.array(
                    [
                        [1, 0, 0],
                        [0, np.cos(incline[i]), -np.sin(incline[i])],
                        [0, np.sin(incline[i]), np.cos(incline[i])],
                    ]
                )
                pc = np.array([cx, cy, positions_z[i]]).reshape((3, 1))

                p0_rot = pc + R_1 @ (R_0 @ p0)
                p1_rot = pc + R_1 @ (R_0 @ p1)
                p2_rot = pc + R_1 @ (R_0 @ p2)
                p3_rot = pc + R_1 @ (R_0 @ p3)

                fractures.append(
                    pp.PlaneFracture(np.hstack([p0_rot, p1_rot, p2_rot, p3_rot]))
                )

        # Set the fractures in the model.
        self._fractures = fractures

    def grid_type(self):
        return "simplex"


class Source:
    def fluid_source(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        domain = self.domain
        box = domain.bounding_box_from_polytope()

        x_mean = (box["xmax"] + box["xmin"]) / 2
        y_mean = (box["ymax"] + box["ymin"]) / 2
        if self.nd == 3:
            z_mean = (box["zmax"] + box["zmin"]) / 2
        else:
            z_mean = 0

        src_ambient = np.zeros(
            sum(sd.num_cells for sd in subdomains if sd.dim == self.nd)
        )
        src_fracture = np.zeros(
            sum(sd.num_cells for sd in subdomains if sd.dim == self.nd - 1)
        )
        src_intersection = np.zeros(
            sum(sd.num_cells for sd in subdomains if sd.dim < self.nd - 1)
        )

        if len(self._fractures) == 0:
            # Domain without fractures. Put the source in the center of the domain.
            sd = subdomains[0]
            closest_cell = sd.closest_cell(
                np.array([x_mean, y_mean, z_mean]).reshape((3, 1))
            )
            src_ambient[closest_cell] = 1.0
        else:
            x, y, z = np.concatenate(
                [sd.cell_centers for sd in subdomains if sd.dim == self.nd - 1], axis=1
            )
            source_loc = np.argmin(
                (x - x_mean) ** 2 + (y - y_mean) ** 2 + (z - z_mean) ** 2
            )
            src_fracture[source_loc] = 1

        return super().fluid_source(subdomains) + pp.ad.DenseArray(
            np.concatenate([src_ambient, src_fracture, src_intersection])
        )


class ModelProperties:
    def _is_nonlinear_problem(self) -> bool:
        """Check if the model is nonlinear."""
        return False


class StochasticModel(
    StochasticGeometry,
    FTHM_Solver.IterativeSolverMixin,
    Source,
    ModelProperties,
    pp.SinglePhaseFlow,
):
    pass


class BenchmarkModel(
    Source,
    FTHM_Solver.IterativeSolverMixin,
    ModelProperties,
    FlowBenchmark2dCase4Model,
):
    # We may want to look at this at some point, but for now, consider it untested and
    # optional.
    pass


if True:
    model_class = StochasticModel
else:
    model_class = BenchmarkModel


def reset_model_state(model):
    """Reset the model state to a clean state.

    This can be useful to run several simulations with the same model (discretization
    etc.) but, say, with different solver parameters.
    """
    mdg = model.mdg

    num_cells = sum(sd.num_cells for sd in mdg.subdomains())
    num_interface_cells = sum(intf.num_cells for intf in mdg.interfaces())

    zeros = np.zeros(num_cells + num_interface_cells, dtype=float)
    model.equation_system.set_variable_values(
        values=zeros, additive=False, time_step_index=0
    )
    model.equation_system.set_variable_values(
        values=zeros, additive=False, iterate_index=0
    )

    model.time_manager = pp.TimeManager(schedule=[0, 1], dt_init=1, constant_dt=True)


model_params = {
    "solid_constants": solid_constants_2d,
    "linear_solver": {"preconditioner_factory": FTHM_Solver.mass_balance_factory},
    # Control the number of fractures here.
    "num_fractures": 5,
    # This is the simplest way to control the cell size. You can also do
    # 'cell_size_fracture' and 'cell_size_boundary'.
    "meshing_arguments": {"cell_size": 0.1},
    # Control the name of the gmsh file here.
    "meshing_kwargs": {"file_name": "mesh_2d"},
    # Set the spatial dimension.
    "nd": 3,
}


model = model_class(model_params)
model.prepare_simulation()

# To get hold of the fractures, we can do this:
fractures = model.fractures
# Pick out a single fracture, get its endpoints, and print them:
frac = fractures[0]
# Note that the fracture endpoints are stored in a 2D array, where the first column
# contains the x-coordinates and the second column contains the y-coordinates.
print(f"Endpoints of the first fracture: {frac.pts[:, 0]}, {frac.pts[:, 1]}")


# Now, based on an analysis of the fractures, or on feeling lucky today (got it?), we
# can control solver parameters. To see the impact, comment out everything by the
# ksp_monitor line, run again, and see how the number of iterations changes.
linear_solver_opts = {
    # This line PETSc print the residual norm at each iteration.
    "ksp_monitor": None,
    "ksp_type": "gmres",  # Change the Krylov method
    # # Change the preconditioner for interface fluxes. Jacobi is probably a bad choice,
    # # but that is not the point here.
    # "interface_darcy_flux": {"pc_type": "jacobi"},
    # "mass_balance": {"pc_type": "ilu"},
}

model.params["linear_solver"].update({"options": linear_solver_opts})


pp.run_time_dependent_model(model, {"prepare_simulation": False})

# To reset the model state and run the simulation again with the same parameters:
reset_model_state(model)
pp.run_time_dependent_model(model, {"prepare_simulation": False})


# To access statistics on the solver performance:
pprint("Solver statistics:")
pprint(f"Number of Krylov iterations {model._krylov_iters}")
pprint(f"Krylov solver status: {model._petsc_converged_reason}")
pprint(f"Time for solver construction: {model._construction_time} s")
pprint(f"Time for solver solve: {model._solve_time} s")


debug = []
