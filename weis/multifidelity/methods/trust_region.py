# base method class
import numpy as np
from scipy.optimize import minimize, basinhopping
import matplotlib.pyplot as plt
from collections import OrderedDict
from weis.multifidelity.models.testbed_components import (
    simple_2D_high_model,
    simple_2D_low_model,
)
from weis.multifidelity.methods.base_method import BaseMethod
fontsize = 18
plt.rcParams["font.size"] = fontsize


class SimpleTrustRegion(BaseMethod):
    """
    An implementation of a simple trust-region method, following
    the literature as presented in multiple places, but especially
    Andrew March's dissertation.

    Attributes
    ----------
    max_trust_radius : float
        Maximum size of the trust region radius. Not scaled, so this value
        is in real dimensioned values of the design space.
    eta : float
        Value to compare the ratio of actual reduction to predicted reduction
        of the objective value. A ratio higher than eta expands the trust region
        whereas a value lower than eta contracts the trust region.
    radius_tol : float
        Lower limit for the trust region radius for convergence criteria.
    trust_radius : float
        The current value of the trust region radius in dimensioned units.
    """

    def __init__(
        self,
        model_low,
        model_high,
        bounds,
        disp=True,
        num_initial_points=5,
        max_trust_radius=1000.0,
        eta=0.25,
        radius_tol=1e-6,
        trust_radius=0.2,
        expansion_ratio=2.0,
        contraction_ratio=0.25,
    ):
        """
        Initialize the trust region method and store the user-defined options.

        Parameters
        ----------
        max_trust_radius : float
            Maximum size of the trust region radius. Not scaled, so this value
            is in real dimensioned values of the design space.
        eta : float
            Value to compare the ratio of actual reduction to predicted reduction
            of the objective value. A ratio higher than eta expands the trust region
            whereas a value lower than eta contracts the trust region.
        radius_tol : float
            Tolerance of the gradient for convergence criteria. If the norm
            of the gradient at the current design point is less than radius_tol,
            terminate the trust region method. Currently not implemented.
        trust_radius : float
            The current value of the trust region radius in dimensioned units.
        expansion_ratio : float
            The scalar value multiplied to the trust region size if expanding.
        contraction_ratio : float
            The scalar value multiplied to the trust region size if contracting.


        """
        super().__init__(model_low, model_high, bounds, disp, num_initial_points)

        self.max_trust_radius = max_trust_radius
        self.eta = eta
        self.radius_tol = radius_tol
        self.trust_radius = trust_radius
        self.expansion_ratio = expansion_ratio
        self.contraction_ratio = contraction_ratio

    def find_next_point(self, num_basinhop_iterations):
        """
        Find the design point corresponding to the minimum value of the
        corrected low-fidelity model within the trust region.

        This method uses the most recent design point in design_vectors as the initial
        point for the local optimization.

        Returns
        -------
        x_new : array
            The design point corresponding to the minimum value of the corrected
            low-fidelity model within the trust region.
        hits_boundary : boolean
            True is the new design point hits one of the boundaries of the
            trust region.
        """
        x0 = self.design_vectors[-1, :]

        # min (m_k(x_k + s_k)) st ||x_k|| <= del K
        trust_region_lower_bounds = x0 - self.trust_radius
        lower_bounds = np.maximum(trust_region_lower_bounds, self.bounds[:, 0])
        trust_region_upper_bounds = x0 + self.trust_radius
        upper_bounds = np.minimum(trust_region_upper_bounds, self.bounds[:, 1])

        bounds = list(zip(lower_bounds, upper_bounds))
        scaled_function = lambda x: self.objective_scaler * np.squeeze(
            self.approximation_functions[self.objective](x)
        )

        if num_basinhop_iterations:
            minimizer_kwargs = {
                "method": "SLSQP",
                "tol": 1e-10,
                "bounds": bounds,
                "constraints": self.list_of_constraints,
                "options": {"disp": self.disp == 2, "maxiter": 20},
            }
            res = basinhopping(
                scaled_function,
                x0,
                stepsize=np.mean(upper_bounds - lower_bounds) * 0.5,
                niter=num_basinhop_iterations,
                disp=self.disp == 2,
                minimizer_kwargs=minimizer_kwargs,
            )
        else:
            res = minimize(
                scaled_function,
                x0,
                method="SLSQP",
                tol=1e-10,
                bounds=bounds,
                constraints=self.list_of_constraints,
                options={"disp": self.disp == 2, "maxiter": 20},
            )
        x_new = res.x

        tol = 1e-6
        if np.any(np.abs(trust_region_lower_bounds - x_new) < tol) or np.any(
            np.abs(trust_region_upper_bounds - x_new) < tol
        ):
            hits_boundary = True
        else:
            hits_boundary = False

        return x_new, hits_boundary

    def update_trust_region(self, x_new, hits_boundary):
        """
        Either expand or contract the trust region radius based on the
        value of the high-fidelity function at the proposed design point.

        Modifies `design_vectors` and `trust_radius` in-place.

        Parameters
        ----------
        x_new : array
            The design point corresponding to the minimum value of the corrected
            low-fidelity model within the trust region.
        hits_boundary : boolean
            True is the new design point hits one of the boundaries of the
            trust region.

        """
        # 3. Compute the ratio of actual improvement to predicted improvement
        prev_point_high = (
            self.objective_scaler
            * self.model_high.run(self.design_vectors[-1])[self.objective]
        )
        new_point_high = (
            self.objective_scaler * self.model_high.run(x_new)[self.objective]
        )
        prev_point_approx = self.objective_scaler * self.approximation_functions[
            self.objective
        ](self.design_vectors[-1])
        new_point_approx = self.objective_scaler * self.approximation_functions[
            self.objective
        ](x_new)

        actual_reduction = prev_point_high - new_point_high
        predicted_reduction = prev_point_approx - new_point_approx

        if predicted_reduction == 0.0:
            rho = 0.0
        else:
            rho = actual_reduction / predicted_reduction

        # 4. Accept or reject the trial point according to that ratio
        # Unclear if this logic is needed; it's better to update the surrogate model with a bad point, even.
        # Removed this logic for now, see git blame for this line to retrive it.
        self.design_vectors = np.vstack((self.design_vectors, np.atleast_2d(x_new)))

        # 5. Update trust region according to rho_k
        if (
            rho >= self.eta and hits_boundary
        ):  # unclear if we need this hits_boundary logic
            self.trust_radius = min(
                self.expansion_ratio * self.trust_radius, self.max_trust_radius
            )
        elif rho < self.eta:  # Unclear if this is the best check
            self.trust_radius *= self.contraction_ratio

        if self.disp:
            print()
            print("Predicted reduction:", np.squeeze(predicted_reduction))
            print("Actual reduction:", actual_reduction)
            print("Rho", np.squeeze(rho))
            print("Trust radius:", self.trust_radius)
            print("Best func value:", new_point_high)

    def optimize(self, plot=False, num_iterations=100, num_basinhop_iterations=False, interp_method="smt"):
        """
        Actually perform the trust-region optimization.

        Parameters
        ----------
        plot : boolean
            If True, plot a 2d representation of the optimization process.
        num_iterations : int
            Number of trust region iterations.

        """
        self.construct_approximations(interp_method=interp_method)
        self.process_constraints()

        if plot:
            self.plot_functions()

        for i in range(num_iterations):
            self.process_constraints()
            x_new, hits_boundary = self.find_next_point(num_basinhop_iterations)

            self.update_trust_region(x_new, hits_boundary)

            self.construct_approximations(interp_method=interp_method)

            if plot:
                self.plot_functions()

            if self.trust_radius <= self.radius_tol:
                break

        results = self.process_results()

        return results

    def plot_functions(self):
        """
        Plot a 2D contour plot of the design space and optimizer progress.

        Saves figures to .png files.

        """

        print("Plotting!")

        if self.n_dims == 2:
            n_plot = 21
            x_plot = np.linspace(self.bounds[0, 0], self.bounds[0, 1], n_plot)
            y_plot = np.linspace(self.bounds[1, 0], self.bounds[1, 1], n_plot)
            X, Y = np.meshgrid(x_plot, y_plot)
            x_values = np.vstack((X.flatten(), Y.flatten())).T

            y_plot_high = self.model_high.run_vec(x_values)[self.objective].reshape(
                n_plot, n_plot
            )

            surrogate = []
            for x_value in x_values:
                surrogate.append(
                    np.squeeze(self.approximation_functions[self.objective](x_value))
                )
            surrogate = np.array(surrogate)
            y_plot_high = surrogate.reshape(n_plot, n_plot)

            fig = plt.figure(figsize=(7.05, 5))
            contour = plt.contourf(X, Y, y_plot_high, levels=201)
            plt.scatter(
                self.design_vectors[:, 0], self.design_vectors[:, 1], color="white"
            )
            ax = plt.gca()
            ax.set_aspect("equal", "box")

            cbar = fig.colorbar(contour)
            cbar.ax.set_ylabel("Value")
            # ticks = np.round(np.linspace(0.305, 0.48286, 6), 3)
            # cbar.set_ticks(ticks)
            # cbar.set_ticklabels(ticks)

            x = self.design_vectors[-1, 0]
            y = self.design_vectors[-1, 1]
            points = np.array(
                [
                    [x + self.trust_radius, y + self.trust_radius],
                    [x + self.trust_radius, y - self.trust_radius],
                    [x - self.trust_radius, y - self.trust_radius],
                    [x - self.trust_radius, y + self.trust_radius],
                    [x + self.trust_radius, y + self.trust_radius],
                ]
            )
            plt.plot(points[:, 0], points[:, 1], "w--")

            plt.xlim(self.bounds[0])
            plt.ylim(self.bounds[1])

            plt.xlabel("DV #1")
            plt.ylabel("DV #2")

            plt.tight_layout()

            num_iter = self.design_vectors.shape[0]
            num_offset = 10

            plt.show()

            # if num_iter <= 5:
            #     for i in range(num_offset):
            #         plt.savefig(
            #             f"image_{self.counter_plot}.png", dpi=300, bbox_inches="tight"
            #         )
            #         self.counter_plot += 1
            # else:
            #     plt.savefig(
            #         f"image_{self.counter_plot}.png", dpi=300, bbox_inches="tight"
            #     )
            #     self.counter_plot += 1

            plt.close()
            if self.disp:
                print("Saved figure")

        else:
            import niceplots

            x_full = np.atleast_2d(np.linspace(0.0, 1.0, 101)).T
            squeezed_x = np.squeeze(x_full)
            y_full = squeezed_x.copy()
            for i, x_val in enumerate(squeezed_x):
                y_full[i] = self.approximation_functions[self.objective](
                    np.atleast_2d(x_val.T)
                )

            y_full_high = self.model_high.run_vec(x_full)[self.objective]
            y_full_low = self.model_low.run_vec(x_full)[self.objective]

            y_low = self.model_low.run_vec(self.design_vectors)[self.objective]
            y_high = self.model_high.run_vec(self.design_vectors)[self.objective]

            plt.figure()

            lw = 4
            s = 100
            plt.plot(squeezed_x, y_full_low, label="low-fidelity", c="tab:green", lw=lw)
            plt.scatter(self.design_vectors, y_low, c="tab:green", s=s)

            plt.plot(squeezed_x, y_full_high, label="high-fidelity", c="tab:orange", lw=lw)
            plt.scatter(self.design_vectors, y_high, c="tab:orange", s=s)

            plt.plot(squeezed_x, np.squeeze(y_full), label="surrogate", c="tab:blue", lw=lw)
            
            # x = self.design_vectors[-1, 0]
            # y_plot = y_high[-1]
            # y_diff = 0.5
            # x_lb = max(x - self.trust_radius, self.bounds[0, 0])
            # x_ub = min(x + self.trust_radius, self.bounds[0, 1])
            # points = np.array(
            #     [
            #         [x_lb, y_plot],
            #         [x_ub, y_plot],
            #     ]
            # )
            # plt.plot(points[:, 0], points[:, 1], "-", color="gray", clip_on=False)
            # 
            # points = np.array(
            #     [
            #         [x_lb, y_plot - y_diff],
            #         [x_lb, y_plot + y_diff],
            #     ]
            # )
            # plt.plot(points[:, 0], points[:, 1], "-", color="gray", clip_on=False)
            # 
            # points = np.array(
            #     [
            #         [x_ub, y_plot - y_diff],
            #         [x_ub, y_plot + y_diff],
            #     ]
            # )
            # plt.plot(points[:, 0], points[:, 1], "-", color="gray", clip_on=False)

            plt.xlim(self.bounds[0])
            plt.ylim([-11, 10])

            plt.xlabel("x")
            plt.ylabel("y", rotation=0)

            ax = plt.gca()
            ax.text(s="Low-fidelity", x=0.1, y=2.5, c="tab:green", fontsize=fontsize)
            ax.text(s="High-fidelity", x=0.2, y=-11., c="tab:orange", fontsize=fontsize)
            ax.text(
                s="Corrected low-fidelity", x=0.45, y=-8.0, c="tab:blue", fontsize=fontsize
            )

            niceplots.adjust_spines(outward=True)

            plt.tight_layout()

            plt.savefig(f"1d_{self.counter_plot}.png", dpi=300, bbox_inches="tight")
            self.counter_plot += 1
