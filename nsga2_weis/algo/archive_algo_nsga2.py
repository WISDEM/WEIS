from threading import local
import time
from itertools import chain, islice

from nsga2_weis.algo.genetic_functions import (
    binary_tournament_selection,
    polynomial_mutation,
    simulated_binary_crossover,
)
import numpy as np
from numpy.typing import ArrayLike

from mpi4py import MPI

from nsga2_weis.algo.fast_nondom_sort import fast_nondom_sort
from nsga2_weis.algo.crowding_distance_assignment import crowding_distance_assignment


class NSGA2:

    N_DV: int = 0  # number of design variables
    N_constr: int = 0  # number of constraints
    N_obj: int = 0  # number of objective functions
    N_population: int = 0  # number of members in the population

    design_vars_population: ArrayLike = np.array([])  # design variables
    objs_population: ArrayLike = np.array([])  # objective values at DV points
    constrs_population: ArrayLike = np.array([])  # constraint values at DV points
    feasibility_population: ArrayLike = np.array([])  # feasibility values at DV points
    fun_objectives: callable = None  # objective function storage
    fun_constraints: callable = None  # constraint value storage

    # front organization
    idx_fronts: list[list[int]] = None  # list of indices into raw DVs to ID fronts
    design_vars_fronts: list[np.ndarray] = None  # list of views into DVs by front
    objs_fronts: list[np.ndarray] = None  # list of views into objectives by front
    feasibility_fronts: list[bool] = None  # list of feasibility values for fronts

    rate_crossover: float = 0.9  # crossover probability
    eta_c: float = 20  # distribution index
    rate_mutation: float = 0.1  # mutation probability
    eta_m: float = 20  # distribution index

    feasibility_dominates: bool = False  # use feasibility in the sorting process

    comm_mpi: MPI.Comm = None  # MPI communicator for parallel evaluation
    model_mpi: tuple[int, int] = None  # parallelization model: size, color
    # follows the format used by openmdao/openmdao/utils/concurrent_utils.py

    verbose: bool = False  # verbosity switch

    def __init__(
        self,
        design_vars_init: ArrayLike,  # initial population
        fun_objectives: callable,  # callable objective functions: DVs -> objs
        fun_constraints: callable = None,  # callable constraint functions: DVs -> constrs
        N_obj: int = None,  # manual counts of objectives
        N_constr: int = None,  # manual counts of constraints
        design_vars_l: ArrayLike = None,  # the lower bound of the DVs
        design_vars_u: ArrayLike = None,  # the upper bound of the DVs
        params_override=(None, None, None, None),  # override params for NSGA-II
        comm_mpi: MPI.Comm = None,  # communicator for parallel implementation
        model_mpi: tuple[int, int] = None,  # model for spreading work across processes
        verbose: bool = False,  # verbose outputs
    ):
        """
        initialize NSGA2 optimizer and its population

        Parameters
        ----------
        design_vars_init : ArrayLike
            initial population for design varibales
        fun_objectives : callable
            a function that, given any row in design_vars_init, returns the objectives
        fun_constraints : callable
            a function that, given any row in design_vars_init, returns the constraints
        N_obj : int, optional
            the number of objectives for an expensive evaluation function, by default None
        design_vars_l : _type_, optional
            lower bounds on the design variables, if None, defaults to -inf, by default None
        design_vars_u : _type_, optional
            upper bounds on the design variables, if None, defaults to inf, by default None
        """

        # install provided settings
        if params_override[0] is not None:
            self.rate_crossover = params_override[0]
        if params_override[1] is not None:
            self.eta_c = params_override[1]
        if params_override[2] is not None:
            self.rate_mutation = params_override[2]
        if params_override[3] is not None:
            self.eta_m = params_override[3]
        self.comm_mpi = comm_mpi
        self.model_mpi = model_mpi
        self.verbose = verbose

        # take in an initial population of design variables
        design_vars_init = np.atleast_2d(design_vars_init)  # convert to numpy if necessary
        self.N_population, self.N_DV = design_vars_init.shape  # extract sizes
        self.design_vars_population = design_vars_init.copy()  # hold on to a copy

        # set up the objectives: for expensive models, take pre-specified
        # objective count if it's cheap, don't specify and just throw away a
        # feval for sizing
        if N_obj:
            self.N_obj = N_obj
        else:
            self.N_obj = len(fun_objectives(design_vars_init[0]))
        # likewise for constraints
        if N_constr:
            self.N_constr = N_constr
        else:
            self.N_constr = 0 if fun_constraints is None else len(fun_constraints(design_vars_init[0]))
        # use feasibility-based domination when a constraint is passed in
        if self.N_constr:
            self.feasibility_dominates = True

        # initialize objectives to all nan
        self.objs_population = np.nan * np.ones((self.N_population, self.N_obj))
        self.needs_recompute_objs = [True for _ in range(self.N_population)]
        # initialize constraints to all nan
        self.constrs_population = np.nan * np.ones((self.N_population, self.N_constr))
        self.needs_recompute_constrs = [True for _ in range(self.N_population)]
        # initialize feasibility to ones
        self.feasibility_population = np.ones((self.N_population,), dtype=bool)

        self.fun_objectives = fun_objectives  # install a function to evaluate
        self.update_objectives()  # now that there's a function, update values
        if self.N_constr:
            self.fun_constraints = fun_constraints  # install a function to evaluate constraints
            self.update_constraints()  # now that there's a constraint, update values

            self.feasibility_population = np.all(self.constrs_population >= 0.0, axis=1)

        design_vars_l = np.array(design_vars_l)  # convert to np.array
        design_vars_u = np.array(design_vars_u)  # convert to np.array
        # validate the sizes of the design variables
        if design_vars_l.size != self.N_DV:
            raise ValueError(f"Lower bounds size mismatch: expected {self.N_DV}, got {design_vars_l.size}.")
        if design_vars_u.size != self.N_DV:
            raise ValueError(f"Upper bounds size mismatch: expected {self.N_DV}, got {design_vars_u.size}.")
        self.design_vars_l = design_vars_l  # save the lower bounds on design variables
        self.design_vars_u = design_vars_u  # save the upper bounds on design variables

    def update_objectives(self):
        """
        Update the internal objectives.

        Returns
        -------
        np.ndarray
            returns a reference to objs_p
        """

        if np.any(self.needs_recompute_objs):
            # any time there's an update, the front data becomes stale
            self.idx_fronts = None
            self.design_vars_fronts = None
            self.objs_fronts = None
            self.constrs_fronts = None
            self.feasibility_fronts = None

        return self.update_objectives_data(
            self.design_vars_population,
            self.objs_population,
            self.needs_recompute_objs,
        )

    def update_objectives_data(
        self,
        design_vars_p: np.ndarray,
        objs_p: np.ndarray,
        needs_recompute_objs: list[bool],
    ):
        """
        Update in objectives that are flagged for recomputation from a provided
        dataset.

        Parameters
        ----------
        design_vars_p : np.ndarray
            incoming population's design variable specifications
        objs_p : np.ndarray
            array of objective outcomes
        needs_recompute_objs : list[bool]
            list of objective evaluations that need re-computation

        Returns
        -------
        np.ndarray
            returns a reference to objs_p

        Raises
        ------
        ValueError
            if the sizes of DV and objective populations are mismatched
        """

        neval_obj_update = 0
        if self.verbose:
            print("UPDATING OBJECTIVES...", end="", flush=True)
            tm_st = time.time()

        # size and validate the input data
        N_pop, N_DV = design_vars_p.shape
        xN_pop, N_obj = objs_p.shape
        if N_pop != xN_pop:
            raise ValueError(
                f"Dimension mismatch: design_vars_p has {N_pop} individuals, objs_p has {xN_pop} individuals."
            )

        # gather indices that need recomputation
        indices_to_update = [i for i, flag in enumerate(needs_recompute_objs) if flag]
        args_to_eval = [design_vars_p[i, :] for i in indices_to_update]

        # evaluate in batch if possible, else fallback to single
        if args_to_eval:
            if self.comm_mpi is None:
                print(f"DEBUG!!!!! RUNNING ALL CASES IN SERIAL")
                results = [self.fun_objectives(arg) for arg in args_to_eval]
            else:
                # distribute the evaluation across MPI processes
                comm = self.comm_mpi
                rank = comm.rank

                if self.model_mpi is not None:  # i.e.: paralleization model is specified
                    size, color = self.model_mpi
                    local_args = islice(args_to_eval, color, None, size)  # slice by color
                else:
                    size = comm.size
                    local_args = islice(args_to_eval, rank, None, size)  # slice by rank

                # scatter the indices to all processes
                local_results = [self.fun_objectives(arg) for arg in local_args]

                # debug print!
                print(f"DEBUG!!!!! model_mpi IS {'' if self.model_mpi is None else 'NOT '}NONE")
                print(f"DEBUG!!!!! ON RANK: {rank} OF {size}, ran {len(local_results)} cases")

                # gather all results at root
                gathered_results = comm.gather(local_results, root=0)

                # flatten the gathered results on rank 0
                if rank == 0:
                    results = [item for sublist in gathered_results for item in sublist]
                else:
                    results = None

                # broadcast results to all processes
                results = comm.bcast(results, root=0)

            # assign results across processors
            for idx, res in zip(indices_to_update, results):
                objs_p[idx, :] = res
                needs_recompute_objs[idx] = False
                neval_obj_update += 1

        if self.verbose:
            tm_end = time.time()
            print(
                f" DONE. USING {neval_obj_update} OBJECTIVE FUNCTION CALLS IN {tm_end-tm_st:.4f}s.",
                flush=True,
            )

        return objs_p

    def update_constraints(self):
        """
        Update the internal constraints.

        Returns
        -------
        np.ndarray
            returns a reference to constrs_p

        Raises
        ------
        ValueError
            if this is called even though there are no constraints
        """

        if not self.N_constr:
            # raise error because this was called when there's no constraints
            raise ValueError("Cannot update constraints when there are no constraints defined.")

        # pass the data to the constraint updater
        rv = self.update_constraints_data(
            self.design_vars_population,
            self.constrs_population,
            self.needs_recompute_constrs,
        )

        # recompute the feasibility
        self.feasibility_population = np.all(self.constrs_population >= 0.0, axis=1)

        return rv

    def update_constraints_data(
        self,
        design_vars_p: np.ndarray,
        constrs_p: np.ndarray,
        needs_recompute_constrs: list[bool],
    ):
        """
        Update in constraints that are flagged for recomputation from a provided
        dataset.

        Parameters
        ----------
        design_vars_p : np.ndarray
            incoming population's design variable specifications
        constrs_p : np.ndarray
            array of constraint outcomes
        needs_recompute_constrs : list[bool]
            list of constraint evaluations that need re-computation

        Returns
        -------
        np.ndarray
            returns a reference to constrs_p

        Raises
        ------
        ValueError
            if the sizes of DV and constraint populations are mismatched
        """

        neval_constr_update = 0
        if self.verbose:
            print("UPDATING CONSTRAINTS...", end="", flush=True)
            tm_st = time.time()

        # size and validate the input data
        N_pop, N_DV = design_vars_p.shape
        xN_pop, N_constr = constrs_p.shape
        if N_pop != xN_pop:
            raise ValueError(
                f"Dimension mismatch: design_vars_p has {N_pop} individuals, " f"constrs_p has {xN_pop} individuals."
            )

        # gather indices that need recomputation
        indices_to_update = [i for i, flag in enumerate(needs_recompute_constrs) if flag]
        args_to_eval = [design_vars_p[i, :] for i in indices_to_update]

        # evaluate in batch if possible, else fallback to single
        if args_to_eval:
            if self.comm_mpi is None:
                print(f"DEBUG!!!!! RUNNING ALL CASES IN SERIAL")
                results = [self.fun_constraints(arg) for arg in args_to_eval]
            else:
                # distribute the evaluation across MPI processes
                comm = self.comm_mpi
                rank = comm.rank

                if self.model_mpi is not None:  # i.e.: paralleization model is specified
                    size, color = self.model_mpi
                    local_args = islice(args_to_eval, color, None, size)  # slice by color
                else:
                    size = comm.size
                    local_args = islice(args_to_eval, rank, None, size)  # slice by rank

                # scatter the indices to all processes
                local_results = [self.fun_constraints(arg) for arg in local_args]

                # debug print!
                print(f"DEBUG!!!!! model_mpi IS {'' if self.model_mpi is None else 'NOT '}NONE")
                print(f"DEBUG!!!!! ON RANK: {rank} OF {size}, ran {len(local_results)} cases")

                # gather all results at root
                gathered_results = comm.gather(local_results, root=0)

                # flatten the gathered results on rank 0
                if rank == 0:
                    results = [item for sublist in gathered_results for item in sublist]
                else:
                    results = None

                # broadcast results to all processes
                results = comm.bcast(results, root=0)

            # assign results across processors
            for idx, res in zip(indices_to_update, results):
                constrs_p[idx, :] = res
                needs_recompute_constrs[idx] = False
                neval_constr_update += 1

        if self.verbose:
            tm_end = time.time()
            print(
                f" DONE. USING {neval_constr_update} CONSTRAINT FUNCTION CALLS IN {tm_end-tm_st:.4f}s.",
                flush=True,
            )

        return constrs_p

    def compute_fronts(self):
        """
        Get the non-dominated front map for the current population.

        This function computes the non-dominated fronts from the optimizer's
        internal design variables (design_vars_population) and objective values
        (objs_population). It updates the objectives if necessary and returns the
        indices of the fronts.

        Returns
        -------
        list[list[int]]
            list of maps into self.design_vars_population and self.objs_population
            representing each front
        """

        # pass internal data to the external front computation pieces
        self.idx_fronts = self.compute_fronts_data(
            design_vars_in=self.design_vars_population,
            objs_in=self.objs_population,
            needs_recompute_objs=self.needs_recompute_objs,
            constrs_in=self.constrs_population,
            needs_recompute_constrs=self.needs_recompute_constrs,
            feasibility_dominates=self.feasibility_dominates,
        )

        return self.idx_fronts

    def compute_fronts_data(
        self,
        design_vars_in: np.ndarray,
        objs_in: np.ndarray,
        needs_recompute_objs: list[bool],
        constrs_in: np.ndarray = None,
        needs_recompute_constrs: list[bool] = None,
        feasibility_dominates: bool = False,
    ):
        """
        Compute the non-dominated fronts data from the provided data.

        This function coputes the non-dominated fronts from the provided design
        variables (design_vars_in) and objective values (objs_in). It updates the
        objectives if necessary and returns the indices of the fronts.

        Parameters
        ----------
        design_vars_in : np.ndarray
            population design varible input
        objs_in : np.ndarray
            population objective input
        needs_recompute_objs : list[bool]
            flags for recomputation of the individual objectives

        Returns
        -------
        list[list[int]]
            indices of the non-dominated fronts in the population
        """

        # first, update any objectives
        self.update_objectives_data(design_vars_in, objs_in, needs_recompute_objs)
        if (constrs_in is not None) and feasibility_dominates:
            self.update_constraints_data(design_vars_in, constrs_in, needs_recompute_constrs)

        # create a list to pass to the sorting algorithm
        objs_tosort = list(map(tuple, objs_in))
        # TODO: figure out how to pass the map object and still have it work

        if self.verbose:  # if verbose, print and start timer
            print("COMPUTING THE PARETO FRONTS...", end="", flush=True)
            tm_st = time.time()
        # do the fast, non-dominated sort algorithm to sort
        idx_fronts = fast_nondom_sort(objs_tosort)  # indices of the fronts
        if self.verbose:  # if verbose, stop timer and print
            tm_end = time.time()
            print(f" DONE. TIME: {tm_end-tm_st:.4f}s", flush=True)

        if (constrs_in is not None) and feasibility_dominates:
            if needs_recompute_constrs is None:
                needs_recompute_constrs = needs_recompute_objs
            self.update_constraints_data(design_vars_in, constrs_in, needs_recompute_constrs)

            feasibility_fronts = [constrs_in[idx_dv, :] >= 0.0 for idx_dv in idx_fronts]

            idx_fronts_feasible = []
            idx_fronts_infeasible = []

            for idx_f, f in enumerate(idx_fronts):
                is_feasible = np.all(feasibility_fronts[idx_f], axis=1)
                points_added = 0
                if np.sum(is_feasible):
                    idx_fronts_feasible.append(np.array(f)[is_feasible].tolist())
                    points_added += len(idx_fronts_feasible[-1])
                if np.sum(~is_feasible):
                    idx_fronts_infeasible.append(np.array(f)[~is_feasible].tolist())
                    points_added += len(idx_fronts_infeasible[-1])
                assert len(f) == points_added

            idx_fronts = idx_fronts_feasible + idx_fronts_infeasible

        return idx_fronts

    def get_fronts(
        self,
        compute_design_vars: bool = True,
        compute_objs: bool = True,
        compute_constrs: bool = True,
        feasibility_dominates: bool = False,
    ):
        """
        Get the non-dominated fronts from the current population.

        This function computes the non-dominated fronts from the optimizer's
        internal design variables (design_vars_population) and objective values
        (objs_population). It updates the objectives if necessary and returns the
        indices of the fronts, along with the corresponding design variables and
        objective values if requested.

        Parameters
        ----------
        compute_design_vars : bool, optional
            should the design var. fronts be computed and returned, by default True
        compute_objs : bool, optional
            should the objective fronts be computed and returned, by default True

        Returns
        -------
        list[list[int]]
            indices of the non-dominated fronts in the population
        list[np.ndarray], optional
            design variables for the fronts, if compute_design_vars is True
        list[np.ndarray], optional
            objective values for the fronts, if compute_objs is True
        """

        values_return = self.get_fronts_data(
            self.design_vars_population,
            self.objs_population,
            self.needs_recompute_objs,
            self.idx_fronts,
            compute_design_vars,
            compute_objs,
            constrs_in=self.constrs_population,
            needs_recompute_constrs=self.needs_recompute_objs,
            compute_constrs=compute_constrs,
            feasibility_dominates=feasibility_dominates,
        )

        # deal with the return values
        rvi = 0
        self.idx_fronts = values_return[rvi]
        rvi += 1
        if compute_design_vars:
            self.design_vars_fronts = values_return[rvi]
            rvi += 1
        if compute_objs:
            self.objs_fronts = values_return[rvi]
            rvi += 1
        if compute_constrs:
            self.constrs_fronts = values_return[rvi]
            rvi += 1

        return values_return

    def get_fronts_data(
        self,
        design_vars_in: np.ndarray,
        objs_in: np.ndarray,
        needs_recompute_objs: list[bool],
        idx_fronts_in: list[list[int]] = None,
        compute_design_vars: bool = True,
        compute_objs: bool = True,
        constrs_in: np.ndarray = None,
        compute_constrs: bool = False,
        needs_recompute_constrs: list[bool] = None,
        feasibility_dominates: bool = False,
    ):
        """
        Get the non-dominated front data from the provided data.

        This function returns the data for the non-dominated fronts from the
        provided design variables (design_vars_in) and objective values (objs_in). It
        updates the objectives if necessary and returns the indices of the
        fronts, along with the corresponding design variables and objective
        values if requested.

        Parameters
        ----------
        design_vars_in : np.ndarray
            population design varible input
        objs_in : np.ndarray
            population objective input
        needs_recompute_objs : list[bool]
            flags for recomputation of the individual objectives
        compute_design_vars : bool, optional
            should the design var. fronts be computed and returned, by default True
        compute_objs : bool, optional
            should the objective fronts be computed and returned, by default True

        Returns
        -------
        list[list[int]]
            indices of the non-dominated fronts in the population
        list[np.ndarray], optional
            design variables for the fronts, if compute_design_vars is True
        list[np.ndarray], optional
            objective values for the fronts, if compute_objs is True
        """

        # compute the fronts data
        idx_fronts = (
            self.compute_fronts_data(
                design_vars_in,
                objs_in,
                needs_recompute_objs,
                constrs_in=constrs_in,
                needs_recompute_constrs=needs_recompute_constrs,
                feasibility_dominates=feasibility_dominates,
            )
            if not idx_fronts_in
            else idx_fronts_in
        )

        # front re-computation
        design_vars_fronts = []
        objs_fronts = []
        constrs_fronts = []
        for idx_f, f in enumerate(idx_fronts):
            if compute_design_vars:
                design_vars_f = design_vars_in[f, :]  # slice index in to create views
                design_vars_fronts.append(design_vars_f)
            if compute_objs:
                objs_f = objs_in[f, :]  # slice index in to create views
                objs_fronts.append(objs_f)
            if compute_constrs:
                if constrs_in is None:
                    raise ValueError("Cannot compute constraints fronts without constrs_in being provided.")
                constrs_f = constrs_in[f, :]  # slice index in to create views
                constrs_fronts.append(constrs_f)

        # compile returns and ship
        to_return = [idx_fronts]
        if compute_design_vars:
            to_return.append([np.array(v) for v in design_vars_fronts])
        if compute_objs:
            to_return.append([np.array(v) for v in objs_fronts])
        if compute_constrs:
            to_return.append([np.array(v) for v in constrs_fronts])

        return tuple(to_return)

    def get_Nfronts(self):
        """get the number of non-dominated fronts we're dealing with"""
        if self.idx_fronts is None:
            return 0  # no data, no fronts
        return len(self.idx_fronts)

    def get_crowding_distance_data(
        self,
        objs_front_in: list[np.ndarray],
    ):

        # return the crowding distances for an input set of fronts

        # get the front
        if self.verbose:
            print("COMPUTING CROWDING DISTANCE...", end="", flush=True)
            tm_st = time.time()
        D_front = [crowding_distance_assignment(f) for f in objs_front_in]
        if self.verbose:
            tm_end = time.time()
            print(f" DONE. TIME: {tm_end-tm_st:.4f}s", flush=True)

        # return the crowding distances
        return D_front

    def get_rank(
        self,
        local: bool = False,
    ):
        return self.get_rank_data(
            objs_front_in=self.objs_fronts,
            constrs_front_in=self.constrs_fronts,
            local=local,
        )

    def get_rank_data(
        self,
        objs_front_in: np.ndarray,
        constrs_front_in: np.ndarray = None,
        local: bool = False,
    ):

        if constrs_front_in:
            # get the feasiblity values
            feasible_constraintwise_front = [f >= 0.0 for f in constrs_front_in]
            feasible_front = [np.all(f, axis=1) for f in feasible_constraintwise_front]
            # print(f"DEBUG!!!!! constrs_front_in: {constrs_front_in}")
            # print(
            #     f"DEBUG!!!!! feasible_constraintwise_front: {feasible_constraintwise_front}"
            # )
            print(f"DEBUG!!!!! feasible_front: {feasible_front}")

        # get the global/local ranking of the points
        D_front = self.get_crowding_distance_data(objs_front_in)

        count_front = ([0] + [len(f) for f in D_front])[:-1]  # get the front sizes
        localR_front = [np.argsort(-D).tolist() for D in D_front]  # sort on crowding distance within fronts

        if local:
            if constrs_front_in:
                # reorder s.t. feasible solutions come first, followed by infeasible
                # solutions, each subset in the order of localR_front
                localR_front = [
                    np.hstack(
                        (
                            np.array(localR_front[i])[feasible_front[i]],
                            np.array(localR_front[i])[~feasible_front[i]],
                        )
                    ).tolist()
                    for i in range(len(localR_front))
                ]
            return localR_front  # we're done if we just want intra-front ranking

        # continue on and return global ranking
        R_front = [(R + np.cumsum(count_front)[i]).tolist() for i, R in enumerate(localR_front)]
        if constrs_front_in:
            # reorder s.t. feasible solutions come first, followed by infeasible
            # solutions, each subset in the order of R_front
            R_front = [
                np.hstack(
                    (
                        np.array(R_front[i])[feasible_front[i]],
                        np.array(R_front[i])[~feasible_front[i]],
                    )
                ).tolist()
                for i in range(len(R_front))
            ]
        return R_front

    def sort_data(self):
        """
        re-sort the raw data so it's in rank order
        """

        # get the data
        _, design_vars_fronts, objs_fronts = self.get_fronts(
            feasibility_dominates=True if self.N_constr else False,
        )  # get the fronts
        D_fronts = self.get_crowding_distance_data(self.objs_fronts)  # crowding distances
        idx_argsort_D = [np.argsort(Df)[::-1] for Df in D_fronts]  # get argsort in the front

        # new placeholders
        idx_fronts_new = []
        design_vars_fronts_new = []
        objs_fronts_new = []

        for idx_f, argsort_Df in enumerate(idx_argsort_D):  # loop over the fronts
            # populate our new fronts
            idx_fronts_new.append(list(range(len(argsort_Df))))
            design_vars_fronts_new.append(design_vars_fronts[idx_f][argsort_Df, :])
            objs_fronts_new.append(objs_fronts[idx_f][argsort_Df, :])

        # store all the results
        self.design_vars_population = np.vstack(design_vars_fronts_new)
        self.objs_population = np.vstack(objs_fronts_new)
        self.idx_fronts = idx_fronts_new
        self.design_vars_fronts = design_vars_fronts_new
        self.objs_fronts = objs_fronts_new

        # sanity check the re-sorting
        if not np.allclose(self.design_vars_population.shape, np.array([self.N_population, self.N_DV])):
            raise ValueError(
                f"new design_vars_population size mismatch: expected ({self.N_population}, {self.N_DV}), got {self.design_vars_population.shape}."
            )
        if not np.allclose(self.objs_population.shape, np.array([self.N_population, self.N_obj])):
            raise ValueError(
                f"new objs_population size mismatch: expected ({self.N_population}, {self.N_obj}), got {self.objs_population.shape}."
            )

    def get_binary_tournament_selection(self, ratio_keep=0.5):
        """run a binary tournament selection"""

        rank = np.hstack(self.get_rank())  # get the overall ranking
        # run, return binary tournament selection on ranking
        idx_select = binary_tournament_selection(rank, ratio_keep=ratio_keep)
        return idx_select

    def _get_default_limits(self):
        design_vars_l = (
            -np.inf * np.ones_like(self.design_vars_raw[0]) if (self.design_vars_l is None) else self.design_vars_l
        )
        design_vars_u = (
            np.inf * np.ones_like(self.design_vars_raw[0]) if (self.design_vars_u is None) else self.design_vars_u
        )
        return design_vars_l, design_vars_u

    def propose_new_generation(self):

        # get the limits
        design_vars_l, design_vars_u = self._get_default_limits()  # get the default limits

        # use the selection to generate a new (elitist) population
        idx_selection = self.get_binary_tournament_selection(ratio_keep=1.0)

        # create new proposed populations
        design_vars_proposal = self.design_vars_population[idx_selection, :].copy()
        objs_proposal = self.objs_population[idx_selection, :].copy()

        # array of changes
        changed = np.array([False for _ in design_vars_proposal])

        # now, try a crossover
        design_vars_proposal, changed_crossover = simulated_binary_crossover(
            design_vars_proposal,
            design_vars_l=design_vars_l,
            design_vars_u=design_vars_u,
            rate_crossover=self.rate_crossover,
            eta_c=self.eta_c,
        )
        assert len(design_vars_proposal) == self.N_population
        changed = np.logical_or(changed, changed_crossover)

        # now, do a mutation
        design_vars_proposal, changed_mutation = polynomial_mutation(
            design_vars_proposal,
            design_vars_l=design_vars_l,
            design_vars_u=design_vars_u,
            rate_mutation=self.rate_mutation,
            eta_m=self.eta_m,
        )
        assert len(design_vars_proposal) == self.N_population
        changed = np.logical_or(changed, changed_mutation)

        # update the objectives that have changed
        self.update_objectives_data(design_vars_proposal, objs_proposal, needs_recompute_objs=changed)

        return design_vars_proposal, objs_proposal, changed  # return the results

    def iterate_population(self):

        # get previous, proposed next populations
        design_vars_prev, objs_prev = self.design_vars_population, self.objs_population
        design_vars_next, objs_next, changed_next = self.propose_new_generation()

        # combine the populations and compute the fronts
        design_vars_combo = np.vstack([design_vars_prev, design_vars_next])
        objs_combo = np.vstack([objs_prev, objs_next])
        changed_combo = np.hstack([self.needs_recompute_objs, changed_next])

        # compute the fronts of the combined dataset
        idx_fronts, design_vars_fronts, objs_fronts = self.get_fronts_data(design_vars_combo, objs_combo, changed_combo)
        R_fronts = self.get_rank_data(objs_fronts, local=True)

        # new data
        self.design_vars_population = []
        self.objs_population = []
        self.idx_fronts = []

        idx_counter = 0  # count how many we get in there
        for idx_f, f in enumerate(R_fronts):
            if idx_counter >= self.N_population:
                break
            self.idx_fronts.append([])  # add a new front to the map
            for idx_v in f:  # for each index in the front
                if idx_counter >= self.N_population:
                    break
                self.design_vars_population.append(design_vars_fronts[idx_f][idx_v])  # add to the re-sort
                self.objs_population.append(objs_fronts[idx_f][idx_v])  # add to the re-sort
                self.idx_fronts[idx_f].append(idx_counter)  # put the new index in the map
                idx_counter += 1  # increment counter

        if len(self.design_vars_population) != self.N_population:
            raise ValueError(
                f"Population size mismatch: expected {self.N_population}, got {len(self.design_vars_population)}"
            )
        self.design_vars_population = np.array(self.design_vars_population)
        self.objs_population = np.array(self.objs_population)
