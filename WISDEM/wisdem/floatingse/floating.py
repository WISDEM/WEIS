import numpy as np
import openmdao.api as om
from wisdem.floatingse.member import Member
from wisdem.floatingse.map_mooring import MapMooring
from wisdem.floatingse.floating_frame import FloatingFrame

# from wisdem.floatingse.substructure import Substructure, SubstructureGeometry


class FloatingSE(om.Group):
    def initialize(self):
        self.options.declare("modeling_options")

    def setup(self):
        opt = self.options["modeling_options"]

        # self.set_input_defaults("mooring_type", "chain")
        # self.set_input_defaults("anchor_type", "SUCTIONPILE")
        # self.set_input_defaults("loading", "hydrostatic")
        # self.set_input_defaults("wave_period_range_low", 2.0, units="s")
        # self.set_input_defaults("wave_period_range_high", 20.0, units="s")
        # self.set_input_defaults("cd_usr", -1.0)
        # self.set_input_defaults("zref", 100.0)
        # self.set_input_defaults("number_of_offset_columns", 0)
        # self.set_input_defaults("material_names", ["steel"])

        n_member = opt["floating"]["members"]["n_members"]
        mem_prom = [
            "E_mat",
            "G_mat",
            "sigma_y_mat",
            "rho_mat",
            "rho_water",
            "unit_cost_mat",
            "material_names",
            "painting_cost_rate",
            "labor_cost_rate",
        ]
        # mem_prom += ["Uref", "zref", "shearExp", "z0", "cd_usr", "cm", "beta_wind", "rho_air", "mu_air", "beta_water",
        #            "rho_water", "mu_water", "Uc", "Hsig_wave","Tsig_wave","rho_water","water_depth"]
        for k in range(n_member):
            self.add_subsystem(
                "member" + str(k),
                Member(column_options=opt["floating"]["members"], idx=k, n_mat=opt["materials"]["n_mat"]),
                promotes=mem_prom,
            )

        # Next run MapMooring
        self.add_subsystem(
            "mm", MapMooring(options=opt["mooring"], gamma=opt["WISDEM"]["FloatingSE"]["gamma_f"]), promotes=["*"]
        )

        # Add in the connecting truss
        self.add_subsystem("load", FloatingFrame(modeling_options=opt), promotes=["*"])

        # Evaluate system constraints
        # self.add_subsystem("cons", FloatingConstraints(modeling_options=opt), promotes=["*"])

        # Connect all input variables from all models
        mem_vars = [
            "nodes_xyz",
            "nodes_r",
            "transition_node",
            "section_D",
            "section_t",
            "section_A",
            "section_Asx",
            "section_Asy",
            "section_Ixx",
            "section_Iyy",
            "section_Izz",
            "section_rho",
            "section_E",
            "section_G",
            "idx_cb",
            "buoyancy_force",
            "displacement",
            "center_of_buoyancy",
            "center_of_mass",
            "total_mass",
            "total_cost",
            "Awater",
            "Iwater",
            "added_mass",
        ]
        for k in range(n_member):
            for var in mem_vars:
                self.connect("member" + str(k) + "." + var, "member" + str(k) + ":" + var)

        """
        self.connect("max_offset_restoring_force", "mooring_surge_restoring_force")
        self.connect("operational_heel_restoring_force", "mooring_pitch_restoring_force")
        """
