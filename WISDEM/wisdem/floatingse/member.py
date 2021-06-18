import copy

import numpy as np
import openmdao.api as om
import wisdem.commonse.frustum as frustum
import wisdem.commonse.utilities as util
import wisdem.commonse.manufacturing as manufacture
import wisdem.commonse.cross_sections as cs
from wisdem.commonse import eps, gravity
from sortedcontainers import SortedDict
from wisdem.commonse.wind_wave_drag import CylinderEnvironment
from wisdem.commonse.utilization_constraints import GeometricConstraints

NULL = -9999
MEMMAX = 200
NREFINE = 1


class CrossSection(object):
    def __init__(
        self, D=0.0, t=0.0, A=0.0, Asx=0.0, Asy=0.0, Ixx=0.0, Iyy=0.0, Izz=0.0, E=0.0, G=0.0, rho=0.0, sigy=0.0
    ):
        self.D, self.t = D, t  # Needed for OpenFAST
        self.A, self.Asx, self.Asy = A, Asx, Asy
        self.Ixx, self.Iyy, self.Izz = Ixx, Iyy, Izz
        self.E, self.G, self.rho, self.sigy = E, G, rho, sigy


def get_nfull(npts, nref=NREFINE):
    n_full = int(1 + nref * (npts - 1))
    return n_full


def I_cyl(r_i, r_o, h, m):
    if type(r_i) == type(np.array([])):
        n = r_i.size
        r_i = r_i.flatten()
        r_o = r_o.flatten()
    else:
        n = 1
    Ixx = Iyy = (m / 12.0) * (3.0 * (r_i ** 2.0 + r_o ** 2.0) + h ** 2.0)
    Izz = 0.5 * m * (r_i ** 2.0 + r_o ** 2.0)
    return np.c_[Ixx, Iyy, Izz, np.zeros((n, 3))]


class DiscretizationYAML(om.ExplicitComponent):

    """
    Convert the YAML inputs into more native and easy to use variables.

    Parameters
    ----------
    s : numpy array[n_height_tow]
        1D array of the non-dimensional grid defined along the member axis (0-member base,
        1-member top)
    joint1 : numpy array[3], [m]
        Global dimensional coordinates (x-y-z) for bottom node of member
    joint2 : numpy array[3], [m]
        Global dimensional coordinates (x-y-z) for top node of member
    layer_materials : list of strings
        1D array of the names of the materials of each layer modeled in the member
        structure.
    layer_thickness : numpy array[n_layers, n_height], [m]
        2D array of the thickness of the layers of the member structure. The first
        dimension represents each layer, the second dimension represents each piecewise-
        constant entry of the member sections.
    outer_diameter_in : numpy array[n_height_tow], [m]
        cylinder diameter at corresponding locations
    material_names : list of strings
        1D array of names of materials.
    E_mat : numpy array[n_mat, 3], [Pa]
        2D array of the Youngs moduli of the materials. Each row represents a material,
        the three members represent E11, E22 and E33.
    G_mat : numpy array[n_mat, 3], [Pa]
        2D array of the shear moduli of the materials. Each row represents a material,
        the three members represent G12, G13 and G23.
    sigma_y_mat : numpy array[n_mat], [Pa]
        2D array of the yield strength of the materials. Each row represents a material,
        the three members represent Xt12, Xt13 and Xt23.
    rho_mat : numpy array[n_mat], [kg/m**3]
        1D array of the density of the materials. For composites, this is the density of
        the laminate.
    unit_cost_mat : numpy array[n_mat], [USD/kg]
        1D array of the unit costs of the materials.
    outfitting_factor_in : float
        Multiplier that accounts for secondary structure mass inside of member
    rho_water : float, [kg/m**3]
        density of water

    Returns
    -------
    height : float, [m]
        Scalar of the member height computed along the z axis.
    section_height : numpy array[n_height-1], [m]
        parameterized section heights along cylinder
    outer_diameter : numpy array[n_height], [m]
        cylinder diameter at corresponding locations
    wall_thickness : numpy array[n_height-1], [m]
        shell thickness at corresponding locations
    E : numpy array[n_height-1], [Pa]
        Isotropic Youngs modulus of the materials along the member sections.
    G : numpy array[n_height-1], [Pa]
        Isotropic shear modulus of the materials along the member sections.
    sigma_y : numpy array[n_height-1], [Pa]
        Isotropic yield strength of the materials along the member sections.
    rho : numpy array[n_height-1], [kg/m**3]
        Density of the materials along the member sections.
    unit_cost : numpy array[n_height-1], [USD/kg]
        Unit costs of the materials along the member sections.
    outfitting_factor : numpy array[n_height-1]
        Additional outfitting multiplier in each section

    """

    def initialize(self):
        self.options.declare("options")
        self.options.declare("idx")
        self.options.declare("n_mat")

    def setup(self):
        n_mat = self.options["n_mat"]
        opt = self.options["options"]
        idx = self.options["idx"]
        n_height = opt["n_height"][idx]
        n_layers = opt["n_layers"][idx]
        n_ballast = opt["n_ballasts"][idx]

        # TODO: Use reference axis and curvature, s, instead of assuming everything is vertical on z
        self.add_input("s", val=np.zeros(n_height))
        self.add_input("joint1", val=np.zeros(3), units="m")
        self.add_input("joint2", val=np.zeros(3), units="m")
        self.add_discrete_input("layer_materials", val=n_layers * [""])
        self.add_discrete_input("ballast_materials", val=n_ballast * [""])
        self.add_input("layer_thickness", val=np.zeros((n_layers, n_height)), units="m")
        self.add_input("outer_diameter_in", np.zeros(n_height), units="m")
        self.add_discrete_input("material_names", val=n_mat * [""])
        self.add_input("E_mat", val=np.zeros([n_mat, 3]), units="Pa")
        self.add_input("G_mat", val=np.zeros([n_mat, 3]), units="Pa")
        self.add_input("sigma_y_mat", val=np.zeros(n_mat), units="Pa")
        self.add_input("rho_mat", val=np.zeros(n_mat), units="kg/m**3")
        self.add_input("unit_cost_mat", val=np.zeros(n_mat), units="USD/kg")
        self.add_input("outfitting_factor_in", val=1.0)
        self.add_input("rho_water", 0.0, units="kg/m**3")

        self.add_output("height", val=0.0, units="m")
        self.add_output("section_height", val=np.zeros(n_height - 1), units="m")
        self.add_output("outer_diameter", val=np.zeros(n_height), units="m")
        self.add_output("wall_thickness", val=np.zeros(n_height - 1), units="m")
        self.add_output("E", val=np.zeros(n_height - 1), units="Pa")
        self.add_output("G", val=np.zeros(n_height - 1), units="Pa")
        self.add_output("sigma_y", val=np.zeros(n_height - 1), units="Pa")
        self.add_output("rho", val=np.zeros(n_height - 1), units="kg/m**3")
        self.add_output("unit_cost", val=np.zeros(n_height - 1), units="USD/kg")
        self.add_output("outfitting_factor", val=np.ones(n_height - 1))
        self.add_output("ballast_density", val=np.zeros(n_ballast), units="kg/m**3")
        self.add_output("ballast_unit_cost", val=np.zeros(n_ballast), units="USD/kg")

        # Distributed Beam Properties (properties needed for ElastoDyn (OpenFAST) inputs or BModes inputs for verification purposes)
        self.add_output("z_param", np.zeros(n_height), units="m")
        self.add_output("sec_loc", np.zeros(n_height - 1), desc="normalized sectional location")
        self.add_output("str_tw", np.zeros(n_height - 1), units="deg", desc="structural twist of section")
        self.add_output("tw_iner", np.zeros(n_height - 1), units="deg", desc="inertial twist of section")
        self.add_output("mass_den", np.zeros(n_height - 1), units="kg/m", desc="sectional mass per unit length")
        self.add_output(
            "foreaft_iner",
            np.zeros(n_height - 1),
            units="kg*m",
            desc="sectional fore-aft intertia per unit length about the Y_G inertia axis",
        )
        self.add_output(
            "sideside_iner",
            np.zeros(n_height - 1),
            units="kg*m",
            desc="sectional side-side intertia per unit length about the Y_G inertia axis",
        )
        self.add_output(
            "foreaft_stff",
            np.zeros(n_height - 1),
            units="N*m**2",
            desc="sectional fore-aft bending stiffness per unit length about the Y_E elastic axis",
        )
        self.add_output(
            "sideside_stff",
            np.zeros(n_height - 1),
            units="N*m**2",
            desc="sectional side-side bending stiffness per unit length about the Y_E elastic axis",
        )
        self.add_output("tor_stff", np.zeros(n_height - 1), units="N*m**2", desc="sectional torsional stiffness")
        self.add_output("axial_stff", np.zeros(n_height - 1), units="N", desc="sectional axial stiffness")
        self.add_output("cg_offst", np.zeros(n_height - 1), units="m", desc="offset from the sectional center of mass")
        self.add_output("sc_offst", np.zeros(n_height - 1), units="m", desc="offset from the sectional shear center")
        self.add_output("tc_offst", np.zeros(n_height - 1), units="m", desc="offset from the sectional tension center")

    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
        # Unpack dimensions
        opt = self.options["options"]
        idx = self.options["idx"]
        n_height = opt["n_height"][idx]
        n_ballast = opt["n_ballasts"][idx]

        # Unpack values
        xyz0 = inputs["joint1"]
        xyz1 = inputs["joint2"]
        dxyz = xyz1 - xyz0
        h_col = np.sqrt(np.sum(dxyz ** 2))
        lthick = inputs["layer_thickness"]
        lthick = 0.5 * (lthick[:, :-1] + lthick[:, 1:])

        outputs["height"] = h_col
        outputs["section_height"] = np.diff(h_col * inputs["s"])
        outputs["wall_thickness"] = np.sum(lthick, axis=0)
        outputs["outer_diameter"] = inputs["outer_diameter_in"]
        outputs["outfitting_factor"] = inputs["outfitting_factor_in"] * np.ones(n_height - 1)
        twall = lthick

        # Check to make sure we have good values
        if np.any(outputs["section_height"] <= 0.0):
            raise ValueError("Section height values must be greater than zero, " + str(outputs["section_height"]))
        if np.any(outputs["wall_thickness"] <= 0.0):
            raise ValueError("Wall thickness values must be greater than zero, " + str(outputs["wall_thickness"]))
        if np.any(outputs["outer_diameter"] <= 0.0):
            raise ValueError("Diameter values must be greater than zero, " + str(outputs["outer_diameter"]))

        # DETERMINE MATERIAL PROPERTIES IN EACH SECTION
        # Convert to isotropic material
        E = np.mean(inputs["E_mat"], axis=1)
        G = np.mean(inputs["G_mat"], axis=1)
        sigy = inputs["sigma_y_mat"]
        rho = inputs["rho_mat"]
        cost = inputs["unit_cost_mat"]
        mat_names = discrete_inputs["material_names"]

        # Initialize sectional data
        E_param = np.zeros(twall.shape)
        G_param = np.zeros(twall.shape)
        sigy_param = np.zeros(twall.shape)
        rho_param = np.zeros(n_height - 1)
        cost_param = np.zeros(n_height - 1)

        # Loop over materials and associate it with its thickness
        layer_mat = discrete_inputs["layer_materials"]
        for k in range(len(layer_mat)):
            # Get the material name for this layer
            iname = layer_mat[k]

            # Get the index into the material list
            imat = mat_names.index(iname)

            imass = rho[imat] * twall[k, :]

            # For density, take mass weighted layer
            rho_param += imass

            # For cost, take mass weighted layer
            cost_param += imass * cost[imat]

            # Store the value associated with this thickness
            E_param[k, :] = E[imat]
            G_param[k, :] = G[imat]
            sigy_param[k, :] = sigy[imat]

        # Mass weighted cost (should really weight by radius too)
        cost_param /= rho_param

        # Thickness weighted density (should really weight by radius too)
        rho_param /= twall.sum(axis=0)

        # Mixtures of material properties: https://en.wikipedia.org/wiki/Rule_of_mixtures

        # Volume fraction
        vol_frac = twall / twall.sum(axis=0)[np.newaxis, :]

        # Average of upper and lower bounds
        E_param = 0.5 * np.sum(vol_frac * E_param, axis=0) + 0.5 / np.sum(vol_frac / E_param, axis=0)
        G_param = 0.5 * np.sum(vol_frac * G_param, axis=0) + 0.5 / np.sum(vol_frac / G_param, axis=0)
        sigy_param = 0.5 * np.sum(vol_frac * sigy_param, axis=0) + 0.5 / np.sum(vol_frac / sigy_param, axis=0)

        # Store values
        outputs["E"] = E_param
        outputs["G"] = G_param
        outputs["rho"] = rho_param
        outputs["sigma_y"] = sigy_param
        outputs["unit_cost"] = cost_param

        # Unpack for Elastodyn
        z_param = min(xyz0[2], xyz1[2]) + (h_col * inputs["s"])
        z = 0.5 * (z_param[:-1] + z_param[1:])
        D, _ = util.nodal2sectional(outputs["outer_diameter"])
        itube = cs.Tube(D, outputs["wall_thickness"])
        Az, Ixx, Iyy, Jz = itube.Area, itube.Jxx, itube.Jyy, itube.J0
        outputs["z_param"] = z_param
        outputs["sec_loc"] = 0.0 if len(z) == 1 else (z - z[0]) / (z[-1] - z[0])
        outputs["mass_den"] = rho_param * Az
        outputs["foreaft_iner"] = rho_param * Ixx
        outputs["sideside_iner"] = rho_param * Iyy
        outputs["foreaft_stff"] = E_param * Ixx
        outputs["sideside_stff"] = E_param * Iyy
        outputs["tor_stff"] = G_param * Jz
        outputs["axial_stff"] = E_param * Az

        # Loop over materials and associate it with its thickness
        rho_ballast = np.zeros(n_ballast)
        cost_ballast = np.zeros(n_ballast)
        ballast_mat = discrete_inputs["ballast_materials"]
        for k in range(n_ballast):
            # Get the material name for this layer
            iname = ballast_mat[k]

            if iname.find("water") >= 0 or iname == "":
                rho_ballast[k] = inputs["rho_water"]
                continue

            # Get the index into the material list
            imat = mat_names.index(iname)

            # Store values
            rho_ballast[k] = rho[imat]
            cost_ballast[k] = cost[imat]

        outputs["ballast_density"] = rho_ballast
        outputs["ballast_unit_cost"] = cost_ballast


class MemberDiscretization(om.ExplicitComponent):
    """
    Discretize geometry into finite element nodes

    Parameters
    ----------
    s : numpy array[n_height_tow]
        1D array of the non-dimensional grid defined along the member axis (0-member base,
        1-member top)
    outer_diameter : numpy array[n_height], [m]
        cylinder diameter at corresponding locations
    wall_thickness : numpy array[n_height-1], [m]
        shell thickness at corresponding locations
    E : numpy array[n_height-1], [Pa]
        Isotropic Youngs modulus of the materials along the member sections.
    G : numpy array[n_height-1], [Pa]
        Isotropic shear modulus of the materials along the member sections.
    sigma_y : numpy array[n_height-1], [Pa]
        Isotropic yield strength of the materials along the member sections.
    rho : numpy array[n_height-1], [kg/m**3]
        Density of the materials along the member sections.
    unit_cost : numpy array[n_height-1], [USD/kg]
        Unit costs of the materials along the member sections.
    outfitting_factor : numpy array[n_height-1]
        Additional outfitting multiplier in each section

    Returns
    -------
    s_full : numpy array[n_full]
        non-dimensional locations along member
    z_full : numpy array[n_full], [m]
        dimensional locations along member axis
    d_full : numpy array[n_full], [m]
        cylinder diameter at corresponding locations
    t_full : numpy array[n_full-1], [m]
        shell thickness at corresponding locations
    E_full : numpy array[n_full-1], [Pa]
        Isotropic Youngs modulus of the materials along the member sections.
    G_full : numpy array[n_full-1], [Pa]
        Isotropic shear modulus of the materials along the member sections.
    sigma_y_full : numpy array[n_full-1], [Pa]
        Isotropic yield strength of the materials along the member sections.
    rho_full : numpy array[n_full-1], [kg/m**3]
        Density of the materials along the member sections.
    unit_cost_full : numpy array[n_full-1], [USD/kg]
        Unit costs of the materials along the member sections.
    nu_full : numpy array[n_full-1]
        Poisson's ratio assuming isotropic material
    outfitting_full : numpy array[n_full-1]
        Additional outfitting multiplier in each section

    """

    """discretize geometry into finite element nodes"""

    def initialize(self):
        self.options.declare("n_height")
        self.options.declare("n_refine", default=NREFINE)

    def setup(self):
        n_height = self.options["n_height"]
        n_full = get_nfull(n_height, nref=self.options["n_refine"])

        self.add_input("s", val=np.zeros(n_height))
        self.add_input("height", val=0.0, units="m")
        self.add_input("outer_diameter", np.zeros(n_height), units="m")
        self.add_input("wall_thickness", np.zeros(n_height - 1), units="m")
        self.add_input("E", val=np.zeros(n_height - 1), units="Pa")
        self.add_input("G", val=np.zeros(n_height - 1), units="Pa")
        self.add_input("sigma_y", val=np.zeros(n_height - 1), units="Pa")
        self.add_input("rho", val=np.zeros(n_height - 1), units="kg/m**3")
        self.add_input("unit_cost", val=np.zeros(n_height - 1), units="USD/kg")
        self.add_input("outfitting_factor", val=np.ones(n_height - 1))

        self.add_output("s_full", np.zeros(n_full), units="m")
        self.add_output("z_full", np.zeros(n_full), units="m")
        self.add_output("d_full", np.zeros(n_full), units="m")
        self.add_output("t_full", np.zeros(n_full - 1), units="m")
        self.add_output("E_full", val=np.zeros(n_full - 1), units="Pa")
        self.add_output("G_full", val=np.zeros(n_full - 1), units="Pa")
        self.add_output("nu_full", val=np.zeros(n_full - 1))
        self.add_output("sigma_y_full", val=np.zeros(n_full - 1), units="Pa")
        self.add_output("rho_full", val=np.zeros(n_full - 1), units="kg/m**3")
        self.add_output("unit_cost_full", val=np.zeros(n_full - 1), units="USD/kg")
        self.add_output("outfitting_full", val=np.ones(n_full - 1))

        # self.declare_partials('*', '*', method='fd', form='central', step=1e-6)

    def compute(self, inputs, outputs):
        # Unpack inputs
        s_param = inputs["s"]
        n_refine = int(np.round(self.options["n_refine"]))

        # TODO: Put these somewhere
        # Create constraint output that draft is less than water depth
        # outputs["draft_margin"] = draft / inputs["max_draft"]

        # Make sure freeboard is more than 20% of Hsig_wave (DNV-OS-J101)
        # outputs["wave_height_freeboard_ratio"] = inputs["Hsig_wave"] / (np.abs(freeboard) + eps)

        # Have to regine each element one at a time so that we preserve input nodes
        s_full = np.array([])
        for k in range(s_param.size - 1):
            sref = np.linspace(s_param[k], s_param[k + 1], n_refine + 1)
            s_full = np.append(s_full, sref)
        s_full = np.unique(s_full)
        s_section = 0.5 * (s_full[:-1] + s_full[1:])

        # Assuming straight (non-curved) members, set dimensional z along the axis
        outputs["s_full"] = s_full
        outputs["z_full"] = s_full * inputs["height"]

        # All other parameters
        outputs["d_full"] = np.interp(s_full, s_param, inputs["outer_diameter"])
        outputs["t_full"] = util.sectionalInterp(s_section, s_param, inputs["wall_thickness"])
        outputs["rho_full"] = util.sectionalInterp(s_section, s_param, inputs["rho"])
        outputs["E_full"] = util.sectionalInterp(s_section, s_param, inputs["E"])
        outputs["G_full"] = util.sectionalInterp(s_section, s_param, inputs["G"])
        outputs["sigma_y_full"] = util.sectionalInterp(s_section, s_param, inputs["sigma_y"])
        outputs["unit_cost_full"] = util.sectionalInterp(s_section, s_param, inputs["unit_cost"])
        outputs["outfitting_full"] = util.sectionalInterp(s_section, s_param, inputs["outfitting_factor"])
        outputs["nu_full"] = 0.5 * outputs["E_full"] / outputs["G_full"] - 1.0


class MemberComponent(om.ExplicitComponent):
    """
    Convert the YAML inputs into more native and easy to use variables.

    Parameters
    ----------
    joint1 : numpy array[3], [m]
        Global dimensional coordinates (x-y-z) for bottom node of member
    joint2 : numpy array[3], [m]
        Global dimensional coordinates (x-y-z) for top node of member
    grid_axial_joints : numpy array[n_axial]
        non-dimensional locations along member for named axial joints
    height : float, [m]
        Scalar of the member height computed along the z axis.
    s_full : numpy array[n_full]
        non-dimensional locations along member
    z_full : numpy array[n_full], [m]
        dimensional locations along member axis
    d_full : numpy array[n_full], [m]
        cylinder diameter at corresponding locations
    t_full : numpy array[n_full-1], [m]
        shell thickness at corresponding locations
    E_full : numpy array[n_full-1], [Pa]
        Isotropic Youngs modulus of the materials along the member sections.
    G_full : numpy array[n_full-1], [Pa]
        Isotropic shear modulus of the materials along the member sections.
    rho_full : numpy array[n_full-1], [kg/m**3]
        Density of the materials along the member sections.
    unit_cost_full : numpy array[n_full-1], [USD/kg]
        Unit costs of the materials along the member sections.
    outfitting_full : numpy array[n_full-1]
        Additional outfitting multiplier in each section
    labor_cost_rate : float, [USD/min]
        Labor cost rate
    painting_cost_rate : float, [USD/m/m]
        Painting / surface finishing cost rate
    bulkhead_grid : numpy array[n_bulk]
        Non-dimensional locations of the bulkheads
    bulkhead_thickness : numpy array[n_bulk], [m]
        Thickness of the bulkheads at the gridded locations
    ring_stiffener_web_height : float, [m]
        height of stiffener web (base of T)
    ring_stiffener_web_thickness : float, [m]
        thickness of stiffener web (base of T)
    ring_stiffener_flange_width : float, [m]
        height of stiffener flange (top of T)
    ring_stiffener_flange_thickness : float, [m]
        thickness of stiffener flange (top of T)
    ring_stiffener_spacing : float, [m]
        Axial distance from one ring stiffener to another
    axial_stiffener_web_height : float, [m]
        height of stiffener web (base of T)
    axial_stiffener_web_thickness : float, [m]
        thickness of stiffener web (base of T)
    axial_stiffener_flange_width : float, [m]
        height of stiffener flange (top of T)
    axial_stiffener_flange_thickness : float, [m]
        thickness of stiffener flange (top of T)
    axial_stiffener_spacing : float, [rad]
        Angular distance from one axial stiffener to another
    ballast_grid : numpy array[n_ballast,2]
        Non-dimensional start and end points for each ballast segment
    ballast_density : numpy array[n_ballast], [kg/m**3]
        density of ballast material
    ballast_volume : numpy array[n_ballast], [m**3]
        Volume of ballast segments.  Should be non-zero for permanent ballast, zero for variable ballast
    ballast_unit_cost : numpy array[n_ballast], [USD/kg]
        Cost per unit mass of ballast

    Returns
    -------
    shell_cost : float, [USD]
        Outer shell cost
    shell_mass : float, [kg]
        Outer shell mass
    shell_z_cg : float, [m]
        z-position of center of mass of member shell
    shell_I_base : numpy array[6], [kg*m**2]
        mass moment of inertia of shell about base [xx yy zz xy xz yz]
    bulkhead_mass : float, [kg]
        mass of column bulkheads
    bulkhead_z_cg : float, [m]
        z-coordinate of center of gravity for all bulkheads
    bulkhead_cost : float, [USD]
        cost of column bulkheads
    bulkhead_I_base : numpy array[6], [kg*m**2]
        Moments of inertia of bulkheads relative to keel point
    stiffener_mass : float, [kg]
        mass of column stiffeners
    stiffener_cost : float, [USD]
        cost of column stiffeners
    stiffener_z_cg : float, [m]
        z-coordinate of center of gravity for all ring stiffeners
    stiffener_I_base : numpy array[6], [kg*m**2]
        Moments of inertia of stiffeners relative to base point
    flange_spacing_ratio : float
        ratio between flange and stiffener spacing
    stiffener_radius_ratio : numpy array[n_full-1]
        ratio between stiffener height and radius
    constr_flange_compactness : float
        Standard check on ring stiffener flange geometry and material properties from API (<1)
    constr_web_compactness : float
        Standard check on ring stiffener web geometry and material properties from API (<1)
    ballast_cost : float, [USD]
        cost of permanent ballast
    ballast_mass : float, [kg]
        mass of permanent ballast
    ballast_z_cg : float, [m]
        z-coordinate or permanent ballast center of gravity
    ballast_height : numpy array[n_ballast]
        Non-dimensional s-coordinate height of each ballast section. Only non-zero here for permanent ballast.
    ballast_I_base : numpy array[6], [kg*m**2]
        Moments of inertia of permanent ballast relative to bottom point
    variable_ballast_capacity : float, [m**3]
        inner radius of column at potential ballast mass
    constr_ballast_capacity : numpy array[n_ballast]
        Used ballast volume relative to total capacity, must be <= 1.0
    total_mass : float, [kg]
        Total mass of member, including permanent ballast, but without variable ballast
    total_cost : float, [USD]
        Total cost of member, including permanent ballast
    structural_mass : float, [kg]
        Total structural mass of member, which does NOT include ballast
    structural_cost : float, [USD]
        Total structural cost of member, which does NOT include ballast
    z_cg : float, [m]
        z-coordinate of center of gravity for the complete member, including permanent ballast but not variable ballast
    I_total : numpy array[6], [kg*m**2]
        Moments of inertia of member at the center of mass
    s_all : numpy array[npts]
        Final non-dimensional points of all internal member nodes
    center_of_mass : numpy array[3], [m]
        Global dimensional coordinates (x-y-z) for member center of mass / gravity
    nodes_xyz : numpy array[npts,3], [m]
        Global dimensional coordinates (x-y-z) for all member nodes in s_all output
    section_D : numpy array[npts-1], [m]
        Cross-sectional diameter of all member segments
    section_t : numpy array[npts-1], [m]
        Cross-sectional effective thickness of all member segments
    section_A : numpy array[npts-1], [m**2]
        Cross-sectional area of all member segments
    section_Asx : numpy array[npts-1], [m**2]
        Cross-sectional shear area in x-direction (member c.s.) of all member segments
    section_Asy : numpy array[npts-1], [m**2]
        Cross-sectional shear area in y-direction (member c.s.) of all member segments
    section_Ixx : numpy array[npts-1], [kg*m**2]
        Cross-sectional moment of inertia about x-axis in member c.s. of all member segments
    section_Iyy : numpy array[npts-1], [kg*m**2]
        Cross-sectional moment of inertia about y-axis in member c.s. of all member segments
    section_Izz : numpy array[npts-1], [kg*m**2]
        Cross-sectional moment of inertia about z-axis in member c.s. of all member segments
    section_rho : numpy array[npts-1], [kg/m**3]
        Cross-sectional density of all member segments
    section_E : numpy array[npts-1], [Pa]
        Cross-sectional Young's modulus (of elasticity) of all member segments
    section_G : numpy array[npts-1], [Pa]
        Cross-sectional shear modulus all member segments
    section_sigma_y : numpy array[npts-1], [Pa]
        Cross-sectional yield stress of all member segments

    """

    def initialize(self):
        self.options.declare("options")
        self.options.declare("idx")
        self.options.declare("n_refine", default=NREFINE)

    def setup(self):
        opt = self.options["options"]
        idx = self.options["idx"]
        n_height = opt["n_height"][idx]
        n_full = get_nfull(n_height, nref=self.options["n_refine"])
        n_axial = opt["n_axial_joints"][idx]
        n_bulk = opt["n_bulkheads"][idx]
        n_ball = opt["n_ballasts"][idx]

        # Inputs
        self.add_input("joint1", val=np.zeros(3), units="m")
        self.add_input("joint2", val=np.zeros(3), units="m")
        self.add_input("height", val=0.0, units="m")
        self.add_input("s_full", np.zeros(n_full), units="m")
        self.add_input("z_full", np.zeros(n_full), units="m")
        self.add_input("d_full", np.zeros(n_full), units="m")
        self.add_input("t_full", np.zeros(n_full - 1), units="m")
        self.add_input("E_full", val=np.zeros(n_full - 1), units="Pa")
        self.add_input("G_full", val=np.zeros(n_full - 1), units="Pa")
        self.add_input("rho_full", val=np.zeros(n_full - 1), units="kg/m**3")
        self.add_input("sigma_y_full", val=np.zeros(n_full - 1), units="Pa")
        self.add_input("unit_cost_full", val=np.zeros(n_full - 1), units="USD/kg")
        self.add_input("outfitting_full", val=np.ones(n_full - 1))
        self.add_input("labor_cost_rate", 0.0, units="USD/min")
        self.add_input("painting_cost_rate", 0.0, units="USD/m/m")

        self.add_input("grid_axial_joints", np.zeros(n_axial))

        self.add_input("bulkhead_grid", np.zeros(n_bulk))
        self.add_input("bulkhead_thickness", np.zeros(n_bulk), units="m")

        self.add_input("ring_stiffener_web_height", 0.0, units="m")
        self.add_input("ring_stiffener_web_thickness", 0.0, units="m")
        self.add_input("ring_stiffener_flange_width", 1e-6, units="m")
        self.add_input("ring_stiffener_flange_thickness", 0.0, units="m")
        self.add_input("ring_stiffener_spacing", 1000.0, units="m")

        self.add_input("axial_stiffener_web_height", 0.0, units="m")
        self.add_input("axial_stiffener_web_thickness", 0.0, units="m")
        self.add_input("axial_stiffener_flange_width", 1e-6, units="m")
        self.add_input("axial_stiffener_flange_thickness", 0.0, units="m")
        self.add_input("axial_stiffener_spacing", 0.0, units="rad")

        self.add_input("ballast_grid", np.zeros((n_ball, 2)))
        self.add_input("ballast_density", np.zeros(n_ball), units="kg/m**3")
        self.add_input("ballast_volume", np.zeros(n_ball), units="m**3")
        self.add_input("ballast_unit_cost", np.zeros(n_ball), units="USD/kg")

        self.add_input("s_ghost1", 0.0)
        self.add_input("s_ghost2", 1.0)

        # Outputs
        self.add_output("shell_cost", val=0.0, units="USD")
        self.add_output("shell_mass", val=0.0, units="kg")
        self.add_output("shell_z_cg", val=0.0, units="m")
        self.add_output("shell_I_base", np.zeros(6), units="kg*m**2")

        self.add_output("bulkhead_mass", 0.0, units="kg")
        self.add_output("bulkhead_z_cg", 0.0, units="m")
        self.add_output("bulkhead_cost", 0.0, units="USD")
        self.add_output("bulkhead_I_base", np.zeros(6), units="kg*m**2")

        self.add_output("stiffener_mass", 0.0, units="kg")
        self.add_output("stiffener_z_cg", 0.0, units="m")
        self.add_output("stiffener_cost", 0.0, units="USD")
        self.add_output("stiffener_I_base", np.zeros(6), units="kg*m**2")
        self.add_output("flange_spacing_ratio", 0.0)
        self.add_output("stiffener_radius_ratio", NULL * np.ones(MEMMAX))
        self.add_output("constr_flange_compactness", 0.0)
        self.add_output("constr_web_compactness", 0.0)

        self.add_output("ballast_cost", 0.0, units="USD")
        self.add_output("ballast_mass", 0.0, units="kg")
        self.add_output("ballast_height", np.zeros(n_ball))
        self.add_output("ballast_z_cg", 0.0, units="m")
        self.add_output("ballast_I_base", np.zeros(6), units="kg*m**2")
        self.add_output("variable_ballast_capacity", 0.0, units="m**3")
        self.add_output("variable_ballast_Vpts", val=np.zeros(10), units="m**3")
        self.add_output("variable_ballast_spts", val=np.zeros(10))
        self.add_output("constr_ballast_capacity", np.zeros(n_ball))

        self.add_output("total_mass", 0.0, units="kg")
        self.add_output("total_cost", 0.0, units="USD")
        self.add_output("structural_mass", 0.0, units="kg")
        self.add_output("structural_cost", 0.0, units="USD")
        self.add_output("z_cg", 0.0, units="m")
        self.add_output("I_total", np.zeros(6), units="kg*m**2")

        self.add_output("s_all", NULL * np.ones(MEMMAX))
        self.add_output("center_of_mass", np.zeros(3), units="m")
        self.add_output("nodes_r", np.zeros(MEMMAX), units="m")
        self.add_output("nodes_xyz", NULL * np.ones((MEMMAX, 3)), units="m")
        self.add_output("section_D", NULL * np.ones(MEMMAX), units="m")
        self.add_output("section_t", NULL * np.ones(MEMMAX), units="m")
        self.add_output("section_A", NULL * np.ones(MEMMAX), units="m**2")
        self.add_output("section_Asx", NULL * np.ones(MEMMAX), units="m**2")
        self.add_output("section_Asy", NULL * np.ones(MEMMAX), units="m**2")
        self.add_output("section_Ixx", NULL * np.ones(MEMMAX), units="kg*m**2")
        self.add_output("section_Iyy", NULL * np.ones(MEMMAX), units="kg*m**2")
        self.add_output("section_Izz", NULL * np.ones(MEMMAX), units="kg*m**2")
        self.add_output("section_rho", NULL * np.ones(MEMMAX), units="kg/m**3")
        self.add_output("section_E", NULL * np.ones(MEMMAX), units="Pa")
        self.add_output("section_G", NULL * np.ones(MEMMAX), units="Pa")
        self.add_output("section_sigma_y", NULL * np.ones(MEMMAX), units="Pa")

    def add_node(self, s_new):
        # Quit if node already exists
        if s_new in self.sections:
            # print('Node already exists,',s_new)
            return

        # Find section we will be interrupting
        idx = self.sections.bisect_left(s_new) - 1
        if idx < 0:
            raise ValueError("Cannot insert node before start of list")

        keys_orig = self.sections.keys()
        self.sections[s_new] = copy.copy(self.sections[keys_orig[idx]])

    def insert_section(self, s0, s1, cross_section0):

        idx0 = self.sections.bisect_left(s0)
        idx1 = self.sections.bisect_left(s1)
        keys_orig = self.sections.keys()

        # Be sure to add new node with old section before adding new section
        self.add_node(s1)

        # If we are straddling an old point, have to convert that to the new section
        if idx0 != idx1:
            self.sections[keys_orig[idx0]] = cross_section0

        # Add new section
        # if s0 in self.sections:
        #    print('Node already exists,',s0)
        self.sections[s0] = cross_section0

    def add_section(self, s0, s1, cross_section0):
        self.sections[s0] = cross_section0
        self.sections[s1] = None

    def compute(self, inputs, outputs):
        # Initialize dictionary that will keep our member nodes so we can convert to OpenFAST format
        self.sections = SortedDict()

        self.add_main_sections(inputs, outputs)
        self.add_bulkhead_sections(inputs, outputs)
        self.add_ring_stiffener_sections(inputs, outputs)
        self.add_ballast_sections(inputs, outputs)
        self.compute_mass_properties(inputs, outputs)
        self.nodal_discretization(inputs, outputs)

    def add_main_sections(self, inputs, outputs):
        # Unpack inputs
        s_full = inputs["s_full"]
        t_full = inputs["t_full"]
        d_full = inputs["d_full"]
        rho = inputs["rho_full"]
        Emat = inputs["E_full"]
        Gmat = inputs["G_full"]
        sigymat = inputs["sigma_y_full"]
        coeff = inputs["outfitting_full"]

        t_web = inputs["axial_stiffener_web_thickness"]
        t_flange = inputs["axial_stiffener_flange_thickness"]
        h_web = inputs["axial_stiffener_web_height"]
        w_flange = inputs["axial_stiffener_flange_width"]
        th_stiffener = inputs["axial_stiffener_spacing"]

        # Number of axial stiffeners
        n_stiff = 0 if th_stiffener == 0.0 else 2 * np.pi / th_stiffener

        # Outer and inner radius of web by section
        d_sec, _ = util.nodal2sectional(d_full)
        R_wo = 0.5 * d_sec - t_full
        R_wi = R_wo - h_web
        R_w = 0.5 * (R_wo + R_wi)

        # Outer and inner radius of flange by section
        R_fo = R_wi
        R_fi = R_fo - t_flange
        R_f = 0.5 * (R_fo + R_fi)

        A_web = h_web * t_web
        A_flange = w_flange * t_flange
        A_stiff = n_stiff * (A_web + A_flange)
        Ix_stiff = 0.5 * n_stiff * (A_web * R_w ** 2 + A_flange * R_f ** 2)
        Iz_stiff = 2 * Ix_stiff
        t_eff = A_stiff / (2 * np.pi * R_w)

        # Add sections for structural analysis
        for k in range(len(s_full) - 1):
            itube = cs.Tube(d_sec[k], t_full[k])
            iprop = CrossSection(
                D=d_sec[k],
                t=coeff[k] * t_full[k] + t_eff[k],
                A=coeff[k] * itube.Area + A_stiff,
                Ixx=coeff[k] * itube.Jxx + Ix_stiff[k],
                Iyy=coeff[k] * itube.Jyy + Ix_stiff[k],
                Izz=coeff[k] * itube.J0 + Iz_stiff[k],
                Asx=itube.Asx,
                Asy=itube.Asy,
                rho=rho[k],
                E=Emat[k],
                G=Gmat[k],
                sigy=sigymat[k],
            )
            self.add_section(s_full[k], s_full[k + 1], iprop)

        # Adjust for ghost sections
        s_ghost1 = float(inputs["s_ghost1"])
        s_ghost2 = float(inputs["s_ghost2"])
        if s_ghost1 > 0.0:
            self.add_node(s_ghost1)
            for s in self.sections:
                if s >= s_ghost1:
                    break
                self.sections[s].rho = 1e-2
                self.sections[s].E *= 1e2
                self.sections[s].G *= 1e2
        if s_ghost2 < 1.0:
            self.add_node(s_ghost2)
            for s in self.sections:
                if s < s_ghost2 or s == 1.0:
                    continue
                self.sections[s].rho = 1e-2
                self.sections[s].E *= 1e2
                self.sections[s].G *= 1e2

        # Shell mass properties with new interpolation in case ghost nodes were added
        s_grid = np.array(list(self.sections.keys()))
        s_section = 0.5 * (s_grid[:-1] + s_grid[1:])
        R = np.interp(s_grid, s_full, 0.5 * d_full)
        Rb = R[:-1]
        Rt = R[1:]
        zz = np.interp(s_grid, s_full, inputs["z_full"])
        H = np.diff(zz)
        t_full = util.sectionalInterp(s_section, s_full, inputs["t_full"])
        rho = util.sectionalInterp(s_section, s_full, inputs["rho_full"])
        rho[s_section < s_ghost1] = 0.0
        rho[s_section > s_ghost2] = 0.0
        coeff = util.sectionalInterp(s_section, s_full, coeff)
        k_m = util.sectionalInterp(s_section, s_full, inputs["unit_cost_full"])
        R_w = 0.5 * (Rb + Rt) - t_full - 0.5 * h_web
        R_f = 0.5 * (Rb + Rt) - t_full - h_web - 0.5 * t_flange
        Ix_stiff = 0.5 * n_stiff * (A_web * R_w ** 2 + A_flange * R_f ** 2)
        Iz_stiff = 2 * Ix_stiff

        # Total mass of cylinder
        V_shell = frustum.frustumShellVol(Rb, Rt, t_full, H)
        mass = coeff * rho * (V_shell + A_stiff * H)
        outputs["shell_mass"] = mass.sum()

        # Center of mass
        cm_section = zz[:-1] + frustum.frustumShellCG(Rb, Rt, t_full, H)
        outputs["shell_z_cg"] = np.dot(cm_section, mass) / mass.sum()

        # Moments of inertia
        Izz_section = coeff * rho * (frustum.frustumShellIzz(Rb, Rt, t_full, H) + H * Iz_stiff)
        Ixx_section = Iyy_section = coeff * rho * (frustum.frustumShellIxx(Rb, Rt, t_full, H) + H * Ix_stiff)

        # Sum up each cylinder section using parallel axis theorem
        I_base = np.zeros((3, 3))
        for k in range(Izz_section.size):
            R = np.array([0.0, 0.0, cm_section[k] - zz[0]])
            Icg = util.assembleI([Ixx_section[k], Iyy_section[k], Izz_section[k], 0.0, 0.0, 0.0])

            I_base += Icg + mass[k] * (np.dot(R, R) * np.eye(3) - np.outer(R, R))

        outputs["shell_I_base"] = util.unassembleI(I_base)

        # Compute costs based on "Optimum Design of Steel Structures" by Farkas and Jarmai
        # All dimensions for correlations based on mm, not meters.
        R_ave = 0.5 * (Rb + Rt)
        taper = np.minimum(Rb / Rt, Rt / Rb)
        nsec = t_full.size
        mshell = rho * V_shell
        mshell_tot = np.sum(rho * V_shell)
        # k_m = inputs["unit_cost_full"]  # 1.1 # USD / kg carbon steel plate
        k_f = inputs["labor_cost_rate"]  # 1.0 # USD / min labor
        k_p = inputs["painting_cost_rate"]  # USD / m^2 painting
        k_e = 0.064  # Industrial electricity rate $/kWh https://www.eia.gov/electricity/monthly/epm_table_grapher.php?t=epmt_5_6_a
        e_f = 15.9  # Electricity usage kWh/kg for steel
        e_fo = 26.9  # Electricity usage kWh/kg for stainless steel

        # Cost Step 1) Cutting flat plates for taper using plasma cutter
        cutLengths = 2.0 * np.sqrt((Rt - Rb) ** 2.0 + H ** 2.0)  # Factor of 2 for both sides
        # Cost Step 2) Rolling plates
        # Cost Step 3) Welding rolled plates into shells (set difficulty factor based on tapering with logistic function)
        theta_F = 4.0 - 3.0 / (1 + np.exp(-5.0 * (taper - 0.75)))
        # Cost Step 4) Circumferential welds to join cans together
        theta_A = 2.0

        # Labor-based expenses
        K_f = k_f * (
            manufacture.steel_cutting_plasma_time(cutLengths, t_full)
            + manufacture.steel_rolling_time(theta_F, R_ave, t_full)
            + manufacture.steel_butt_welding_time(theta_A, nsec, mshell_tot, cutLengths, t_full)
            + manufacture.steel_butt_welding_time(theta_A, nsec, mshell_tot, 2 * np.pi * Rb[1:], t_full[1:])
        )

        # Cost step 5) Painting- outside and inside
        theta_p = 2
        K_p = k_p * theta_p * 2 * (2 * np.pi * R_ave * H).sum()

        # Cost step 6) Outfitting with electricity usage
        K_o = np.sum(1.5 * k_m * (coeff - 1.0) * mshell)

        # Material cost with waste fraction, but without outfitting,
        K_m = 1.21 * np.sum(k_m * mshell)

        # Electricity usage
        K_e = np.sum(k_e * (e_f * mshell + e_fo * (coeff - 1.0) * mshell))

        # Assemble all costs for now
        tempSum = K_m + K_e + K_o + K_p + K_f

        # Capital cost share from BLS MFP by NAICS
        K_c = 0.118 * tempSum / (1.0 - 0.118)

        outputs["shell_cost"] = tempSum + K_c

    def add_bulkhead_sections(self, inputs, outputs):
        # Unpack variables
        s_full = inputs["s_full"]
        z_full = inputs["z_full"]
        R_od = 0.5 * inputs["d_full"]
        twall = inputs["t_full"]
        rho = inputs["rho_full"]
        E = inputs["E_full"]
        G = inputs["G_full"]
        sigy = inputs["sigma_y_full"]
        s_bulk = inputs["bulkhead_grid"]
        t_bulk = inputs["bulkhead_thickness"]
        coeff = inputs["outfitting_full"]
        L = inputs["height"]
        s_ghost1 = float(inputs["s_ghost1"])
        s_ghost2 = float(inputs["s_ghost2"])
        nbulk = s_bulk.size
        if nbulk == 0:
            return

        # Make sure we are not working in ghost regions
        s_bulk = np.unique(np.minimum(np.maximum(s_bulk, s_ghost1), s_ghost2))

        # Get z and R_id values of bulkhead locations
        z_bulk = np.interp(s_bulk, s_full, z_full)
        twall_bulk = util.sectionalInterp(s_bulk, s_full, twall)
        rho_bulk = util.sectionalInterp(s_bulk, s_full, rho)
        E_bulk = util.sectionalInterp(s_bulk, s_full, E)
        G_bulk = util.sectionalInterp(s_bulk, s_full, G)
        sigy_bulk = util.sectionalInterp(s_bulk, s_full, sigy)
        coeff_bulk = util.sectionalInterp(s_bulk, s_full, coeff)
        R_od_bulk = np.interp(s_bulk, s_full, R_od)
        R_id_bulk = R_od_bulk - twall_bulk

        # Add bulkhead sections: assumes bulkhead and shell are made of same material!
        s0 = s_bulk - 0.5 * t_bulk / L
        s1 = s_bulk + 0.5 * t_bulk / L
        if s0[0] < s_ghost1:
            s0[0] = s_ghost1
            s1[0] = s_ghost1 + t_bulk[0] / L
        if s1[-1] > s_ghost2:
            s0[-1] = s_ghost2 - t_bulk[-1] / L
            s1[-1] = s_ghost2
        for k in range(nbulk):
            itube = cs.Tube(2 * R_od_bulk[k], R_od_bulk[k])  # thickness=radius for solid disk
            iprop = CrossSection(
                D=2 * R_od_bulk[k],
                t=R_od_bulk[k],
                A=itube.Area,
                Ixx=itube.Jxx,
                Iyy=itube.Jyy,
                Izz=itube.J0,
                Asx=itube.Asx,
                Asy=itube.Asy,
                rho=coeff_bulk[k] * rho_bulk[k],
                E=E_bulk[k],
                G=G_bulk[k],
                sigy=sigy_bulk[k],
            )
            self.insert_section(s0[k], s1[k], iprop)

        # Compute bulkhead mass independent of shell
        m_bulk = coeff_bulk * rho_bulk * np.pi * R_id_bulk ** 2 * t_bulk
        outputs["bulkhead_mass"] = m_bulk.sum()

        z_cg = 0.0 if outputs["bulkhead_mass"] == 0.0 else np.dot(z_bulk, m_bulk) / m_bulk.sum()
        outputs["bulkhead_z_cg"] = z_cg

        # Compute moments of inertia at keel
        # Assume bulkheads are just simple thin discs with radius R_od-t_wall and mass already computed
        Izz = 0.5 * m_bulk * R_id_bulk ** 2
        Ixx = Iyy = 0.5 * Izz
        dz = z_bulk - z_full[0]
        I_keel = np.zeros((3, 3))
        for k in range(nbulk):
            R = np.array([0.0, 0.0, dz[k]])
            Icg = util.assembleI([Ixx[k], Iyy[k], Izz[k], 0.0, 0.0, 0.0])
            I_keel += Icg + m_bulk[k] * (np.dot(R, R) * np.eye(3) - np.outer(R, R))

        outputs["bulkhead_I_base"] = util.unassembleI(I_keel)

        # Compute costs based on "Optimum Design of Steel Structures" by Farkas and Jarmai
        # All dimensions for correlations based on mm, not meters.
        k_m = util.sectionalInterp(s_bulk, s_full, inputs["unit_cost_full"])
        k_f = inputs["labor_cost_rate"]  # 1.0 # USD / min labor
        k_p = inputs["painting_cost_rate"]  # USD / m^2 painting
        m_shell = outputs["shell_mass"]

        # Cost Step 1) Cutting flat plates using plasma cutter
        cutLengths = 2.0 * np.pi * R_id_bulk
        # Cost Step 2) Fillet welds with GMAW-C (gas metal arc welding with CO2) of bulkheads to shell
        theta_w = 3.0  # Difficulty factor

        # Labor-based expenses
        K_f = k_f * (
            manufacture.steel_cutting_plasma_time(cutLengths, t_bulk)
            + manufacture.steel_filett_welding_time(theta_w, nbulk, m_bulk + m_shell, 2 * np.pi * R_id_bulk, t_bulk)
        )

        # Cost Step 3) Painting (two sided)
        theta_p = 1.0
        K_p = k_p * theta_p * 2 * (np.pi * R_id_bulk ** 2.0).sum()

        # Material cost, without outfitting
        K_m = np.sum(k_m * m_bulk)

        # Total cost
        c_bulk = K_m + K_f + K_p
        outputs["bulkhead_cost"] = c_bulk

    def add_ring_stiffener_sections(self, inputs, outputs):
        # Unpack variables
        s_full = inputs["s_full"]
        z_full = inputs["z_full"]
        L = inputs["height"]
        R_od = 0.5 * inputs["d_full"]
        twall = inputs["t_full"]
        rho = inputs["rho_full"]
        E = inputs["E_full"]
        G = inputs["G_full"]
        sigy = inputs["sigma_y_full"]
        sigma_y = inputs["sigma_y_full"]
        coeff = inputs["outfitting_full"]
        s_bulk = inputs["bulkhead_grid"]
        s_ghost1 = float(inputs["s_ghost1"])
        s_ghost2 = float(inputs["s_ghost2"])

        t_web = inputs["ring_stiffener_web_thickness"]
        t_flange = inputs["ring_stiffener_flange_thickness"]
        h_web = inputs["ring_stiffener_web_height"]
        w_flange = inputs["ring_stiffener_flange_width"]
        L_stiffener = inputs["ring_stiffener_spacing"]

        n_stiff = 0 if L_stiffener == 0.0 else int(np.floor(L / L_stiffener))
        if n_stiff == 0:
            return

        web_frac = t_web / w_flange

        # Calculate stiffener spots along the member axis and deconflict with bulkheads
        s_stiff = (np.arange(1, n_stiff + 0.1) - 0.5) * (L_stiffener / L)

        # Make sure we are not working in ghost regions
        s_stiff = s_stiff[s_stiff > s_ghost1]
        s_stiff = s_stiff[s_stiff < s_ghost2]
        n_stiff = s_stiff.size

        tol = w_flange / L
        for k, s in enumerate(s_stiff):
            while np.any(np.abs(s_bulk - s) <= tol) and s > tol:
                s -= tol
            s_stiff[k] = s

        s0 = s_stiff - 0.5 * w_flange / L
        s1 = s_stiff + 0.5 * w_flange / L
        if s0[0] < 0.0:
            s0[0] = 0.0
            s1[0] = w_flange / L
        if s1[-1] > 1.0:
            s0[-1] = 1 - w_flange / L
            s1[-1] = 1.0

        # Get z and R_id values of bulkhead locations
        z_stiff = np.interp(s_stiff, s_full, z_full)
        twall_stiff = util.sectionalInterp(s_stiff, s_full, twall)
        rho_stiff = util.sectionalInterp(s_stiff, s_full, rho)
        E_stiff = util.sectionalInterp(s_stiff, s_full, E)
        G_stiff = util.sectionalInterp(s_stiff, s_full, G)
        sigy_stiff = util.sectionalInterp(s_stiff, s_full, sigy)
        coeff_stiff = util.sectionalInterp(s_stiff, s_full, coeff)
        R_od_stiff = np.interp(s_stiff, s_full, R_od)
        R_id_stiff = R_od_stiff - twall_stiff

        # Create some constraints for reasonable stiffener designs for an optimizer
        outputs["flange_spacing_ratio"] = w_flange / (0.5 * L_stiffener)
        outputs["stiffener_radius_ratio"] = NULL * np.ones(MEMMAX)
        outputs["stiffener_radius_ratio"][:n_stiff] = (h_web + t_flange + twall_stiff) / R_od_stiff
        # "compactness" check on stiffener geometry (must be >= 1)
        fact = np.sqrt(E / sigma_y).min()
        outputs["constr_flange_compactness"] = 0.375 * (t_flange / (0.5 * w_flange)) * fact
        outputs["constr_web_compactness"] = 1.0 * (t_web / h_web) * fact

        # Outer and inner radius of web by section
        R_wo = R_id_stiff
        R_wi = R_wo - h_web
        # Outer and inner radius of flange by section
        R_fo = R_wi
        R_fi = R_fo - t_flange

        # Make stiffener sections
        for k in range(n_stiff):
            ishell = cs.Tube(2 * R_od_stiff[k], twall_stiff[k])
            iweb = cs.Tube(2 * R_wo[k], h_web)
            iflange = cs.Tube(2 * R_fo[k], t_flange)
            Ak = coeff_stiff[k] * ishell.Area + iflange.Area + iweb.Area * web_frac
            # Find effective thickness for OpenFAST
            t_eff = R_od_stiff[k] - np.sqrt(R_od_stiff[k] ** 2 - Ak / np.pi)
            iprop = CrossSection(
                D=2 * R_od_stiff[k],
                t=t_eff,
                A=Ak,
                Ixx=coeff_stiff[k] * ishell.Jxx + iflange.Jxx + iweb.Jxx * web_frac,
                Iyy=coeff_stiff[k] * ishell.Jyy + iflange.Jyy + iweb.Jyy * web_frac,
                Izz=coeff_stiff[k] * ishell.J0 + iflange.J0 + iweb.J0 * web_frac,
                Asx=ishell.Asx + iflange.Asx + iweb.Asx * web_frac,
                Asy=ishell.Asy + iflange.Asy + iweb.Asy * web_frac,
                rho=rho_stiff[k],
                E=E_stiff[k],
                G=G_stiff[k],
                sigy=sigy_stiff[k],
            )
            self.insert_section(s0[k], s1[k], iprop)

        # Material volumes by section
        V_web = np.pi * (R_wo ** 2 - R_wi ** 2) * t_web
        V_flange = np.pi * (R_fo ** 2 - R_fi ** 2) * w_flange

        # Ring mass by volume by section
        m_web = rho_stiff * V_web
        m_flange = rho_stiff * V_flange
        m_ring = m_web + m_flange
        outputs["stiffener_mass"] = m_ring.sum()
        outputs["stiffener_z_cg"] = np.dot(z_stiff, m_ring) / m_ring.sum()

        # Compute moments of inertia for stiffeners (lumped by section for simplicity) at keel
        I_web = I_cyl(R_wi, R_wo, t_web, m_web)
        I_flange = I_cyl(R_fi, R_fo, w_flange, m_flange)
        I_keel = np.zeros((3, 3))
        for k in range(n_stiff):
            R = np.array([0.0, 0.0, (z_stiff[k] - z_full[0])])
            I_ring = util.assembleI(I_web[k, :] + I_flange[k, :])
            I_keel += I_ring + m_ring[k] * (np.dot(R, R) * np.eye(3) - np.outer(R, R))
        outputs["stiffener_I_base"] = util.unassembleI(I_keel)

        # Compute costs based on "Optimum Design of Steel Structures" by Farkas and Jarmai
        # All dimensions for correlations based on mm, not meters.
        k_m = util.sectionalInterp(s_stiff, s_full, inputs["unit_cost_full"])
        k_f = inputs["labor_cost_rate"]  # 1.0 # USD / min labor
        k_p = inputs["painting_cost_rate"]  # USD / m^2 painting
        m_shell = outputs["shell_mass"]

        # Cost Step 1) Cutting stiffener strips from flat plates using plasma cutter
        cutLengths_w = 2.0 * np.pi * 0.5 * (R_wo + R_wi)
        cutLengths_f = 2.0 * np.pi * R_fo
        # Cost Step 2) Welding T-stiffeners together GMAW-C (gas metal arc welding with CO2) fillet welds
        theta_w = 3.0  # Difficulty factor
        # Cost Step 3) Welding stiffeners to shell GMAW-C (gas metal arc welding with CO2) fillet welds
        # Will likely fillet weld twice (top & bottom), so factor of 2 on second welding terms

        # Labor-based expenses
        K_f = k_f * (
            manufacture.steel_cutting_plasma_time(cutLengths_w.sum(), t_web)
            + manufacture.steel_cutting_plasma_time(cutLengths_f.sum(), t_flange)
            + manufacture.steel_filett_welding_time(theta_w, 1, m_ring, 2 * np.pi * R_fo, t_web)
            + manufacture.steel_filett_welding_time(theta_w, 1, m_ring + m_shell, 2 * np.pi * R_wo, t_web)
        )

        # Cost Step 4) Painting
        theta_p = 2.0
        K_p = (
            k_p
            * theta_p
            * 2
            * np.pi
            * np.sum(R_wo ** 2.0 - R_wi ** 2.0 + 0.5 * (R_fo + R_fi) * (2 * w_flange + 2 * t_flange) - R_fo * t_web)
        )

        # Material cost, without outfitting
        K_m = np.sum(k_m * m_ring)

        # Total cost
        c_ring = K_m + K_f + K_p
        outputs["stiffener_cost"] = c_ring

    def add_ballast_sections(self, inputs, outputs):
        # Unpack variables
        s_full = inputs["s_full"]
        z_full = inputs["z_full"]
        R_od = 0.5 * inputs["d_full"]
        twall = inputs["t_full"]
        s_ballast = inputs["ballast_grid"]
        rho_ballast = inputs["ballast_density"]
        V_ballast = inputs["ballast_volume"]
        km_ballast = inputs["ballast_unit_cost"]
        s_ghost1 = float(inputs["s_ghost1"])
        s_ghost2 = float(inputs["s_ghost2"])
        n_ballast = len(V_ballast)
        if n_ballast == 0:
            return

        # Move away from ghost regions
        s_ballast += s_ghost1
        s_ballast = np.minimum(s_ballast, s_ghost2)

        # Number of points for volume integration
        npts = 10
        m_ballast = rho_ballast * V_ballast
        I_ballast = np.zeros(6)
        s_end = s_ballast[:, 0].copy()
        z_cg = np.zeros(n_ballast)
        V_avail = np.zeros(n_ballast)
        for k in range(n_ballast):
            # Find geometry of the ballast space
            sinterp = np.linspace(s_ballast[k, 0], s_ballast[k, 1], npts)
            zpts = np.interp(sinterp, s_full, z_full)
            H = zpts[-1] - zpts[0]
            R_od_pts = np.interp(sinterp, s_full, R_od)
            twall_pts = util.sectionalInterp(sinterp, s_full, twall)
            R_id_pts = R_od_pts - twall_pts

            # Available volume in this ballast space
            V_pts = frustum.frustumVol(R_id_pts[:-1], R_id_pts[1:], np.diff(zpts))
            V_avail[k] = V_pts.sum()

            # Augment density for these sections (should already be bulkheads at boundaries)
            for s in self.sections:
                if s >= s_ballast[k, 0] and s < s_ballast[k, 1]:
                    self.sections[s].rho += m_ballast[k] / self.sections[s].A / H

            # If permanent ballast, compute mass properties, but have to find where ballast extends to in cavity
            if V_ballast[k] > 0.0:
                s_end[k] = np.interp(V_ballast[k], np.cumsum(np.r_[0, V_pts]), sinterp)
                z_end = np.interp(V_ballast[k], np.cumsum(np.r_[0, V_pts]), zpts)
                zpts = np.linspace(zpts[0], z_end, npts)
                H = np.diff(zpts)

                R_od_pts = np.interp(zpts, z_full, R_od)
                twall_pts = util.sectionalInterp(zpts, z_full, twall)
                R_id_pts = R_od_pts - twall_pts

                V_pts = frustum.frustumVol(R_id_pts[:-1], R_id_pts[1:], H)
                cg_pts = frustum.frustumCG(R_id_pts[:-1], R_id_pts[1:], H) + zpts[:-1]
                z_cg[k] = np.dot(cg_pts, V_pts) / V_pts.sum()

                Ixx = Iyy = frustum.frustumIxx(R_id_pts[:-1], R_id_pts[1:], H)
                Izz = frustum.frustumIzz(R_id_pts[:-1], R_id_pts[1:], H)
                I_temp = np.zeros((3, 3))
                for ii in range(npts - 1):
                    R = np.array([0.0, 0.0, cg_pts[ii]])
                    Icg = util.assembleI([Ixx[ii], Iyy[ii], Izz[ii], 0.0, 0.0, 0.0])
                    I_temp += Icg + V_pts[ii] * (np.dot(R, R) * np.eye(3) - np.outer(R, R))
                I_ballast += rho_ballast[k] * util.unassembleI(I_temp)
            else:
                outputs["variable_ballast_capacity"] = V_avail[k]
                outputs["variable_ballast_Vpts"] = np.cumsum(np.r_[0, V_pts])
                outputs["variable_ballast_spts"] = sinterp

        # Save permanent ballast mass and variable height
        outputs["ballast_mass"] = m_ballast.sum()
        outputs["ballast_I_base"] = I_ballast
        outputs["ballast_z_cg"] = np.dot(z_cg, m_ballast) / (m_ballast.sum() + eps)
        outputs["ballast_cost"] = np.dot(km_ballast, m_ballast)
        outputs["ballast_height"] = s_end - s_ballast[:, 0]
        outputs["constr_ballast_capacity"] = V_ballast / V_avail

    def compute_mass_properties(self, inputs, outputs):
        # Unpack variables
        z_full = inputs["z_full"]

        z_shell = outputs["shell_z_cg"]
        z_ballast = outputs["ballast_z_cg"]
        z_bulkhead = outputs["bulkhead_z_cg"]
        z_stiffener = outputs["stiffener_z_cg"]

        m_shell = outputs["shell_mass"]
        m_ballast = outputs["ballast_mass"]
        m_bulkhead = outputs["bulkhead_mass"]
        m_stiffener = outputs["stiffener_mass"]

        c_shell = outputs["shell_cost"]
        c_ballast = outputs["ballast_cost"]
        c_bulkhead = outputs["bulkhead_cost"]
        c_stiffener = outputs["stiffener_cost"]

        I_shell = outputs["shell_I_base"]
        I_ballast = outputs["ballast_I_base"]
        I_bulkhead = outputs["bulkhead_I_base"]
        I_stiffener = outputs["stiffener_I_base"]

        # Find mass of all of the sub-components of the member
        m_total = m_shell + m_ballast + m_bulkhead + m_stiffener
        c_total = c_shell + c_ballast + c_bulkhead + c_stiffener

        # Masses assumed to be focused at section centroids
        z_cg = (
            m_shell * z_shell + m_ballast * z_ballast + m_bulkhead * z_bulkhead + m_stiffener * z_stiffener
        ) / m_total

        # Add up moments of inertia at keel, make sure to scale mass appropriately
        I_total = I_shell + I_ballast + I_bulkhead + I_stiffener

        # Move moments of inertia from keel to cg
        I_total -= m_total * ((z_cg - z_full[0]) ** 2.0) * np.r_[1.0, 1.0, np.zeros(4)]

        # Store outputs addressed so far
        outputs["total_mass"] = m_total
        outputs["structural_mass"] = m_total - m_ballast
        outputs["z_cg"] = z_cg
        outputs["I_total"] = I_total
        outputs["total_cost"] = c_total
        outputs["structural_cost"] = c_total - c_ballast
        # outputs["cost_rate"] = c_total / m_total

    def nodal_discretization(self, inputs, outputs):
        # Unpack inputs
        s_full = inputs["s_full"]
        d_full = inputs["d_full"]
        z_full = inputs["z_full"]
        s_axial = inputs["grid_axial_joints"]
        xyz0 = inputs["joint1"]
        xyz1 = inputs["joint2"]
        dxyz = xyz1 - xyz0

        # Add in axial nodes
        for s in s_axial:
            self.add_node(s)

        # Convert non-dimensional to dimensional
        s_grid = np.array(list(self.sections.keys()))
        r_grid = 0.5 * np.interp(s_grid, s_full, d_full)
        n_nodes = s_grid.size
        nodes = np.outer(s_grid, dxyz) + xyz0[np.newaxis, :]

        # Convert axial to absolute
        outputs["center_of_mass"] = (outputs["z_cg"] / z_full[-1]) * dxyz + xyz0

        # Store all nodes and sections
        outputs["s_all"] = NULL * np.ones(MEMMAX)
        outputs["nodes_r"] = NULL * np.ones(MEMMAX)
        outputs["nodes_xyz"] = NULL * np.ones((MEMMAX, 3))
        outputs["section_D"] = NULL * np.ones(MEMMAX)
        outputs["section_t"] = NULL * np.ones(MEMMAX)
        outputs["section_A"] = NULL * np.ones(MEMMAX)
        outputs["section_Asx"] = NULL * np.ones(MEMMAX)
        outputs["section_Asy"] = NULL * np.ones(MEMMAX)
        outputs["section_rho"] = NULL * np.ones(MEMMAX)
        outputs["section_Ixx"] = NULL * np.ones(MEMMAX)
        outputs["section_Iyy"] = NULL * np.ones(MEMMAX)
        outputs["section_Izz"] = NULL * np.ones(MEMMAX)
        outputs["section_E"] = NULL * np.ones(MEMMAX)
        outputs["section_G"] = NULL * np.ones(MEMMAX)
        outputs["section_sigma_y"] = NULL * np.ones(MEMMAX)

        outputs["s_all"][:n_nodes] = s_grid
        outputs["nodes_xyz"][:n_nodes, :] = nodes
        outputs["nodes_r"][:n_nodes] = r_grid

        for k, s in enumerate(s_grid):
            if s == s_grid[-1]:
                continue
            outputs["section_D"][k] = self.sections[s].D
            outputs["section_t"][k] = self.sections[s].t
            outputs["section_A"][k] = self.sections[s].A
            outputs["section_Asx"][k] = self.sections[s].Asx
            outputs["section_Asy"][k] = self.sections[s].Asy
            outputs["section_rho"][k] = self.sections[s].rho
            outputs["section_Ixx"][k] = self.sections[s].Ixx
            outputs["section_Iyy"][k] = self.sections[s].Iyy
            outputs["section_Izz"][k] = self.sections[s].Izz
            outputs["section_E"][k] = self.sections[s].E
            outputs["section_G"][k] = self.sections[s].G
            outputs["section_sigma_y"][k] = self.sections[s].sigy


class MemberHydro(om.ExplicitComponent):
    """
    Compute member substructure elements in floating offshore wind turbines.

    Parameters
    ----------
    rho_water : float, [kg/m**3]
        density of water
    s_full : numpy array[n_full], [m]
        non-dimensional coordinates of section nodes
    z_full : numpy array[n_full], [m]
        z-coordinates of section nodes
    d_full : numpy array[n_full], [m]
        outer diameter at each section node bottom to top (length = nsection + 1)
    s_all : numpy array[MEMAX]
        Final non-dimensional points of all internal member nodes
    nodes_xyz : numpy array[MEMAX,3], [m]
        Global dimensional coordinates (x-y-z) for all member nodes in s_all output


    Returns
    -------
    center_of_buoyancy : numpy array[3], [m]
        z-position CofB of member
    displacement : float, [m**3]
        Volume of water displaced by member
    buoyancy_force : float, [N]
        Net z-force from buoyancy on member
    idx_cb : int
        Index of closest node to center of buoyancy
    Awater : float, [m**2]
        Area of waterplace cross section
    Iwater : float, [m**4]
        Second moment of area of waterplace cross section
    added_mass : numpy array[6], [kg]
        hydrodynamic added mass matrix diagonal
    waterline_centroid : numpy array[2], [m]
        x-y center of waterplane crossing for this member

    """

    def initialize(self):
        self.options.declare("n_full")

    def setup(self):
        n_full = self.options["n_full"]

        # Variables local to the class and not OpenMDAO
        self.ibox = None

        self.add_input("rho_water", 0.0, units="kg/m**3")
        self.add_input("s_full", np.zeros(n_full), units="m")
        self.add_input("z_full", np.zeros(n_full), units="m")
        self.add_input("d_full", np.zeros(n_full), units="m")
        self.add_input("s_all", NULL * np.ones(MEMMAX))
        self.add_input("nodes_xyz", NULL * np.ones((MEMMAX, 3)), units="m")

        self.add_output("center_of_buoyancy", np.zeros(3), units="m")
        self.add_output("displacement", 0.0, units="m**3")
        self.add_output("buoyancy_force", 0.0, units="N")
        self.add_output("idx_cb", 0)
        self.add_output("Awater", 0.0, units="m**2")
        self.add_output("Iwater", 0.0, units="m**4")
        self.add_output("added_mass", np.zeros(6), units="kg")
        self.add_output("waterline_centroid", np.zeros(2), units="m")
        self.add_output("z_dim", np.zeros(n_full), units="m")
        self.add_output("d_eff", np.zeros(n_full), units="m")

    def compute(self, inputs, outputs):
        # Unpack variables
        nnode = np.where(inputs["s_all"] == NULL)[0][0]
        s_grid = inputs["s_all"][:nnode]
        xyz = inputs["nodes_xyz"][:nnode, :]
        s_full = inputs["s_full"]
        z_full = inputs["z_full"]
        R_od = 0.5 * inputs["d_full"]
        rho_water = inputs["rho_water"]
        xyz0 = xyz[0, :]
        dxyz = xyz[-1, :] - xyz[0, :]

        # Dimensions away from vertical
        tilt = np.arctan(dxyz[0] / (1e-8 + dxyz[2]))
        outputs["z_dim"] = xyz0[2] + s_full * dxyz[2]
        outputs["d_eff"] = inputs["d_full"] / np.cos(tilt)

        # Compute volume of each section and mass of displaced water by section
        # Find the radius at the waterline so that we can compute the submerged volume as a sum of frustum sections
        if xyz[:, 2].min() < 0.0 and xyz[:, 2].max() > 0.0:
            s_waterline = np.interp(0.0, xyz[:, 2], s_grid)
            ind = np.where(xyz[:, 2] < 0.0)[0]
            s_under = np.r_[s_grid[ind], s_waterline]
            waterline = True
            outputs["waterline_centroid"] = (xyz0 + s_waterline * dxyz)[:2]
        elif xyz[:, 2].max() < 0.0:
            s_under = s_grid
            waterline = False
            r_waterline = 0.0
            outputs["waterline_centroid"] = np.zeros(2)
        else:
            return
        z_under = np.interp(s_under, s_full, z_full)
        r_under = np.interp(s_under, s_full, R_od)
        if waterline:
            r_waterline = r_under[-1]

        # Submerged volume (with zero-padding)
        dz = np.diff(z_under)
        V_under = frustum.frustumVol(r_under[:-1], r_under[1:], dz)
        V_under_tot = V_under.sum()
        outputs["displacement"] = V_under_tot
        outputs["buoyancy_force"] = rho_water * outputs["displacement"] * gravity

        # Compute Center of Buoyancy in z-coordinates (0=waterline)
        # First get z-coordinates of CG of all frustums
        z_cg_under = frustum.frustumCG(r_under[:-1], r_under[1:], dz) + z_under[:-1]
        z_cb = np.dot(z_cg_under, V_under) / V_under_tot
        s_cb = np.interp(z_cb, z_under, s_under)
        cb = xyz0 + s_cb * dxyz
        outputs["center_of_buoyancy"] = cb
        outputs["idx_cb"] = util.closest_node(xyz, cb)

        # 2nd moment of area for circular cross section
        # Note: Assuming Iwater here depends on "water displacement" cross-section
        # and not actual moment of inertia type of cross section (thin hoop)
        outputs["Iwater"] = 0.25 * np.pi * r_waterline ** 4.0
        outputs["Awater"] = np.pi * r_waterline ** 2.0

        # Calculate diagonal entries of added mass matrix
        temp = np.linspace(z_under[0], z_under[-1], 200)
        r_under = np.interp(temp, z_under, r_under)
        z_under = temp
        m_a = np.zeros(6)
        m_a[:2] = rho_water * V_under_tot  # A11 surge, A22 sway

        # Lxy = np.sqrt((xyz[:, 0].max() - xyz[:, 0].min()) ** 2 + (xyz[:, 1].max() - xyz[:, 1].min()) ** 2)
        D = 2 * r_under.max()
        # Lxy = np.maximum(Lxy, D)
        m_a[2] = (1.0 / 6.0) * rho_water * D ** 3.0  # A33 heave * Lxy *
        m_a[3:5] = (
            np.pi * rho_water * np.trapz((z_under - z_cb) ** 2.0 * r_under ** 2.0, z_under)
        )  # A44 roll, A55 pitch
        m_a[5] = 0.0  # A66 yaw
        outputs["added_mass"] = m_a


class Global2MemberLoads(om.ExplicitComponent):
    """
    Converts the loading from the global c.s. to the member (element) c.s.

    Parameters
    ----------
    s_full : numpy array[n_full], [m]
        non-dimensional coordinates of section nodes
    s_all : numpy array[MEMMAX]
        Final non-dimensional points of all internal member nodes
    nodes_xyz : numpy array[MEMMAX,3], [m]
        Global dimensional coordinates (x-y-z) for all member nodes in s_all output
    Px_global : numpy array[npts], [N/m]
        x-Force density in global coordinates at member nodes
    Py_global : numpy array[npts], [N/m]
        y-Force density in global coordinates at member nodes
    Pz_global : numpy array[npts], [N/m]
        z-Force density in global coordinates at member nodes


    Returns
    -------
    Px : numpy array[MEMMAX], [N/m]
        x-Force density in element coordinates (x along axis) at member nodes
    Py : numpy array[MEMMAX], [N/m]
        y-Force density in element coordinates (x along axis) at member nodes
    Pz : numpy array[MEMMAX], [N/m]
        z-Force density in element coordinates (x along axis) at member nodes

    """

    def initialize(self):
        self.options.declare("n_full")

    def setup(self):
        n_full = self.options["n_full"]

        # Variables local to the class and not OpenMDAO
        self.ibox = None

        self.add_input("s_full", np.zeros(n_full), units="m")
        self.add_input("s_all", NULL * np.ones(MEMMAX))
        self.add_input("nodes_xyz", NULL * np.ones((MEMMAX, 3)), units="m")
        self.add_input("Px_global", np.zeros(n_full), units="N/m")
        self.add_input("Py_global", np.zeros(n_full), units="N/m")
        self.add_input("Pz_global", np.zeros(n_full), units="N/m")
        self.add_input("qdyn_global", np.zeros(n_full), units="Pa")

        self.add_output("Px", NULL * np.ones(MEMMAX), units="N/m")
        self.add_output("Py", NULL * np.ones(MEMMAX), units="N/m")
        self.add_output("Pz", NULL * np.ones(MEMMAX), units="N/m")
        self.add_output("qdyn", NULL * np.ones(MEMMAX), units="Pa")

    def compute(self, inputs, outputs):
        # Unpack variables
        s_full = inputs["s_full"]
        nnode = np.where(inputs["s_all"] == NULL)[0][0]
        s_grid = inputs["s_all"][:nnode]
        xyz = inputs["nodes_xyz"][:nnode, :]

        # Put global loads on denser grid
        Px_g = np.interp(s_grid, s_full, inputs["Px_global"])
        Py_g = np.interp(s_grid, s_full, inputs["Py_global"])
        Pz_g = np.interp(s_grid, s_full, inputs["Pz_global"])
        qdyn_g = np.interp(s_grid, s_full, inputs["qdyn_global"])

        # Get rotation matrix that puts x along member axis
        unit_x = np.array([1.0, 0.0, 0.0])
        dxyz = xyz[-1, :] - xyz[0, :]
        R = util.rotate_align_vectors(dxyz, unit_x)
        P_local = R @ np.c_[Px_g, Py_g, Pz_g].T

        # Store local loads
        Px = NULL * np.ones(MEMMAX)
        Py = NULL * np.ones(MEMMAX)
        Pz = NULL * np.ones(MEMMAX)
        Px[:nnode] = P_local[0, :]
        Py[:nnode] = P_local[1, :]
        Pz[:nnode] = P_local[2, :]
        outputs["Px"] = Px
        outputs["Py"] = Py
        outputs["Pz"] = Pz

        qdyn = NULL * np.ones(MEMMAX)
        qdyn[: (nnode - 1)], _ = util.nodal2sectional(qdyn_g)
        outputs["qdyn"] = qdyn


class Member(om.Group):
    def initialize(self):
        self.options.declare("column_options")
        self.options.declare("idx")
        self.options.declare("n_mat")

    def setup(self):
        opt = self.options["column_options"]
        idx = self.options["idx"]

        n_height = opt["n_height"][idx]
        n_refine = NREFINE if n_height > 2 else np.maximum(NREFINE, 2)
        n_full = get_nfull(n_height, nref=n_refine)

        # TODO: Use reference axis and curvature, s, instead of assuming everything is vertical on z
        self.add_subsystem(
            "yaml", DiscretizationYAML(options=opt, idx=idx, n_mat=self.options["n_mat"]), promotes=["*"]
        )

        self.add_subsystem(
            "gc", GeometricConstraints(nPoints=n_height, diamFlag=True), promotes=["constr_taper", "constr_d_to_t"]
        )
        self.connect("outer_diameter", "gc.d")
        self.connect("wall_thickness", "gc.t")

        self.add_subsystem("geom", MemberDiscretization(n_height=n_height, n_refine=n_refine), promotes=["*"])

        self.add_subsystem("comp", MemberComponent(options=opt, idx=idx, n_refine=n_refine), promotes=["*"])

        self.add_subsystem("hydro", MemberHydro(n_full=n_full), promotes=["*"])

        prom = [
            "Uref",
            "zref",
            "shearExp",
            "z0",
            "cd_usr",
            "cm",
            "beta_wind",
            "rho_air",
            "mu_air",
            "beta_wave",
            "rho_water",
            "mu_water",
            "Uc",
            "Hsig_wave",
            "Tsig_wave",
            "water_depth",
            "yaw",
        ]
        self.add_subsystem("env", CylinderEnvironment(nPoints=n_full, water_flag=True), promotes=prom)
        self.connect("z_dim", "env.z")
        self.connect("d_eff", "env.d")

        self.add_subsystem(
            "g2e",
            Global2MemberLoads(n_full=n_full),
            promotes=["nodes_xyz", "s_all", "s_full", "Px", "Py", "Pz", "qdyn"],
        )
        self.connect("env.Px", "g2e.Px_global")
        self.connect("env.Py", "g2e.Py_global")
        self.connect("env.Pz", "g2e.Pz_global")
        self.connect("env.qdyn", "g2e.qdyn_global")
