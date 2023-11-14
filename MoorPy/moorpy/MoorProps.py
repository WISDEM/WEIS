
# MoorProps
# start of a script to hold the functions that hold the mooring line and anchor property info
# just simple functions for now, can import functions to other scripts if necessary
# very much in progress
# MoorProps started 4-26-21, getLineProps first initialized back in Sept/Oct 2020


import numpy as np
import moorpy as mp
import math


def getLineProps(dmm, type="chain", stud="studless", source="Orcaflex-altered", name=""):
    ''' getLineProps version 3.2: Restructuring v3.1 to 'Orcaflex-original' and 'Orcaflex-altered' 
    
    Motivation: The existing public, and NREL-internal references for mooring line component property
    data are either proprietary, or unreliable and incomplete. The best way to derive new equations
    as a function of diameter is to use data from mooring line manufacturer's catalogs. Once developed,
    these new equations will serve as an updated version to compare against existing expressions.
    
    The goal is to have NREL's own library of mooring line property equations, but more research is needed.
    The original Orcaflex equations are the best we have right now and have been altered to include
    a quadratic chain MBL equation rather than a cubic, to avoid negative MBLs.
    Also, different cost models are put in to the altered version to avoid the Equimar data. Many sources
    researched for cost data, coefficients used are close to the old NREL internal data, but still an approximation.
    
    For more info, see the Mooring Component Properties Word doc.

    - This function requires at least one input: the line diameter in millimeters.
    - The rest of the inputs are optional: describe the desired type of line (chain, polyester, wire, etc.),
    the type of chain (studless or studlink), the source of data (Orcaflex-original or altered), or a name identifier
    - The function will output a MoorPy linetype object

    '''
    
    
    
    if source=="Orcaflex-original":
        d = dmm/1000  # orcaflex uses meters https://www.orcina.com/webhelp/OrcaFlex/
        
        if type=="chain":
            c = 1.96e4 #grade 2=1.37e4; grade 3=1.96e4; ORQ=2.11e4; R4=2.74e4
            MBL = c*d**2*(44-80*d)*1000     #[N]  The same for both studless and studlink
            if stud=="studless":
                massden = 19.9*d**2*1000    #[kg/m]
                EA = 0.854e8*d**2*1000      #[N]
                d_vol = 1.8*d               #[m]
            elif stud=="studlink" or stud=="stud":
                massden = 21.9*d**2*1000    #[kg/m]
                EA = 1.010e8*d**2*1000      #[N]
                d_vol = 1.89*d              #[m]
            else:
                raise ValueError("getLineProps error: Choose either studless or stud chain type ")
        
        elif type=="nylon":
            massden = 0.6476*d**2*1000      #[kg/m]
            EA = 1.18e5*d**2*1000           #[N]
            MBL = 139357*d**2*1000          #[N] for wet nylon line, 163950d^2 for dry nylon line
            d_vol = 0.85*d                  #[m]
        elif type=="polyester":
            massden = 0.7978*d**2*1000      #[kg/m]
            EA = 1.09e6*d**2*1000           #[N]
            MBL = 170466*d**2*1000          #[N]
            d_vol = 0.86*d                  #[m]
        elif type=="polypropylene":
            massden = 0.4526*d**2*1000      #[kg/m]
            EA = 1.06e6*d**2*1000           #[N]
            MBL = 105990*d**2*1000          #[N]
            d_vol = 0.80*d                  #[m]
        elif type=="wire-fiber" or type=="fiber":
            massden = 3.6109*d**2*1000      #[kg/m]
            EA = 3.67e7*d**2*1000           #[N]
            MBL = 584175*d**2*1000          #[N]
            d_vol = 0.82*d                  #[m]
        elif type=="wire-wire" or type=="wire" or type=="IWRC":
            massden = 3.9897*d**2*1000      #[kg/m]
            EA = 4.04e7*d**2*1000           #[N]
            MBL = 633358*d**2*1000          #[N]
            d_vol = 0.80*d                  #[m]
        else:
            raise ValueError("getLineProps error: Linetype not valid. Choose from given rope types or chain ")
            
        
        
        # cost
        # Derived from Equimar graph: https://tethys.pnnl.gov/sites/default/files/publications/EquiMar_D7.3.2.pdf
        if type=="chain":
            cost = (0.21*(MBL/9.81/1000))*1.29              #[$/m]
        elif type=="nylon" or type=="polyester" or type=="polypropylene":
            cost = (0.235*(MBL/9.81/1000))*1.29         #[$/m]
        elif type=="wire" or type=="wire-wire" or type=="IWRC" or type=='fiber' or type=='wire-fiber':
            cost = (0.18*(MBL/9.81/1000) + 90)*1.29     #[$/m]
        else:
            raise ValueError("getLineProps error: Linetype not valid. Choose from given rope types or chain ")



    elif source=="Orcaflex-altered":
        d = dmm/1000  # orcaflex uses meters https://www.orcina.com/webhelp/OrcaFlex/
        
        if type=="chain":
            c = 2.74e4 #grade 2=1.37e4; grade 3=1.96e4; ORQ=2.11e4; R4=2.74e4
            MBL = (371360*d**2 + 51382.72*d)*(c/2.11e4)*1000 # this is a fit quadratic term to the cubic MBL equation. No negatives
            if stud=="studless":
                massden = 19.9*d**2*1000        #[kg/m]
                EA = 0.854e8*d**2*1000          #[N]
                d_vol = 1.8*d                   #[m]
            elif stud=="studlink" or stud=="stud":
                massden = 21.9*d**2*1000        #[kg/m]
                EA = 1.010e8*d**2*1000          #[N]
                d_vol = 1.89*d                  #[m]
            else:
                raise ValueError("getLineProps error: Choose either studless or stud chain type ")
                
            #cost = 2.5*massden   # a ballpark for R4 chain
            #cost = (0.58*MBL/1000/9.81) - 87.6          # [$/m] from old NREL-internal
            #cost = 3.0*massden     # rough value similar to old NREL-internal
            cost = 2.585*massden     # [($/kg)*(kg/m)=($/m)]
            #cost = 0.0
        
        elif type=="nylon":
            massden = 0.6476*d**2*1000      #[kg/m]
            EA = 1.18e5*d**2*1000           #[N]
            MBL = 139357*d**2*1000          #[N] for wet nylon line, 163950d^2 for dry nylon line
            d_vol = 0.85*d                  #[m]
            cost = (0.42059603*MBL/1000/9.81) + 109.5   # [$/m] from old NREL-internal
        elif type=="polyester":
            massden = 0.7978*d**2*1000      #[kg/m]
            EA = 1.09e6*d**2*1000           #[N]
            MBL = 170466*d**2*1000          #[N]
            d_vol = 0.86*d                  #[m]
            
            #cost = (0.42059603*MBL/1000/9.81) + 109.5   # [$/m] from old NREL-internal
            #cost = 1.1e-4*MBL               # rough value similar to old NREL-internal
            cost = 0.162*(MBL/9.81/1000)    #[$/m]
            
        elif type=="polypropylene":
            massden = 0.4526*d**2*1000      #[kg/m]
            EA = 1.06e6*d**2*1000           #[N]
            MBL = 105990*d**2*1000          #[N]
            d_vol = 0.80*d                  #[m]
            cost = 1.0*((0.42059603*MBL/1000/9.81) + 109.5)   # [$/m] from old NREL-internal
        elif type=="hmpe":
            massden = 0.4526*d**2*1000      #[kg/m]
            EA = 38.17e6*d**2*1000          #[N]
            MBL = 619000*d**2*1000          #[N]
            d_vol = 1.01*d                  #[m]
            cost = (0.01*MBL/1000/9.81)     # [$/m] from old NREL-internal
        elif type=="wire-fiber" or type=="fiber":
            massden = 3.6109*d**2*1000      #[kg/m]
            EA = 3.67e7*d**2*1000           #[N]
            MBL = 584175*d**2*1000          #[N]
            d_vol = 0.82*d                  #[m]
            cost = 0.53676471*MBL/1000/9.81             # [$/m] from old NREL-internal
        elif type=="wire-wire" or type=="wire" or type=="IWRC":
            massden = 3.9897*d**2*1000      #[kg/m]
            EA = 4.04e7*d**2*1000           #[N]
            MBL = 633358*d**2*1000          #[N]
            d_vol = 0.80*d                  #[m]
            #cost = MBL * 900./15.0e6
            #cost = (0.33*MBL/1000/9.81) + 139.5         # [$/m] from old NREL-internal
            cost = 5.6e-5*MBL               # rough value similar to old NREL-internal
        else:
            raise ValueError("getLineProps error: Linetype not valid. Choose from given rope types or chain ")


    elif source=="NREL":
        '''
        getLineProps v3.1 used to have old NREL-internal equations here as a placeholder, but they were not trustworthy.
         - The chain equations used data from Vicinay which matched OrcaFlex data. The wire rope equations matched OrcaFlex well,
           the synthetic rope equations did not
           
        The idea is to have NREL's own library of mooring line property equations, but more research needs to be done.
        The 'OrcaFlex-altered' source version is a start and can change name to 'NREL' in the future, but it is
        still 'OrcaFlex-altered' because most of the equations are from OrcaFlex, which is the best we have right now.
        
        Future equations need to be optimization proof = no negative numbers anywhere (need to write an interpolation function)
        Can add new line types as well, such as Aramid or HMPE
        '''
        pass
        
    
    
    
    # Set up a main identifier for the linetype. Useful for things like "chain_bot" or "chain_top"
    if name=="":
        typestring = f"{type}{dmm:.0f}"
    else:
        typestring = name
    
    
    notes = f"made with getLineProps - source: {source}"
    

    return mp.LineType(typestring, d_vol, massden, EA, MBL=MBL, cost=cost, notes=notes, input_type=type, input_d=dmm)


#----NOTES----

#def getAnchorProps(value, anchor="drag-embedment", soil_type='medium clay',uhc_mode = True, method = 'static', display=0):
    # need to make 'value' input all the same as either Mass or UHC
    #'value' can be both capacity or mass, 'uhc_mode' defines what it is. However, this method does not take into account fx, fz loading directions or safety factor

# applied safety factors make input UHC smaller resulting in smaller mass > uhc_mode = True, calulating for capacity | INPUT: mass | does not account for this 
    #added 'capacity_sf' variable: outputs capacity for applied safety factor, while showing original input capacities (fx,fz)

# VLA calculation for thickness may be wrong: may not need to divide by 4. When we dont it matched STEVMANTA table (in excel sheet)

# Do i want to scale fx,fz by 20% when considering dynamic loads like in Moorpy 'capacity_x = 1,2*fx ; capacity_z = 1.2*fz

# Need to validate suction curves and compare to Ahmed's curves. His is pure horizontal while these ones are for both horizontal and vertical 

#--------------

def getAnchorMass( uhc_mode = True, fx=0, fz=0, mass_int=0, anchor="drag-embedment", soil_type='medium clay', method = 'static', display=0):
    '''Calculates anchor required capacity, mass, and cost based on specified loadings and anchor type
    
    Parameters
    ----------
    uhc_mode: boolean
        True: INPUT Mass to find UHC
        False: INPUT fx,fz to find Mass and UHC
    fx : float
        horizontal maximum anchor load [N]
    fz : float 
        vertical maximum anchor load [N]
    massint : float 
        anchor mass [kg]    
    anchor : string
        anchor type name
    soil_type: string
        soil type name
    method: string
        anchor analysis method for applied safety factors 

    Returns
    -------
    UHC: float
        required anchor ultimate holding capacity [N]
    mass: float
        anchor mass [kg]
    info: float
        dictionary
    '''

    #--MAGIC NUMBERS -- assign varibles to values
    density_st = 7.85*1000       #[kg/m^3]
    gravity = 9.81               #[m/sec^2]? or [N/kg], if so may need to add conversion *1000 so [kN/mT]
    info = {}


    # Define anchor type
    if anchor == "drag-embedment":
       
        # Define soil type (coefficients are from Vryhof for Stevpris MK6)
        if soil_type == 'soft clay':
            a, b = 509.96, 0.93
        elif soil_type == 'medium clay':
            a, b = 701.49, 0.93
        elif soil_type == 'hard clay' or soil_type == 'sand':
            a, b = 904.21, 0.92
        else:
            raise ValueError('Error: Invalid soil type')
        
        # Define mode
        # Calculate capacity | INPUT: Mass in kg 
        if uhc_mode:
            uhc = a * (mass_int/1000)**b*1000           #[N] (from Vryhof eqn)
            # y = m*(x)^b

            print(f"Mass input: {mass_int} -- UHC: {uhc}")
            info["Mass"] = mass_int   #[kg]
            info["UHC"] = uhc               #[N]
            
        
        # Calculate mass | INPUT: UHC 
        else:
            # Define method for safety factor
            if method == 'static':
                uhc = 1.8*fx     #[N]
            elif method == 'dynamic':
                uhc = 1.5*fx      #[N]
            else:
                raise Exception("Error - invalid method")

            mass = (uhc /1000 /a)**(1/b)*1000     # [kg] (from Vryhof eqn)
            
            print(f"UHC input: {fx} -- Mass: {mass}")
            info["UHC"] = uhc           #[N]
            info["Mass"] = mass                 #[kg]
        
    elif anchor == "VLA":
        
        # Vryhof coefficients
        if soil_type == 'soft clay':
            c, d = 0.003, -0.2
        elif soil_type == 'medium clay':
            c, d = 0.0017, -0.32
        else:
            raise ValueError('Error: Invalid soil type')

        if uhc_mode: # Calculate capacity | INPUT: Mass
            #t2_m_ratio = 20933.3                            #[m^2]

            #Ericka note - I don't know where this ratio is coming from. Matt may have estimated from Vryhof manual
            t2_m_ratio = 1308.33
            
            #Ericka note -not sure about this equation - should it be to the 1/2 power? 
            thickness = (mass_int/1000 / t2_m_ratio)**(1/3)      #[m]     #DONE: magic number - assign to variable (think its from t^2/A = 6000; thickness ratio)
            area = mass_int/ (thickness * density_st)            #[m^2]   Area (1-30)
            uhc = (area - d) / c * 1000                          #[N] Vryhof equation


            print(f"Mass input: {mass_int} -- UHC: {uhc}, Area: {area}")
            info["Mass"] = mass_int                 #[kg]
            info["UHC"] = uhc                       #[N]
            info["Area"] = area                     #[m^2]

        else: # Calculate mass | INPUT: UHC
            capacity_x = 2.0 * fx  # N
            capacity_z = 2.0 * fz  # N
            uhc = np.linalg.norm([capacity_x,capacity_z])
            
            #Vryhof equation
            area = c * uhc/1000 + d          #[m^2] Vryhof equation
            # y = m*x + b
            
            #Ericka note - I don't know where this ratio is coming from. Matt may have estimated from Vryhof manual
            t2_a_ratio = 0.006                  #[m^2]
            if area < 0:
                thickness = 0
                raise ValueError ("Error: Negative area)")
            else:
                #thickness = math.sqrt(t2_a_ratio * area)/4          #[m]    #### why divide by 4 - looking at this in exel sheet
                thickness = math.sqrt(t2_a_ratio * area)
                mass = area * thickness * density_st                #[kg]


                print(f"UHC input: fx:{fx} fz:{fz} -- Mass: {mass}, Area: {area}")
                info["UHC"] = uhc   #[N]
                info["Mass"] = mass                 #[kg]
                info["Area"] = area                 #[m^2]

    
    ###### VERIFICATION REQUIRED, THESE CURVES CONSIDER INCLINED LOADS -> While Ahmed's is pure horizontal 
    elif anchor == "suction":
        
        # c and d numbers are from ABS FOWT Anchor Report
        # m and b numbers are from Brian's excel sheet curve fit to Mass vs UHC graph
        if soil_type == 'soft clay':
            m = 88.152
            b = 1.1058
            c_L, d_L = 1.1161, 0.3442
            c_D, d_D = 0.3095, 0.2798
            c_T, d_T = 2.058, 0.2803
            
        elif soil_type == 'medium clay':
            m = 384.15
            b = 0.8995
            c_L, d_L = 0.5166, 0.3995
            c_D, d_D = 0.126, 0.3561
            c_T, d_T = 0.8398, 0.3561
           
        else:
            raise ValueError("Invalid Soil Type")
        
        
        if uhc_mode: # Calculate capacity | INPUT: Mass
           
            ### ABS (Matts curves) slightly off  (from Brian's excel sheet curve fit for Mass vs UHC)
            uhc = m * (mass_int/1000) ** b * 1000 # [N]


            print(f"Mass INPUT: {mass_int} -- UHC: {uhc}")
            info["Mass"] = mass_int       #[kg]
            info["UHC"] = uhc                   #[N]

        else: # Calculate mass | INPUT: UHC
        
            capacity_x = 1.6*fx                           #[N]
            #capacity_x = fx
            capacity_z = 2.0*fz                           #[N]
            uhc = np.linalg.norm([capacity_x, capacity_z])  #[kN]

            # Equations from ABS Anchors Report
            L = c_L * (uhc/1000) **d_L                 #[m]
            D = c_D * (uhc/1000) **d_D                 #[m]
            T = (c_T * (uhc/1000) **d_T) /1000        #[m] 

            # y = m*(x)^b
            
            # volume of one end + open cylinder wall
            volume = (math.pi/4 * D ** 2 + math.pi * D * L) * T     #[m^3]
            mass = volume * density_st                              #[kg]


            print(f"UHC input: fx:{fx} fz:{fz} -- Mass: {mass}")
            info["UHC"] = uhc           #[N]
            info["Mass"] = mass                 #[kg]
            #info["Length"] = L
          

    
    
    #-----IN PROGRESS -----------------------------------

    elif anchor == "SEPLA":
        # Ericka note - we think these coefficients come from Ahmed's intermediate model
        if soil_type == 'soft clay':
            c_B, d_B = 0.0225, 0.5588
            c_L, d_L = 0.0450, 0.5588
            c_T, d_T = 0.0006, 0.5588
            
        else:
            raise ValueError("Invalid Soil Type")
        
        if uhc_mode: # Calculate UHC | INPUT: Mass       ### A little bit off from uhc = FALSE 

            # from excel
            m = 1557.4
            b = 0.5956
            uhc = m *(mass_int/1000) ** b  *10000 # [N]

        
            print(f'Work in progress -- Mass input: {mass_int} -- UHC {uhc}')
            info["Mass"] = mass_int
            info["UHC"] = uhc

        else: # Calculate Mass | INPUT: UHC
            capacity_x = 2.0*fx
            uhc = capacity_x
        
            B = c_B * (uhc/1000) ** d_B        #[m]
            L = c_L * (uhc/1000) ** d_L        #[m]
            T = c_T * (uhc/1000) ** d_T        #[m]
            # y = m*(x)^b

            area = B*L                          #[m^2]
            volume = area * T                   #[m^3]
            mass = volume * density_st          #[mT]


            print(f"UHC input: fx:{fx} fz:{fz} -- Mass {mass}")
            info["UHC"] = uhc   #[N]
            info["Mass"] = mass                 #[kg]
            info["Area"] = area                 #[m^2]

    elif anchor == "micropile":
        raise ValueError ("Not supported yet")    


    else:
        return Exception("Error - invalid anchor type")
    


    if uhc_mode:
        mass = mass_int

    return uhc, mass, info
    #return info


def getAnchorCost(fx, fz, type="drag-embedment",soil_type='medium clay', method = 'static'):
    ''' applies factors to material cost '''
    

    uhc, mass, info = getAnchorMass( uhc_mode = False, fx=fx, fz=fz, anchor= type, soil_type=soil_type, method = method)       
    euros2dollars = 1.18    # the number of dollars there currently are in a euro (3-31-21)
    
    if type == "drag-embedment":

 
        anchorMatCost = 6.7*mass # $ per kg mass
        anchorInstCost = 163548*euros2dollars       # installation cost
        anchorDecomCost = 228967*euros2dollars      # decommissioning cost
        
    elif type == "suction":

        anchorMatCost = 10.25 *mass # $ per kg mass
        anchorInstCost = 179331*euros2dollars   # installation cost
        anchorDecomCost = 125532*euros2dollars  # decommissioning cost
    
        
    elif type == "micropile":   
        
        raise ValueError('Micropile material costs are not yet supported')
        #anchorMatCost = 0.48 * mass # $ per kg mass
        anchorInstCost = 0                                  # installation cost
        anchorDecomCost = 0                                 # decommissioning cost
    
    elif type == "plate":   # cross between suction and plate
        
        raise ValueError('Plate material costs are not yet supported')
        #anchorMatCost = 0.45 * mass# $ per kg mass
        anchorInstCost = 0                      # installation cost
        anchorDecomCost = 0                     # decommissioning cost
    
    else:
        raise ValueError(f'getAnchorProps received an unsupported anchor type ({type})')
        
    # mooring line sizing:  Tension limit for QS: 50% MBS.  Or FOS = 2
    

    return anchorMatCost, anchorInstCost, anchorDecomCost   # [USD]


def getAnchorProps(fx, fz, type="drag-embedment", display=0):
    ''' ****OLD VERSION**** Calculates anchor required capacity and cost based on specified loadings and anchor type'''
    
    # for now this is based on API RP-2SK guidance for static analysis of permanent mooring systems
    # fx and fz are horizontal and vertical load components assumed to come from a dynamic (or equivalent) analysis.
    
    # mooring line tenLimit specified in yaml and inversed for a SF in constraints
    # take the line forces on the anchor and give 20% consideration for dynamic loads on lines
    # coefficients in front of fx and fz in each anchorType are the SF for that anchor for quasi-static (pages 30-31 of RP-2SK)
    
    # scale QS loads by 20% to approximate dynamic loads
    # fx = 1.2*fx
    # fz = 1.2*fz
    
    # note: capacity is measured here in kg force
    
    euros2dollars = 1.18    # the number of dollars there currently are in a euro (3-31-21)
    
    if type == "drag-embedment":
        capacity_x = 1.5*fx/9.81
        
        fzCost = 0
        #if fz > 0: 
        #    fzCost = 1e3*fz
        #    if display > 0:  print('WARNING: Nonzero vertical load specified for a drag embedment anchor.')   
        
        anchorMatCost = 0.188*capacity_x + fzCost   # material cost
        anchorInstCost = 163548*euros2dollars       # installation cost
        anchorDecomCost = 228967*euros2dollars      # decommissioning cost
        
    elif type == "suction":
        capacity_x = 1.6*fx/9.81
        capacity_z = 2.0*fz/9.81
        capacity = np.linalg.norm([capacity_x, capacity_z])    # overall capacity, assuming in any direction for now
        anchorMatCost = 1.08*capacity           # material cost
        anchorInstCost = 179331*euros2dollars   # installation cost
        anchorDecomCost = 125532*euros2dollars  # decommissioning cost
    
    elif type == 'deadweight-granite':
        capacity_x = 1.6*fx/9.81    # no safety factors given explicitly for deadweight anchors in the standards; assuming these safety factors are the same as piles/gravity/plates
        capacity_z = 2.0*fz/9.81    # no safety factors given explicitly for deadweight anchors in the standards; assuming these safety factors are the same as piles/gravity/plates
        capacity = np.linalg.norm([capacity_x, capacity_z])
        anchorMatCost = 0.05*capacity   # cost of a granite deadweight anchor is about $50 per ton (from Senu)
        anchorInstCost = 0
        anchorDecomCost = 0
    
    elif type == 'deadweight-concrete':
        capacity_x = 1.6*fx/9.81    # no safety factors given explicitly for deadweight anchors in the standards; assuming these safety factors are the same as piles/gravity/plates
        capacity_z = 2.0*fz/9.81    # no safety factors given explicitly for deadweight anchors in the standards; assuming these safety factors are the same as piles/gravity/plates
        capacity = np.linalg.norm([capacity_x, capacity_z])
        anchorMatCost = 0.075*capacity   # cost of a concrete deadweight anchor is about $75 per ton (from Senu)
        # reinforced concrete is about $1000 per ton ($1 per kg), but we don't need reinforced concrete for deadweight anchors
        anchorInstCost = 0
        anchorDecomCost = 0
    
    elif type == "plate":
        capacity_x = 2.0*fx/9.81
        capacity_z = 2.0*fz/9.81
        capacity = np.linalg.norm([capacity_x, capacity_z])    # overall capacity, assuming in any direction for now
        raise ValueError("plate anchors not yet supported")
        
    elif type == "micropile":    # note: no API guidance on this one
        capacity_x = 2.0*fx/9.81
        capacity_z = 2.0*fz/9.81
        capacity = np.linalg.norm([capacity_x, capacity_z]) # overall capacity, assuming in any direction for now
        anchorMatCost = (200000*1.2/500000)*capacity        # [(Euros*($/Euros)/kg)*kg] linear interpolation of material cost
        anchorInstCost = 0                                  # installation cost
        anchorDecomCost = 0                                 # decommissioning cost
    
    elif type == "SEPLA":   # cross between suction and plate
        capacity_x = 2.0*fx/9.81
        capacity_z = 2.0*fz/9.81
        capacity = np.linalg.norm([capacity_x, capacity_z])    # overall capacity, assuming in any direction for now
        anchorMatCost = 0.45*capacity           # material cost
        anchorInstCost = 0                      # installation cost
        anchorDecomCost = 0                     # decommissioning cost
    
    else:
        raise ValueError(f'getAnchorProps received an unsupported anchor type ({type})')
        
    # mooring line sizing:  Tension limit for QS: 50% MBS.  Or FOS = 2
    

    return anchorMatCost, anchorInstCost, anchorDecomCost   # [USD]

