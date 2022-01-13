import yaml
import numpy as np
import matplotlib.pyplot as plt
import pickle
from raft_model import runRAFT



def getOutputs(model):
    
    model.calcOutputs()
    fowt = model.fowtList[0]
    
    mass     = fowt.M_struc[0,0]
    displ    = fowt.V*1025
    GMT      = fowt.body.rM[2]-fowt.rCG_TOT[2]
    offset   = np.hypot(model.results['means']['platform offset'][0],model.results['means']['platform offset'][1])
    pitch    = model.results['means']['platform offset'][4]*180/np.pi
    #tensions = model.results['means']['fairlead tensions']
    
    return (mass, displ, GMT, offset, pitch)


name = 'VolturnUS-S'
#name = 'OC3spar'

with open('../designs/'+name+'.yaml') as file:
    design = yaml.load(file, Loader=yaml.FullLoader)

baseline = runRAFT(design)


ccD = design['platform']['members'][0]['d']
ocD = design['platform']['members'][1]['d']
T   = design['platform']['members'][0]['rA'][2]
ocR = design['platform']['members'][1]['rA'][0]
pH  = (design['platform']['members'][2]['rA'][2]-T)*2

lower = 0.75
upper = 1.25
ballast = True

ccDs = [ccD*lower, ccD, ccD*upper]
ocDs = [ocD*lower, ocD, ocD*upper]
Ts   = [  T*lower,   T,   T*upper]
ocRs = [ocR*lower, ocR, ocR*upper]
pHs  = [ pH*lower,  pH,  pH*upper]

M = np.zeros([3,3,3,3,3])
pV = np.zeros([3,3,3,3,3])
GMT = np.zeros([3,3,3,3,3])
XY = np.zeros([3,3,3,3,3])
P = np.zeros([3,3,3,3,3])

#%%
for a in ccDs:
    design['platform']['members'][0]['d'] = a
    design['platform']['members'][2]['rA'][0] = design['platform']['members'][2]['rA'][0]*(a/ccD)
    design['platform']['members'][3]['rA'][0] = design['platform']['members'][3]['rA'][0]*(a/ccD)
    for b in ocDs:
        design['platform']['members'][1]['d'] = b
        design['platform']['members'][2]['rB'][0] = design['platform']['members'][1]['rA'][0] - b/2
        design['platform']['members'][3]['rB'][0] = design['platform']['members'][1]['rB'][0] - b/2
        design['mooring']['points'][3]['location'][0] = design['platform']['members'][1]['rA'][0] - b/2 # minus to be relative, others are absolute
        design['mooring']['points'][4]['location'][0] = (design['platform']['members'][1]['rA'][0] + b/2)*np.cos(np.deg2rad(60))
        design['mooring']['points'][4]['location'][1] = (design['platform']['members'][1]['rA'][0] + b/2)*np.sin(np.deg2rad(60))
        design['mooring']['points'][5]['location'][0] = (design['platform']['members'][1]['rA'][0] + b/2)*np.cos(np.deg2rad(300))
        design['mooring']['points'][5]['location'][1] = (design['platform']['members'][1]['rA'][0] + b/2)*np.sin(np.deg2rad(300))
        for c in Ts:
            design['platform']['members'][0]['rA'][2] = c
            design['platform']['members'][1]['rA'][2] = c
            design['platform']['members'][2]['rA'][2] = c + (design['platform']['members'][2]['d'][1]/2)
            design['platform']['members'][2]['rB'][2] = c + (design['platform']['members'][2]['d'][1]/2)
            for d in ocRs:
                design['platform']['members'][1]['rA'][0] = d
                design['platform']['members'][1]['rB'][0] = d
                design['platform']['members'][2]['rB'][0] = d - design['platform']['members'][1]['d']/2
                design['platform']['members'][3]['rB'][0] = d - design['platform']['members'][1]['d']/2
                design['mooring']['points'][3]['location'][0] = -d - design['platform']['members'][1]['d']/2
                design['mooring']['points'][4]['location'][0] = (d + design['platform']['members'][1]['d']/2)*np.cos(np.deg2rad(60))
                design['mooring']['points'][4]['location'][1] = (d + design['platform']['members'][1]['d']/2)*np.sin(np.deg2rad(60))
                design['mooring']['points'][5]['location'][0] = (d + design['platform']['members'][1]['d']/2)*np.cos(np.deg2rad(300))
                design['mooring']['points'][5]['location'][1] = (d + design['platform']['members'][1]['d']/2)*np.sin(np.deg2rad(300))
                for e in pHs:
                    design['platform']['members'][2]['d'][1] = e
                    design['platform']['members'][2]['rA'][2] = design['platform']['members'][0]['rA'][2] + e/2
                    design['platform']['members'][2]['rB'][2] = design['platform']['members'][1]['rA'][2] + e/2
                    
                    print('------')
                    print(f'RUNNING {a}-{b}-{c}-{d}-{e}')
                    print('------')
                    
                    model = runRAFT(design, ballast=ballast)
                    
                    (mass, displ, gmt, offset, pitch) = getOutputs(model)
                    M[ccDs.index(a),ocDs.index(b),Ts.index(c),ocRs.index(d),pHs.index(e)] = mass
                    pV[ccDs.index(a),ocDs.index(b),Ts.index(c),ocRs.index(d),pHs.index(e)] = displ
                    GMT[ccDs.index(a),ocDs.index(b),Ts.index(c),ocRs.index(d),pHs.index(e)] = gmt
                    XY[ccDs.index(a),ocDs.index(b),Ts.index(c),ocRs.index(d),pHs.index(e)] = offset
                    P[ccDs.index(a),ocDs.index(b),Ts.index(c),ocRs.index(d),pHs.index(e)] = pitch

#%%
'''
sweep = dict(mass=M, displ=pV, gmt=GMT, offset=XY, pitch=P)

with open(f'sweep_{lower}-{upper}_ballast={ballast}.pkl', 'wb') as pfile:
    pickle.dump(sweep, pfile)
'''
with open(f'sweep-{lower}-{upper}.pkl', 'rb') as pfile:
    sweep = pickle.load(pfile)

M = sweep['mass']
pV = sweep['displ']
GMT = sweep['gmt']
XY = sweep['offset']
P = sweep['pitch']


#%%
#### Mass
         
fig, ax = plt.subplots(4,4, figsize=(24,16))

F = np.zeros([len(ocRs),len(pHs)])

title = False

pontoon_heights, outer_column_radii = np.meshgrid(pHs, ocRs)

for iocR in range(len(ocRs)):
    for ipH in range(len(pHs)):
        F[iocR,ipH] = M[1,1,1,iocR,ipH]

pHocR = ax[0,0].contourf(pontoon_heights, outer_column_radii, F)
cbar = fig.colorbar(pHocR, ax=ax[0,0], label='Platform Mass (kg)')
ax[0,0].set_ylabel('Outer Column Radius (m)')
ax[0,0].set_xlabel('Pontoon Height (m)')
if title: ax[0,0].set_title('Platform Mass pH vs. ocR')


pontoon_heights, drafts = np.meshgrid(pHs, Ts)

for iT in range(len(Ts)):
    for ipH in range(len(pHs)):
        F[iT,ipH] = M[1,1,iT,1,ipH]

pHT = ax[0,1].contourf(pontoon_heights, drafts, F)
cbar = fig.colorbar(pHT, ax=ax[0,1], label='Platform Mass (kg)')
ax[0,1].set_ylabel('Draft (m)')
ax[0,1].set_xlabel('Pontoon Height (m)')
if title: ax[0,1].set_title('Platform Mass pH vs. T')


pontoon_heights, outer_column_diameters = np.meshgrid(pHs, ocDs)

for iocD in range(len(ocDs)):
    for ipH in range(len(pHs)):
        F[iocD,ipH] = M[1,iocD,1,1,ipH]

pHocD = ax[0,2].contourf(pontoon_heights, outer_column_diameters, F)
cbar = fig.colorbar(pHocD, ax=ax[0,2], label='Platform Mass (kg)')
ax[0,2].set_ylabel('Outer Column Diameter (m)')
ax[0,2].set_xlabel('Pontoon Height (m)')
if title: ax[0,2].set_title('Platform Mass pH vs. ocD')

pontoon_heights, center_column_diameters = np.meshgrid(pHs, ccDs)

for iccD in range(len(ccDs)):
    for ipH in range(len(pHs)):
        F[iccD,ipH] = M[iccD,1,1,1,ipH]

pHccD = ax[0,3].contourf(pontoon_heights, center_column_diameters, F)
cbar = fig.colorbar(pHccD, ax=ax[0,3], label='Platform Mass (kg)')
ax[0,3].set_ylabel('Center Column Diameter (m)')
ax[0,3].set_xlabel('Pontoon Height (m)')
if title: ax[0,3].set_title('Platform Mass pH vs. ccD')

# --------------

outer_column_radii, pontoon_heights = np.meshgrid(ocRs, pHs)

for ipH in range(len(pHs)):
    for iocR in range(len(ocRs)):
        F[ipH,iocR] = M[1,1,1,iocR,ipH]

ocRpH = ax[1,0].contourf(outer_column_radii,pontoon_heights, F)
cbar = fig.colorbar(ocRpH, ax=ax[1,0], label='Platform Mass (kg)')
ax[1,0].set_xlabel('Outer Column Radius (m)')
ax[1,0].set_ylabel('Pontoon Height (m)')
if title: ax[1,0].set_title('Platform Mass ocR vs. pH')


outer_column_radii, drafts = np.meshgrid(ocRs, Ts)

for iT in range(len(Ts)):
    for iocR in range(len(ocRs)):
        F[iT,iocR] = M[1,1,iT,iocR,1]

ocRT = ax[1,1].contourf(outer_column_radii, drafts, F)
cbar = fig.colorbar(ocRT, ax=ax[1,1], label='Platform Mass (kg)')
ax[1,1].set_ylabel('Draft (m)')
ax[1,1].set_xlabel('Outer Column Radius (m)')
if title: ax[1,1].set_title('Platform Mass ocR vs. T')


outer_column_radii, outer_column_diameters = np.meshgrid(ocRs, ocDs)

for iocD in range(len(ocDs)):
    for iocR in range(len(ocRs)):
        F[iocD,iocR] = M[1,iocD,1,iocR,1]

ocRocD = ax[1,2].contourf(outer_column_radii, outer_column_diameters, F)
cbar = fig.colorbar(ocRocD, ax=ax[1,2], label='Platform Mass (kg)')
ax[1,2].set_ylabel('Outer Column Diameter (m)')
ax[1,2].set_xlabel('Outer Column Radius (m)')
if title: ax[1,2].set_title('Platform Mass ocR vs. ocD')

outer_column_radii, center_column_diameters = np.meshgrid(ocRs, ccDs)

for iccD in range(len(ccDs)):
    for iocR in range(len(ocRs)):
        F[iccD,iocR] = M[iccD,1,1,iocR,1]

ocRccD = ax[1,3].contourf(outer_column_radii, center_column_diameters, F)
cbar = fig.colorbar(ocRccD, ax=ax[1,3], label='Platform Mass (kg)')
ax[1,3].set_ylabel('Center Column Diameter (m)')
ax[1,3].set_xlabel('Outer Column Radius (m)')
if title: ax[1,3].set_title('Platform Mass ocR vs. ccD')


# --------------

drafts, pontoon_heights = np.meshgrid(Ts, pHs)

for ipH in range(len(pHs)):
    for iT in range(len(Ts)):
        F[ipH,iT] = M[1,1,iT,1,ipH]

TpH = ax[2,0].contourf(drafts, pontoon_heights, F)
cbar = fig.colorbar(TpH, ax=ax[2,0], label='Platform Mass (kg)')
ax[2,0].set_xlabel('Draft (m)')
ax[2,0].set_ylabel('Pontoon Height (m)')
if title: ax[2,0].set_title('Platform Mass T vs. pH')


drafts, outer_column_radii = np.meshgrid(Ts, ocRs)

for iocR in range(len(ocRs)):
    for iT in range(len(Ts)):
        F[iocR,iT] = M[1,1,iT,iocR,1]

TocR = ax[2,1].contourf(drafts, outer_column_radii, F)
cbar = fig.colorbar(TocR, ax=ax[2,1], label='Platform Mass (kg)')
ax[2,1].set_xlabel('Draft (m)')
ax[2,1].set_ylabel('Outer Column Radius (m)')
if title: ax[2,1].set_title('Platform Mass T vs. ocR')


drafts, outer_column_diameters = np.meshgrid(Ts, ocDs)

for iocD in range(len(ocDs)):
    for iT in range(len(Ts)):
        F[iocD,iT] = M[1,iocD,iT,1,1]

TocD = ax[2,2].contourf(drafts, outer_column_diameters, F)
cbar = fig.colorbar(TocD, ax=ax[2,2], label='Platform Mass (kg)')
ax[2,2].set_ylabel('Outer Column Diameter (m)')
ax[2,2].set_xlabel('Draft (m)')
if title: ax[2,2].set_title('Platform Mass T vs. ocD')

drafts, center_column_diameters = np.meshgrid(Ts, ccDs)

for iccD in range(len(ccDs)):
    for iT in range(len(Ts)):
        F[iccD,iT] = M[iccD,1,iT,1,1]

TccD = ax[2,3].contourf(drafts, center_column_diameters, F)
cbar = fig.colorbar(TccD, ax=ax[2,3], label='Platform Mass (kg)')
ax[2,3].set_ylabel('Center Column Diameter (m)')
ax[2,3].set_xlabel('Draft (m)')
if title: ax[2,3].set_title('Platform Mass T vs. ccD')


# --------------

outer_column_diameters, pontoon_heights = np.meshgrid(ocDs, pHs)

for ipH in range(len(pHs)):
    for iocD in range(len(ocDs)):
        F[ipH,iocD] = M[1,iocD,1,1,ipH]

ocDpH = ax[3,0].contourf(outer_column_diameters, pontoon_heights, F)
cbar = fig.colorbar(ocDpH, ax=ax[3,0], label='Platform Mass (kg)')
ax[3,0].set_xlabel('Outer Column Diameter (m)')
ax[3,0].set_ylabel('Pontoon Height (m)')
if title: ax[3,0].set_title('Platform Mass ocD vs. pH')


outer_column_diameters, outer_column_radii = np.meshgrid(ocDs, ocRs)

for iocR in range(len(ocRs)):
    for iocD in range(len(ocDs)):
        F[iocR,iocD] = M[1,iocD,1,iocR,1]

ocDocR = ax[3,1].contourf(outer_column_diameters, outer_column_radii, F)
cbar = fig.colorbar(ocDocR, ax=ax[3,1], label='Platform Mass (kg)')
ax[3,1].set_xlabel('Outer Column Diameter (m)')
ax[3,1].set_ylabel('Outer Column Radius (m)')
if title: ax[3,1].set_title('Platform Mass ocD vs. ocR')


outer_column_diameters, drafts = np.meshgrid(ocDs, Ts)

for iT in range(len(Ts)):
    for iocD in range(len(ocDs)):
        F[iT,iocD] = M[1,iocD,iT,1,1]

ocDT = ax[3,2].contourf(outer_column_diameters, drafts, F)
cbar = fig.colorbar(ocDT, ax=ax[3,2], label='Platform Mass (kg)')
ax[3,2].set_xlabel('Outer Column Diameter (m)')
ax[3,2].set_ylabel('Draft (m)')
if title: ax[3,2].set_title('Platform Mass ocD vs. T')

outer_column_diameters, center_column_diameters = np.meshgrid(ocDs, ccDs)

for iccD in range(len(ccDs)):
    for iocD in range(len(ocDs)):
        F[iccD,iocD] = M[iccD,iocD,1,1,1]

ocDccD = ax[3,3].contourf(outer_column_diameters, center_column_diameters, F)
cbar = fig.colorbar(ocDccD, ax=ax[3,3], label='Platform Mass (kg)')
ax[3,3].set_ylabel('Center Column Diameter (m)')
ax[3,3].set_xlabel('Outer Column Diameter (m)')
if title: ax[3,3].set_title('Platform Mass ocD vs. ccD')



fig.tight_layout()




#### Displacement

fig, ax = plt.subplots(4,4, figsize=(24,16))

F = np.zeros([len(ocRs),len(pHs)])



pontoon_heights, outer_column_radii = np.meshgrid(pHs, ocRs)

for iocR in range(len(ocRs)):
    for ipH in range(len(pHs)):
        F[iocR,ipH] = pV[1,1,1,iocR,ipH]

pHocR = ax[0,0].contourf(pontoon_heights, outer_column_radii, F)
cbar = fig.colorbar(pHocR, ax=ax[0,0], label='Platform Displacement (kg)')
ax[0,0].set_ylabel('Outer Column Radius (m)')
ax[0,0].set_xlabel('Pontoon Height (m)')
ax[0,0].set_title('Platform Displacement pH vs. ocR')


pontoon_heights, drafts = np.meshgrid(pHs, Ts)

for iT in range(len(Ts)):
    for ipH in range(len(pHs)):
        F[iT,ipH] = pV[1,1,iT,1,ipH]

pHT = ax[0,1].contourf(pontoon_heights, drafts, F)
cbar = fig.colorbar(pHT, ax=ax[0,1], label='Platform Displacement (kg)')
ax[0,1].set_ylabel('Draft (m)')
ax[0,1].set_xlabel('Pontoon Height (m)')
ax[0,1].set_title('Platform Displacement pH vs. T')


pontoon_heights, outer_column_diameters = np.meshgrid(pHs, ocDs)

for iocD in range(len(ocDs)):
    for ipH in range(len(pHs)):
        F[iocD,ipH] = pV[1,iocD,1,1,ipH]

pHocD = ax[0,2].contourf(pontoon_heights, outer_column_diameters, F)
cbar = fig.colorbar(pHocD, ax=ax[0,2], label='Platform Displacement (kg)')
ax[0,2].set_ylabel('Outer Column Diameter (m)')
ax[0,2].set_xlabel('Pontoon Height (m)')
ax[0,2].set_title('Platform Displacement pH vs. ocD')

pontoon_heights, center_column_diameters = np.meshgrid(pHs, ccDs)

for iccD in range(len(ccDs)):
    for ipH in range(len(pHs)):
        F[iccD,ipH] = pV[iccD,1,1,1,ipH]

pHccD = ax[0,3].contourf(pontoon_heights, center_column_diameters, F)
cbar = fig.colorbar(pHccD, ax=ax[0,3], label='Platform Displacement (kg)')
ax[0,3].set_ylabel('Center Column Diameter (m)')
ax[0,3].set_xlabel('Pontoon Height (m)')
ax[0,3].set_title('Platform Displacement pH vs. ccD')

# --------------

outer_column_radii, pontoon_heights = np.meshgrid(ocRs, pHs)

for ipH in range(len(pHs)):
    for iocR in range(len(ocRs)):
        F[ipH,iocR] = pV[1,1,1,iocR,ipH]

ocRpH = ax[1,0].contourf(outer_column_radii,pontoon_heights, F)
cbar = fig.colorbar(ocRpH, ax=ax[1,0], label='Platform Displacement (kg)')
ax[1,0].set_xlabel('Outer Column Radius (m)')
ax[1,0].set_ylabel('Pontoon Height (m)')
ax[1,0].set_title('Platform Displacement ocR vs. pH')


outer_column_radii, drafts = np.meshgrid(ocRs, Ts)

for iT in range(len(Ts)):
    for iocR in range(len(ocRs)):
        F[iT,iocR] = pV[1,1,iT,iocR,1]

ocRT = ax[1,1].contourf(outer_column_radii, drafts, F)
cbar = fig.colorbar(ocRT, ax=ax[1,1], label='Platform Displacement (kg)')
ax[1,1].set_ylabel('Draft (m)')
ax[1,1].set_xlabel('Outer Column Radius (m)')
ax[1,1].set_title('Platform Displacement ocR vs. T')


outer_column_radii, outer_column_diameters = np.meshgrid(ocRs, ocDs)

for iocD in range(len(ocDs)):
    for iocR in range(len(ocRs)):
        F[iocD,iocR] = pV[1,iocD,1,iocR,1]

ocRocD = ax[1,2].contourf(outer_column_radii, outer_column_diameters, F)
cbar = fig.colorbar(ocRocD, ax=ax[1,2], label='Platform Displacement (kg)')
ax[1,2].set_ylabel('Outer Column Diameter (m)')
ax[1,2].set_xlabel('Outer Column Radius (m)')
ax[1,2].set_title('Platform Displacement ocR vs. ocD')

outer_column_radii, center_column_diameters = np.meshgrid(ocRs, ccDs)

for iccD in range(len(ccDs)):
    for iocR in range(len(ocRs)):
        F[iccD,iocR] = pV[iccD,1,1,iocR,1]

ocRccD = ax[1,3].contourf(outer_column_radii, center_column_diameters, F)
cbar = fig.colorbar(ocRccD, ax=ax[1,3], label='Platform Displacement (kg)')
ax[1,3].set_ylabel('Center Column Diameter (m)')
ax[1,3].set_xlabel('Outer Column Radius (m)')
ax[1,3].set_title('Platform Displacement ocR vs. ccD')


# --------------

drafts, pontoon_heights = np.meshgrid(Ts, pHs)

for ipH in range(len(pHs)):
    for iT in range(len(Ts)):
        F[ipH,iT] = pV[1,1,iT,1,ipH]

TpH = ax[2,0].contourf(drafts, pontoon_heights, F)
cbar = fig.colorbar(TpH, ax=ax[2,0], label='Platform Displacement (kg)')
ax[2,0].set_xlabel('Draft (m)')
ax[2,0].set_ylabel('Pontoon Height (m)')
ax[2,0].set_title('Platform Displacement T vs. pH')


drafts, outer_column_radii = np.meshgrid(Ts, ocRs)

for iocR in range(len(ocRs)):
    for iT in range(len(Ts)):
        F[iocR,iT] = pV[1,1,iT,iocR,1]

TocR = ax[2,1].contourf(drafts, outer_column_radii, F)
cbar = fig.colorbar(TocR, ax=ax[2,1], label='Platform Displacement (kg)')
ax[2,1].set_xlabel('Draft (m)')
ax[2,1].set_ylabel('Outer Column Radius (m)')
ax[2,1].set_title('Platform Displacement T vs. ocR')


drafts, outer_column_diameters = np.meshgrid(Ts, ocDs)

for iocD in range(len(ocDs)):
    for iT in range(len(Ts)):
        F[iocD,iT] = pV[1,iocD,iT,1,1]

TocD = ax[2,2].contourf(drafts, outer_column_diameters, F)
cbar = fig.colorbar(TocD, ax=ax[2,2], label='Platform Displacement (kg)')
ax[2,2].set_ylabel('Outer Column Diameter (m)')
ax[2,2].set_xlabel('Draft (m)')
ax[2,2].set_title('Platform Displacement T vs. ocD')

drafts, center_column_diameters = np.meshgrid(Ts, ccDs)

for iccD in range(len(ccDs)):
    for iT in range(len(Ts)):
        F[iccD,iT] = pV[iccD,1,iT,1,1]

TccD = ax[2,3].contourf(drafts, center_column_diameters, F)
cbar = fig.colorbar(TccD, ax=ax[2,3], label='Platform Displacement (kg)')
ax[2,3].set_ylabel('Center Column Diameter (m)')
ax[2,3].set_xlabel('Draft (m)')
ax[2,3].set_title('Platform Displacement T vs. ccD')


# --------------

outer_column_diameters, pontoon_heights = np.meshgrid(ocDs, pHs)

for ipH in range(len(pHs)):
    for iocD in range(len(ocDs)):
        F[ipH,iocD] = pV[1,iocD,1,1,ipH]

ocDpH = ax[3,0].contourf(outer_column_diameters, pontoon_heights, F)
cbar = fig.colorbar(ocDpH, ax=ax[3,0], label='Platform Displacement (kg)')
ax[3,0].set_xlabel('Outer Column Diameter (m)')
ax[3,0].set_ylabel('Pontoon Height (m)')
ax[3,0].set_title('Platform Displacement ocD vs. pH')


outer_column_diameters, outer_column_radii = np.meshgrid(ocDs, ocRs)

for iocR in range(len(ocRs)):
    for iocD in range(len(ocDs)):
        F[iocR,iocD] = pV[1,iocD,1,iocR,1]

ocDocR = ax[3,1].contourf(outer_column_diameters, outer_column_radii, F)
cbar = fig.colorbar(ocDocR, ax=ax[3,1], label='Platform Displacement (kg)')
ax[3,1].set_xlabel('Outer Column Diameter (m)')
ax[3,1].set_ylabel('Outer Column Radius (m)')
ax[3,1].set_title('Platform Displacement ocD vs. ocR')


outer_column_diameters, drafts = np.meshgrid(ocDs, Ts)

for iT in range(len(Ts)):
    for iocD in range(len(ocDs)):
        F[iT,iocD] = pV[1,iocD,iT,1,1]

ocDT = ax[3,2].contourf(outer_column_diameters, drafts, F)
cbar = fig.colorbar(ocDT, ax=ax[3,2], label='Platform Displacement (kg)')
ax[3,2].set_xlabel('Outer Column Diameter (m)')
ax[3,2].set_ylabel('Draft (m)')
ax[3,2].set_title('Platform Displacement ocD vs. T')

outer_column_diameters, center_column_diameters = np.meshgrid(ocDs, ccDs)

for iccD in range(len(ccDs)):
    for iocD in range(len(ocDs)):
        F[iccD,iocD] = pV[iccD,iocD,1,1,1]

ocDccD = ax[3,3].contourf(outer_column_diameters, center_column_diameters, F)
cbar = fig.colorbar(ocDccD, ax=ax[3,3], label='Platform Displacement (kg)')
ax[3,3].set_ylabel('Center Column Diameter (m)')
ax[3,3].set_xlabel('Outer Column Diameter (m)')
ax[3,3].set_title('Platform Displacement ocD vs. ccD')



fig.tight_layout()









