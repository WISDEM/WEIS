import pytest
import sys
import numpy as np

# test local code; consider src layout in future to test installed code
sys.path.append('..')
import FrequencyDomain as fdo

capyTestFile = f'./test_data/mesh_converge_0.750_1.250.nc'

def test_read_capy_nc_wCapyShape():
    wCapy, addedMass, damping, fEx = fdo.read_capy_nc(capyTestFile)
    assert len(wCapy) == 28

def test_read_capy_nc_addedMassShape():
    wCapy, addedMass, damping, fEx = fdo.read_capy_nc(capyTestFile)
    assert addedMass.shape == (6, 6, 28)

def test_read_capy_nc_dampingShape():
    wCapy, addedMass, damping, fEx = fdo.read_capy_nc(capyTestFile)
    assert damping.shape == (6, 6, 28)

def test_read_capy_nc_fExShape():
    wCapy, addedMass, damping, fEx = fdo.read_capy_nc(capyTestFile)
    assert fEx.shape == (6, 28)

def test_read_capy_nc_fExComplex():
    wCapy, addedMass, damping, fEx = fdo.read_capy_nc(capyTestFile)
    assert fEx.dtype == 'complex128'

def test_read_capy_nc_wRangeCheck():
    with pytest.raises(ValueError):
        wDes = np.arange(0.01, 3, 0.01)
        wCapy, addedMass, damping, fEx = fdo.read_capy_nc(capyTestFile, wDes=wDes)

def test_read_capy_nc_addedMassVals():
    wCapy, addedMass, damping, fEx = fdo.read_capy_nc(capyTestFile)
    refAddedMass = np.loadtxt(f'./ref_data/capytaine_integration/wCapy-addedMass-surge.txt')
    assert max(abs(refAddedMass[:,1] - addedMass[0,0,:])) < 1e-12

def test_read_capy_nc_dampingVals():
    wCapy, addedMass, damping, fEx = fdo.read_capy_nc(capyTestFile)
    refDamping = np.loadtxt(f'./ref_data/capytaine_integration/wCapy-damping-surge.txt')
    assert max(abs(refDamping[:,1] - damping[0,0,:])) < 1e-12

def test_read_capy_nc_fExRealVals():
    wCapy, addedMass, damping, fEx = fdo.read_capy_nc(capyTestFile)
    refFExReal = np.loadtxt(f'./ref_data/capytaine_integration/wCapy-fExcitationReal-surge.txt')
    assert max(abs(refFExReal[:,1] - fEx[0,:].real)) < 1e-12

def test_read_capy_nc_fExImagVals():
    wCapy, addedMass, damping, fEx = fdo.read_capy_nc(capyTestFile)
    refFExImag = np.loadtxt(f'./ref_data/capytaine_integration/wCapy-fExcitationImag-surge.txt')
    assert max(abs(refFExImag[:,1] - fEx[0,:].imag)) < 1e-12

def test_read_capy_nc_addedMassInterpVals():
    wDes = np.arange(0.1, 2.8, 0.01)
    wDes, addedMassInterp, dampingInterp, fExInterp = fdo.read_capy_nc(capyTestFile, wDes=wDes)
    refAddedMassInterp = np.loadtxt(f'./ref_data/capytaine_integration/wDes-addedMassInterp-surge.txt')
    assert max(abs(refAddedMassInterp[:,1] - addedMassInterp[0,0,:])) < 1e-12

def test_read_capy_nc_dampingInterpVals():
    wDes = np.arange(0.1, 2.8, 0.01)
    wDes, addedMassInterp, dampingInterp, fExInterp = fdo.read_capy_nc(capyTestFile, wDes=wDes)
    refDampingInterp = np.loadtxt(f'./ref_data/capytaine_integration/wDes-dampingInterp-surge.txt')
    assert max(abs(refDampingInterp[:,1] - dampingInterp[0,0,:])) < 1e-12

def test_read_capy_nc_fExInterpRealVals():
    wDes = np.arange(0.1, 2.8, 0.01)
    wDes, addedMassInterp, dampingInterp, fExInterp = fdo.read_capy_nc(capyTestFile, wDes=wDes)
    refFExInterpReal = np.loadtxt(f'./ref_data/capytaine_integration/wDes-fExcitationInterpReal-surge.txt')
    assert max(abs(refFExInterpReal[:,1] - fExInterp[0,:].real)) < 1e-12

def test_read_capy_nc_fExInterpImagVals():
    wDes = np.arange(0.1, 2.8, 0.01)
    wDes, addedMassInterp, dampingInterp, fExInterp = fdo.read_capy_nc(capyTestFile, wDes=wDes)
    refFExInterpImag = np.loadtxt(f'./ref_data/capytaine_integration/wDes-fExcitationInterpImag-surge.txt')
    assert max(abs(refFExInterpImag[:,1] - fExInterp[0,:].imag)) < 1e-12

def test_call_capy_addedMassShape():
    wRange = np.arange(0.1, 2.9, 0.1)
    meshFName = f'./test_data/float.gdf'
    wCapy, addedMass, damping, fEx = fdo.call_capy(meshFName, wRange)
    assert addedMass.shape == (6, 6, 28)

def test_call_capy_dampingShape():
    wRange = np.arange(0.1, 2.9, 0.1)
    meshFName = f'./test_data/float.gdf'
    wCapy, addedMass, damping, fEx = fdo.call_capy(meshFName, wRange)
    assert damping.shape == (6, 6, 28)

def test_call_capy_fExShape():
    wRange = np.arange(0.1, 2.9, 0.1)
    meshFName = f'./test_data/float.gdf'
    wCapy, addedMass, damping, fEx = fdo.call_capy(meshFName, wRange)
    assert fEx.shape == (6, 28)

def test_call_capy_fExComplex():
    wRange = np.arange(0.1, 2.9, 0.1)
    meshFName = f'./test_data/float.gdf'
    wCapy, addedMass, damping, fEx = fdo.call_capy(meshFName, wRange)
    assert fEx.dtype == 'complex128'

def test_call_capy_addedMassVals():
    wRange = np.arange(0.1, 2.9, 0.1)
    meshFName = f'./test_data/float.gdf'
    refNc = f'./ref_data/capytaine_integration/floatData.nc'
    wCapy, addedMass, damping, fEx = fdo.call_capy(meshFName, wRange)
    refwCapy, refAddedMass, refDamping, refFEx = fdo.read_capy_nc(refNc)
    assert max(abs(refAddedMass[0,0,:] - addedMass[0,0,:])) < 1e-12

def test_call_capy_dampingVals():
    wRange = np.arange(0.1, 2.9, 0.1)
    meshFName = f'./test_data/float.gdf'
    refNc = f'./ref_data/capytaine_integration/floatData.nc'
    wCapy, addedMass, damping, fEx = fdo.call_capy(meshFName, wRange)
    refwCapy, refAddedMass, refDamping, refFEx = fdo.read_capy_nc(refNc)
    assert max(abs(refDamping[0,0,:] - damping[0,0,:])) < 1e-12

def test_call_capy_fExRealVals():
    wRange = np.arange(0.1, 2.9, 0.1)
    meshFName = f'./test_data/float.gdf'
    refNc = f'./ref_data/capytaine_integration/floatData.nc'
    wCapy, addedMass, damping, fEx = fdo.call_capy(meshFName, wRange)
    refwCapy, refAddedMass, refDamping, refFEx = fdo.read_capy_nc(refNc)
    assert max(abs(refFEx[0,:].real - fEx[0,:].real)) < 1e-12

def test_call_capy_fExImagVals():
    wRange = np.arange(0.1, 2.9, 0.1)
    meshFName = f'./test_data/float.gdf'
    refNc = f'./ref_data/capytaine_integration/floatData.nc'
    wCapy, addedMass, damping, fEx = fdo.call_capy(meshFName, wRange)
    refwCapy, refAddedMass, refDamping, refFEx = fdo.read_capy_nc(refNc)
    assert max(abs(refFEx[0,:].imag - fEx[0,:].imag)) < 1e-12
