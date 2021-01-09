module test_MD_Types_C

    use pFUnit_mod
    use MoorDyn_Types_C
    use MoorDyn_Types

    implicit none

    real, parameter :: test_float = 1.0
    character, parameter :: test_string = "test_string"
    logical, parameter :: test_bool = .TRUE.
    real, parameter :: test_array(6) = (/ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 /)

contains

    @test
    subroutine test_MD_InitInputType_C()

        ! Implicitly test the C-based type instantiation by supplying the
        ! expected value-types; for example, provide a float where a float is
        ! expected.
        ! Excplicitly test the C2F routine.
        ! Excplicitly test the F2C routine.

        type(MD_InitInputType) :: f_type
        type(MD_InitInputType_C) :: c_type

        ! Instantiate the C-based type with the expected value-types
        c_type%g = test_float
        c_type%rhoW = test_float
        c_type%WtrDepth = test_float
        c_type%PtfmInit = test_array
        c_type%FileName = test_string
        c_type%RootName = test_string
        c_type%Echo = test_bool
        c_type%DTIC = test_float
        c_type%TMaxIC = test_float
        c_type%CdScaleIC = test_float
        c_type%threshIC = test_float

        ! Do the C to Fortran conversion and test
        ! TODO: develop an array test
        ! TODO: develop a string comparison test
        f_type = MD_InitInputType_C2F(c_type)
        @assertEqual( test_float, f_type%g )
        @assertEqual( test_float, f_type%rhoW )
        @assertEqual( test_float, f_type%WtrDepth )
        @assertEqual( test_array, f_type%PtfmInit )
        ! @assertEqual( c_type%FileName, f_type%FileName )
        ! @assertEqual( c_type%RootName, f_type%RootName )
        @assertTrue( f_type%Echo )
        @assertEqual( test_float, f_type%DTIC )
        @assertEqual( test_float, f_type%TMaxIC )
        @assertEqual( test_float, f_type%CdScaleIC )
        @assertEqual( test_float, f_type%threshIC )

        ! Do the Fortran to C conversion and test
        c_type = MD_InitInputType_F2C(f_type)
        @assertEqual( test_float, real(c_type%g) )
        @assertEqual( test_float, real(c_type%rhoW) )
        @assertEqual( test_float, real(c_type%WtrDepth) )
        @assertEqual( test_array, real(c_type%PtfmInit) )
        ! ! @assertEqual( f_type%FileName, c_type%FileName )
        ! ! @assertEqual( f_type%RootName, c_type%RootName )
        @assertTrue( logical(c_type%Echo) )
        @assertEqual( test_float, real(c_type%DTIC) )
        @assertEqual( test_float, real(c_type%TMaxIC) )
        @assertEqual( test_float, real(c_type%CdScaleIC) )
        @assertEqual( test_float, real(c_type%threshIC) )

    end subroutine

    @test
    subroutine test_MD_ParameterType_C()

        type(MD_ParameterType) :: f_type
        type(MD_ParameterType_C) :: c_type

        ! ! Instantiate the C-based type with the expected value-types
        ! c_type%g = test_float
        ! c_type%rhoW = test_float
        ! c_type%WtrDepth = test_float
        ! c_type%PtfmInit = test_array
        ! c_type%FileName = test_string
        ! c_type%RootName = test_string
        ! c_type%Echo = test_bool
        ! c_type%DTIC = test_float
        ! c_type%TMaxIC = test_float
        ! c_type%CdScaleIC = test_float
        ! c_type%threshIC = test_float

        ! ! Do the C to Fortran conversion and test
        ! ! TODO: develop an array test
        ! ! TODO: develop a string comparison test
        ! f_type = MD_InitInputType_C2F(c_type)
        ! @assertEqual( test_float, f_type%g )
        ! @assertEqual( test_float, f_type%rhoW )
        ! @assertEqual( test_float, f_type%WtrDepth )
        ! @assertEqual( test_array, f_type%PtfmInit )
        ! ! @assertEqual( c_type%FileName, f_type%FileName )
        ! ! @assertEqual( c_type%RootName, f_type%RootName )
        ! @assertTrue( f_type%Echo )
        ! @assertEqual( test_float, f_type%DTIC )
        ! @assertEqual( test_float, f_type%TMaxIC )
        ! @assertEqual( test_float, f_type%CdScaleIC )
        ! @assertEqual( test_float, f_type%threshIC )

        ! ! Do the Fortran to C conversion and test
        ! c_type = MD_InitInputType_F2C(f_type)
        ! @assertEqual( test_float, real(c_type%g) )
        ! @assertEqual( test_float, real(c_type%rhoW) )
        ! @assertEqual( test_float, real(c_type%WtrDepth) )
        ! @assertEqual( test_array, real(c_type%PtfmInit) )
        ! ! ! @assertEqual( f_type%FileName, c_type%FileName )
        ! ! ! @assertEqual( f_type%RootName, c_type%RootName )
        ! @assertTrue( logical(c_type%Echo) )
        ! @assertEqual( test_float, real(c_type%DTIC) )
        ! @assertEqual( test_float, real(c_type%TMaxIC) )
        ! @assertEqual( test_float, real(c_type%CdScaleIC) )
        ! @assertEqual( test_float, real(c_type%threshIC) )

    end subroutine

end module
