!**********************************************************************************************************************************
! LICENSING
! Copyright (C) 2015  Matthew Hall
!
!    This file is part of MoorDyn.
!
! Licensed under the Apache License, Version 2.0 (the "License");
! you may not use this file except in compliance with the License.
! You may obtain a copy of the License at
!
!     http://www.apache.org/licenses/LICENSE-2.0
!
! Unless required by applicable law or agreed to in writing, software
! distributed under the License is distributed on an "AS IS" BASIS,
! WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
! See the License for the specific language governing permissions and
! limitations under the License.
!
!**********************************************************************************************************************************
MODULE MoorDynAPI

USE MoorDyn
USE MoorDyn_Types
USE MoorDyn_Types_C
USE iso_c_binding

implicit none

PUBLIC :: MD_Init_C
PUBLIC :: MD_UpdateStates_C
PUBLIC :: MD_End_C

CONTAINS

SUBROUTINE MD_Init_C(InitInp_C, u_C, p_C, x_C, xd_C, z_C, other_C, y_C, m_C, DTcoupling_C, InitOut_C) BIND(C, NAME="MD_Init_C")
   TYPE(MD_InitInputType_C),       INTENT(INOUT)  :: InitInp_C
   TYPE(MD_InputType_C),           INTENT(  OUT)  :: u_C
   TYPE(MD_ParameterType_C),       INTENT(  OUT)  :: p_C
   TYPE(MD_ContinuousStateType_C), INTENT(  OUT)  :: x_C
   TYPE(MD_DiscreteStateType_C),   INTENT(  OUT)  :: xd_C
   TYPE(MD_ConstraintStateType_C), INTENT(  OUT)  :: z_C
   TYPE(MD_OtherStateType_C),      INTENT(  OUT)  :: other_C
   TYPE(MD_OutputType_C),          INTENT(  OUT)  :: y_C
   TYPE(MD_MiscVarType_C),         INTENT(  OUT)  :: m_C
   REAL(C_DOUBLE),                 INTENT(INOUT)  :: DTcoupling_C
   TYPE(MD_InitOutputType_C),      INTENT(INOUT)  :: InitOut_C
   ! INTEGER(IntKi),               INTENT(  OUT)  :: ErrStat     ! Error status of the operation
   ! CHARACTER(*),                 INTENT(  OUT)  :: ErrMsg      ! Error message if ErrStat /= ErrID_None

   ! Local variables
   TYPE(MD_InitInputType)        :: InitInp
   TYPE(MD_InputType)            :: u
   TYPE(MD_ParameterType)        :: p
   TYPE(MD_ContinuousStateType)  :: x
   TYPE(MD_DiscreteStateType)    :: xd
   TYPE(MD_ConstraintStateType)  :: z
   TYPE(MD_OtherStateType)       :: other
   TYPE(MD_OutputType)           :: y
   TYPE(MD_MiscVarType)          :: m
   REAL(DbKi)                    :: DTcoupling
   TYPE(MD_InitOutputType)       :: InitOut
   INTEGER(IntKi)                :: ErrStat
   CHARACTER(ErrMsgLen)          :: ErrMsg

   integer, pointer :: pa(:), ps
   real, pointer :: a(:)

   print *, "Starting MD_Init_C"

   ! Convert the C types to Fortran for all INTENT(IN) arguments
   DTcoupling = REAL(DTcoupling_C, DbKi)

   InitInp = MD_InitInputType_C2F(InitInp_C)

   ! CHARACTER(ChanLen) , DIMENSION(:), ALLOCATABLE  :: OutList      !< string containing list of output channels requested in input file [-]

   call MD_Init(InitInp, u, p, x, xd, z, other, y, m, DTcoupling, InitOut, ErrStat, ErrMsg)

   
   ! Convert the Fortran types to C for all INTENT(OUT) arguments
   DTcoupling_C = REAL(DTcoupling, C_DOUBLE)

   InitInp_C = MD_InitInputType_F2C(InitInp)

   ! struct_MD_InputType - Fortran to C
   ! u_C%object = u%object

   p_C = MD_ParameterType_F2C(p)

   ! struct_MD_ContinuousStateType - Fortran to C
   ! x_C%object = x%object
   ! x_C%states = x%states
   ! x_C%states_Len = x%states_Len

   ! struct_MD_DiscreteStateType - Fortran to C
   ! xd_C%object = xd%object
   xd_C%dummy = xd%dummy

   ! struct_MD_ConstraintStateType - Fortran to C
   ! z_C%object = z%object
   z_C%dummy = z%dummy

   ! struct_MD_OtherStateType - Fortran to C
   ! other_C%object = other%object
   other_C%dummy = other%dummy

   ! struct_MD_OutputType - Fortran to C
   ! y_C%object = y%object
   ! y_C%WriteOutput = y%WriteOutput
   ! y_C%WriteOutput_Len = y%WriteOutput_Len

   ! struct_MD_MiscVarType - Fortran to C
   ! m_C%object = m%object
   ! m_C%FairIdList = m%FairIdList
   ! m_C%FairIdList_Len = m%FairIdList_Len
   ! m_C%ConnIdList = m%ConnIdList
   ! m_C%ConnIdList_Len = m%ConnIdList_Len
   ! m_C%LineStateIndList = m%LineStateIndList
   ! m_C%LineStateIndList_Len = m%LineStateIndList_Len
   ! m_C%MDWrOutput = m%MDWrOutput
   ! m_C%MDWrOutput_Len = m%MDWrOutput_Len

   ! struct_MD_InitOutputType - Fortran to C
   ! InitOut_C%object = InitOut%object
   ! InitOut_C%writeOutputHdr = InitOut%writeOutputHdr
   ! InitOut_C%writeOutputHdr_Len = InitOut%writeOutputHdr_Len
   ! InitOut_C%writeOutputUnt = InitOut%writeOutputUnt
   ! InitOut_C%writeOutputUnt_Len = InitOut%writeOutputUnt_Len

   IF ( ErrStat >= AbortErrLev ) THEN
      print *, TRIM(ErrMsg)//' MD_Init:'//TRIM(ErrMsg)
      RETURN
   END IF
   print *, "Ending MD_Init_C"
end subroutine

SUBROUTINE MD_UpdateStates_C( t_C, n, u_len, u_C, p_C, x_C, xd_C, z_C, other_C, m_C) BIND(C, NAME="MD_UpdateStates_C")
   REAL(C_DOUBLE)                    , INTENT(IN   ) :: t_C
   INTEGER(C_INT)                    , INTENT(IN   ) :: n
   INTEGER(C_INT)                    , INTENT(IN   ) :: u_len
   TYPE(MD_InputType_C)              , INTENT(INOUT) :: u_C(u_len)  ! INTENT(INOUT) ! had to change this to INOUT
   ! REAL(DbKi)                      , INTENT(IN   ) :: utimes(:)
   TYPE(MD_ParameterType_C)          , INTENT(IN   ) :: p_C
   TYPE(MD_ContinuousStateType_C)    , INTENT(INOUT) :: x_C
   TYPE(MD_DiscreteStateType_C)      , INTENT(INOUT) :: xd_C
   TYPE(MD_ConstraintStateType_C)    , INTENT(INOUT) :: z_C
   TYPE(MD_OtherStateType_C)         , INTENT(INOUT) :: other_C
   TYPE(MD_MiscVarType_C)            , INTENT(INOUT) :: m_C

   REAL(DbKi)                      :: t
   TYPE(MD_InputType)              :: u(1)       ! INTENT(INOUT) ! had to change this to INOUT
   TYPE(MD_ParameterType)          :: p          ! INTENT(IN   )
   TYPE(MD_ContinuousStateType)    :: x          ! INTENT(INOUT)
   TYPE(MD_DiscreteStateType)      :: xd         ! INTENT(INOUT)
   TYPE(MD_ConstraintStateType)    :: z          ! INTENT(INOUT)
   TYPE(MD_OtherStateType)         :: other      ! INTENT(INOUT)
   TYPE(MD_MiscVarType)            :: m          ! INTENT(INOUT)
   INTEGER(IntKi)                  :: ErrStat    ! Error status of the operation
   CHARACTER(ErrMsgLen)            :: ErrMsg     ! Error message if ErrStat /= ErrID_None

   REAL(DbKi) :: utimes(1)

   print *, "Starting MD_UpdateStates_C"

   utimes = (/ 0.0 /) !, 0.1 /)

   print *, "1"
   ! Convert the C types to Fortran for all INTENT(IN) arguments
   t = REAL(t_C, DbKi)

   ! struct_MD_ParameterType - C to Fortran
   ! p%object = p_C%object
   p%NTypes = p_C%NTypes
   p%NConnects = p_C%NConnects
   p%NFairs = p_C%NFairs
   p%NConns = p_C%NConns
   p%NAnchs = p_C%NAnchs
   p%NLines = p_C%NLines
   p%g = p_C%g
   p%rhoW = p_C%rhoW
   p%WtrDpth = p_C%WtrDpth
   p%kBot = p_C%kBot
   p%cBot = p_C%cBot
   p%dtM0 = p_C%dtM0
   p%dtCoupling = p_C%dtCoupling
   p%NumOuts = p_C%NumOuts
   ! p%RootName = p_C%RootName
   ! call c2f_string(p_C%RootName, p%RootName)
   ! p%Delim = p_C%Delim
   p%MDUnOut = p_C%MDUnOut
   print *, "3"
   ! struct_MD_ContinuousStateType - C to Fortran
   ! x%object = x_C%object
   ! x%states = x_C%states
   ! x%states_Len = x_C%states_Len

   ! ! struct_MD_DiscreteStateType - C to Fortran
   ! ! xd%object = xd_C%object
   ! xd%dummy = xd_C%dummy

   ! struct_MD_ConstraintStateType - C to Fortran
   ! z%object = z_C%object
   ! z%dummy = z_C%dummy

   ! struct_MD_OtherStateType - C to Fortran
   ! other%object = other_C%object
   ! other%dummy = other_C%dummy

   ! struct_MD_MiscVarType - C to Fortran
   ! m%object = m_C%object
   ! m%FairIdList = m_C%FairIdList
   ! m%FairIdList_Len = m_C%FairIdList_Len
   ! m%ConnIdList = m_C%ConnIdList
   ! m%ConnIdList_Len = m_C%ConnIdList_Len
   ! m%LineStateIndList = m_C%LineStateIndList
   ! m%LineStateIndList_Len = m_C%LineStateIndList_Len
   ! m%MDWrOutput = m_C%MDWrOutput
   ! m%MDWrOutput_Len = m_C%MDWrOutput_Len
   ! print *, "4"
   CALL MD_UpdateStates( t, n, u, utimes, p, x, xd, z, other, m, ErrStat, ErrMsg)
   ! print *, "5"
   ! ! Convert the Fortran types to C for all INTENT(OUT) arguments

   ! ! struct_MD_ContinuousStateType - Fortran to C
   ! ! x_C%object = x%object
   ! ! x_C%states = x%states
   ! ! x_C%states_Len = x%states_Len

   ! ! struct_MD_DiscreteStateType - Fortran to C
   ! ! xd_C%object = xd%object
   ! xd_C%dummy = xd%dummy

   ! ! struct_MD_ConstraintStateType - Fortran to C
   ! ! z_C%object = z%object
   ! z_C%dummy = z%dummy

   ! ! struct_MD_OtherStateType - Fortran to C
   ! ! other_C%object = other%object
   ! other_C%dummy = other%dummy

   ! struct_MD_MiscVarType - Fortran to C
   ! m_C%object = m%object
   ! m_C%FairIdList = m%FairIdList
   ! m_C%FairIdList_Len = m%FairIdList_Len
   ! m_C%ConnIdList = m%ConnIdList
   ! m_C%ConnIdList_Len = m%ConnIdList_Len
   ! m_C%LineStateIndList = m%LineStateIndList
   ! m_C%LineStateIndList_Len = m%LineStateIndList_Len
   ! m_C%MDWrOutput = m%MDWrOutput
   ! m_C%MDWrOutput_Len = m%MDWrOutput_Len
   print *, "6"

   print *, "Ending MD_UpdateStates_C"
end subroutine


subroutine MD_End_C(u_C, p_C, x_C, xd_C, z_C, other_C, y_C, m_C) BIND(C, NAME="MD_End_C")
   TYPE(MD_InputType_C),           INTENT(INOUT)  :: u_C
   TYPE(MD_ParameterType_C),       INTENT(  OUT)  :: p_C
   TYPE(MD_ContinuousStateType_C), INTENT(  OUT)  :: x_C
   TYPE(MD_DiscreteStateType_C),   INTENT(  OUT)  :: xd_C
   TYPE(MD_ConstraintStateType_C), INTENT(  OUT)  :: z_C
   TYPE(MD_OtherStateType_C),      INTENT(  OUT)  :: other_C
   TYPE(MD_OutputType_C),          INTENT(  OUT)  :: y_C
   TYPE(MD_MiscVarType_C),         INTENT(  OUT)  :: m_C
   ! INTEGER(IntKi),               INTENT(  OUT)  :: ErrStat     ! Error status of the operation
   ! CHARACTER(*),                 INTENT(  OUT)  :: ErrMsg      ! Error message if ErrStat /= ErrID_None

   TYPE(MD_InputType) :: u
   TYPE(MD_ParameterType) :: p
   TYPE(MD_ContinuousStateType) :: x
   TYPE(MD_DiscreteStateType) :: xd
   TYPE(MD_ConstraintStateType) :: z
   TYPE(MD_OtherStateType) :: other
   TYPE(MD_OutputType) :: y
   TYPE(MD_MiscVarType):: m
   INTEGER(IntKi) :: ErrStat
   CHARACTER(ErrMsgLen) :: ErrMsg

   print *, "Starting MD_End_C"

   ! Convert the C types to Fortran for all INTENT(IN) arguments

   ! struct_MD_InputType - C to Fortran
   ! u%object = u_C%object

   call MD_END(u, p, x, xd, z, other, y, m, ErrStat , ErrMsg)

   ! Convert the Fortran types to C for all INTENT(OUT) arguments

   ! struct_MD_InputType - Fortran to C
   ! u_C%object = u%object

   ! struct_MD_ParameterType - Fortran to C
   ! p_C%object = p%object
   p_C%NTypes = p%NTypes
   p_C%NConnects = p%NConnects
   p_C%NFairs = p%NFairs
   p_C%NConns = p%NConns
   p_C%NAnchs = p%NAnchs
   p_C%NLines = p%NLines
   p_C%g = p%g
   p_C%rhoW = p%rhoW
   p_C%WtrDpth = p%WtrDpth
   p_C%kBot = p%kBot
   p_C%cBot = p%cBot
   p_C%dtM0 = p%dtM0
   p_C%dtCoupling = p%dtCoupling
   p_C%NumOuts = p%NumOuts
   ! p_C%RootName = p%RootName
   ! call f2c_string(p%RootName, p_C%RootName)
   ! p_C%Delim = p%Delim
   ! call f2c_string(p%Delim, p_C%Delim)
   p_C%MDUnOut = p%MDUnOut

   ! struct_MD_ContinuousStateType - Fortran to C
   ! x_C%object = x%object
   ! x_C%states = x%states
   ! x_C%states_Len = x%states_Len

   ! struct_MD_DiscreteStateType - Fortran to C
   ! xd_C%object = xd%object
   xd_C%dummy = xd%dummy

   ! struct_MD_ConstraintStateType - Fortran to C
   ! z_C%object = z%object
   z_C%dummy = z%dummy

   ! struct_MD_OtherStateType - Fortran to C
   ! other_C%object = other%object
   other_C%dummy = other%dummy

   ! struct_MD_OutputType - Fortran to C
   ! y_C%object = y%object
   ! y_C%WriteOutput = y%WriteOutput
   ! y_C%WriteOutput_Len = y%WriteOutput_Len

   ! struct_MD_MiscVarType - Fortran to C
   ! m_C%object = m%object
   ! m_C%FairIdList = m%FairIdList
   ! m_C%FairIdList_Len = m%FairIdList_Len
   ! m_C%ConnIdList = m%ConnIdList
   ! m_C%ConnIdList_Len = m%ConnIdList_Len
   ! m_C%LineStateIndList = m%LineStateIndList
   ! m_C%LineStateIndList_Len = m%LineStateIndList_Len
   ! m_C%MDWrOutput = m%MDWrOutput
   ! m_C%MDWrOutput_Len = m%MDWrOutput_Len

   print *, "Ending MD_End_C"
end subroutine

END MODULE
