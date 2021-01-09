!*********************************************************************************************************************************
! MoorDyn_Types_C
!.................................................................................................................................
! This file is part of MoorDyn.
!
! Copyright (C) 2012-2016 National Renewable Energy Laboratory
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
!
! This is the C-interface of the derived types for MoorDyn.
! All are generated with the OpenFAST Registry, but some modifications were made
! and some types commented since they arent currently used.
!
!*********************************************************************************************************************************


MODULE MoorDyn_Types_C
USE, INTRINSIC :: ISO_C_Binding
USE NWTC_Library
USE MoorDyn_Types

IMPLICIT NONE

   TYPE, BIND(C) :: MD_InitInputType_C
      !  TYPE(C_PTR) :: object = C_NULL_PTR
      REAL(KIND=C_FLOAT) :: g 
      REAL(KIND=C_FLOAT) :: rhoW 
      REAL(KIND=C_FLOAT) :: WtrDepth 
      REAL(KIND=C_FLOAT) :: PtfmInit(6) 
      CHARACTER(KIND=C_CHAR), DIMENSION(1025) :: FileName 
      CHARACTER(KIND=C_CHAR), DIMENSION(1025) :: RootName 
      LOGICAL(KIND=C_BOOL) :: Echo 
      REAL(KIND=C_FLOAT) :: DTIC 
      REAL(KIND=C_FLOAT) :: TMaxIC 
      REAL(KIND=C_FLOAT) :: CdScaleIC 
      REAL(KIND=C_FLOAT) :: threshIC 
      !  TYPE(C_ptr) :: OutList = C_NULL_PTR 
      !  INTEGER(C_int) :: OutList_Len = 0 
   END TYPE MD_InitInputType_C

   TYPE, BIND(C) :: MD_InputType_C
      TYPE(C_PTR) :: object = C_NULL_PTR
   END TYPE MD_InputType_C

   TYPE, BIND(C) :: MD_InitOutputType_C
      TYPE(C_PTR) :: object = C_NULL_PTR
      TYPE(C_ptr) :: writeOutputHdr = C_NULL_PTR 
      INTEGER(C_int) :: writeOutputHdr_Len = 0 
      TYPE(C_ptr) :: writeOutputUnt = C_NULL_PTR 
      INTEGER(C_int) :: writeOutputUnt_Len = 0 
   END TYPE MD_InitOutputType_C

   TYPE, BIND(C) :: MD_OutputType_C
      TYPE(C_PTR) :: object = C_NULL_PTR
      TYPE(C_ptr) :: WriteOutput = C_NULL_PTR 
      INTEGER(C_int) :: WriteOutput_Len = 0 
   END TYPE MD_OutputType_C

   TYPE, BIND(C) :: MD_OutParmType_C
      TYPE(C_PTR) :: object = C_NULL_PTR
      CHARACTER(KIND=C_CHAR), DIMENSION(10) :: Name 
      CHARACTER(KIND=C_CHAR), DIMENSION(10) :: Units 
      INTEGER(KIND=C_INT) :: QType 
      INTEGER(KIND=C_INT) :: OType 
      INTEGER(KIND=C_INT) :: NodeID 
      INTEGER(KIND=C_INT) :: ObjID 
   END TYPE MD_OutParmType_C

   TYPE, BIND(C) :: MD_ParameterType_C
   !  TYPE(C_PTR) :: object = C_NULL_PTR
      INTEGER(KIND=C_INT) :: NTypes 
      INTEGER(KIND=C_INT) :: NConnects 
      INTEGER(KIND=C_INT) :: NFairs 
      INTEGER(KIND=C_INT) :: NConns 
      INTEGER(KIND=C_INT) :: NAnchs 
      INTEGER(KIND=C_INT) :: NLines 
      REAL(KIND=C_FLOAT) :: g 
      REAL(KIND=C_FLOAT) :: rhoW 
      REAL(KIND=C_FLOAT) :: WtrDpth 
      REAL(KIND=C_FLOAT) :: kBot 
      REAL(KIND=C_FLOAT) :: cBot 
      REAL(KIND=C_FLOAT) :: dtM0 
      REAL(KIND=C_FLOAT) :: dtCoupling 
      INTEGER(KIND=C_INT) :: NumOuts 
      CHARACTER(KIND=C_CHAR), DIMENSION(1025) :: RootName 
      CHARACTER(KIND=C_CHAR), DIMENSION(2) :: Delim 
      INTEGER(KIND=C_INT) :: MDUnOut 
   END TYPE MD_ParameterType_C

   TYPE, BIND(C) :: MD_ContinuousStateType_C
      TYPE(C_PTR) :: object = C_NULL_PTR
      TYPE(C_ptr) :: states = C_NULL_PTR 
      INTEGER(C_int) :: states_Len = 0 
   END TYPE MD_ContinuousStateType_C

   TYPE, BIND(C) :: MD_DiscreteStateType_C
      TYPE(C_PTR) :: object = C_NULL_PTR
      REAL(KIND=C_FLOAT) :: dummy 
   END TYPE MD_DiscreteStateType_C

   TYPE, BIND(C) :: MD_ConstraintStateType_C
      TYPE(C_PTR) :: object = C_NULL_PTR
      REAL(KIND=C_FLOAT) :: dummy 
   END TYPE MD_ConstraintStateType_C

   TYPE, BIND(C) :: MD_OtherStateType_C
      TYPE(C_PTR) :: object = C_NULL_PTR
      REAL(KIND=C_FLOAT) :: dummy 
   END TYPE MD_OtherStateType_C

   TYPE, BIND(C) :: MD_MiscVarType_C
      TYPE(C_PTR) :: object = C_NULL_PTR
      TYPE(C_ptr) :: FairIdList = C_NULL_PTR 
      INTEGER(C_int) :: FairIdList_Len = 0 
      TYPE(C_ptr) :: ConnIdList = C_NULL_PTR 
      INTEGER(C_int) :: ConnIdList_Len = 0 
      TYPE(C_ptr) :: LineStateIndList = C_NULL_PTR 
      INTEGER(C_int) :: LineStateIndList_Len = 0 
      TYPE(C_ptr) :: MDWrOutput = C_NULL_PTR 
      INTEGER(C_int) :: MDWrOutput_Len = 0 
   END TYPE MD_MiscVarType_C






  TYPE, BIND(C) :: MD_LineProp_C
    TYPE(C_PTR) :: object = C_NULL_PTR
    INTEGER(KIND=C_INT) :: IdNum 
    CHARACTER(KIND=C_CHAR), DIMENSION(10) :: name 
    REAL(KIND=C_FLOAT) :: d 
    REAL(KIND=C_FLOAT) :: w 
    REAL(KIND=C_FLOAT) :: EA 
    REAL(KIND=C_FLOAT) :: BA 
    REAL(KIND=C_FLOAT) :: Can 
    REAL(KIND=C_FLOAT) :: Cat 
    REAL(KIND=C_FLOAT) :: Cdn 
    REAL(KIND=C_FLOAT) :: Cdt 
  END TYPE MD_LineProp_C

  TYPE, BIND(C) :: MD_Connect_C
    TYPE(C_PTR) :: object = C_NULL_PTR
    INTEGER(KIND=C_INT) :: IdNum 
    CHARACTER(KIND=C_CHAR), DIMENSION(10) :: type 
    INTEGER(KIND=C_INT) :: TypeNum 
    TYPE(C_ptr) :: AttachedFairs = C_NULL_PTR 
    INTEGER(C_int) :: AttachedFairs_Len = 0 
    TYPE(C_ptr) :: AttachedAnchs = C_NULL_PTR 
    INTEGER(C_int) :: AttachedAnchs_Len = 0 
    REAL(KIND=C_FLOAT) :: conX 
    REAL(KIND=C_FLOAT) :: conY 
    REAL(KIND=C_FLOAT) :: conZ 
    REAL(KIND=C_FLOAT) :: conM 
    REAL(KIND=C_FLOAT) :: conV 
    REAL(KIND=C_FLOAT) :: conFX 
    REAL(KIND=C_FLOAT) :: conFY 
    REAL(KIND=C_FLOAT) :: conFZ 
    REAL(KIND=C_FLOAT) :: conCa 
    REAL(KIND=C_FLOAT) :: conCdA 
    REAL(KIND=C_FLOAT) :: Ftot(3)
    REAL(KIND=C_FLOAT) :: Mtot(3,3)
    REAL(KIND=C_FLOAT) :: S(3,3)
    REAL(KIND=C_FLOAT) :: r(3)
    REAL(KIND=C_FLOAT) :: rd(3)
  END TYPE MD_Connect_C

  ! TYPE, BIND(C) :: MD_Line_C
  !   TYPE(C_PTR) :: object = C_NULL_PTR
  !   INTEGER(KIND=C_INT) :: IdNum 
  !   CHARACTER(KIND=C_CHAR), DIMENSION(10) :: type 
  !   TYPE(C_PTR) :: OutFlagList(20)
  !   INTEGER(KIND=C_INT) :: FairConnect 
  !   INTEGER(KIND=C_INT) :: AnchConnect 
  !   INTEGER(KIND=C_INT) :: PropsIdNum 
  !   INTEGER(KIND=C_INT) :: N 
  !   REAL(KIND=C_FLOAT) :: UnstrLen 
  !   REAL(KIND=C_FLOAT) :: BA 
  !   TYPE(C_ptr) :: r = C_NULL_PTR 
  !   INTEGER(C_int) :: r_Len = 0 
  !   TYPE(C_ptr) :: rd = C_NULL_PTR 
  !   INTEGER(C_int) :: rd_Len = 0 
  !   TYPE(C_ptr) :: q = C_NULL_PTR 
  !   INTEGER(C_int) :: q_Len = 0 
  !   TYPE(C_ptr) :: l = C_NULL_PTR 
  !   INTEGER(C_int) :: l_Len = 0 
  !   TYPE(C_ptr) :: lstr = C_NULL_PTR 
  !   INTEGER(C_int) :: lstr_Len = 0 
  !   TYPE(C_ptr) :: lstrd = C_NULL_PTR 
  !   INTEGER(C_int) :: lstrd_Len = 0 
  !   TYPE(C_ptr) :: V = C_NULL_PTR 
  !   INTEGER(C_int) :: V_Len = 0 
  !   TYPE(C_ptr) :: T = C_NULL_PTR 
  !   INTEGER(C_int) :: T_Len = 0 
  !   TYPE(C_ptr) :: Td = C_NULL_PTR 
  !   INTEGER(C_int) :: Td_Len = 0 
  !   TYPE(C_ptr) :: W = C_NULL_PTR 
  !   INTEGER(C_int) :: W_Len = 0 
  !   TYPE(C_ptr) :: Dp = C_NULL_PTR 
  !   INTEGER(C_int) :: Dp_Len = 0 
  !   TYPE(C_ptr) :: Dq = C_NULL_PTR 
  !   INTEGER(C_int) :: Dq_Len = 0 
  !   TYPE(C_ptr) :: Ap = C_NULL_PTR 
  !   INTEGER(C_int) :: Ap_Len = 0 
  !   TYPE(C_ptr) :: Aq = C_NULL_PTR 
  !   INTEGER(C_int) :: Aq_Len = 0 
  !   TYPE(C_ptr) :: B = C_NULL_PTR 
  !   INTEGER(C_int) :: B_Len = 0 
  !   TYPE(C_ptr) :: F = C_NULL_PTR 
  !   INTEGER(C_int) :: F_Len = 0 
  !   TYPE(C_ptr) :: S = C_NULL_PTR 
  !   INTEGER(C_int) :: S_Len = 0 
  !   TYPE(C_ptr) :: M = C_NULL_PTR 
  !   INTEGER(C_int) :: M_Len = 0 
  !   INTEGER(KIND=C_INT) :: LineUnOut 
  !   TYPE(C_ptr) :: LineWrOutput = C_NULL_PTR 
  !   INTEGER(C_int) :: LineWrOutput_Len = 0 
  ! END TYPE MD_Line_C

CONTAINS

subroutine f2c_string(f_string, c_string)
   character(*), intent(in) :: f_string
   character(kind=C_CHAR), intent(out) :: c_string(CStringLen)
   c_string = transfer( trim(f_string)//C_NULL_CHAR, c_string )
end subroutine

subroutine c2f_string(c_string, f_string)
   character(kind=C_CHAR), intent(in) :: c_string(CStringLen)
   character(*), intent(out) :: f_string
   character(CStringLen) :: temp_string
   temp_string = transfer( c_string(1:CStringLen-1), f_string )
   call RemoveNullChar(temp_string)
   f_string = trim(temp_string)
end subroutine

FUNCTION MD_InitInputType_C2F(InitInp_C) RESULT(InitInp_F)

   TYPE(MD_InitInputType_C), INTENT(IN) :: InitInp_C
   TYPE(MD_InitInputType) :: InitInp_F

   InitInp_F%g = InitInp_C%g
   InitInp_F%rhoW = InitInp_C%rhoW
   InitInp_F%WtrDepth = InitInp_C%WtrDepth
   InitInp_F%PtfmInit = InitInp_C%PtfmInit
   call c2f_string(InitInp_C%FileName, InitInp_F%FileName)
   call c2f_string(InitInp_C%RootName, InitInp_F%RootName)
   InitInp_F%Echo = InitInp_C%Echo
   InitInp_F%DTIC = InitInp_C%DTIC
   InitInp_F%TMaxIC = InitInp_C%TMaxIC
   InitInp_F%CdScaleIC = InitInp_C%CdScaleIC
   InitInp_F%threshIC = InitInp_C%threshIC
   ! InitInp_F%OutList = InitInp_C%OutList
   ! InitInp_F%OutList_Len = InitInp_C%OutList_LenS

END FUNCTION

FUNCTION MD_InitInputType_F2C(InitInp_F) RESULT(InitInp_C)

   TYPE(MD_InitInputType), INTENT(IN) :: InitInp_F
   TYPE(MD_InitInputType_C) :: InitInp_C

   InitInp_C%g = InitInp_F%g
   InitInp_C%rhoW = InitInp_F%rhoW
   InitInp_C%WtrDepth = InitInp_F%WtrDepth
   InitInp_C%PtfmInit = InitInp_F%PtfmInit
   call f2c_string(InitInp_F%FileName, InitInp_C%FileName)
   call f2c_string(InitInp_F%RootName, InitInp_C%RootName)
   InitInp_C%Echo = InitInp_F%Echo
   InitInp_C%DTIC = InitInp_F%DTIC
   InitInp_C%TMaxIC = InitInp_F%TMaxIC
   InitInp_C%CdScaleIC = InitInp_F%CdScaleIC
   InitInp_C%threshIC = InitInp_F%threshIC
   ! InitInp_C%OutList = InitInp_F%OutList
   ! InitInp_C%OutList_Len = InitInp_F%OutList_Len

END FUNCTION

FUNCTION MD_ParameterType_F2C(p_F) RESULT(p_C)

   TYPE(MD_ParameterType), INTENT(IN) :: p_F
   TYPE(MD_ParameterType_C) :: p_C

   ! struct_MD_ParameterType - Fortran to C
   p_C%NTypes = p_F%NTypes
   p_C%NConnects = p_F%NConnects
   p_C%NFairs = p_F%NFairs
   p_C%NConns = p_F%NConns
   p_C%NAnchs = p_F%NAnchs
   p_C%NLines = p_F%NLines
   p_C%g = p_F%g
   p_C%rhoW = p_F%rhoW
   p_C%WtrDpth = p_F%WtrDpth
   p_C%kBot = p_F%kBot
   p_C%cBot = p_F%cBot
   p_C%dtM0 = p_F%dtM0
   p_C%dtCoupling = p_F%dtCoupling
   p_C%NumOuts = p_F%NumOuts
   call f2c_string(p_F%RootName, p_C%RootName)
   call f2c_string(p_F%Delim, p_C%Delim)
   p_C%MDUnOut = p_F%MDUnOut

END FUNCTION

END MODULE MoorDyn_Types_C
