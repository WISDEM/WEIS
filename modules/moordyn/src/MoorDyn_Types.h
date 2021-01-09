//!STARTOFREGISTRYGENERATEDFILE 'MoorDyn_Types.h'
//!
//! WARNING This file is generated automatically by the FAST registry.
//! Do not edit.  Your changes to this file will be lost.
//!

#ifndef _MoorDyn_TYPES_H
#define _MoorDyn_TYPES_H


#ifdef _WIN32 //define something for Windows (32-bit)
#  include "stdbool.h"
#  define CALL __declspec( dllexport )
#elif _WIN64 //define something for Windows (64-bit)
#  include "stdbool.h"
#  define CALL __declspec( dllexport ) 
#else
#  include <stdbool.h>
#  define CALL 
#endif


  typedef struct MD_InitInputType {
    void * object ;
    float g ;
    float rhoW ;
    float WtrDepth ;
    float PtfmInit[6] ;
    char FileName[1024] ;
    char RootName[1024] ;
    bool Echo ;
    float DTIC ;
    float TMaxIC ;
    float CdScaleIC ;
    float threshIC ;
    char * OutList ;     int OutList_Len ;
  } MD_InitInputType_t ;
  // typedef struct MD_LineProp {
  //   void * object ;
  //   int IdNum ;
  //   char name[10] ;
  //   float d ;
  //   float w ;
  //   float EA ;
  //   float BA ;
  //   float Can ;
  //   float Cat ;
  //   float Cdn ;
  //   float Cdt ;
  // } MD_LineProp_t ;
  // typedef struct MD_Connect {
  //   void * object ;
  //   int IdNum ;
  //   char type[10] ;
  //   int TypeNum ;
  //   int * AttachedFairs ;     int AttachedFairs_Len ;
  //   int * AttachedAnchs ;     int AttachedAnchs_Len ;
  //   float conX ;
  //   float conY ;
  //   float conZ ;
  //   float conM ;
  //   float conV ;
  //   float conFX ;
  //   float conFY ;
  //   float conFZ ;
  //   float conCa ;
  //   float conCdA ;
  //   float Ftot[3] ;
  //   float Mtot[3][3] ;
  //   float S[3][3] ;
  //   float r[3] ;
  //   float rd[3] ;
  // } MD_Connect_t ;
  // typedef struct MD_Line {
  //   void * object ;
  //   int IdNum ;
  //   char type[10] ;
  //   int OutFlagList[20] ;
  //   int FairConnect ;
  //   int AnchConnect ;
  //   int PropsIdNum ;
  //   int N ;
  //   float UnstrLen ;
  //   float BA ;
  //   float * r ;
  //   int r_Len ;
  //   float * rd ;
  //   int rd_Len ;
  //   float * q ;
  //   int q_Len ;
  //   float * l ;
  //   int l_Len ;
  //   float * lstr ;
  //   int lstr_Len ;
  //   float * lstrd ;
  //   int lstrd_Len ;
  //   float * V ;
  //   int V_Len ;
  //   float * T ;
  //   int T_Len ;
  //   float * Td ;
  //   int Td_Len ;
  //   float * W ;
  //   int W_Len ;
  //   float * Dp ;
  //   int Dp_Len ;
  //   float * Dq ;
  //   int Dq_Len ;
  //   float * Ap ;
  //   int Ap_Len ;
  //   float * Aq ;
  //   int Aq_Len ;
  //   float * B ;
  //   int B_Len ;
  //   float * F ;
  //   int F_Len ;
  //   float * S ;
  //   int S_Len ;
  //   float * M ;
  //   int M_Len ;
  //   int LineUnOut ;
  //   float * LineWrOutput ;
  //   int LineWrOutput_Len ;
  // } MD_Line_t ;
  typedef struct MD_OutParmType {
    void * object ;
    char Name[10] ;
    char Units[10] ;
    int QType ;
    int OType ;
    int NodeID ;
    int ObjID ;
  } MD_OutParmType_t ;
  typedef struct MD_InitOutputType {
    void * object ;
    char * writeOutputHdr ;
    int writeOutputHdr_Len ;
    char * writeOutputUnt ;
    int writeOutputUnt_Len ;

  } MD_InitOutputType_t ;
  typedef struct MD_ContinuousStateType {
    void * object ;
    float * states ;
    int states_Len ;
  } MD_ContinuousStateType_t ;
  typedef struct MD_DiscreteStateType {
    void * object ;
    float dummy ;
  } MD_DiscreteStateType_t ;
  typedef struct MD_ConstraintStateType {
    void * object ;
    float dummy ;
  } MD_ConstraintStateType_t ;
  typedef struct MD_OtherStateType {
    void * object ;
    float dummy ;
  } MD_OtherStateType_t ;
  typedef struct MD_MiscVarType {
    void * object ;
    int * FairIdList ;
    int FairIdList_Len ;
    int * ConnIdList ;
    int ConnIdList_Len ;
    int * LineStateIndList ;
    int LineStateIndList_Len ;
    float * MDWrOutput ;
    int MDWrOutput_Len ;
  } MD_MiscVarType_t ;
  typedef struct MD_ParameterType {
    void * object ;
    int NTypes ;
    int NConnects ;
    int NFairs ;
    int NConns ;
    int NAnchs ;
    int NLines ;
    float g ;
    float rhoW ;
    float WtrDpth ;
    float kBot ;
    float cBot ;
    float dtM0 ;
    float dtCoupling ;
    int NumOuts ;
    char RootName[1024] ;
    char Delim[1] ;
    int MDUnOut ;
  } MD_ParameterType_t ;
  typedef struct MD_InputType {
    void * object ;
  } MD_InputType_t ;
  typedef struct MD_OutputType {
    void * object ;
    float * WriteOutput ;
    int WriteOutput_Len ;
  } MD_OutputType_t ;
  typedef struct MD_UserData {
    MD_InitInputType_t             MD_InitInput ;
    MD_InitOutputType_t            MD_InitOutput ;
    MD_ContinuousStateType_t       MD_ContState ;
    MD_DiscreteStateType_t         MD_DiscState ;
    MD_ConstraintStateType_t       MD_ConstrState ;
    MD_OtherStateType_t            MD_OtherState ;
    MD_MiscVarType_t               MD_Misc ;
    MD_ParameterType_t             MD_Param ;
    MD_InputType_t                 MD_Input ;
    MD_OutputType_t                MD_Output ;
  } MD_t ;

#endif // _MoorDyn_TYPES_H


//!ENDOFREGISTRYGENERATEDFILE
