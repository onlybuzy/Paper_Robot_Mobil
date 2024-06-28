
clc 
clear 
%% Training Data

C1_R_T=readmatrix("Train\C1_vr_train.csv");
C1_L_T=readmatrix("Train\C1_vl_train.csv");

%% Validation Data

C1_R_V=readmatrix("Validation\C1_vr_val.csv");
C1_L_V=readmatrix("Validation\C1_vl_val.csv");