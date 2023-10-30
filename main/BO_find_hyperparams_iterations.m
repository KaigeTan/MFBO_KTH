clear all
clc
close all
parentFolder = fileparts(pwd);
addpath(genpath(parentFolder + "\utils"));
addpath(genpath(parentFolder + "\Simulink_models"));
addpath(genpath(parentFolder + "\data"));
addpath(genpath(parentFolder + "\mtgp-master"));
%% initial data
%rng(5)
% Load dataset with 3D input (X) and 1D output (y)
% initial data for GP training
load(parentFolder + "\data\random_search\init_pts_5DOFs300.mat")
num_data = 1;
X = sampled_mat_in(1: num_data, :);
Y_spd = sampled_mat_out(1: num_data, 1);
Y_eng = sampled_mat_out(1: num_data, 2);
num_iter = 70;
% run the loop, BO -> evaluate -> add in dataset -> ...
tic
for i_iter = 1: num_iter
    % append new data point into the dataset
    if i_iter > 1
        X = [X; param_vec];
        Y_spd = [Y_spd; vx_avg];
    end
    % Define the objective function for Bayesian optimization
    fun = @(v) bayesian_optimization_function(v, X, Y_spd);
    
    % Run Bayesian optimization
    % x1 - x5: T, alpha, alpha_b_gain, z_l_gain, z_l_diff
    x1 = optimizableVariable('x1',[0.5, 2.5], 'Transform', 'log');
    x2 = optimizableVariable('x2',[0.1, 0.9], 'Transform', 'log');
    x3 = optimizableVariable('x3',[0.1, 0.7], 'Transform', 'log');
    x4 = optimizableVariable('x4',[0.001, 0.008], 'Transform', 'log');
    x5 = optimizableVariable('x5',[0, 2*pi]); %, 'Transform', 'log');
    
    vars = [x1, x2, x3, x4, x5];
    results = bayesopt(fun, vars, 'MaxObjectiveEvaluations', 100, 'Verbose', 0, 'PlotFcn', []);
    
    % Display the best point found
    best_point = results.XAtMinObjective;
    best_objective = results.MinObjective;
    fprintf('********************************************\n');
    fprintf('Best point: %d %d %d %d %d\n', table2array(best_point));
    disp(['Best objective: ', num2str(best_objective)]);
    fprintf('********************************************\n');
    
    %% test the selected hyperparameters
    rng(1)
    mdl = 'energy_rigid_wireOnly_quadruped_CPG_BO';
    % parameters for oscillators, constant
    k = 1000;       % positive constant, regulates the speed of convergence
    A = 1;          % positive constant, determines the amplitude of steady-state oscillation
    tau = 1;        % positive time constant, tunes the speed of switching
    epsilon = 5;    % positive constant, determines the coupling strength
    phi_vec = pi*[-1; 1; -1; 1]; % phase difference
    alpha_r_gain = pi; % alpha_r, robot direction
    
    % parameters for one oscillator
    param_vec = table2array(best_point);
    %% test the selected hyperparameters
    out = test_sim_CPG_model(mdl, param_vec);
    vx_avg = mean(out.observation.signals.values(61: end, 4));
    energy = out.energy.Data(end);
    COT = out.COT.Data(end);
    fprintf('speed: %d, energy, %d, COT: %d\n', vx_avg, energy, COT);
end
t_cost = toc;
%% save results
file_name = [parentFolder, '\data\BO\BO_training_', num2str(num_data), 'pts', num2str(num_iter), 'iters.mat'];
if exist([parentFolder, '\data\BO'], 'dir') ~= 7
    mkdir(parentFolder, '\data\BO');
end
save(file_name, 'X', 'Y_spd');

%% Define the objective function for Bayesian optimization
function obj = bayesian_optimization_function(x, X, y)
    % Fit a Gaussian process regressor on the dataset
    gpr = fitrgp(X, y, 'KernelFunction', 'squaredexponential', 'Sigma', 1e-5, 'BasisFunction', 'none');
    % Predict the output for the given input
    obj = -predict(gpr, x);
end
