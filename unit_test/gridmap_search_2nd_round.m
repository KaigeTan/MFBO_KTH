clear all
clc
close all
parentFolder = fileparts(pwd);
addpath(genpath(parentFolder + "\utils"));
addpath(genpath(parentFolder + "\Simulink_models"));
addpath(genpath(parentFolder + "\data"));
mdl = 'energy_rigid_wireOnly_quadruped_CPG_BO';
%% parameters for oscillators, constant
k = 1000;       % positive constant, regulates the speed of convergence
A = 1;          % positive constant, determines the amplitude of steady-state oscillation
tau = 1;        % positive time constant, tunes the speed of switching
epsilon = 5;    % positive constant, determines the coupling strength
phi_vec = pi*[-1; 1; -1; 1]; % phase difference
alpha_r_gain = pi; % alpha_r, robot direction
%rng(1)
%% get training samples
num_grid = 4;
num_var = 5;
n_round = num_var^num_grid;
sampled_mat_out = zeros(n_round, 3);    % x_spd, enegry, COT
% generate a table which contain the grid search combination
base_vec = linspace(0, 1, num_grid);
T_vec = (1.17-0.5)*round(base_vec, 2) + 0.5;        % oscillation period, 0.5 - 1.17 s
alpha_vec = (0.88-0.616)*round(base_vec, 2) + 0.616;   % shape ratio, determines the time of rising phase (var), 0.616 - 0.88
% from oscillator to the actuation signal
alpha_b_gain_vec = (0.7-0.5)*round(base_vec, 2) + 0.5;    % maximal bending angle, 0.5 - 0.7 rad
z_l_gain_vec = (0.008-0.0057)*round(base_vec, 4)+ 0.0057;    % maximal bending length, 0.0057 - 0.008 m
z_l_diff_vec = (4.1888-2.0944)*round(base_vec, 4) + 2.0944;             % phase difference between 2.0944 - 4.1888 rad
% generate all combinations of the variables using combvec
sampled_mat_in = combvec(T_vec, alpha_vec, alpha_b_gain_vec, z_l_gain_vec, z_l_diff_vec)';

% 5DOF: T, alpha, alpha_b_gain, z_l_gain, z_l_diff
tic
for i_iter = 1: length(sampled_mat_in)
    % prevent some configurations lead to extreme states and simulation error
    try
        out = test_sim_CPG_model(mdl, sampled_mat_in(i_iter, :));
    catch
        fprintf('round: %d, simulation failed!\n', i_iter);
        sampled_mat_out(i_iter, :) = [0, 0, 0];
        continue
    end
    
    vx_avg = mean(out.observation.signals.values(61: end, 4));
    energy = out.energy.Data(end);
    COT = out.COT.Data(end);
    sampled_mat_out(i_iter, :) = [vx_avg, energy, COT];
    fprintf('round: %d, spd: %.3d, energy: %.3d, COT: %.3d\n', ...
        i_iter, vx_avg, energy, COT);
end
t_cost = toc;
% save results
file_name = [parentFolder, '\data\exhausted_search\2nd_grid_search_5DOF_5pts_', 'time', num2str(t_cost), '.mat'];
if exist([parentFolder, '\data\exhausted_search\'], 'dir') ~= 7
    mkdir([parentFolder, '\data\exhausted_search\']);
end
save(file_name, 'sampled_mat_in', 'sampled_mat_out');
