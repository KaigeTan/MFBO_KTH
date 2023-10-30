clear all
clc
close all
parentFolder = fileparts(pwd);
addpath(genpath(parentFolder + "\utils"));
addpath(genpath(parentFolder + "\Simulink_models"));
addpath(genpath(parentFolder + "\data"));
addpath(genpath(parentFolder + "\mtgp-master"));
%% parameters for one oscillator
mdl = 'energy_rigid_wireOnly_quadruped_CPG_BO';
k = 1000;       % positive constant, regulates the speed of convergence
A = 1;          % positive constant, determines the amplitude of steady-state oscillation
tau = 1;        % positive time constant, tunes the speed of switching
epsilon = 5;    % positive constant, determines the coupling strength
phi_vec = pi*[-1; 1; -1; 1]; % phase difference
alpha_r_gain = pi; % alpha_r, robot direction
%% get training samples
n_round = 300;
%rng(5);
sampled_mat_in = zeros(n_round, 5); % design space: 4 params
sampled_mat_out = zeros(n_round, 3);    % x_spd, enegry, COT

% 5DOF: T, alpha, alpha_b_gain, z_l_gain, z_l_diff
for i_iter = 1: n_round
    % oscillator parameters, randomized
    T = round(2*rand(1) + 0.5, 2);        % oscillation period, 0.5 - 2.5 s
    f = 1/T;          % oscillation frequency (var)
    alpha = 0.8*round(rand(1) + 0.1, 2);   % shape ratio, determines the time of rising phase (var), 0.1 - 0.9
    % from oscillator to the actuation signal
    alpha_b_gain = round(0.6*rand(1) + 0.1, 2);    % maximal bending angle, 0.1 - 0.7 rad
    z_l_gain = round(0.007*rand(1) + 0.001, 4);    % maximal bending length, 0.001 - 0.008 m
    z_l_diff = round(2*pi*rand(1), 4);             % phase difference between alpha_b and z_l
    % define the parameter vector
    param_vec = [T, alpha, alpha_b_gain, z_l_gain, z_l_diff];
    % simulate and collect data
    out = test_sim_CPG_model(mdl, param_vec);
    vx_avg = mean(out.observation.signals.values(61: end, 4));
    energy = out.energy.Data(end);
    COT = out.COT.Data(end);
    sampled_mat_in(i_iter, :) = [T, alpha, alpha_b_gain, z_l_gain, z_l_diff];
    sampled_mat_out(i_iter, :) = [vx_avg, energy, COT];
    fprintf('round: %d, spd: %.3d, energy: %.3d, COT: %.3d\n', ...
        i_iter, vx_avg, energy, COT);
end
% save results
file_name = [parentFolder, '\data\random_search\init_pts_5DOFs', num2str(n_round), '.mat'];
if exist([parentFolder, '\data\random_search\'], 'dir') ~= 7
    mkdir([parentFolder, '\data\random_search\']);
end
save(file_name, 'sampled_mat_in', 'sampled_mat_out');
