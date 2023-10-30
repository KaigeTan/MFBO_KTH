clear;clc;
addpath(genpath('.\mtgp-master'));
addpath(genpath('.\data'));
%% initial data
rng(5)
% Load dataset with 3D input (X) and 1D output (y)
load('.\data\BO\10pts_50iters.mat')
num_data = 20;
datax_fl_vec = X(1: num_data, :); % fl training inputs, N X 3
Y_spd_fl_vec = Y_spd(1: num_data, 1); % fl training targets, N X 1
num_iter = 10;
model_name = 'CPG_online';
% hyperparameter setting
covfunc_x = {'covSEard'};
M = 2;    % Number of tasks
D = 5;    % Dimensionality of input space
irank = M;    % rank for Kf (1, ... M). irank=M -> Full rank
density_fl = num_data;
density_fh = 1;
pred_index = 2;     % predict the output for fh
% test the best point on the low fidelity model
opt_param = X(21, :); % dense training inputs, N X 3
datax_fh_vec = [];
datay_fh_vec = [];
vel_x_mat = [];
dx_mat = [];
% constant
Tf = 5;
Ts = 5e-2;
Ts_cal = 5e-4;
alpha_r_gain = pi;
k = 1000;
A_cpg = 1;
tau = 1;
epsilon = 5;
phi_vec = pi*[-1; 1; -1; 1];
init_state = rand(1, 8)*2 - 1;
% KF params
A = [0 Ts; 0 1]; % State transition matrix
B = [0.5*Ts^2; Ts]; % Input control matrix
C = [1 0]; % Measurement matrix
Q = diag([1e-2,1]); % Process noise covariance
R = 0.15; % Measurement noise covariance
x = [0; 0]; % Initial state vector
P = eye(2); % Initial covariance matrix

for i = 6: num_iter
    %% add an human input reset command
    prompt = 'Press Enter to continue ...';
    str = input(prompt,'s');
    
    %% set parameters
    % variable
    T = opt_param(1);
    f = 1/T;
    alpha = opt_param(2);
    alpha_b_gain = opt_param(3);
    z_l_gain = opt_param(4);
    z_l_diff = opt_param(5);

    %% run the simulation -- connnected to the real robot
    fprintf('iter: %d\n', i);
    out = sim(model_name);
    ask_time = 'Enter start time: ';
    start_time = str2double(input(ask_time,'s'));
    vx_avg = mean(out.vx(20*start_time: 20*(start_time+3))); % the last 3 secs data
    dx = 0.5*(out.front + out.back); 
    fprintf('********************************************\n');
    fprintf('iter: %d, speed: %d\n', i, vx_avg);
    fprintf('********************************************\n');
    datax_fh_vec = [datax_fh_vec; opt_param];
    datay_fh_vec = [datay_fh_vec; vx_avg];
    if length(out.vx) < 10*20+1
        vel_x_mat = [vel_x_mat, [out.vx(61: end); zeros(10*20-length(out.vx), 1)]];
    else
        vel_x_mat = [vel_x_mat, out.vx(62: end)];
        %dx_mat = [dx_mat, dx];
    end
    
    %% train MTGP model
    % append collected data from physical test
    xtrain = [datax_fl_vec; datax_fh_vec];
    ytrain = [Y_spd_fl_vec; datay_fh_vec];
    ind_kf_train = [ones(length(Y_spd_fl_vec), 1); 2*ones(length(datay_fh_vec), 1)];
    ind_kx_train = linspace(1, length(ytrain), length(ytrain));
    nx = ones(length(ytrain), 1); % observations on each task-input point
    data_MTGP  = {covfunc_x, xtrain, ytrain, M, irank, nx, ind_kf_train, ind_kx_train};
    
    % Hyper-parameter learning
    [logtheta_all, deriv_range] = init_mtgp_default(xtrain, covfunc_x, M, irank);
    [MTGP_params, nl]           = learn_mtgp(logtheta_all, deriv_range, data_MTGP);
    % MTGP inference: GP prediction on all data points
    [Ypred, ~] = predict_mtgp_all_tasks(MTGP_params, data_MTGP, xtrain);
    % we only focus on a specific column for fh output (based on pred_index)
    datay_fh_list = Ypred(:, pred_index);
    
    %% Bayesian optimization
    % Define the objective function for Bayesian optimization based on fh
    fun = @(v) bayesian_optimization_function(v, xtrain, datay_fh_list);
    % Run Bayesian optimization
    % x1 - x5: T, alpha, alpha_b_gain, z_l_gain, z_l_diff
    x1 = optimizableVariable('x1',[0.5, 2.5], 'Transform', 'log');
    x2 = optimizableVariable('x2',[0.1, 0.9], 'Transform', 'log');
    x3 = optimizableVariable('x3',[0.1, 0.7], 'Transform', 'log');
    x4 = optimizableVariable('x4',[0.003, 0.008], 'Transform', 'log');
    x5 = optimizableVariable('x5',[0, 2*pi]); %, 'Transform', 'log');
    
    vars = [x1, x2, x3, x4, x5];
    results = bayesopt(fun, vars, 'MaxObjectiveEvaluations', 100, 'Verbose', 0, 'PlotFcn', []);
    
    % Display the best point found
    best_point = results.XAtMinObjective;
    % best_objective = results.MinObjective;
    fprintf('********************************************\n');
    fprintf('Next candidate point: %d %d %d %d %d\n', table2array(best_point));
    fprintf('********************************************\n');
    
    % test the selected hyperparameters
    opt_param = table2array(best_point);
    
end
