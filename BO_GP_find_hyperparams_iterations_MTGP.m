clear;clc;
% initial data
rng(5)
% Load dataset with 3D input (X) and 1D output (y)
load('.\results\10pts_50iters.mat')
num_data = 30;
datax_fl_vec = X(1: num_data, :); % fl training inputs, N X 3
Y_spd_fl_vec = Y_spd(1: num_data, 1); % fl training targets, N X 1
num_iter = 100;

%% build MTGP model to predict outputs on fh
covfunc_x = {'covSEard'};
M = 2;    % Number of tasks
D = 3;    % Dimensionality of input space
irank = M;    % rank for Kf (1, ... M). irank=M -> Full rank
density_fl = num_data;
density_fh = 1;
pred_index = 2;     % predict the output for fh

% test the best point on the low fidelity model
datax_fh = X(29, :); % dense training inputs, N X 3

%%%%%%%%%%%%%%%%%
% send to simulink(robot), and wait for the result, 
% result either 20hz, raw v_x, moving average in the script
% or stabilized 

% define the vector for the fh data
datax_fh_vec = datax_fh;
datay_fh_vec = datay_fh;

for i = 1: num_iter
    % train MTGP model
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
    
    % test the selected hyperparameters
    datax_fh = table2array(best_point);
    [datay_fh, ~, ~] = check_fh(datax_fh, 'energy_rigid_wireOnly_quadruped_CPG_BO_fh'); % dense training targets, N X 1
    % append the new data test result on fh in the training set
    
    datax_fh_vec = [datax_fh_vec; datax_fh];
    datay_fh_vec = [datay_fh_vec; datay_fh];
end



% Define the objective function for Bayesian optimization
function obj = bayesian_optimization_function(x, X, y)
    % Fit a Gaussian process regressor on the dataset
    gpr = fitrgp(X, y, 'KernelFunction', 'squaredexponential', 'Sigma', 1e-5, 'BasisFunction', 'none');
    % Predict the output for the given input
    obj = -predict(gpr, x);
end
