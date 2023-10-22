% Define the objective function for Bayesian optimization
function obj = bayesian_optimization_function(x, X, y)
    % Fit a Gaussian process regressor on the dataset
    gpr = fitrgp(X, y, 'KernelFunction', 'squaredexponential', 'Sigma', 1e-5, 'BasisFunction', 'none');
    % Predict the output for the given input
    obj = -predict(gpr, x);
end