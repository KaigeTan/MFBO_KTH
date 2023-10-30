%% this is the function to run and test the simulink model soft robot based on the pattern with CPG
function out = test_sim_CPG_model(mdl_name, param_vec)
    %% configurations
    parentFolder = fileparts(pwd);
    addpath(genpath(parentFolder + "\Simulink_models"));
    open(mdl_name);
    % robot animation on/off
    set_param(mdl_name,'SimMechanicsOpenEditorOnUpdate','on') 
    isCheckDone = 0;
    Tf = 5;
    Ts = 0.05;
    %% variables definition
    T = param_vec(1);
    alpha = param_vec(2);
    alpha_b_gain = param_vec(3);
    z_l_gain  = param_vec(4);
    z_l_diff  = param_vec(5);
    f = 1/T;          % oscillation frequency (var)
    % initial states for Hopf oscillator states
    init_state = rand(1, 8)*2 - 1;
    %% assignin to the base workspace, o.w. simulink doesnt recognize
    assignin("base", "alpha", alpha)
    assignin("base", "alpha_b_gain", alpha_b_gain)
    assignin("base", "z_l_gain", z_l_gain)
    assignin("base", "z_l_diff", z_l_diff)
    assignin("base", "f", f)
    assignin("base", "init_state", init_state)
    assignin("base", "isCheckDone", isCheckDone)
    assignin("base", "Tf", Tf)
    assignin("base", "Ts", Ts)
    %% test the selected hyperparameters
    out = sim(mdl_name);
end