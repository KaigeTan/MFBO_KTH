% Check the BO searched point performance on the high fidelity model
function [vx_avg, energy, COT] = check_fh(datax_fh, model_name)
    %% test the selected hyperparameters
    % robot animation on/off
    set_param(model_name,'SimMechanicsOpenEditorOnUpdate','on') 
    assignin('base', 'isCheckDone', 0);
    assignin('base', 'Tf', 5);
    assignin('base', 'Ts', 0.05);
    % from oscillator to the actuation signal
    assignin('base', 'alpha_r_gain', pi);
    %% parameters for one oscillator
    T = datax_fh(1);
    assignin('base', 'T', datax_fh(1));
    assignin('base', 'alpha', datax_fh(2));
    assignin('base', 'alpha_b_gain', datax_fh(3));
    assignin('base', 'z_l_gain', datax_fh(4));
    assignin('base', 'z_l_diff', datax_fh(5));
    
    
    assignin('base', 'f', 1/T);          % oscillation frequency (var)
    assignin('base', 'k', 1000);         % positive constant, regulates the speed of convergence
    assignin('base', 'A', 1);            % positive constant, determines the amplitude of steady-state oscillation
    assignin('base', 'tau', 1);          % positive time constant, tunes the speed of switching
    % parameters for coupled oscillators
    assignin('base', 'epsilon', 5);      % positive constant, determines the coupling strength
    assignin('base', 'phi_vec', pi*[-1; 1; -1; 1]);  % phase difference
    % initial states for Hopf oscillator states
    assignin('base', 'init_state', rand(1, 8)*2 - 1);
    
    %% from oscillator to the actuation signal
    out = sim(model_name);
    vx_avg = mean(out.observation.signals.values(61: end, 4));
    energy = out.energy.Data(end);
    COT = out.COT.Data(end);
    fprintf('speed: %d, energy, %d, COT: %d\n', vx_avg, energy, COT);
end