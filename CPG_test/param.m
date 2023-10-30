clear;clc;
%% from oscillator to the actuation signal
alpha_r_gain = pi;
alpha_b_gain = 0.46;
z_l_gain = 6.8e-03;
%% parameters for one oscillator
k = 1000;       % positive constant, regulates the speed of convergence
A = 2;          % positive constant, determines the amplitude of steady-state oscillation
T = 1.25;        % oscillation period, 0.8
f = 1/T;          % oscillation frequency (var)
alpha = 0.856;   % shape ratio, determines the time of rising phase (var)
tau = 0.5;        % positive time constant, tunes the speed of switching
%% parameters for coupled oscillators
epsilon =5;    % positive constant, determines the coupling strength
phi_vec = pi*[0; 0.5; 0; 0.5]; % phase difference
%% initial states for Hopf oscillator states
rng(1)
init_state = rand(1, 8)*2 - 1;