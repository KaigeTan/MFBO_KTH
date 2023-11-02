# MFBO_KTH
Learning-enhanced Optimal Gait Design for a Tendon-driven Soft Quadruped Robot via Multi-fidelity Bayesian Optimization

<h3 align="left">Connect with me:</h3>
<p align="left">
</p>

- 📫**kaiget@kth.se**
- Or 📫**xuezhin@kth.se**

<h3 align="left">Languages and Tools:</h3>
<p align="left"> <a href="https://www.mathworks.com/" target="_blank" rel="noreferrer"> <img src="https://upload.wikimedia.org/wikipedia/commons/2/21/Matlab_Logo.png" alt="matlab" width="40" height="40"/> </a> </p>

## Structure

The repository is structured as follows:

- `Simulink_models`: This directory contains all Simulink files, including soft robot plant models and CPG oscillators.

- `data`: This directory contains generated data files for 1) offline training results, 2) benchmark method testing, and 3) prerequest data for online training.

- `mtgp-master`: This directory contains all of the functions used in the MTBO. 

- `unit_test`: This directory contains the main functions for the benchmark method testing and MFBO physical training Python code.

- `MFGP_main.m`: This m function file is for MFBO physical training.

- `CPG_online.slx`: This .slx files contains simulink model used for SoftQ online training.

## Getting Started
To get started with this project, clone the repository to your local machine using the following command:

```
git clone https://github.com/n7729697/MTBO_KTH.git
```
