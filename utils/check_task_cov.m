function [Kf, Lf, Kf_normal] = check_task_cov(MTGP_params)
    theta_lf = MTGP_params(1:3);
    m = 2;
    irank = 2;
    Lf = vec2lowtri_inchol(theta_lf,m,irank);
    Kf = Lf*Lf';
    %D = diag(diag(1./Kf));
    %Kf_normal = Kf*D;
    theta_2 = sign(Lf(2))*sqrt(Lf(2)^2/(Lf(1)^2+Lf(2)^2+Lf(3)^2));
    Kf_normal = [1 theta_2; theta_2 1];
end