function [gradients,MSE,Dkl] = MG_AutoEncoder_v1(dX,VAE,parameters)

    [dXp,Codep] = VAE_Network(dX,VAE,parameters,1,VAE.Sigma);
    
    MSE = mean((dXp-dX).^2,'all');
    
    Code_mu = mean(Codep,2);
    Code_std = std(Codep,[],2);

    Dkl = log(Code_std) + (1 + Code_mu.^2)./(2.*Code_std.^2) - 1/2;
    Dkl = mean(Dkl);

    Dkl = Dkl.*VAE.alpha;

    Loss = MSE + Dkl;

    gradients = dlgradient(Loss,parameters);

end