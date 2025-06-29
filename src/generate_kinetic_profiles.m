


function [n,T,r] = generate_kinetic_profiles(channels,N,Noise)

% radius
r = linspace(0,1,channels)';

% density parameters
Na = 0.5 + rand(1,N)*2;
Nb = 0.5 + rand(1,N)*2;
N0 = 1e19 + rand(1,N)*1e20;

% temperature parameters
Ta = 0.5 + rand(1,N)*2;
Tb = 0.5 + rand(1,N)*2;
T0 = 1000 + rand(1,N)*20000;

% density and temperature profiles
n = N0.*(1 - r.^Na).^Nb;
T = T0.*(1 - r.^Ta).^Tb;

% add noise 
n = n + Noise.*normrnd(0,std(n,[],1).*ones(size(mean(n,2))));
T = T + Noise.*normrnd(0,std(T,[],1).*ones(size(mean(T,2))));

end