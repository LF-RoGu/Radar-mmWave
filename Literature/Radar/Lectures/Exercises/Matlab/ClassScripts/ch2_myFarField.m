clc
clear
close all
f = 76e9;
speedOfLight = 299792458;
lambda = 0.3/(f/1e9);
fprintf('Lambda is %g\n',lambda);
wavenumber = 2*pi/lambda;
AntennaSize = 0.3; 
% Far field:
R0 = 2*AntennaSize^2/lambda;
fprintf('Far field starts at %g\n',R0)