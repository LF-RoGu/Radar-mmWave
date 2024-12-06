clc
clear
close all
theta_in = linspace(-pi/2,pi/2,360);
NoAntennas = 7;
fprintf('NoAntennas:%d\n',NoAntennas);
% fprintf('Total number of antennas: %d\n',NoAntennas*2+1)
f = 76e9;
speedOfLight = 299792458;
lambda = 0.3/(f/1e9);
fprintf('Lambda is %g\n',lambda);
wavenumber = 2*pi/lambda;
d = lambda/2;
fprintf('Antenna distance: %g\n',d);
fprintf('Equals %g nanoseconds\n',d/speedOfLight*1e9);
%pos = [-NoAntennas:NoAntennas]*d;
pos = [1:NoAntennas]*d;
KillThemAll = ones(size(pos));
KillThemAll(3:end-1) = 0;
fprintf('Size of antenna: %g\n',pos(end)-pos(1))

plot(pos,0,'.')
for k = 1:length(theta_in)
    arg = -j*wavenumber*pos*sin(theta_in(k));
    S(k) = sum(exp(arg).*KillThemAll);
end
plot(rad2deg(theta_in),abs(S))