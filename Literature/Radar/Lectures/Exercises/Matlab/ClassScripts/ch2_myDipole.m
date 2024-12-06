clc
clear
close all
f = 76e9;
speedOfLight = 299792458;
lambda = 0.3/(f/1e9);
fprintf('Lambda is %g\n',lambda);
wavenumber = 2*pi/lambda;
Range = 1000*lambda;
phi = linspace(0,pi,360);
% create observation points in xy-plane
ObservationPoints = Range*[sin(phi);cos(phi)];
plot(ObservationPoints(1,:),ObservationPoints(2,:));
title("Observations points","FontSize",20)
xlabel("x in m","FontSize",20)
ylabel("y in m","FontSize",20)
axis equal
SizeOrAperture = 3.5*lambda;
spacing = 0.005*lambda;
% Place dipoles
N = SizeOrAperture/spacing;

for k = 1:length(phi)
    Ex(k) = 0;
    Ey(k) = 0;
    for inner = 1:N
        posOfDipole = [0;inner*spacing];
        tmp = getMyField(ObservationPoints(:,k),posOfDipole,wavenumber); 
        Ex(k) = Ex(k)+tmp(1);
        Ey(k) = Ey(k)+tmp(2);
    end
end
% Calculate magnitude
Mag = sqrt(Ex.*conj(Ex)+Ey.*conj(Ey));
figure
plot(rad2deg(phi),Mag)
xlabel("phi in degree","FontSize",20)
ylabel("Magnitude","FontSize",20)



function field = getMyField(pos,loc,wavenumber)
    % Far field of Herzian Dipole in xy-plane orientated in y-direction 
    % loc: Localization of Dipole in xy-plane
    % pos: Where to calculate the field in xy-plane
    % wavenumber: 2pi/lambda
    phi = atan2(loc(2)-pos(2),loc(1)-pos(1));
    ephi(1) = -sin(phi);
    ephi(2) = cos(phi);
    Range = norm([loc(1)-pos(1);loc(2)-pos(2)]);
    fak   = exp(-j*wavenumber*Range)/Range*cos(phi);
    Ex    = ephi(1)*fak;
    Ey    = ephi(2)*fak;
    field = [Ex Ey];
end
