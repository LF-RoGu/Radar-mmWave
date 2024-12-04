function [phi theta] = EulerAccel(ax, ay)
%
%
g = 9.8; % average of sqrt(ax^2 + ay^2 + az^2)

theta = asin(  ax / g );
phi   = asin( -ay / (g*cos(theta)) );

 