function [ydot] = EP_KDE(t,y,wt,w)

% Input t in seconds, y is the Euler parameters
% Output ydot is the time rate of change of Euler parameters

w = interp1(wt, w, t)'; % 3x1 column, angular velocity [rad/s]

b0=y(1);
b1=y(2);
b2=y(3);
b3=y(4);

B = [-b1 -b2 -b3 ; ...
      b0 -b3  b2 ; ...
      b3  b0 -b1 ; ...
     -b2  b1  b0 ];

beta_dot = 0.5*B*w ;

ydot = beta_dot;

end

