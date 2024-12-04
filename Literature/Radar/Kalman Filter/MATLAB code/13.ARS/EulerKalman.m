function [psi theta phi] = EulerKalman(A, z)
%
%
persistent H Q R
persistent x P
persistent firstRun


if isempty(firstRun)
  H = eye(4);

  Q = 0.0001*eye(4); 
  R = 10*eye(4);

  x = [1 0 0 0]';  
  P = 1*eye(4);
  
  firstRun = 1;  
end

% x = Euler Parameter, beta (aka, quaternion)

% Kalman Filter algorithm
xp = A*x;
Pp = A*P*A' + Q;

K = Pp*H'*inv(H*Pp*H' + R);

x = xp + K*(z - H*xp);   
P = Pp - K*H*Pp;

e=EP2Euler321(x); % Euler param. to yaw-pitch-roll
psi  =e(1); % yaw [rad]
theta=e(2); % pitch [rad]
phi  =e(3); % roll [rad]