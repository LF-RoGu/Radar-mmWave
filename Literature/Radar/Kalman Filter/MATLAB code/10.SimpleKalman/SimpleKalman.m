function volt = SimpleKalman(z)
%
%
persistent A H Q R 
persistent x P
persistent firstRun


if isempty(firstRun)
  A = 1;
  H = 1;
  
  Q = 0;
  R = 4;

  x = 14;
  P =  6;
  
  firstRun = 1;  
end

% Kalman algorithm  
xp = A*x;           % I. Prediction of estimate
Pp = A*P*A' + Q;    %    Prediction of error cov

K = Pp*H'*inv(H*Pp*H' + R); % II. Computation of Kalman gain

x = xp + K*(z - H*xp); % III. Comp. of state estimate
P = Pp - K*H*Pp;       % IV. Comp. of error cov.


volt = x;