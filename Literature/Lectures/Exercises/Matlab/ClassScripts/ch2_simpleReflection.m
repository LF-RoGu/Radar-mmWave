clc
clear 
close all
reflectingPlaneXmin = -10;
reflectingPlaneXmax =  2;
reflectingPlaneZmin = 0;
reflectingPlaneZmax = 0;
reflectingPlane.start = [reflectingPlaneXmin;reflectingPlaneZmin];
reflectingPlane.end = [reflectingPlaneXmax;reflectingPlaneZmax];


if reflectingPlaneZmin ~ reflectingPlaneZmax
    fprintf('Not yet')
end
line([reflectingPlane.start(1) reflectingPlane.end(1)],...
     [reflectingPlane.start(2) reflectingPlane.end(2)],'LineWidth',10)
source = [-1;2];
hold on
plot(source(1),source(2),'x','MarkerSize',20)
mirror = source; 
mirror(2) = -mirror(2);
plot(mirror(1),mirror(2),'x','MarkerSize',20)
theta = deg2rad(0:20:180);
for k = 1:length(theta)
    x = 10*sin(theta(k));
    y = 10*cos(theta(k));
   
    L2.start = mirror;
    L2.end = [x;y];
    [point,flag] = intersectionPoint(reflectingPlane,L2);
    if flag == 1
        line([point(1) x],[point(2) y]);
        line([point(1) source(1)],[point(2) source(2)]);
    end
    line([source(1) x],[source(2) y]);
    
    plot(x,y,'.','MarkerSize',20)
   % line([mirror(1) x],[mirror(2) y]);

end
xlim([-10 10])

function [point,flag] = intersectionPoint(L1,L2)
    % Find the two vector parallel to lines
    N1 = (L1.end-L1.start);
    n1 = N1/norm(N1);
    %
    N2 = (L2.end-L2.start);
    n2 = N2/norm(N2);
    %
    A = [n2 -n1];
    tmp = inv(A)*(L1.start-L2.start);
    point = L1.start + n1*tmp(2);
    if tmp(2) > norm(N1) || tmp(2) < 0 || tmp(1) < 0 || tmp(1) > norm(N2)
        flag = 0;
    else
        flag = 1;
    end

end
axis equal


