clc
clear
close all


OVerlayAllScans = 0; 
LaneInformation.y01 = 8;
LaneInformation.y02 = 4;

Vehicles.Lane1.Xpos = [1];
Vehicles.Lane2.Xpos = [5];
Vehicles.Lane1.Ypos = [8];
Vehicles.Lane2.Ypos = [4];

Vehicles.Lane1.Length = [4];
Vehicles.Lane2.Length = [10];
Vehicles.Lane1.Speed = [1];
Vehicles.Lane2.Speed = [0.5];
Vehicles.Lane1.PTruck = 0.25;
Vehicles.Lane2.PTruck = 0.7;
Vehicles.Lane1.Distance = randn(1,1)*10+20;
Vehicles.Lane2.Distance = randn(1,1)*10+20;
Vehicles.Lane1.TmpLength = 4+floor(rand()*0.25)*10;
Vehicles.Lane2.TmpLength = 4+floor(rand()*0.25)*10;

Radar.x = 50;
Radar.y = 0;
Radar.Ray = [0:0.1:300];



fig1 = figure();
exFile = fopen('targets.csv','w');
dispString = [];
N0 = 200;
for outer = 1:N0
    for inner = 1:length(dispString)
        fprintf('\b')
    end
    dispString = sprintf('%d/%d',outer,N0);
    fprintf('%s',dispString);
    %pause(0.1);
    if OVerlayAllScans == 0 
        clf(fig1);
    end
    subplot(2,1,1)
    % Move vehicles
    Vehicles.Lane1.Xpos = Vehicles.Lane1.Xpos + Vehicles.Lane1.Speed;
    Vehicles.Lane2.Xpos = Vehicles.Lane2.Xpos + Vehicles.Lane2.Speed;
    % Add new vehicles
    Vehicles.Lane1 = createNewBoxes(Vehicles.Lane1);
    Vehicles.Lane2 = createNewBoxes(Vehicles.Lane2);
    % Remove old vehicles
    ind = find(Vehicles.Lane1.Xpos > 100);
    if ~isempty(ind)
        Vehicles.Lane1.Xpos = Vehicles.Lane1.Xpos(1:ind-1);
        Vehicles.Lane1.Ypos = Vehicles.Lane1.Ypos(1:ind-1);
        Vehicles.Lane1.Length = Vehicles.Lane1.Length(1:ind-1);
        Vehicles.Lane1.Speed = Vehicles.Lane1.Speed(1:ind-1);
    end
    ind = find(Vehicles.Lane2.Xpos > 100);
    if ~isempty(ind)
        Vehicles.Lane2.Xpos = Vehicles.Lane2.Xpos(1:ind-1);
        Vehicles.Lane2.Ypos = Vehicles.Lane2.Ypos(1:ind-1);
        Vehicles.Lane2.Length = Vehicles.Lane2.Length(1:ind-1);
        Vehicles.Lane2.Speed = Vehicles.Lane2.Speed(1:ind-1);
    end
   
    
    % Plot vehicles
    plotBoxes(Vehicles.Lane1);
    plotBoxes(Vehicles.Lane2);
    xlim([0 100]);
    ylim([0 10]);
    tmps = sprintf('%d/%d',length(Vehicles.Lane1.Xpos),length(Vehicles.Lane1.Xpos));
    title(tmps);
    [x1,y1,v1] = sampleAllBoxes(Vehicles.Lane1);
    [x1,y1,v1] = seeIfShadows(x1,y1,v1,Vehicles.Lane2,Radar);
    [x2,y2,v2] = sampleAllBoxes(Vehicles.Lane2);
    plotShadows(Vehicles.Lane2,Radar);
    x = [x1 x2];
    y = [y1 y2];
    v = [v1 v2];
    hold on;
    %plot(x,y,'.')
    [x,y,R,RR,phi] = takeRandom(x,y,v,Radar);
    plot(x,y,'x')
    fprintf(exFile,'%d',outer);
    for exCount = 1:256
        if exCount <= length(x)
            fprintf(exFile,',%4.2f,%4.2f,%4.2f',R(exCount),RR(exCount),phi(exCount));
        else
            fprintf(exFile,',0,0,0');
        end
    end
    fprintf(exFile,'\n');

    
    xlabel('x [m]','FontSize',20)
    ylabel('y [m]','FontSize',20)
    subplot(2,1,2)
    plot(x,RR,'.')
    hold on
    xlim([0 100])
    ylim([-1 1])
    xlabel('x [m]','FontSize',20)
    ylabel('RR [m/sec]','FontSize',20)
    if mod(outer,10) == 0
        drawnow
    end
end
fclose(exFile);

function [x,y,R,RR,phi] = takeRandom(x0,y0,v0,radar)
    R0 = sqrt((x0-radar.x).^2+(y0-radar.y).^2);
    N = 50; % number of detections 
    x = [];
    y = [];
    R = [];
    RR = [];
    v = [];

    phi = [];
    for k = 1:N
        ind = rand()*length(x0);
        ind = floor(ind)+1;
        x = [x x0(ind(1))];
        y = [y y0(ind(1))];
        R = [R R0(ind(1))];
        v = [v v0(ind(1))];
        
    end
    phi = atan2((y-radar.y),(x-radar.x));
    phi = phi+randn(size(phi))*0.000;
    x   = R.*cos(phi)+radar.x;
    y   = R.*sin(phi)+radar.y;
    RR  = v.*cos(phi);
end

function plotBoxes(Vehicles)
    for k = 1:length(Vehicles.Xpos)
        rectangle('Position',[Vehicles.Xpos(k),Vehicles.Ypos(k),Vehicles.Length(k),2])
    end
end

function plotShadows(L1,radar)
    for k = 1:length(L1.Xpos)
        x01 = L1.Xpos(k);
        % sketch all corners
        if x01 < 100
            y01 = L1.Ypos(1);
            dx = x01 - radar.x;
            dy = y01 - radar.y;
            line([radar.x radar.x+3*dx],[radar.y radar.y + 3*dy]);
            y01 = L1.Ypos(1) + 2;
            dx = x01 - radar.x;
            dy = y01 - radar.y;
            line([radar.x radar.x+3*dx],[radar.y radar.y + 3*dy]);
           
            x01 = x01+L1.Length(k);
            y01 = L1.Ypos(1);
            dx = x01 - radar.x;
            dy = y01 - radar.y;
            line([radar.x radar.x+3*dx],[radar.y radar.y + 3*dy]);
           
            y01 = L1.Ypos(1)+2;
            dx = x01 - radar.x;
            dy = y01 - radar.y;
            line([radar.x radar.x+3*dx],[radar.y radar.y + 3*dy]);
        end
    end
end

function [x,y,v] = seeIfShadows(x1,y1,v1,Lane2Vehicles,Radar)
    % BRUTE force
    x = [];
    y = [];
    v = [];
   
    for k = 1:length(x1)
        shadowed = 0;
        if x1(k) < 100
            % Shoot a ray in target direction from radar
            d = [x1(k) - Radar.x;y1(k) - Radar.y];
            dis = norm(d);
            d = d/dis;
            tmpRay = Radar.Ray(Radar.Ray<dis);
            d = d.*tmpRay;
            d(1,:) = d(1,:) + Radar.x;
            d(2,:) = d(2,:) + Radar.y;
            
            ik = 1;
            while ik <= length(Lane2Vehicles.Xpos) & shadowed == 0
                vec.x = Lane2Vehicles.Xpos(ik);
                vec.y = Lane2Vehicles.Ypos(ik);
                vec.L = Lane2Vehicles.Length(ik);
                vec.v = Lane2Vehicles.Speed(ik);
                shadowed = shadowed + findIfRayHitsVehicle(d,vec);
       
                ik = ik + 1;
            end
        end
    
        if shadowed == 0
            x = [x x1(k)];
            y = [y y1(k)];
            v = [v v1(k)];
        end
   end
end

function [x,y,v] = sampleAllBoxes(Vehicles)
    x = [];
    y = [];
    v = [];
    for k = 1:length(Vehicles.Xpos)
        Vehicle.x = Vehicles.Xpos(k);
        Vehicle.y = Vehicles.Ypos(k);
        Vehicle.L = Vehicles.Length(k);
        Vehicle.v = Vehicles.Speed(k);

        [tmpx,tmpy,tmpv] = sampleBox(Vehicle);
        x = [x tmpx];
        y = [y tmpy];
        v = [v tmpv];

    end
    
end

function [x,y,v] = sampleBox(Vehicle)
    x = [];
    y = [];
    v = [];
    posx = Vehicle.x;

    tmpx = posx:0.1:posx+Vehicle.L;
    x = [x tmpx];
    y = [y Vehicle.y*ones(size(tmpx))];
    v = [v Vehicle.v*ones(size(tmpx))];
    tmpy = Vehicle.y:0.1:Vehicle.y+2;
    x = [x posx*ones(size(tmpy))];
    y = [y tmpy]; 
    v = [v Vehicle.v*ones(size(tmpy))];
    tmpy = Vehicle.y:0.1:Vehicle.y+2;
    x = [x (posx+Vehicle.L)*ones(size(tmpy))];
    y = [y tmpy]; 
    v = [v Vehicle.v*ones(size(tmpy))];
end

function [Vehicles] = createNewBoxes(Vehicles)
    if min(Vehicles.Xpos) > Vehicles.Distance+Vehicles.TmpLength;
        Vehicles.Xpos = [1 Vehicles.Xpos];
        Vehicles.Ypos = [Vehicles.Ypos(end) Vehicles.Ypos];
        Vehicles.Speed = [Vehicles.Speed(end) Vehicles.Speed];
        Vehicles.Length = [Vehicles.TmpLength Vehicles.Length];
        Vehicles.Distance = randn(1,1)*10+20;
        Vehicles.TmpLength = 4+floor(rand<Vehicles.PTruck)*10;
    end 
end

function hit = findIfRayHitsVehicle(ray,vec)
    hit = 0;
    [xBox,yBox,vBox] = sampleBox(vec);
    xRay = ray(1,:);
    yRay = ray(2,:);

    k = 1;
    while k <= length(xBox) & hit == 0
        dis = (xRay-xBox(k)).^2+(yRay-yBox(k)).^2;
        ind = find(dis < 0.1);
        if length(ind) > 0 
            hit = 1;
        end
        k = k+1;  
    end
end