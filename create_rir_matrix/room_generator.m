function [H, selected_zones, debug_cell] = room_generator(room_dim, reception_zones, num_speakers,...
                                                            mics_pos , sitting_rad, sitting_var, beta)
%% Documentation
% --- Inputs ---
% room_dim: vector [x, y, z] in meters.
% reception_zones: matrix with dim=(n,2), where n are the number of zones, 
%                  and each row represent the zone start and end.  In Deg                             
% mics_pos: matrix with dim=(m,3), where m is thr number of mics.
%           each row represent mic location [x,y,z] in meters.
%           the first one is the reference.
% sitting_rad: number denoting the avg siting radius
% sitting_var: number denoting by how much a speaker's radius can change 
% beta: room reverberation.
% --- Outputs ---
% H: matrix with dim=(n,m,k), where m is the number of mics.
%                n is the number of zones.
%                k is the length of the rir.
% selected_zones: vector of length num_speakers signifing the selected reception zone for each speaker
% debug_cell: contains statistics about random selections in the function
%% -- Find Speaker Locations -- 
    % - - - Find Angles - - - 
    % takes the input reception zones and randomly chooses a
    % zone for each speaker. then, we randmoly choose an angle within 
    % each selected reception zone.
    num_zones = size(reception_zones,1);
    selected_zones = randsample(num_zones , num_speakers);
    reception_zones = reception_zones( selected_zones ,: );
    ang_step=reception_zones(:,2)-reception_zones(:,1);
    angles = (rand(num_speakers,1)*(0.7-0.3) +0.3).*ang_step + reception_zones(:,1);
    
    % - - - Find Radiuses - - -
    % selects a random radius for each speaker. the radius is unifomly selected
    % from [sitting_rad-0.3, sitting_rad+0.3]
    room_center = room_dim/2;
    rads = rand(num_speakers,1)*sitting_var*2 +(sitting_rad-sitting_var);
    
    % - - - Convert To X,Y,Z - - -
    % converts selected location to x,y,z from rad,ang,z
    angles = deg2rad(angles);
    [x,y] = pol2cart(angles(:),rads(:));
    z = rand(num_speakers,1)*(1.4-1) +1 +(randi(2,num_speakers,1)-1)*0.6;
    locs = [ x + room_center(1) , y + room_center(2) , z ];
    
%% -- Create RIRs --
    % - - - Choose Beta - - -
    % randomly chooses RT60 from truncated poison distribution acording to beta parmaters
    beta = 10*beta;
    cur_beta = random( truncate( makedist('poisson','lambda',beta(1)), beta(2), beta(3)))/10;
    
    % - - - Calculate RIRs - - -
    H=[];
    for i=1:num_speakers
        H = cat(3, H , auxRIRgenerator(room_dim, locs(i,:) , mics_pos, cur_beta));
    end
    
%% --- debug ---
    %figure;
    % scatter(locs(:,1),locs(:,2))
    % scatter(room_center(1),room_center(2))

    
    debug_cell = cell(3,1);
    debug_cell{1} = angles;
    debug_cell{2} = rads;
    debug_cell{3} = cur_beta;
end