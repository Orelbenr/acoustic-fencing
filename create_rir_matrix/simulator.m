clear; close all; clc;
addpath RIR-Generator

%% Parameters - change only here
save_ON = true;         % wheter to save the results or only plot grapgh
folder_name ='demo';    % output folder name
file_name = 'demo.mat'; % output file name
num_samples = 100;        % number of situation (AKA N)
      
% disributaion of the rooms dimension
room_pd = truncate( makedist('normal','mu',(6+2.5)/2, 'sigma', 1), 2.5, 6);

reception_zones = [-60 0;0 60]; % reception zones [degrees]
max_sitting_rad = 1.5;          % max distance of speaker from room center [meters]
max_sitting_var = 0.3;          % statistics of speakers distance

% RT60 distribution defined by beta
beta= [0.3, 0.2, 0.8] ; %[avg, min ,max] truncated poisson 0.1 steps  (old [0.2 0.3 0.4])

num_speakers = 2;   % number of total speakers
mics_num = 9;       % numer of microphones
mic_rad = 0.13 / 2; % radius of mics array
mic_rad_var = 0.05; % statistics of mics array displacement from center of room
mic_phase = 0;      % phase shift of the mics array



%% Calculate Mics Locations
mic_angle= linspace(mic_phase , 360+mic_phase , mics_num)';
mic_angle= deg2rad(mic_angle(1:end-1));
mic_rad= mic_rad*ones(size(mic_angle));
[x,y] = pol2cart(mic_angle,mic_rad);
relative_mics_pos = [ x , y , zeros(size(x)) ];
relative_mics_pos = [zeros(1,3); relative_mics_pos]; 

%% Generate RIRs
H=[]; zone_dict = []; selected_angles =[]; selected_rads=[]; selected_beta =[];
selected_x = [];
for i = 1:num_samples
    room_dim = random(room_pd,1,3);
    room_center = room_dim/2;
    mic_center = [room_center(1), room_center(2), 0.75] + rand(1,3)*mic_rad_var*2 - mic_rad_var;
    mics_pos = relative_mics_pos + mic_center;
    min_XY_rad = min(room_dim(1:2))/2;
    sitting_rad = min( max_sitting_rad, min_XY_rad -0.4);
    sitting_var = (max_sitting_var/max_sitting_rad) * sitting_rad ;
   
    
    [cur_H, cur_selection, debug_cell ] = room_generator(room_dim, reception_zones, num_speakers,...
            mics_pos, sitting_rad, sitting_var, beta);
    H = cat(4 , H , cur_H );
    zone_dict = cat(1, zone_dict, cur_selection');
    selected_angles = [selected_angles; debug_cell{1}];
    selected_rads = [selected_rads; debug_cell{2}];
    selected_beta = [selected_beta; debug_cell{3}];
    selected_x = [selected_x; room_dim(1)];
end
zone_dict = zone_dict -1;

fig2 = figure;
subplot(4,1,1); 
histogram(rad2deg(selected_angles),50);
title('Speaker Angle Distribution'); xlabel('Speaker Angle [deg]'); ylabel('Count');
subplot(4,1,2); 
histogram(selected_rads,50)
title('Speaker Distance From Center Distribution'); xlabel('Radius [meter]'); ylabel('Count');
subplot(4,1,3);
histogram(selected_x,50);
title('X Dimention Of The Room Distribution'); xlabel('X dimention [meter]');ylabel('Count');
subplot(4,1,4); 
histogram(selected_beta,7)
title('RT60 Distribution'); xlabel('RT60 [sec]');ylabel('Count');


%% Save
if save_ON
    folder = ['.\output_dir\' folder_name '\'];
    mkdir(folder)
    saveas(fig2,[folder 'Sumulation Statistics.png'])

    save([folder file_name],'H', 'zone_dict' ,'num_samples', 'room_dims',...
        'reception_zones', 'max_sitting_rad', 'max_sitting_var', 'beta', 'num_speakers',...
        'mics_num', 'mic_rad', 'mic_phase')

    fid=fopen([folder 'README.txt'],'wt');
    fprintf(fid,' reception_zones =');
    fprintf(fid,'\r\n   %f, %f',reception_zones);
    fprintf(fid,'\r\n beta=');
    fprintf(fid,'\r\n    %f',beta);
    fprintf(fid,'\r\n num_samples = %f \r\n sitting_rad = %f \r\n num_speakers = %f \r\n mics_num = %f \r\n mic_center = %f, %f, %f \r\n mic_rad = %f \r\n mic_phase = %f',...
                num_samples, sitting_rad, num_speakers,mics_num, mic_center, mic_rad(1), mic_phase);
    fclose(fid);
end

