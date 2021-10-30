function [h] = auxRIRgenerator(roomDimensions, sourcePosition, recieversPosition, beta)

% roomDimensions = [4 4 3];
% sourcePosition = [2 3 1.5];
% recieversPosition = [2 1 2.9; 2 3 2.9];
% beta = 0.3;
%% Default Parameters
    c = 343;                    % Sound velocity (m/s)
    fs = 16000;                 % Sample frequency (samples/s)
    order = -1;                 % Reflection order
    dim = 3;                    % Room dimension
    hp_filter = 1;              % Enable high-pass filter
    n_samples= 4096;
    recieversType = 'omnidirectional';
    receiversOrientation = [0,0];

%% Run RIR Generation
    n_mics=size(recieversPosition,1);
    h=zeros(n_samples,n_mics);
%     figure; hold on;
    for i=1:n_mics
        h(:,i) = rir_generator(c, fs, recieversPosition(i,:), sourcePosition, roomDimensions,...
                          beta, n_samples, recieversType, order, dim,...
                          receiversOrientation, hp_filter);
%         plot(h(i,:))
    end
end
