roomDimensions = [6 4 3];
sourcePosition = [2 3 1.5];
recieversPosition = [4 1 2];

recieversType = {'omnidirectional'};
receiversOrientation = [0,0];
beta = [0.3];

[h] = auxRIRgenerator(roomDimensions, sourcePosition, recieversPosition, recieversType, receiversOrientation, beta);