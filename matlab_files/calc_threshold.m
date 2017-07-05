% calc_threshold.m

% calculate threshold for weak classifiers
% delta      : weak classifier matrix (256 x 6416)
% delta_size : # of weak classifiers 6416 


delta_face_means    = zeros(delta_size, 1);       % store the mean of faces
delta_face_sd       = zeros(delta_size, 1);       % store the sd of faces

delta_nonface_means = zeros(delta_size, 1);  	  % store the mean of nonfaces
delta_nonface_sd    = zeros(delta_size, 1);  	  % store the sd of nonfaces

% FACE_SIZE    = FACE_SIZE;                       % change these in load_data.m
% NONFACE_SIZE = NONFACE_SIZE;

faces = faces(:, 1:FACE_SIZE);
nonfaces = nonfaces(:, 1:NONFACE_SIZE);

% store result of inner product of each weak classifier and each face/nonface
faces    = faces';      % FACE_SIZE    x 256  -- faces stored in rows
nonfaces = nonfaces';   % NONFACE_SIZE x 256  -- faces stored in cols

positive = zeros(FACE_SIZE, 1);
negative = zeros(NONFACE_SIZE, 1);

faces    = double(faces);
nonfaces = double(nonfaces);

% project each face/nonface onto each weak classifier
% column i of pos/neg corresponds to all faces/nonfaces projected
% onto the i-th weak classifer

pos = faces * delta;                             % 800 x 36480
delta_face_means = mean(pos, 1)';
delta_face_sd = std(pos, 1)';
clear pos;

neg = nonfaces * delta;                          % 3200 x 36480
delta_nonface_means = mean(neg, 1)';
delta_nonface_sd = std(neg, 1)';
clear neg;


% end calc_threshold.m
