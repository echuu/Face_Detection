% calculate threshold for weak classifiers
% delta      : weak classifier matrix (256 x 6416)
% delta_size : # of weak classifiers 6416 



delta_face_means    = zeros(delta_size, 1); % store the mean of faces
delta_face_sd       = zeros(delta_size, 1); % store the sd of faces

delta_nonface_means = zeros(delta_size, 1);  % store the mean of nonfaces
delta_nonface_sd    = zeros(delta_size, 1);  % store the sd of nonfaces

n_faces    = face_size; % change these in load_data.m
n_nonfaces = nonface_size;

faces = faces(:, 1:n_faces);
nonfaces = nonfaces(:, 1:n_nonfaces);


% store result of inner product of each weak classifier and each face/nonface
faces = faces';         % n_faces    x 256  -- faces stored in rows
nonfaces = nonfaces';   % n_nonfaces x 256  -- faces stored in cols

positive = zeros(face_size, 1);
negative = zeros(nonface_size, 1);

faces    = double(faces);
nonfaces = double(nonfaces);


% project each face/nonface onto each weak classifier
% column i of pos/neg corresponds to all faces/nonfaces projected
% onto the i-th weak classifer

tic
pos = faces * delta;     % 800 x 36480
delta_face_means = mean(pos, 1)';
delta_face_sd = std(pos, 1)';
clear pos;
toc

tic
neg = nonfaces * delta;  % 3200 x 36480
delta_nonface_means = mean(neg, 1)';
delta_nonface_sd = std(neg, 1)';
clear neg;
toc
