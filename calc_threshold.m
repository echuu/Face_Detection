% calculate threshold for weak classifiers
% delta      : weak classifier matrix (256 x 6416)
% delta_size : # of weak classifiers 6416 


delta_face_means    = zeros(delta_size, 1); % store the mean of faces
delta_face_sd       = zeros(delta_size, 1); % store the sd of faces

delta_nonface_means = zeros(delta_size, 1);  % store the mean of nonfaces
delta_nonface_sd    = zeros(delta_size, 1);  % store the sd of nonfaces

% store result of inner product of each weak classifier and each face/nonface
positive = zeros(face_size, 1);
negative = zeros(nonface_size, 1);

faces    = double(faces);
nonfaces = double(nonfaces);

tic
for i = 1:delta_size
    
    for j = 1:size(faces,2)
        % positive(j) = dot(delta(:,i),face16(:,j));
        positive(j) = delta(:,i)' * faces(:,j);
    end
    
    delta_face_means(i) = mean(positive);
    delta_face_sd(i)  = std(positive);
    
    for j = 1:size(nonfaces,2)
        % negative(j) = dot(delta(:,i),nonface16(:,j));
        negative(j) = delta(:,i)' * nonfaces(:,j);
    end
    
    delta_nonface_means(i) = mean(negative);
    delta_nonface_sd(i) = std(negative);
    disp(['calculating weak classifier ' int2str(i)])
end
toc