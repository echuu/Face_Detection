% process_negs has to be run first -- makes references to negative images
% create training set for adaboost
% return values:
%     X: training set with faces, non_faces, and negatives
%     Y: correct classification of each face (1), non_face, negative (-1)

function [X, Y] = createTrain(face16, nonface16, sub_images,...
								n_faces, n_nonfaces, n_negs)
	dim = 16;
	m = n_faces + n_nonfaces + n_negs;
	Y(1:m)       = -1;
	Y(1:n_faces) =  1;
	Y = Y';
	
	% X = [faces | non_faces | negatives]
	X = zeros(dim * dim, m);
	X(:, 1:n_faces)                        = face16(:, 1:n_faces);
	X(:, n_faces + 1:n_faces + n_nonfaces) = nonface16(:, 1:n_nonfaces);
	X(:, n_faces + n_nonfaces + 1:end)     = sub_images;
