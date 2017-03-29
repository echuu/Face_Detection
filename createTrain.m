% process_negs has to be run first -- makes references to negative images
% create training set for adaboost
% return values:
%     X: training set with faces, non_faces, and negatives
%     Y: correct classification of each face (1), non_face, negative (-1)

function [X, Y] = createTrain(faces, nonfaces, sub_images,...
								n_faces, n_nonfaces, n_negs)
	m = n_faces + n_nonfaces + n_negs;
	Y = zeros(m, 1);
	Y(1:m)       = -1;
	Y(1:n_faces) =  1;

	X = [faces; nonfaces];

