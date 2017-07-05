% batchMinimize.m

% findMinWtErr() rewritten to minimize over batches

% weights (m x 1) holds weights for each of the images
function [err, ind, h] = testBatch(weights, dim, num_batches, train_batch, X, delta,...
			delta_face_means, delta_face_sd, delta_nonface_means, delta_nonface_sd, Y)

	% return quantities:
		% weighted error (needed for updating alphas)
		% batch_id: used to determine which train_batch to find the wk class.
		% ind: index of the w.c. within the train_batch
		% h: the classification that gives the lowest weighted error

	min_id     = -1;
	min_wt_err = 9999;

	% read in error_matrix, class_matrix for each of the k batches

	err_vec_i = zeros(1, dim);
	tic
	computeErrClass();
	toc

	% find minimum weighted error across all batches
	for j = 1:dim % iterate thru weak classifiers
		if err_vec_i(j) < min_wt_err
			min_wt_err = err_vec_i(j);
			min_ind = j;
		end
	end % inner for loop

	% use cached train_batch
	h = zeros(size(weights,1), 1); % m x 1

	p_i = X * delta(:,min_ind);

	[h, ~] = gauss_classify(p_i, delta_face_means(min_ind),...
			delta_face_sd(min_ind), delta_nonface_means(min_ind),...
			delta_nonface_sd(min_ind));

	err      = min_wt_err;
	ind      = min_ind;

% end batchMinimize.m
