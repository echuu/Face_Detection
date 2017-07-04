% batchMinimize.m

% findMinWtErr() rewritten to minimize over batches

% weights (m x 1) holds weights for each of the images
function [err, batch_id, ind, h] = batchMinimize(weights, dim, num_batches)
	% return quantities:
		% weighted error (needed for updating alphas)
		% batch_id: used to determine which batch to find the wk class.
		% ind: index of the w.c. within the batch
		% h: the classification that gives the lowest weighted error

	min_id     = -1;
	min_wt_err = 9999;
	batch_id   = 0;

	% read in error_matrix, class_matrix for each of the k batches

	for i = 1:num_batches

		batch_error_name = ['err_mat_', num2str(i), '.csv'];
		batch_class_name = ['class_mat_', num2str(i), '.csv'];

		err_mat_i   = csvread(batch_error_name);
		class_mat_i = csvread(batch_class_name); 

		err_vec_i = weights' * err_mat_i; % 1 x delta_size

		% find minimum weighted error across all batches
		for j = 1:dim % iterate thru weak classifiers
			if err_vec(j) < min_wt_err
				min_wt_err = err_vec(j);
				min_ind = j;
				batch_id = i;
				h = class_mat_i(:, min_ind); % 'best' class. (w.r.t. wt. error)
			end
		end % inner for loop

	end % outer for loop

	err      = min_wt_err;
	ind      = min_ind;
	batch_id = batch;

% end batchMinimize.m
