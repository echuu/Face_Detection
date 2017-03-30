function [err, ind] = findMinWtErr(weights, error_matrix, dim, DEBUG, t)

	min_ind = -1;
	min_wt_err = 9999;

	error_vec = weights' * error_matrix;

	for i = 1:dim
		if error_vec(i) < min_wt_err
			min_wt_err = error_vec(i);
			min_ind = i;
		end
	end

	err = min_wt_err;
	ind = min_ind;

	if DEBUG ~= 0
	    disp(['iter: ' num2str(t)... 
	    	  ' weak classifier: ' num2str(ind) ' -- '...
	           'weighted error: ' num2str(err)]);
	end