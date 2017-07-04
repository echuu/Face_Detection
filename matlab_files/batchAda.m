% batchAda.m

% see first part of adaboost.m

% rewrite the the loop that precomputes the error matrix, classification matrix
% rewrite findMinWtErrr() function to iterate thru each batch to find the
% index, batch of weak classifier that minimizes the weighted error

% ---------------------- ADABOOST PRECOMPUTATIONS ---------------------------- %

k = 4; % split the number of faces/nonface into k batches
       % X is 5000 x 256 --> dim(X_i) = 1250 x 256
batch = linspace(0, m, k + 1);
for i = 1:k

	m_i    = batch(2) - batch(1); % number of images
	X_i    = X(batch(i) + 1:batch(i+1), :);
	proj_i = X_i * delta; % m_i x delta_size

	% initialize err_i, class_i with proper dimensions (both m_i x delta_size)
	err_i   = zeros(m_i, delta_size);
	class_i = zeros(m_i, delta_size);

	% calculate error matrix, classification matrix for the i-th batch
	% 1 <= i <= k
	for j = 1:delta_size
		[h_i, ~]   = gauss_classify(proj_i(:,j), delta_face_means(j),...
			delta_face_sd(j), delta_nonface_means(j), delta_nonface_sd(j));
		err_i(:,j)   = h_i ~= Y(batch(i) + 1:batch(i+1));
		class_i(:,j) = h_i; 
	end % inner for loop

	batch_error_name = ['err_mat_', num2str(i), '.csv'];
	batch_class_name = ['class_mat_', num2str(i), '.csv'];

	csvwrite(batch_error_name, err_i);
	csvwrite(batch_class_name, class_i);

	disp(['Finished batch ', num2str(i), ' calculations']);
end % outer for loop
disp('Finished batch calculations');

% ---------------------- END ADABOOST PRECOMPUTATIONS ------------------------ %
T = 10;

for t = 1:T

	tic

	[err, batch_id, ind, h] = batchMinimize(D_cur, delta_size, k);

	alpha(t) = 0.5 * log((1 - err) / err);
	
	batch_ind(t) = batch_id;
	wc_ind       = ind;

	F     = F + alpha(t) .* h;
	yh    = exp(-Y .* F);
	Z     = sum(yh) / m;
	D_cur = 1/Z * 1/m * yh; % update weights

	calcClassError(Y, F); 

	toc

end % adaboost iterations

% end batchAda.m
