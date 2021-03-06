% batchAda.m

if 1 == 1
	% load_data();
	% generate_weak_classifiers();
	calc_threshold();
end

if 1 == 1
	DEBUG = 0; % debug != 0 for extra iteration info

	% when faces, nonfaces come into adaboost, they have already been transposed
	% faces    -- FACE_SIZE x 256
	% nonfaces -- NONFACE_SIZE x 256
	NEG_SIZE = 0;

	m = FACE_SIZE + NONFACE_SIZE + NEG_SIZE;
	% [X, Y] = createTrain(faces, nonfaces, sub_images, FACE_SIZE, NONFACE_SIZE, NEG_SIZE);
	[X, Y] = createTrain(faces, nonfaces, 0, FACE_SIZE, NONFACE_SIZE, 0);

	% X : m x 256
	% clear faces; 
	% clear nonfaces;
	
	T = 30;

	disp('first chunk complete');
end

% rewrite the the loop that precomputes the error matrix, classification matrix
% rewrite findMinWtErrr() function to iterate thru each train_batch to find the
% index, train_batch of weak classifier that minimizes the weighted error

k = 4;     % split the number of faces/nonface into k batches
	       % X is 5000 x 256 --> dim(X_i) = 1250 x 256

train_batch = linspace(0, m, k + 1);

[F, Z, D_cur, D_prev, wc_ind, alpha, ...
	class_matrix, error_matrix] = initializeAdaBoost(m, T, delta_size);

disp('begin adaboost calculations');

for t = 1:T

	% disp(['iter ' num2str(t)]);
	tic

	[err, ind, h] = testBatch(D_cur, delta_size, k, train_batch, X, ...
								delta,delta_face_means, delta_face_sd, ...
								delta_nonface_means, delta_nonface_sd, Y);

	alpha(t) = 0.5 * log((1 - err) / err);
	
	wc_ind(t) = ind; 

	% generate strong classifier, perform updates
	F     = F + alpha(t) .* h;
	yh    = exp(-Y .* F);
	Z     = sum(yh) / m;
	D_cur = 1/Z * 1/m * yh; % update weights

	calcClassError(Y, F);
	disp(['iter ' num2str(t), ': weak classifier ', num2str(wc_ind(t))]);

	toc

end % adaboost iterations

% end batchAda.m
