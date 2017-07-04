-- boost.lua

local boost = {}

local function adaboost(T)

	print('entered adaboost function');

	local torch = require('torch');

	print('loaded torch');

	local ld    = require('load_data');
	local ext   = require('externalFunctions');
	local g     = require('common_defs');
	local class = require('classify');
	local calc  = require('calculate');
	local csv2tensor  = require('csv2tensor');

	print('finished loading all libraries');

	-- global constants---------------------------------------------------------
	FIRST_TIME = 0;
	DEBUG      = 0;
	-- global constants---------------------------------------------------------

	local h_mat   = torch.Tensor(g.total_imgs, g.delta_size);
	local err_mat = torch.Tensor(g.total_imgs, g.delta_size);
	local X, Y_train;

	-- load in faces, nonfaces (faces, nonfaces stored as rows)-----------------
	local faces    = ld.importFaces(g.csvpath, g.subset_faces, 0); --  800 x 256
	local nonfaces = ld.importNonfaces(g.csvpath, 
								g.subset_nonfaces, 0);             -- 3200 x 256
	X, Y_train     = ext.createTrain(faces, nonfaces);             -- 4000 x   1
	----------------------------------------------------------------------------

	if DEBUG == 1 then
		print(faces:size());     -- num_faces    x 256
		print(nonfaces:size());  -- num_nonfaces x 256
		print(Y_train:size());   -- total_imgs   x 256
	end

	-- generate weak classifiers
	-- each weak classifiers stored as a column vector
	local delta = ext.generateWC(g.dim, g.delta_size);

	-- calculate threshold
	--[[ 
		face_mean    (delta_size x 1) -- the average value for a face per w.c.
		face_sd      (delta_size x 1) -- the average value for a nonface per w.c.
		nonface_mean (delta_size x 1)
		nonface_sd   (delta_size x 1)
		proj         (total_imgs x delta_size) -- used for classification
	--]]

	if FIRST_TIME == 1 then 
		face_mean, face_sd, nonface_mean, nonface_sd = ext.calcThreshold(delta,
			g.delta_size, faces, nonfaces);

		torch.save('face_mean.dat', face_mean);
		torch.save('face_sd.dat', face_sd);
		torch.save('nonface_mean.dat', nonface_mean);
		torch.save('nonface_sd.dat', nonface_sd);
	else
		face_mean    = torch.load('data_files/face_mean.dat');
		face_sd      = torch.load('data_files/face_sd.dat');
		nonface_mean = torch.load('data_files/nonface_mean.dat');
		nonface_sd   = torch.load('data_files/nonface_sd.dat');
	end

	start_time = os.time();

	-- precompute projections for classification
	local proj = X * delta;
	--torch.save('proj.dat', proj);
	--proj = torch.load('proj.dat');
	end_time = os.time();
	elapsed_time = os.difftime(end_time, start_time);
	print('Projection matrix computed. Total time: '..elapsed_time);

	--------------------- free memory ----------------------------------------------
	faces    = nil;
	nonfaces = nil;
	delta    = nil;
	--------------------- free memory ----------------------------------------------

	-- precompute classifications (these don't change per iteration)
	-- store errors in matrix

	local classify = class.ll_classify;

	start_time = os.time();
	if FIRST_TIME == 0 then
		for i = 1, g.delta_size do
			local proj_i = proj[{{}, {i}}];
			local f1, f2, nf1, nf2;
			f1  = face_mean[i];
			f2  = face_sd[i];
			nf1 = nonface_mean[i];
			nf2 = nonface_sd[i];

			
			local class_vector = classify(proj_i, f1, f2, nf1, nf2);

			h_mat[{{}, {i}}] = class_vector

			err_mat[{{}, {i}}] = torch.ne(Y_train, class_vector);
		end

		-- these take really long to save
		--torch.save('classification_matrix.dat', h_mat);
		--torch.save('error_matrix.dat', err_mat);
	else
		print("Reading in classification matrix and error matrix");
		h_mat   = torch.load('classification_matrix.dat');
		err_mat = torch.load('error_matrix.dat');

		--h_mat   = csv2tensor.load('h_mat.csv');
		--err_mat = csv2tensor.load('err_mat.csv');
	end
	end_time = os.time();
	elapsed_time = os.difftime(end_time, start_time);
	print('Finished Classifications. Total time: '..elapsed_time); -- 30 seconds

	-- can free up proj matrix
	proj = nil;

	--local T = 30;
	local inv_total = 1 / g.total_imgs;

	local F = torch.Tensor(g.total_imgs, 1):zero();	     -- strong classifier
	local Z = 0; 										 -- normalizing factor

	local D_cur  = torch.Tensor(g.total_imgs, 1):fill(inv_total);
	local D_prev = torch.Tensor(g.total_imgs, 1):fill(inv_total);

	local min_ada_index = torch.Tensor(T, 1):zero(); -- store chosen w.c.'s
	local alpha = torch.Tensor(T, 1):zero();

	------ begin adaboost ------------------------------------------------------
	for t = 1, T do
		-- weighted_error = torch.Tensor(g.delta_size, 1):zero();

		error, index = class.findMinWtErr(D_cur, err_mat, g.delta_size, 0, t);
		print('wk class: '.. index);

		alpha[t]     = 0.5 * torch.log((1 - error) / error);
		--print(torch.squeeze(alpha[t]));

		min_ada_index[t] = index;

		D_prev = D_cur;

		h = h_mat[{{},{index}}];

		F     = F + torch.squeeze(alpha[t]) * h;

		yh    = torch.exp(torch.cmul(-Y_train, F)); 

		Z     = torch.sum(yh) * inv_total;

		D_cur = 1/Z * inv_total * yh;

		-- calculate empirical error:
		calc.classError(Y_train, F);
	end
	------ end adaboost --------------------------------------------------------
end

boost.adaboost = adaboost;

return boost;