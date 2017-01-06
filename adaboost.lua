--[[ -------------- adaboost.lua
	
	this file is run after train.lua is finished
	read in the following .dat files before running adaboost.lua:

		(1) delta.dat
		(2) projections.dat  ---- this file is huge
		(3) face_mean.dat
		(4) face_sd.dat
		(5) nonface_mean.dat
		(6) nonface_sd.dat
		(7) Y_train.dat

	train adaboost algorithm
	
	matrices:
		delta  : 256   x 36480
		X      : 256   x 57194
		Y      : 57194 x 1

	functions:
		adaboost()

--]] 


local boost = {}

local ext = require "externalFunctions";
local classify = require "classifly.lua"; -- access kmeans() function

--[[ ----- uncomment if want these to be global

	boost.DELTA        = torch.load('delta.dat');
	boost.proj         = torch.load('projections.dat');
	boost.face_mean    = torch.load('face_mean.dat');
	boost.nonface_mean = torch.load('nonface_mean.dat');
	boost.face_sd      = torch.load('face_sd.dat');
	boost.nonface_sd   = torch.load('nonface_sd.dat');
	boost.Y_train      = torch.load('Y_train.dat');

--]]


boost.adaboost()

local function adaboost(T)
	-- precompute the projections


	start_time = os.time();

	delta        = torch.load('delta.dat');
	proj         = torch.load('projections.dat');
	face_mean    = torch.load('face_mean.dat');
	nonface_mean = torch.load('nonface_mean.dat');
	face_sd      = torch.load('face_sd.dat');
	nonface_sd   = torch.load('nonface_sd.dat');
	Y_train      = torch.load('Y_train.dat');

	end_time = os.time();
	elapsed_time = os.difftime(end_time, start_time);
	print('Finished reading in data files. Total time: '..elapsed_time);

	num_imgs     = proj:size()[1]; -- num of total images (# rows of proj)
	err_mat      = torch.Tensor(num_imgs, proj:size()[2]);

	-- precompute the classifications
	start_time = os.time();
	for i = 1, proj:size()[2] do
		-- pass in i-th column of proj to classiifer
		-- class: +1 if correct class vector holds correct class., else -1
		class = classify.ll_classify(proj[{{}, {i}}],
			face_mean[i], face_sd[i], nonface_mean[i], nonface_sd[i]);
		
		-- # classified incorrectly = sum(err_indicator)
		err_indicator = torch.ne(Y_train, class); -- num_imgs x delta_size

		-- error     = indicator:sum() / num_imgs; -- STORE THIS FOR USE IN ADA
		-- print('iter '..i..' classifcation error: '..error);

		err_mat[{{}, {i}}] = err_indicator;
	end
	
	end_time = os.time();
	elapsed_time = os.difftime(end_time, start_time);
	print('Finished classifications. Total time: '..elapsed_time);


	---- BOOSTING STEP ---------------------------------------------------------

	-- initialize current weights, previous weights for images
	wts_cur  = torch.Tensor(1, num_imgs);
	wts_prev = torch.Tensor(1, num_imgs);

	wts_cur  = 1 / num_imgs;
	wts_prev = 1 / num_imgs;


	print('Begin adaboost');
	start_time = os.time();

	weighted_err = torch.Tensor(1, delta_size);
	alpha        = torch.Tensor(1, T);
	ada_index    = torch.Tensor(1, T);
	F_ip         = torch.Tensor(num_imgs, 1); -- store the inner product of all
											  -- imgs with every weak classifier

	for t = 1, T do
		iter_start = os.time();

		-- calculate weighted error
		wt_err = wts_cur * err_mat; -- 1 x delta_size, wt_error correspond
									      -- to each weak classifier

		-- find weighted classifier with min. weighted error
		min_err, min_ind = torch.min(wt_err, 2); -- 2 b/c weighted_err is
											     -- row vector

		ada_index[t] = min_ind;

		-- update wts_prev, alpha (alpha)
		wts_prev = wts_cur;
		alpha(t) = 0.5 * torch.log((1 - wt_err) / wt_err);

		-- calculate empirical error (minimize)


		------------------------ end calculations ------------------------------
		end_iter_time = os.time();
		elapsed_time = os.difftime(end_iter_time, iter_start);
		print('iter '..i.. ' -- '..elapsed_time ' seconds');
	end ------------------------ end adaboost iteration ------------------------
	
	end_time = os.time();
	elapsed_time = os.difftime(end_time, start_time);
	print('total runtime of classifications: ' .. elapsed_time .. 'seconds');
	print('End adaboost');

	---- END BOOSTING STEP -----------------------------------------------------
	
end


------- function declarations ---------------
 boost.adaboost = adaboost;
 boost.computeClassifications = computeClassifications;
--------------- end function declarations ---



return boost;