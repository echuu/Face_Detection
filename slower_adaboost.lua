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
local classify = require "classify"; -- access ll_classify() function
local calc = require "calculate";

--[[ ----- uncomment if want these to be global

	boost.DELTA        = torch.load('delta.dat');
	boost.proj         = torch.load('projections.dat');     -- bottleneck here
	boost.face_mean    = torch.load('face_mean.dat');
	boost.nonface_mean = torch.load('nonface_mean.dat');
	boost.face_sd      = torch.load('face_sd.dat');
	boost.nonface_sd   = torch.load('nonface_sd.dat');
	boost.Y_train      = torch.load('Y_train.dat');

--]]

local function adaboost(proj, face_mean, nonface_mean, 
	face_sd, nonface_sd, Y_train, T)
	-- precompute the projections

	debug = 0;
	start_time = os.time();

	

	--delta        = torch.load(pathname..'delta.dat');
	if debug == 1 then
		print('Start reading in data files.');

		pathname = "/home/wayne/Desktop/data_files/";
		proj         = torch.load(pathname..'projections.dat');
		face_mean    = torch.load(pathname..'face_mean.dat');
		nonface_mean = torch.load(pathname..'nonface_mean.dat');
		face_sd      = torch.load(pathname..'face_sd.dat');
		nonface_sd   = torch.load(pathname..'nonface_sd.dat');
		Y_train      = torch.load(pathname..'Y_train.dat');

		end_time = os.time();
		elapsed_time = os.difftime(end_time, start_time);
		print('Finished reading in data files. Total time: '..elapsed_time);
	end


	--print('num rows in projection matrix: '..proj:size()[1]);
	--print('num cols in projection matrix: '..proj:size()[2]);

	total_imgs  = proj:size()[1]; -- num of total images (# rows of proj)
	delta_size  = proj:size()[2];
	--err_mat     = torch.Tensor(total_imgs, delta_size);
	-- total_imgs x delta_size

	-- precompute the classifications
	--start_time = os.time();
	
	
	--end_time = os.time();
	--elapsed_time = os.difftime(end_time, start_time);
	--print('Finished classifications. Total time: '..elapsed_time);


	---- BOOSTING STEP ---------------------------------------------------------

	-- initialize current weights, previous weights for images
	wts_cur  = torch.Tensor(1, total_imgs):fill(1 / total_imgs);
	wts_prev = torch.Tensor(1, total_imgs):fill(1 / total_imgs);

	print('Begin adaboost');
	start_time = os.time();

	wt_err       = torch.Tensor(1, delta_size);
	alpha        = torch.Tensor(1, T);
	ada_index    = torch.Tensor(1, T);
	

	-- below calculations can use the precomputed projections, just need
	-- to multiply by alpha[t]
	F_T          = torch.Tensor(total_imgs, T); -- weighted dot products 
	--H_T          = torch.Tensor(T, total_imgs); -- (weighted) classifications
	--Z_T          = torch.Tensor(T, 1);        -- normalizing function
	Err_T        = torch.Tensor(T, 1);        -- empirical error


	--[[ free memory
	face_mean		= nil;
	nonface_mean	= nil;
	face_sd			= nil;
	nonface_sd		= nil;
	--]]

	for t = 1, T do
		start_time = os.time();

		for i = 1, delta_size do
			-- pass in i-th column of proj to classiifer
			-- class: +1 if correct class vector holds correct class., else -1
			class = classify.ll_classify(proj[{{}, {i}}],
				torch.squeeze(face_mean[i]), torch.squeeze(face_sd[i]), 
				torch.squeeze(nonface_mean[i]), torch.squeeze(nonface_sd[i]));

			--print('size of class : '..class:size()[1]);
			--print('size of Y_train : '..Y_train:size()[1]);
			
			-- # classified incorrectly = sum(err_indicator)
			err_indicator = torch.ne(Y_train, class:double()):double(); -- total_imgs x 1
			--print('num rows in indicator: '..err_indicator:size()[1]);
			--err_mat[{{}, {i}}] = err_indicator;
			wt_err[{{},{i}}] = wts_cur * err_indicator;
		end

		--print('iter: '..t.. ' -- done calculating weighted error');

		-- calculate weighted error
		--wt_err = wts_cur * err_mat; -- 1 x delta_size, wt_error correspond
									      -- to each weak classifier

		-- find weighted classifier with min. weighted error
		min_err, min_ind = torch.min(wt_err, 2); -- 2 b/c weighted_err is
											     -- row vector

        min_err = torch.squeeze(min_err);
        min_ind = torch.squeeze(min_ind);
		--print(min_ind);
		--print(min_err);									     
		print('Weak Classifier: '..min_ind..
			' chosen to minimize weighted error');									     

		ada_index[{{},{t}}] = min_ind;

		-- update wts_prev, alpha
		wts_prev = wts_cur;
		alpha[{{},{t}}] = 0.5 * torch.log((1 - min_err) / min_err);

		-- calculate empirical error (minimize)
		proj_i = proj[{{},{min_ind}}];
		Err_T[t], F_T[{{},{t}}] = calc.getEmpiricalError(Y_train, proj_i,
			alpha[{{},{t}}], F_T, t);

		-- call update weight function for wts_cur
		wts_cur = calc.updateWeights(Y_train, F_T[{{},{t}}]):t();

		-- display empirical error for this iteration
		calc.displayErrorTime(t, Err_T[t], start_time);

	end ------------------------ end adaboost iteration ------------------------
	
	end_time = os.time();
	elapsed_time = os.difftime(end_time, start_time);
	print('total runtime of classifications: ' .. elapsed_time .. 'seconds');
	print('End adaboost');

	---- END BOOSTING STEP -----------------------------------------------------
	
end


------- function declarations ---------------
 boost.adaboost = adaboost;
--------------- end function declarations ---

return boost;
