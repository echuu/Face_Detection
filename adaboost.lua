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

local function adaboost()
	-- precompute the projections


	delta        = torch.load('delta.dat');
	proj         = torch.load('projections.dat');
	face_mean    = torch.load('face_mean.dat');
	nonface_mean = torch.load('nonface_mean.dat');
	face_sd      = torch.load('face_sd.dat');
	nonface_sd   = torch.load('nonface_sd.dat');
	Y_train      = torch.load('Y_train.dat');

	-- precompute the classifications
		-- make call to classify.kmeans()
	print('Begin classification');
	start_time = os.time();

	weighted_err = torch.Tensor(delta_size, 1);
	for i = 1, proj:size()[2] do
		-- pass in i-th column of proj to classiifer
		class = classify.ll_classify(proj[{{}, {i}}],
			face_mean[i], face_sd[i], nonface_mean[i], nonface_sd[i]);
		
		-- indicator: +1 if correct class vector holds correct class., else -1
		indicator = torch.eq(Y_train, class);
		
		-- store weighted error
		-- sorted ratio

	end


	end_time = os.time();
	elapsed_time = os.difftime(end_time, start_time);
	print('total runtime of classifications: ' .. elapsed_time .. 'seconds');

	
	
	-- boost weak classifiers


	print('in adaboost');

end

local function computeClassifications()

	print('in computeClassifications');

end


------- function declarations ---------------
 boost.adaboost = adaboost;
 boost.computeClassifications = computeClassifications;
--------------- end function declarations ---



return boost;