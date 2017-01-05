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

	-- center the projections w.r.t. faces, nonfaces
	-- cent_faces     = proj -   
	-- cent_nonfaces  = 

	-- precompute the classifications
		-- make call to classify.kmeans()
	
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