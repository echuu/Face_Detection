--[[ -------------- adaboost.lua
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


boost.DELTA = ext.generateWC(ext.DIM, 36480);

--print('weak classifier matrix generated: '.. boost.DELTA:size()[1]..' x '.. 
--	boost.DELTA:size()[2]);

boost.adaboost()

local function adaboost()
	-- precompute the projections
	-- precompute the classifications
	-- boost weak classifiers
	print('weak classifier matrix generated: '.. boost.DELTA:size()[1]..' x '.. 
	boost.DELTA:size()[2]);

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