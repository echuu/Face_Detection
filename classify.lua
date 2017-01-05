---------------- classify.lua
--[[
		classification method to determine face/non-face

		functions:
					kmeans()
	
--]]

local classify{}


local function ll_classify(proj_i, m0, s0, m1, s1)
	--[[ 
		element-wise intuition: 
			- each element in proj_i is the projection of the j-th image 
			  onto the i-th weak classifier, so each element corresponds to an
			  image (face, nonface)
			- for each of the j projections, we compare to the mean/sd of the
			  face/nonfaces calculated in calcThreshold(), and do kmeans class.
			- if the ratio > 0, then classify as face, else classify as nonface
	--]]

	-- center the projections w.r.t. faces, nonfaces

	cent_faces     = torch.pow((proj_i - m0), 2) / s0^2;
	cent_nonfaces  = torch.pow((proj_i - m1), 2) / s1^2;


	-- calculate ratios, take sign to classify
	ratio = -1/2 * (cent_faces - cent_nonfaces + torch.log(s0) + torch.log(s1));
	
	-- caculate indicator matrix to find class. error
	class              = torch.gt(ratio, 0); -- positive class.
	class[class:eq(0)] = -1;                 -- negative class.



end

--------function declarations -------------------------
classify.ll_classify = ll_classify;
------------------- end function declarations ---------


return classify



