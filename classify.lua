---------------- classify.lua
--[[
		classification method to determine face/non-face

		functions:
					kmeans()
	
--]]

local classify = {}


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
	--[[
		total_imgs = proj_i:size()[1];
		m0_vec = torch.Tensor(total_imgs, 1):fill(m0);
		--s0_vec = torch.Tensor(total_imgs, 1):fill(s0);
		m1_vec = torch.Tensor(total_imgs, 1):fill(m1);
		--s1_vec = torch.Tensor(total_imgs, 1):fill(s1);
	--]]

	--print('ll-classify: size of projection: '..proj_i:size()[1]);

	cent_faces     = torch.pow((proj_i - m0), 2) / s0^2;
	cent_nonfaces  = torch.pow((proj_i - m1), 2) / s1^2;


	--print('size of centered face: '..cent_faces:size()[1]);
	--print('size of centered nonface: '..cent_nonfaces:size()[1]);

	-- calculate ratios, take sign to classify
	ratio = -0.5 * (cent_faces - cent_nonfaces + torch.log(s0) - torch.log(s1));
	
	--print('size of ratio: '..ratio:size()[1]);

	-- caculate indicator matrix to find class. error
	class = torch.gt(ratio, 0):double(); -- positive class.
	class[class:eq(0)] = -1;             -- negative class.

	return class:double();

end

--------function declarations -------------------------
classify.ll_classify = ll_classify;
------------------- end function declarations ---------


return classify



