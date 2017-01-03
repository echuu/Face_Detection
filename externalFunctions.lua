------ externalFunctions.lua
-- createTrain()
--
-------------------------------------

local M = {}


local function createTrain()
	print("Printing from externalFunctions.lua");
end


local function createWindow(pos_i, pos_j, h, w, dim, type)
	--[[    create the haar-feature window
			types:
				(1) : 2-rectangle
				(2) : 3-rectangle
				(3) : diagonal
	--]] 
	window = torch.Tensor(dim, dim):zero();
	if type == 1 then
		print("Creating 2-rectangle feature");
	elseif type == 2 then
		print("Creating 3-rectangle feature");
	elseif type == 3 then
		print('Creating diagonal feature');
	else
		error('invalid feature type');
	end


end



M.createTrain = createTrain;

return M;