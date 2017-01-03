------ externalFunctions.lua
-- createTrain()
--
-------------------------------------

local M = {}


local function createTrain(dim)
	print("Printing from externalFunctions.lua");

	
	window = torch.Tensor(dim, dim):zero();
	for i = 1, dim do
		window[i][i] = -999;
	end
	window_t = window + torch.eye(16);
	

	M.test(); -- call to function within same file
	return window, window_t;
end

local function test()
	print('printing from test function inside externalFunctions');
end


local function createWindow(pos_i, pos_j, height, width, dim, type)
	--[[    create the haar-feature window
			types:
				(1) : 2-rectangle
				(2) : 3-rectangle
				(3) : diagonal
	--]] 
	window     = torch.Tensor(dim, dim):zero();
	window_rot = torch.Tensor(dim, dim):zero();
	if type == 1 then
		print("Creating 2-rectangle feature");
		for h = 1, height do
			for w = 1, width do
				if h <= height / 2 then
					window[pos_i + h][pos_j + w] = 1;          --     left rect.
				else
					window[pos_i + h][pos_j + w] = -1;         --    right rect.
				end
			end
		end 
		window_rot = window:t();                      --------------- end type 1
	elseif type == 2 then
		print("Creating 3-rectangle feature");
		for h = 1, height do
			for w = 1, width do
				if h > height / 3 and h <= 2 * height / 3 then  --    mid. rect.
					window[pos_i + h][pos_j + w] = 1;
				else
					window[pos_i + h][pos_j + w] = -1;          --    out. rect.
				end
			end
		end  
		window_rot = window:t();                      --------------- end type 2
	elseif type == 3 then
		print('Creating diagonal feature');
		for h = 1, height do
			for w = 1, width do
				if h <= height / 2 and w <= width / 2 then      --  up-left rect
					window[pos_i + h][pos_j + w] = 1;
				elseif h > height / 2 and w > width / 2 then    -- lo-right rect
					window[pos_i + h][pos_j + w] = 1;
				else
					window[pos_i + h][pos_j + w] = -1;
				end
			end
		end  
		window_rot = -window;                         --------------- end type 3
	else
		error('invalid feature type');
	end

	window_t = window + torch.eye(16);
	return window, window_rot;

end

M.createTrain = createTrain;
M.test = test;

return M;