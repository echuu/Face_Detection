----------------------------- externalFunctions.lua
------ FUNCTIONS:
-- createTrain()
-- generateWC()
-- createWindow
-------------------------------------

local M = {}

-- feature 1
M.HAAR_1_COLS = 2;
M.HAAR_1_ROWS = 1;

-- feature 2
M.HAAR_2_COLS = 3;
M.HAAR_2_ROWS = 1;

-- feature 3
M.HAAR_3_ROWS = 2;
M.HAAR_3_COLS = 2;

local function createTrain(dim)
	print("Printing from externalFunctions.lua");

	
	window = torch.Tensor(dim, dim):zero();
	for i = 1, dim do
		window[i][i] = -999;
	end
	window_t = window + torch.eye(dim);
	

	M.test(); -- call to function within same file
	return window, window_t;
end

local function generateWC(dim, delta_size)
	local delta = torch.Tensor(dim * dim, delta_size):zero();

	local xSize, ySize, enlarge, step;

	-- create type 1 features
	enlarge = 1;
	step    = 1;
	xSize   = enlarge * M.HAAR_1_COLS;
	ySize   = M.HAAR_1_ROWS;

	col     = 1;
	for i = 0, dim, step do
		for j = 0, dim, step do
			for height = xSize, dim - i, 2 do
				for width = ySize, dim - j do

					w1, w2 = M.createWindow(i, j, height, width, dim, 1);

					delta[{{}, {col}}]      = w1;
					delta[{{}, {col + 1}}]  = w2;
					print('Generating weak classifier: '.. col);
					print('Generating weak classifier: '.. col + 1);

					col = col + 2;
				end
			end
		end
	end

	-- create type 2 features
	xSize   = enlarge * M.HAAR_2_COLS;
	ySize   = M.HAAR_2_ROWS;

	for i = 0, dim, step do
		for j = 0, dim, step do
			for height = xSize, dim - i, xSize do
				for width = ySize, dim - j, ySize do

					w1, w2 = M.createWindow(i, j, height, width, dim, 2);

					delta[{{}, {col}}]      = w1;
					delta[{{}, {col + 1}}]  = w2;

					print('Generating weak classifier: '.. col);
					print('Generating weak classifier: '.. col + 1);

					col = col + 2;
				end
			end
		end
	end

	-- create type 3 features
	xSize   = enlarge * M.HAAR_3_COLS;
	ySize   = M.HAAR_3_ROWS;
	for i = 0, dim, step do
		for j = 0, dim, step do
			for height = xSize, dim - i, xSize do
				for width = ySize, dim - j, ySize do

					w1, w2 = M.createWindow(i, j, height, width, dim, 3);

					delta[{{}, {col}}]      = w1;
					delta[{{}, {col + 1}}]  = w2;

					print('Generating weak classifier: '.. col);
					print('Generating weak classifier: '.. col + 1);

					col = col + 2;
				end
			end
		end
	end

	print("Generated ".. col-1 .. " weak classifiers.");

end ----------------------------------------------------  - end of createTrain()


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

	w1_col     = torch.Tensor(dim * dim, 1):zero();
	w2_col     = torch.Tensor(dim * dim, 1):zero();

	if type == 1 then
		--print("Creating 2-rectangle feature");
		for h = 1, height do
			for w = 1, width do
				if h <= height / 2 then
					window[pos_i + h][pos_j + w] = 1;          --     left rect.
				else
					window[pos_i + h][pos_j + w] = -1;         --    right rect.
				end
			end
		end 
		w1_col     = torch.reshape(window, dim * dim, 1);
		w2_col     = torch.reshape(window:t(), dim * dim, 1); ------- end type 1
	elseif type == 2 then
		--print("Creating 3-rectangle feature");
		for h = 1, height do
			for w = 1, width do
				if h > height / 3 and h <= 2 * height / 3 then  --    mid. rect.
					window[pos_i + h][pos_j + w] = 1;
				else
					window[pos_i + h][pos_j + w] = -1;          --    out. rect.
				end
			end
		end  
		w1_col     = torch.reshape(window, dim * dim, 1);
		w2_col     = torch.reshape(window:t(), dim * dim, 1); ------- end type 2
	elseif type == 3 then
		--print('Creating diagonal feature');
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
		w1_col     = torch.reshape(window, dim * dim, 1);
		w2_col     = torch.reshape(-window, dim * dim, 1); ---------- end type 3
	else
		error('invalid feature type');
	end

	return w1_col, w2_col;

end

------- function declarations ---------------
M.createTrain   = createTrain;
M.test          = test;
M.generateWC    = generateWC;
M.createWindow  = createWindow;
--------------- end function declarations ---


return M;