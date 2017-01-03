----------------------------- externalFunctions.lua
------ FUNCTIONS:
-- generateWC()
-- createWindow()
-- calcThreshold()
-------------------------------------

local M = {}

----------------------- GLOBAL VARIABLE DECLARATION -------------------------
-- feature 1
M.HAAR_1_COLS = 2;
M.HAAR_1_ROWS = 1;

-- feature 2
M.HAAR_2_COLS = 3;
M.HAAR_2_ROWS = 1;

-- feature 3
M.HAAR_3_ROWS = 2;
M.HAAR_3_COLS = 2;

-- dimension of faces
M.DIM           = 16;
M.NUM_FACES     = 11838;
M.NUM_NONFACES  = 45356;

-- negatives


---------------------- END GLOBAL VARIABLE DECLARATION -------------------------


--[[
	create training matrix   X : [ faces | nonfaces | negatives ]
	create true value vector Y : Y[i] = 1 for face, Y[i] = -1 for nonface
--]]
local function createTrain(pos, neg)

	local X, Y, total_images;

	total_images = M.NUM_FACES + M.NUM_NONFACES; -- 57194 total faces + nonfaces

	X = torch.Tensor(M.DIM * M.DIM, total_images);
	Y = torch.Tensor(total_images, 1):fill(-1);

	Y[{{1, M.NUM_FACES}}] = 1; -- faces <=> 1
	X[{{}, {1, M.NUM_FACES}}] = pos[{{}, {1, M.NUM_FACES}}];
	X[{{}, {M.NUM_FACES+1, total_images}}] = neg[{{}, {1, M.NUM_NONFACES}}];
	-- add line to incorporate hard negatives

	return X, Y

end ------------------------------------------------------- end of createTrain()

local function calcThreshold(X, delta_size, faces, nonfaces)
	-- X : 36480 x 256
	local face_mean, face_sd, nonface_mean, nonface_sd, pos, neg;

	face_mean    = torch.FloatTensor(delta_size, 1):zero();
	face_sd      = torch.FloatTensor(delta_size, 1):zero();

	nonface_mean = torch.FloatTensor(delta_size, 1):zero();
	nonface_sd   = torch.FloatTensor(delta_size, 1):zero();

	-- store result of dot product of each weak classifier of each face/nonface
	pos          = torch.Tensor(M.NUM_FACES, 1):zero();
	neg          = torch.Tensor(M.NUM_NONFACES, 1):zero();

	print('dim of delta: ' .. X:size()[1] .. ' x '.. X:size()[2]);


	start_time = os.time();
	
	pos_X = X * faces;    -- 36480 x 11838

	for i = 1, 40 do
		sum = torch.mean(pos_X[{{i},{}}]);
		print(sum);
	end

	end_time = os.time();
	elapsed_time = os.difftime(end_time, start_time);
	print('done with positives - total elapsed: ' .. elapsed_time .. 'seconds');

	-- neg_X = X * nonfaces; -- 36480 x 45356

	end_time = os.time();
	elapsed_time = os.difftime(end_time, start_time);
	-- print('done with negatives - total elapsed: ' .. elapsed_time .. 'seconds');

	


	
	return face_mean, face_sd, nonface_mean, nonface_sd;

end ------------------------------------------------ end of calculateThreshold()



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

					delta[{{}, {col}}]      = w2;
					delta[{{}, {col + 1}}]  = w1;
					--print('Generating weak classifier: '.. col);
					--print('Generating weak classifier: '.. col + 1);

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

					delta[{{}, {col}}]      = w2;
					delta[{{}, {col + 1}}]  = w1;

					--print('Generating weak classifier: '.. col);
					--print('Generating weak classifier: '.. col + 1);

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

					--print('Generating weak classifier: '.. col);
					--print('Generating weak classifier: '.. col + 1);

					col = col + 2;
				end
			end
		end
	end

	print("Generated ".. col-1 .. " weak classifiers.");

	return delta;

end ----------------------------------------------------- end of generateWC()


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

end ----------------------------------------------------- end of createWindow()

------- function declarations ---------------
M.generateWC    = generateWC;
M.createWindow  = createWindow;
M.calcThreshold = calcThreshold;
M.createTrain   = createTrain;
--------------- end function declarations ---


return M;