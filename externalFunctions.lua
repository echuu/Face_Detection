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
--M.NUM_FACES     = 1000;
--M.NUM_NONFACES  = 4000;

-- negatives


-- debug
M.DEBUG = 0;


---------------------- END GLOBAL VARIABLE DECLARATION -------------------------


--[[
	create training matrix   X : [ faces | nonfaces | negatives ]
	create true value vector Y : Y[i] = 1 for face, Y[i] = -1 for nonface
--]]
local function createTrain(pos, neg)

	local Y, total_images;

	total_images = pos:size()[1] + neg:size()[1];-- 57194 total faces + nonfaces

	-- X = torch.Tensor(M.DIM * M.DIM, total_images);
	Y = torch.Tensor(total_images, 1):fill(-1);

	Y[{{1, pos:size()[1]}}] = 1; -- faces <=> 1
	-- add line to incorporate hard negatives

	X = torch.cat(pos, neg, 1);


	return X, Y;

end ------------------------------------------------------- end of createTrain()

local function calcThreshold(delta, delta_size, faces, nonfaces)
	-- delta : 36480 x 256
	start_time = os.time();

	local face_mean, face_sd, nonface_mean, nonface_sd, pos, neg;

	face_mean    = torch.FloatTensor(delta_size, 1):zero();
	face_sd      = torch.FloatTensor(delta_size, 1):zero();

	nonface_mean = torch.FloatTensor(delta_size, 1):zero();
	nonface_sd   = torch.FloatTensor(delta_size, 1):zero();

	-- store result of dot product of each weak classifier of each face/nonface
	pos          = torch.Tensor(faces:size()[1], 1):zero();
	neg          = torch.Tensor(nonfaces:size()[1], 1):zero();

	-- print('dim of delta: ' .. delta:size()[1] .. ' x '.. delta:size()[2]);


	start_time = os.time();

	-- project each face onto every weak classifier	
	pos_X     = faces * delta;    -- 11838 x 36480 (rows <=> faces)
	faces = nil;

	face_mean = torch.mean(pos_X, 1):t();
	face_sd   = torch.std(pos_X, 1):t();

	if M.DEBUG == 1 then
		print('displaying first 50 values of face_mean:');
		print(face_mean[{{1,50}}]);
		end_time = os.time();
		elapsed_time = os.difftime(end_time, start_time);
		print('positives done - total elapsed: ' .. elapsed_time .. ' seconds');
	end

	neg_X         = nonfaces * delta; -- 45356 x 36480
	nonfaces = nil;

	nonface_mean  = torch.mean(neg_X, 1):t();
	nonface_sd   = torch.std(neg_X, 1):t();

	if M.DEBUG == 1 then
		print('displaying first 50 values of nonface_mean:');
		print(nonface_mean[{{1,50}}]);
		end_time = os.time();
		elapsed_time = os.difftime(end_time, start_time);
		print('negatives done - total elapsed: ' .. elapsed_time .. ' seconds');
	end

	

	--- store delta * faces, delta * nonfaces, this is used in training step
	posX = nil;
	negX = nil;

	end_time = os.time();
	elapsed_time = os.difftime(end_time, start_time);
	print('threshold calculation runtime: ' .. elapsed_time .. ' seconds');

	return torch.squeeze(face_mean), torch.squeeze(face_sd),
	 		torch.squeeze(nonface_mean), torch.squeeze(nonface_sd);

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