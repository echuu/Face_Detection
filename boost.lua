local ld    = require('load_data');
local ext   = require('externalFunctions');
local g     = require('common_defs');
local class = require('classify.lua');
local calc  = require('calculate.lua');
csv2tensor  = require('csv2tensor');

-- global constants-------------------------------------------------------------
FIRST_TIME = 1;
DEBUG      = 1;
-- global constants-------------------------------------------------------------

h_mat   = torch.DoubleTensor(g.total_imgs, g.delta_size);
err_mat = torch.DoubleTensor(g.total_imgs, g.delta_size);

-- load in faces, nonfaces (faces, nonfaces stored as rows)---------------------
faces       = ld.importFaces(g.csvpath, g.subset_faces, 1);       --  800 x 256
nonfaces    = ld.importNonfaces(g.csvpath, g.subset_nonfaces, 1); -- 3200 x 256
X, Y_train  = ext.createTrain(faces, nonfaces);                    -- 4000 x   1
--------------------------------------------------------------------------------

if DEBUG == 1 then
	--print(faces:size());     -- num_faces    x 256
	--print(nonfaces:size());  -- num_nonfaces x 256
	--print(Y_train:size());   -- total_imgs   x 256
end

-- generate weak classifiers
-- each weak classifiers stored as a column vector
delta = ext.generateWC(g.dim, g.delta_size);

-- calculate threshold
--[[ 
	face_mean    (delta_size x 1) -- the average value for a face per w.c.
	face_sd      (delta_size x 1) -- the average value for a nonface per w.c.
	nonface_mean (delta_size x 1)
	nonface_sd   (delta_size x 1)
	proj         (total_imgs x delta_size) -- used for classification
--]]

if FIRST_TIME == 1 then 
	face_mean, face_sd, nonface_mean, nonface_sd = ext.calcThreshold(delta,
		g.delta_size, faces, nonfaces);

	torch.save('face_mean.dat', face_mean);
	torch.save('face_sd.dat', face_sd);
	torch.save('nonface_mean.dat', nonface_mean);
	torch.save('nonface_sd.dat', nonface_sd);
else
	face_mean    = torch.load('face_mean.dat');
	face_sd      = torch.load('face_sd.dat');
	nonface_mean = torch.load('nonface_mean.dat');
	nonface_sd   = torch.load('nonface_sd.dat');
end

-- precompute projections for classification
proj = X * delta;
--torch.save('proj.dat', proj);
--proj = torch.load('proj.dat');


--------------------- free memory ----------------------------------------------
faces    = nil;
nonfaces = nil;
delta    = nil;
--------------------- free memory ----------------------------------------------

-- precompute classifications (these don't change per iteration)
-- store errors in matrix

--
start_time = os.time();
if FIRST_TIME == 1 then
	--start_time = os.time();
	for i = 1, g.delta_size do

		h_mat[{{}, {i}}] = class.ll_classify(proj[{{}, {i}}],
			face_mean[i], face_sd[i], nonface_mean[i], nonface_sd[i]);

		err_mat[{{}, {i}}] = torch.ne(Y_train, h_mat[{{}, {i}}]);
	end

	print("Classifications complete");

	-- these take really long to save
	--torch.save('classification_matrix.dat', h_mat);
	--torch.save('error_matrix.dat', err_mat);

	--end_time = os.time();
	--elapsed_time = os.difftime(end_time, start_time);
	--print('Finished classifications. Total time: '..elapsed_time);
else
	print("Reading in classification matrix and error matrix");
	h_mat   = torch.load('classification_matrix.dat');
	err_mat = torch.load('error_matrix.dat');

	--h_mat   = csv2tensor.load('h_mat.csv');
	--err_mat = csv2tensor.load('err_mat.csv');
end

-- can free up proj matrix

T = 20;
local inv_total = 1 / g.total_imgs;

F = torch.Tensor(g.total_imgs, 1):zero();	-- strong classifier
Z = 0; 										-- normalizing factor

D_cur  = torch.Tensor(g.total_imgs, 1):fill(inv_total);
D_prev = torch.Tensor(g.total_imgs, 1):fill(inv_total);

min_ada_index = torch.Tensor(T, 1):zero(); -- index of w.c. w/ lowest wt. error
alpha = torch.Tensor(T, 1):zero();

------ begin adaboost ----------------------------------------------------------
for t = 1, T do
	-- weighted_error = torch.Tensor(g.delta_size, 1):zero();

	error, index = class.findMinWtErr(D_cur, err_mat, g.delta_size, 0, t);
	print('wk class: '.. index..' error: '..error);

	alpha[t]     = 0.5 * torch.log((1 - error) / error);
	--print(torch.squeeze(alpha[t]));

	min_ada_index[t] = index;

	D_prev = D_cur;

	h = h_mat[{{},{index}}];

	F     = F + torch.squeeze(alpha[t]) * h;

	yh    = torch.exp(torch.cmul(-Y_train, F)); 

	Z     = torch.sum(yh) * inv_total;

	D_cur = 1/Z * inv_total * yh;


	-- calculate empirical error:
	calc.classError(Y_train, F);
end


end_time = os.time();
elapsed_time = os.difftime(end_time, start_time);
print('Finished running. Total time: '..elapsed_time);

------ end adaboost ------------------------------------------------------------
--print(min_ada_index);
