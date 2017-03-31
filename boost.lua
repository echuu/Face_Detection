local ld    = require('load_data');
local ext   = require('externalFunctions');
local g     = require('common_defs');
local class = require('classify.lua');

h_mat   = torch.DoubleTensor(g.total_imgs, g.delta_size);
err_mat = torch.DoubleTensor(g.total_imgs, g.delta_size);

-- faces, nonfaces stored as rows
faces    = ld.importFaces(g.pathname, g.subset_faces);       --  800 x 256
nonfaces = ld.importNonfaces(g.pathname, g.subset_nonfaces); -- 3200 x 256
Y_train  = ext.createTrain(faces, nonfaces);                 -- 4000 x 1 
print("True classes: ".. Y_train:size()[1].." images");

-- generate weak classifiers
-- each weak classifiers stored as a column vector
delta = ext.generateWC(g.dim, g.delta_size);

delta_sum = delta[{{1,20},{1,20}}];
print(delta_sum);

-- calculate threshold
--[[ 
	face_mean    (delta_size x 1) -- the average value for a face per w.c.
	face_sd      (delta_size x 1) -- the average value for a nonface per w.c.
	nonface_mean (delta_size x 1)
	nonface_sd   (delta_size x 1)
	proj         (total_imgs x delta_size) -- used for classification
--]]
face_mean, face_sd, nonface_mean, nonface_sd, proj = ext.calcThreshold(delta,
	g.delta_size, faces, nonfaces);

torch.save('face_mean.dat', face_mean);
torch.save('face_sd.dat', face_sd);
torch.save('nonface_mean.dat', nonface_mean);
torch.save('nonface_sd.dat', nonface_sd);
torch.save('proj.dat', proj);

--[[ for debug purposes
	face_mean = torch.load('face_mean.dat');
	face_sd = torch.load('face_sd.dat');
	nonface_mean = torch.load('nonface_mean.dat');
	nonface_sd = torch.load('nonface_sd.dat');
	proj = torch.load('proj.dat');
--]]

--------------------- free memory ----------------------------------------------
faces    = nil;
nonfaces = nil;
delta    = nil;
--------------------- free memory ----------------------------------------------

-- precompute classifications (these don't change per iteration)
-- store errors in matrix

--

if 1 == 1 then
	start_time = os.time();
	for i = 1, g.delta_size do

		h_mat[{{}, {i}}] = class.ll_classify(proj[{{}, {i}}],
			face_mean[i], face_sd[i], nonface_mean[i], nonface_sd[i]);

		err_mat[{{}, {i}}] = torch.ne(Y_train, h_mat[{{}, {i}}]);
	end

	print("Classifications complete");
	torch.save('classification_matrix.dat', h_mat);
	torch.save('error_matrix.dat', err_mat);
	end_time = os.time();
	elapsed_time = os.difftime(end_time, start_time);
	print('Finished classifications. Total time: '..elapsed_time);
else
	h_mat   = torch.load('classification_matrix.dat');
	err_mat = torch.load('error_mat.dat');
end

-- can free up proj matrix

T = 10;
local inv_total = 1 / g.total_imgs;

F = torch.Tensor(g.total_imgs, 1):zero();	-- strong classifier
Z = 0; 										-- normalizing factor

D_cur  = torch.Tensor(g.total_imgs, 1):fill(inv_total);
D_prev = torch.Tensor(g.total_imgs, 1):fill(inv_total);

min_ada_index = torch.Tensor(T, 1):zero(); -- index of w.c. w/ lowest wt. error
alpha = torch.Tensor(T, 1):zero();

------ begin adaboost ----------------------------------------------------------
for t = 1, T do
	weighted_error = torch.Tensor(g.delta_size, 1):zero();

	error, index = class.findMinWtErr(D_cur, err_mat, g.delta_size, 0, t)
	alpha[t]     = -0.5 * torch.log((1 - error) / error);
	min_ada_index[t] = index;
	D_prev = D_cur;

	-- needs to be fixed
	chosen = min_ada_index[{},{1,t};
	chosen_class = h_mat[{{},{]}}];
	--- needs to be fixed


	F     = F + torch.squeeze(alpha[t]) * h;
	yh    = torch.exp(torch.cmul(-Y_train, F)); 
	Z     = torch.sum(yh) * inv_total;
	D_cur = 1/Z * inv_total * yh;
end
------ end adaboost ------------------------------------------------------------
print(min_ada_index);
