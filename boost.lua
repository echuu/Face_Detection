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

for i = 1, 50 do

	print('face mean '.. i.. ': '..face_mean[i]);
end


--------------------- free memory ----------------------------------------------
faces    = nil;
nonfaces = nil;
delta    = nil;
--------------------- free memory ----------------------------------------------

-- precompute classifications (these don't change per iteration)
-- store errors in matrix

--[[
start_time = os.time();
for i = 1, g.delta_size do

	h_mat[{{}, {i}}] = class.ll_classify(proj[{{}, {i}}],
		face_mean[i], face_sd[i], nonface_mean[i], nonface_sd[i]);

	err_mat[{{}, {i}}] = torch.ne(Y_train, h_mat[{{}, {i}}]);
end

subset_errors = err_mat[{{1, 1000}, {1, 1000}}];
torch.save('error_matrix.dat', subset_errors);


print("Classifications complete");
end_time = os.time();
elapsed_time = os.difftime(end_time, start_time);
print('Finished classifications. Total time: '..elapsed_time);


--]]

