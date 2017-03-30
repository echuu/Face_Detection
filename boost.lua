local ld  = require('load_data');
local ext = require('externalFunctions');

local subset_faces    = 800;
local subset_nonfaces = 3200;
local dim             = 16;
local delta_size      = 36480;
local total_imgs      = subset_faces + subset_nonfaces;

pathname = "/home/eric/data_files/";

-- faces, nonfaces stored as rows
faces    = ld.importFaces(pathname, subset_faces);       --  800 x 256
nonfaces = ld.importNonfaces(pathname, subset_nonfaces); -- 3200 x 256

-- generate weak classifiers
-- each weak classifiers stored as a column vector
delta = ext.generateWC(dim, delta_size);


-- calculate threshold
--[[ 
	face_mean    (delta_size x 1) -- the average value for a face per w.c.
	face_sd      (delta_size x 1) -- the average value for a nonface per w.c.
	nonface_mean (delta_size x 1)
	nonface_sd   (delta_size x 1)
	proj         (total_imgs x delta_size) -- used for classification
--]]
face_mean, face_sd, nonface_mean, nonface_sd, proj = ext.calcThreshold(delta,
	delta_size, faces, nonfaces);

