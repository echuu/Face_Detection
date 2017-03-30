local ld  = require('load_data');
local ext = require('externalFunctions');
local g   = require('common_defs')

-- faces, nonfaces stored as rows
faces    = ld.importFaces(g.pathname, g.subset_faces);       --  800 x 256
nonfaces = ld.importNonfaces(g.pathname, g.subset_nonfaces); -- 3200 x 256

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

