--[[
	 Common variable definitions used
--]]

local g = {}

g.subset_faces    = 1000;
g.subset_nonfaces = 4000;
g.dim             = 16;
g.delta_size      = 36480;
g.total_imgs      = g.subset_faces + g.subset_nonfaces;
g.pathname        = "/home/eric/data_files/";

g.csvpath         = "/home/eric/Face_Detection/data_files/";

return g;