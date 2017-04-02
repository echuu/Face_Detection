local load_data = {}

csv2tensor = require('csv2tensor');

local function importFaces(pathname, num_faces, csv)

	print("Importing faces");
	if csv == 0 then
		faces = torch.load(pathname.."faces.dat");
	else
		faces = csv2tensor.load(pathname.."faces.csv");
	end
	faces = faces[{{}, {1, num_faces}}];
	faces = faces:t();

	return faces;
end

local function importNonfaces(pathname, num_nonfaces, csv)

	print("Importing nonfaces");
	if csv == 0 then
		nonfaces = torch.load(pathname.."nonfaces.dat");
	else
		nonfaces = csv2tensor.load(pathname.."nonfaces.csv");
	end
	nonfaces = nonfaces[{{}, {1, num_nonfaces}}];
	nonfaces = nonfaces:t();

	return nonfaces;
end


---------------- function delcarations -------------------------
load_data.importFaces    = importFaces;
load_data.importNonfaces = importNonfaces;
---------------- end function declarations ---------------------

return load_data;