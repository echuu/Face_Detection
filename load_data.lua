local load_data = {}

local function importFaces(pathname, num_faces, csv)

	local csv2tensor = require('csv2tensor');

	print("Importing faces");
	if csv == 0 then
		faces = torch.load(pathname.."faces.dat");
	else
		faces = csv2tensor.load(pathname.."faces.csv");
		--torch.save('faces.dat', faces);
	end
	faces = faces[{{}, {1, num_faces}}];
	faces = faces:t();

	return faces;
end

local function importNonfaces(pathname, num_nonfaces, csv)

	local csv2tensor = require('csv2tensor');

	print("Importing nonfaces");
	if csv == 0 then
		nonfaces = torch.load(pathname.."nonfaces.dat");
	else
		nonfaces = csv2tensor.load(pathname.."nonfaces.csv");
		--torch.save('nonfaces.dat', nonfaces);
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