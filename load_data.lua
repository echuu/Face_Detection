local load_data = {}


local function importFaces(pathname, num_faces)

	print("Importing faces");
	faces = torch.load(pathname.."faces.dat");
	faces = faces[{{}, {1, num_faces}}];
	faces = faces:t();

	return faces;
end

local function importNonfaces(pathname, num_nonfaces)

	print("Importing nonfaces");
	nonfaces = torch.load(pathname.."nonfaces.dat");
	nonfaces = nonfaces[{{}, {1, num_nonfaces}}];
	nonfaces = nonfaces:t();

	return nonfaces;
end


---------------- function delcarations -------------------------
load_data.importFaces    = importFaces;
load_data.importNonfaces = importNonfaces;
---------------- end function declarations ---------------------

return load_data;