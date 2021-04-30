#include "MeshModel.h"

MeshModel::MeshModel()
{
}

MeshModel::MeshModel(std::vector<Mesh> newMeshList)
{
	opaqueMeshes = newMeshList;
	model = glm::mat4(1.f);
	model = glm::scale(model, glm::vec3(1.f, 1.f, 1.f));
	preprocessModel();
}

size_t MeshModel::getMeshCount()
{
	return opaqueMeshes.size();
}

Mesh* MeshModel::getMesh(size_t index)
{
	if (index >= opaqueMeshes.size()) {
		throw std::runtime_error("Attemted to Access invalid mesh index");
	}

	return &opaqueMeshes[index];
}

glm::mat4 MeshModel::getModel()
{
	return model;
}

void MeshModel::setModel(glm::mat4 newModel)
{
	model = newModel;
}

std::vector<std::string> MeshModel::LoadMaterials(const aiScene *scene)
{
	std::vector<std::string> textureList(scene -> mNumMaterials);  //Create 1:1 sized listy of the textures
	for (size_t i = 0; i < scene->mNumMaterials; i++) {
		//Go through each material, and copy over the name of the diffuse texture file
		
		//get the material
		aiMaterial* material = scene->mMaterials[i];

		//Initialize texture to empty string, then replace name if the texture exists
		textureList[i] = ""; 
		//Check for a diffuse texture
		if (material->GetTextureCount(aiTextureType_DIFFUSE)) {

			aiString path;
			if (material->GetTexture(aiTextureType_DIFFUSE, 0, &path) == AI_SUCCESS) {
				//Cut off any directory information already present
				int idx = std::string(path.data).rfind("\\"); //rfind is reversde find, it will find the last \\ in the pathname
				std::string fileName = std::string(path.data).substr(idx + 1);  //Get the filename from the full path
				textureList[i] = fileName;
			}
		}
	
	}
	return textureList;
}

std::vector<Mesh> MeshModel::LoadNode(VkPhysicalDevice newPhysicalDevice, VkDevice newDevice, VkQueue transferQueue, VkCommandPool transferCommandPool, aiNode* node, const aiScene* scene, std::vector<int> matToTex, std::vector<GeometryPass>& geoPasses)
{
	std::vector<Mesh> meshList;
	//Go through eachmesh at this node and create it,  and then add it to the mesh list function
	for (size_t i = 0; i < node->mNumMeshes; i++) {
		//LOAD MESH HERE
		meshList.push_back(
			LoadMesh(newPhysicalDevice, newDevice, transferQueue, transferCommandPool, scene->mMeshes[node->mMeshes[i]], scene, matToTex, geoPasses)
		);
	}

	//Go through each node attached to this node and load it. then append the nodes' meshes to this nodes mesh list
	for (size_t i = 0; i < node->mNumChildren; i++) {

		std::vector<Mesh> newList = LoadNode(newPhysicalDevice, newDevice, transferQueue, transferCommandPool, node->mChildren[i], scene, matToTex, geoPasses);
		
		//Add the child nodes meshes to the meshlist
		meshList.insert(meshList.end(), newList.begin(), newList.end());
	}

	return meshList;
}

Mesh MeshModel::LoadMesh(VkPhysicalDevice newPhysicalDevice, VkDevice newDevice, VkQueue transferQueue, VkCommandPool transferCommandPool, aiMesh* mesh, const aiScene* scene, std::vector<int> matToTex, std::vector<GeometryPass>& geoPasses)
{
	std::vector<Vertex> vertices;
	std::vector<uint32_t> indices;

	vertices.resize(mesh->mNumVertices);
	//Iterate over the vertices and copy them over
	for (size_t i = 0; i < mesh->mNumVertices; i++) {
		//Set position
		vertices[i].pos = { mesh->mVertices[i].x, mesh->mVertices[i].y, mesh->mVertices[i].z};
		
		//And texture co ordinates if they exist
		if (mesh->mTextureCoords[0]) {

			vertices[i].tex = { mesh->mTextureCoords[0][i].x, mesh->mTextureCoords[0][i].y };
		}
		else {
			vertices[i].tex = { 0.f, 0.f };
		}

		//And colour
		vertices[i].col = { 1.f, 1.f, 1.f };
	}


	// Iterate over indices through faces and copy across
	for (size_t i = 0; i < mesh->mNumFaces; i++)
	{
		// Get a face
		aiFace face = mesh->mFaces[i];

		// Go through face's indices and add to list
		for (size_t j = 0; j < face.mNumIndices; j++)
		{
			indices.push_back(face.mIndices[j]);
		}
	}

	// Create new mesh with details and return it
	Mesh newMesh = Mesh(newPhysicalDevice, newDevice, transferQueue, transferCommandPool, &vertices, &indices, matToTex[mesh->mMaterialIndex], geoPasses[mesh->mMaterialIndex]);

	return newMesh;
}

void MeshModel::destroyMeshModel()
{
	for (auto mesh : opaqueMeshes) {
		mesh.destroyBuffers();
	}
	for (auto mesh: translucentMeshes) {
		mesh.destroyBuffers();
	}
}

MeshModel::~MeshModel()
{
}

void MeshModel::preprocessModel()
{
	//iterate through the meshlist and separate the translucent and opaque meshes
	std::vector<Mesh>::iterator it;
	std::vector<Mesh> tempOpaqueMeshes;
	for (it = opaqueMeshes.begin(); it < opaqueMeshes.end(); it++) {
		if (it->isTranslucent()) {
			translucentMeshes.push_back(*it);
			//opaqueMeshes.erase(it);
		}
		else {
			tempOpaqueMeshes.push_back(*it);
		}

	}
	opaqueMeshes = tempOpaqueMeshes;
}
