#pragma once

#include <vector>
#include <glm/glm.hpp>
#include <glm//gtc/matrix_transform.hpp>
#include <assimp/scene.h>
#include "Mesh.h"
#include <iostream>

class MeshModel
{
public:
	MeshModel();
	MeshModel(std::vector<Mesh> newMeshList);

	size_t getOpaqueMeshCount();
	Mesh* getOpaqueMesh(size_t index);

	size_t getTranslucentMeshCount();
	Mesh* getTranslucentMesh(size_t index);

	glm::mat4 getModel();  //Talking about the model matrix here
	void setModel(glm::mat4 newModel);

	static std::vector<std::string> LoadMaterials(const aiScene* scene);   //Static function --> only used outside the models
	static std::vector<Mesh> LoadNode(VkPhysicalDevice newPhysicalDevice, VkDevice newDevice, VkQueue transferQueue, VkCommandPool transferCommandPool, aiNode* node, const aiScene* scene, std::vector<int> matToTex, std::vector<GeometryPass>& geoPasses);
	static Mesh LoadMesh(VkPhysicalDevice newPhysicalDevice, VkDevice newDevice, VkQueue transferQueue, VkCommandPool transferCommandPool, aiMesh* mesh, const aiScene* scene, std::vector<int> matToTex, std::vector<GeometryPass>& geoPasses);

	void destroyMeshModel();

	~MeshModel();
	glm::mat4 model;

private:
	std::vector<Mesh> opaqueMeshes;
	std::vector<Mesh> translucentMeshes;
	void preprocessModel();
};
