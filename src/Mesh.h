#pragma once
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <vector>
#include "utilities.h"

struct Model {
	glm::mat4 model;
};

enum class GeometryPass { TRANSLUCENCY_PASS, OPAQUE_PASS, DUAL_PASS };

class Mesh
{
public:
	Mesh();
	Mesh(VkPhysicalDevice newPhysicalDevice, VkDevice newDevice,
		VkQueue transferQueue, VkCommandPool transferCommandPool,
		std::vector<Vertex> *vertices, std::vector<uint32_t>* indices, int newTexId, GeometryPass& geoPass);

	~Mesh();

	int getVertexCount();
	int getIndexCount();
	VkBuffer getVertexBuffer();
	VkBuffer getIndexBuffer();
	void destroyBuffers();

	void setModel(glm::mat4 newModel);
	int getTexId();
	Model getModel();
	bool isTranslucent();

private:
	//Model data
	Model model;

	int texId;
	GeometryPass geometryPass;

	//Vertex data
	int vertexCount;
	VkBuffer vertexBuffer;
	VkDeviceMemory vertexBufferMemory;

	//Index data
	int indexCount;
	VkBuffer indexBuffer;
	VkDeviceMemory indexBufferMemory;

	VkPhysicalDevice physicalDevice;
	VkDevice device;

	void createVertexBuffer(std::vector<Vertex>* vertices, VkQueue transferQueue, VkCommandPool transferCommandPool);
	void createIndexBuffer(std::vector<uint32_t>* indices, VkQueue transferQueue, VkCommandPool transferCommandPool);
};
