#include "Mesh.h"

Mesh::Mesh()
{
}

Mesh::Mesh(VkPhysicalDevice newPhysicalDevice, VkDevice newDevice, VkQueue transferQueue, VkCommandPool transferCommandPool, std::vector<Vertex>* vertices, std::vector<uint32_t>* indices, int newTexId, GeometryPass& geoPass)
{
	vertexCount = vertices->size();
	indexCount = indices->size();
	physicalDevice = newPhysicalDevice;
	device = newDevice;
	model.model = glm::mat4(1.0f);
	createVertexBuffer(vertices, transferQueue, transferCommandPool);
	createIndexBuffer(indices, transferQueue, transferCommandPool);
	texId = newTexId;
	geometryPass = geoPass;
}

Mesh::~Mesh()
{
}

int Mesh::getVertexCount()
{
	return vertexCount;
}

int Mesh::getIndexCount()
{
	return indexCount;
}

VkBuffer Mesh::getVertexBuffer()
{
	return vertexBuffer;
}

VkBuffer Mesh::getIndexBuffer()
{
	return indexBuffer;
}

void Mesh::destroyBuffers()
{
	vkDestroyBuffer(device, vertexBuffer, nullptr);
	vkFreeMemory(device, vertexBufferMemory, nullptr);
	vkDestroyBuffer(device, indexBuffer, nullptr);
	vkFreeMemory(device, indexBufferMemory, nullptr);
}

void Mesh::setModel(glm::mat4 newModel)
{
	model.model = newModel;
}

int Mesh::getTexId()
{
	return texId;
}

Model Mesh::getModel()
{
	return model;
}

bool Mesh::isTranslucent()
{
	if (geometryPass == GeometryPass::TRANSLUCENCY_PASS) {
		return true;

	}
	return false;
}

void Mesh::createVertexBuffer(std::vector<Vertex>* vertices, VkQueue transferQueue, VkCommandPool transferCommandPool)
{
	//Get size of vertex buffer to create
	VkDeviceSize bufferSize = sizeof(Vertex) * vertices->size();

	//We create a tempoarary buffer to stage vertex data before transferring it to an optimal location in the GPU (Staging buffers)
	VkBuffer stagingBuffer;
	VkDeviceMemory stagingBufferMemory;

	//Create staging buffer and allocate memory to it		
	createBuffer(physicalDevice, device, bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &stagingBuffer, &stagingBufferMemory);

	//Now map memory to vertex buffer
	void* data;  //Pointer to data in host memory
	vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data); //Now we mapped the memory to the pointer
	memcpy(data, vertices->data(), (size_t)bufferSize);  //We copy the vertex data to out host ppointer, the map ensures that thesame is copied to the vertexBufferMemory
	//Now we have the vertex data in vertexBufferMemory so we just unmap
	vkUnmapMemory(device, stagingBufferMemory);
	//If we altered the memory on the host now, the vertexBufferMemory will not be affeted, because we already unmapped

	//Now we have the vertex data on the Staging buffer, we need to transfer it to a memory type tats more optimal for the vertex data
	//Now we create a vertex buffer with the transfer dst bit to mark as a recipient for the data from the staging buffer
	//Pretty cool that we can have combinations of usage types for a buffer eh
	createBuffer(physicalDevice, device, bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &vertexBuffer, &vertexBufferMemory);

	//Now copy over the vertex data to the vertex buffer
	copyBuffer(device, transferQueue, transferCommandPool, stagingBuffer, vertexBuffer, bufferSize);

	//Now that the vertex buffer is in memory of type device local bit, we have optimal usage of memory 
	//Now we can clean up the staging buffer
	vkDestroyBuffer(device, stagingBuffer, nullptr);
	vkFreeMemory(device, stagingBufferMemory, nullptr);
}

void Mesh::createIndexBuffer(std::vector<uint32_t>* indices, VkQueue transferQueue, VkCommandPool transferCommandPool)
{
	// Get size of buffer needed for indices
	VkDeviceSize bufferSize = sizeof(uint32_t) * indices->size();

	// Temporary buffer to "stage" index data before transferring to GPU
	VkBuffer stagingBuffer;
	VkDeviceMemory stagingBufferMemory;
	createBuffer(physicalDevice, device, bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
		VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &stagingBuffer, &stagingBufferMemory);

	// MAP MEMORY TO INDEX BUFFER
	void* data;
	vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
	memcpy(data, indices->data(), (size_t)bufferSize);
	vkUnmapMemory(device, stagingBufferMemory);

	// Create buffer for INDEX data on GPU access only area
	createBuffer(physicalDevice, device, bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
		VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &indexBuffer, &indexBufferMemory);

	// Copy from staging buffer to GPU access buffer
	copyBuffer(device, transferQueue, transferCommandPool, stagingBuffer, indexBuffer, bufferSize);

	// Destroy + Release Staging Buffer resources
	vkDestroyBuffer(device, stagingBuffer, nullptr);
	vkFreeMemory(device, stagingBufferMemory, nullptr);

}


