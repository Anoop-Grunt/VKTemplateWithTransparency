#pragma once
//Device extensions
#include <fstream>
#include <glm/glm.hpp>
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

const int MAX_FRAMES_IN_FLIGHT = 2;  //The maximum number of frames that are being drawn at once
const int MAX_OBJECTS = 40;

const std::vector<const char*> deviceExtensions = {
	VK_KHR_SWAPCHAIN_EXTENSION_NAME
};

struct Vertex {
	//Vertex data representation
	glm::vec3 pos;
	glm::vec3 col; // Vertex Colour (r, g, b)
	glm::vec2 tex; //Texture co ods
};

// Indices (locations) of Queue Families (if they exist at all)
struct QueueFamilyIndices {
	int graphicsFamily = -1;			// Location of graphics queue family
	int presentationFamily = -1;        // Location of the presentation queue family
	// Check if queue families are valid
	bool isValid()
	{
		return graphicsFamily >= 0 && presentationFamily >= 0;
	}
};

struct SwapChainDetails {
	VkSurfaceCapabilitiesKHR surfaceCapabilities;    //surface properties ex: image size/extent
	std::vector< VkSurfaceFormatKHR> formats;        //supported image formats ex:rgba8
	std::vector< VkPresentModeKHR> presentationModes;//SwapChain presentation modes  
};

struct SwapchainImage {
	VkImage image;
	VkImageView imageView;
};

static std::vector<char> readFile(const std::string& filepath) {
	//we want to read the file into a vector so we need the size(in bytes), so we use ate(put the pointer at end of file)
	std::ifstream file(filepath, std::ios::binary | std::ios::ate);

	//Check if file stream successfully opened
	if (!file.is_open()) {
		throw std::runtime_error("Error opening shader file ");
	}

	//We use the current read position as the size for the vector because, the SPIR-V files are in bytecode anyway
	size_t fileSize = (size_t)file.tellg();
	std::vector<char> fileBuffer(fileSize);

	//Now populate the buffer with the shader intermediate bytecode
	file.seekg(0);    //move read position to the start of the file
	file.read(fileBuffer.data(), fileSize);  //Read filesize number of bytes from the pointer into our buffer

	//Close the stream
	file.close();

	return fileBuffer;
}


static uint32_t findMemoryTypeIndex(VkPhysicalDevice physicalDevice ,uint32_t allowedTypes, VkMemoryPropertyFlags properties)
{
	// Get properties of physical device memory
	VkPhysicalDeviceMemoryProperties memoryProperties;
	vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memoryProperties);

	//Now we check which index supports the properties, and the allowed types
	//Physical device memory is dividd into many types of memories--> we want a suitable one
	//If allowed types  = 000000110 --> it means that the second and third memory types are allowed
	//So if we were at memory type index 2 in the iteration, our and operation wold return non zero (000000010 & 000000110)
	//Then we also check if the memory type we just choose also has the required properties
	//For that if the supported properties is (11010) and the required is (01010) the and operation returns (01010) which is just requiredProperties
	//So the and operation needs to return the requiredProperties, if it does, the memory type is Valid, so we return that
	for (uint32_t i = 0; i < memoryProperties.memoryTypeCount; i++)
	{
		if ((allowedTypes & (1 << i))														// Index of memory type must match corresponding bit in allowedTypes
			&& (memoryProperties.memoryTypes[i].propertyFlags & properties) == properties)	// Desired property bit flags are part of memory type's property flags
		{
			// This memory type is valid, so return its index
			return i;
		}
	}
}


static void createBuffer(VkPhysicalDevice physicalDevice, VkDevice device, VkDeviceSize bufferSize, VkBufferUsageFlags bufferUsage,
	VkMemoryPropertyFlags buffrerProperties, VkBuffer* buffer, VkDeviceMemory* bufferMemory) {
	VkBufferCreateInfo bufferInfo = {};
	bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	bufferInfo.size = sizeof(Vertex) * bufferSize; //The size of the vertex buffer
	bufferInfo.usage = bufferUsage;   //Specifying the type of buffer
	bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;    //We dont want the vertex buffer being shared by multiple queues

	VkResult result = vkCreateBuffer(device, &bufferInfo, nullptr, buffer);
	if (result != VK_SUCCESS) {
		throw std::runtime_error("Error creating buffer");
	}

	//Get buffer memory requirements
	VkMemoryRequirements memRequirements = {};    //The type of memory requires for allocating the vertex buffer
	vkGetBufferMemoryRequirements(device, *buffer, &memRequirements);   //We get the memory requirements(like size, alignment etc)

	VkMemoryAllocateInfo memoryAllocInfo = {};
	memoryAllocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	memoryAllocInfo.allocationSize = memRequirements.size;   //How much we want to allocate for our buffer
	memoryAllocInfo.memoryTypeIndex = findMemoryTypeIndex(physicalDevice, memRequirements.memoryTypeBits, buffrerProperties); //We get the index of requirede memory type
	// VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT	: CPU can interact with memory	// VK_MEMORY_PROPERTY_HOST_COHERENT_BIT	: Allows placement of data straight into buffer after mapping (otherwise would have to specify manually)

	// Allocate memory to VkDeviceMemory	
	result = vkAllocateMemory(device, &memoryAllocInfo, nullptr, bufferMemory);
	if (result != VK_SUCCESS) {
		throw std::runtime_error("Error Allocating Buffer Memory");
	}

	//Now Allocate the memory to the vertex  buffer
	vkBindBufferMemory(device, *buffer, *bufferMemory, 0); //The last parameter specifies where in the memory the vertex buffer needs to be allocated, here since the size is the same as the vertex buffer anyways, we just put 0, in other cases we canallocate multiple buffers in the segment

}

static VkCommandBuffer beginCommandBuffer(VkDevice device, VkCommandPool commandPool) {
	//Command buffer to hold  commands (Now it's not just for the trnasfer commands)
	VkCommandBuffer commandBuffer;

	//Command buffer allocate info
	VkCommandBufferAllocateInfo allocInfo = {};
	allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
	allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;  //Meaning it wont be executed by another command buffer, if you rememmber
	allocInfo.commandPool = commandPool;
	allocInfo.commandBufferCount = 1;

	//Now allocate the tranfer command buffer in the transfer command pool
	vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer);

	//Now begin the command buffer and record the transfer command
	VkCommandBufferBeginInfo beginInfo = {};
	beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;  //Because it's a one time operation (the command bufrfer becomes invalid after one usage)

	//Begin recording the  commands
	vkBeginCommandBuffer(commandBuffer, &beginInfo);

	//Now return the command buffer
	return commandBuffer;

};

static void endAndSubmitCommandBuffer(VkDevice device, VkCommandPool commandPool, VkQueue queue, VkCommandBuffer commandBuffer) {

	//Now end the command buffer (stop recording)
	vkEndCommandBuffer(commandBuffer);

	//Queue submission (no sync objects neede because it's a one-time operation)
	VkSubmitInfo submitInfo = {};
	submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	submitInfo.commandBufferCount = 1;
	submitInfo.pCommandBuffers = &commandBuffer;

	//Now submit the transfer command and wait until it finishes
	vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE);
	vkQueueWaitIdle(queue);

	//Free temporary command buffer back to pool
	vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
}

static void copyBuffer(VkDevice device, VkQueue transferQueue, VkCommandPool transferCommandPool, VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize bufferSize) {

	//Create buffer
	VkCommandBuffer transferCommandBuffer = beginCommandBuffer(device, transferCommandPool);

	//We specify which region of the buffer to  copy from and to
	VkBufferCopy copyRegion = {};
	copyRegion.srcOffset = 0; //From the beginning of the src buffer
	copyRegion.dstOffset = 0; //To the beginning of the dst buffer
	copyRegion.size = bufferSize;
	
	
	//record
	vkCmdCopyBuffer(transferCommandBuffer, srcBuffer, dstBuffer, 1, &copyRegion);

	endAndSubmitCommandBuffer(device, transferCommandPool, transferQueue, transferCommandBuffer);
}

static void copyImageBuffer(VkDevice device, VkQueue transferQueue, VkCommandPool transferCommandPool, VkBuffer srcBuffer, VkImage image, uint32_t width, uint32_t height) {
	
	//Create buffer
	VkCommandBuffer transferCommandBuffer = beginCommandBuffer(device, transferCommandPool);

	VkBufferImageCopy imageRegion = {};
	imageRegion.bufferOffset = 0;    //Offset into data
	imageRegion.bufferRowLength = 0; //Row length of data to calculate spacing
	imageRegion.bufferImageHeight = 0; //Again zero, to indicate that data is tightly packed
	imageRegion.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;   //Which aspect of the image to copy
	imageRegion.imageSubresource.mipLevel = 0;  //Which mip level to copy
	imageRegion.imageSubresource.baseArrayLayer = 0; //If we ahave an array of layered images, which layer to copy
	imageRegion.imageSubresource.layerCount = 1;    //How many layers to copy starting from the baseArrayLayer
	imageRegion.imageOffset = { 0, 0, 0 };			//Offset into image as oppsed to offset into rau buffer data
	imageRegion.imageExtent = { width, height, 1 };  //Size of region to copy as x, y, z values 

	//Now record the command to copy from buffer to image
	//The layout needs to be such that the tansfer to destination image is optimal
	vkCmdCopyBufferToImage(transferCommandBuffer, srcBuffer, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &imageRegion);

	//stop recording and submit the command buffer
	endAndSubmitCommandBuffer(device, transferCommandPool, transferQueue, transferCommandBuffer);
};

static void transitionImageLayout(VkDevice device, VkQueue queue, VkCommandPool commandPool, VkImage image, VkImageLayout oldLayout, VkImageLayout newLayout) {
	//Create buffer
	VkCommandBuffer commandBuffer = beginCommandBuffer(device, commandPool);

	VkImageMemoryBarrier imageMemoryBarrier = {};  //Makes sure that a stage has fininshed before another starts, but more importantly it allows us to transition an image layout
	imageMemoryBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
	imageMemoryBarrier.oldLayout = oldLayout; //Transitions from this layout
	imageMemoryBarrier.newLayout = newLayout; //Transitions to this layout
	imageMemoryBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;  //We can transfer to another queue family if we want
	imageMemoryBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED; //We don't want to transfer the resource between the queues
	imageMemoryBarrier.image = image; //Image being accessed and modified as part of the barrier
	imageMemoryBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT; //	Aspect of image being altered
	imageMemoryBarrier.subresourceRange.baseMipLevel = 0;					  //First mip level to start alterations on
	imageMemoryBarrier.subresourceRange.levelCount = 1;     //We only want to alter the first mip level
	imageMemoryBarrier.subresourceRange.baseArrayLayer = 0; //First layer to start alterations on
	imageMemoryBarrier.subresourceRange.layerCount = 1; //We only want to alter the first layer

	VkPipelineStageFlags srcStage;
	VkPipelineStageFlags dstStage;

	// If transitioning from new image to image ready to receive data...
	if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)
	{
		imageMemoryBarrier.srcAccessMask = 0;								// Memory access stage transition must after...
		imageMemoryBarrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;		// Memory access stage transition must before...

		srcStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
		dstStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
	}
	else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
	{
		imageMemoryBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
		imageMemoryBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

		srcStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
		dstStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
	}

	vkCmdPipelineBarrier(
		commandBuffer,
		srcStage, dstStage,		// Pipeline stages (match to src and dst AccessMasks)
		0,						// Dependency flags
		0, nullptr,				// Memory Barrier count + data
		0, nullptr,				// Buffer Memory Barrier count + data
		1, &imageMemoryBarrier	// Image Memory Barrier count + data
	);

	//stop recording and submit the command buffer
	endAndSubmitCommandBuffer(device, commandPool, queue, commandBuffer);
};