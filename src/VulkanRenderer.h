#pragma once
#define GLFW_INCLUDE_VULKAN
#ifdef NDEBUG
const bool enableValidationLayers = false;
#else
const bool enableValidationLayers = true;
#endif
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm//gtc/matrix_transform.hpp>
#include <stdexcept>
#include <vector>
#include <set>
#include <array>
#include <algorithm>
#include "stb_image.h"
#include "utilities.h"
#include "Mesh.h"
#include "VulkanValidation.h"
#include "MeshModel.h"
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

class VulkanRenderer
{
public:
	VulkanRenderer();

	int init(GLFWwindow* newWindow);
	void draw();
	void cleanup();

	void updateCamera(glm::mat4 view, glm::mat4 projection);
	void updateModel(int modelId, glm::mat4 newModel);
	~VulkanRenderer();
	void createMeshModel(std::string modelFile);
private:

	//Scene objects
	std::vector<Mesh> meshList;
	struct UboViewProjection {
		glm::mat4 projection;
		glm::mat4 view;
	} uboViewProjection;

	glm::vec3 cameraPosition;

	GLFWwindow* window;
	int currentFrame = 0;
	VkSampler textureSampler;

	std::vector<MeshModel> modelList;

	//Assets
	std::vector<VkImage> textureImages;
	std::vector<VkDeviceMemory> textureImageMemory;
	std::vector<VkImageView> textureImageViews;

	//Vulkan Components
	//-- MAIN COMPONENTS -- //
	VkInstance instance;
	VkDebugUtilsMessengerEXT debugMessenger;
	struct {
		VkPhysicalDevice physicalDevice;
		VkDevice logicalDevice;
	} mainDevice;
	const std::vector<const char*> validationLayers = {
	"VK_LAYER_KHRONOS_validation"
	};
	VkQueue graphicsQueue;
	VkQueue presentationQueue;
	VkSurfaceKHR surface;
	VkSwapchainKHR swapChain;

	std::vector<SwapchainImage> swapChainImages;
	std::vector<VkFramebuffer> swapChainFrameBuffers;
	std::vector<VkCommandBuffer> commandBuffers;

	std::vector<VkImage> opaqueColorBufferImage;   //One for each swapchain image
	std::vector<VkDeviceMemory> opaqueColourBufferImageMemory; //One for each swapchain image
	std::vector <VkImageView> opaqueColourBufferImageView;   //One for each image

	std::vector<VkImage> accumulationColourBufferImage;   //One for each swapchain image
	std::vector<VkDeviceMemory> accumulationColourBufferImageMemory; //One for each swapchain image
	std::vector <VkImageView> accumulationColourBufferImageView;   //One for each image

	std::vector<VkImage> revealageColourBufferImage;   //One for each swapchain image
	std::vector<VkDeviceMemory> revealageColourBufferImageMemory; //One for each swapchain image
	std::vector <VkImageView> revealageColourBufferImageView;   //One for each image

	std::vector<VkImage> depthBufferImage;    //Now we need one for every image, because we dont want to use the  depth buffer image after the first subpass finished rendering(for the second subpass)
	std::vector<VkDeviceMemory> depthBufferImageMemory; //Again one for each image
	std::vector <VkImageView> depthBufferImageView;   //One for each image

	// --PIPELINE COMPONENTS -- //
	VkPipeline graphicsPipeline;
	VkPipelineLayout pipelineLayout;

	VkPipeline compositionPipeline;
	VkPipelineLayout compositionPipelineLayout;

	VkPipeline translucentGeometryPipeline;
	VkPipelineLayout translucentGeometryPipelineLayout;

	VkRenderPass renderPass;

	//--Synchronization Objects
	std::vector<VkSemaphore> imageAvailable;
	std::vector<VkSemaphore> renderFinished;
	std::vector<VkFence> drawFences;

	// --POOLS--//
	VkCommandPool graphicsCommandPool;

	//-->utilities
	VkFormat swapChainImageFormat;
	VkFormat depthBufferImageFormat;
	VkExtent2D swapChainExtent;

	//--Descriptors
	VkDescriptorSetLayout descriptorSetLayout;
	VkDescriptorSetLayout samplerSetLayout;
	VkDescriptorSetLayout inputSetLayout;

	VkDescriptorPool samplerDescriptorPool;
	VkDescriptorPool descriptorPool;
	VkDescriptorPool inputDescriptorPool;

	std::vector<VkDescriptorSet> descriptorSets;
	std::vector<VkDescriptorSet> samplerDescriptorSets;
	std::vector<VkDescriptorSet> inputDescriptorSets;

	std::vector<VkBuffer> vpUniformBuffer;
	std::vector<VkDeviceMemory> vpUniformBufferMemory;

	VkPushConstantRange pushConstantRange;

	//std::vector<VkBuffer> modelDUniformBuffer;
	//std::vector<VkDeviceMemory> modelDUniformBufferMemory;

	//VkDeviceSize minUniformBufferOffset;
	//size_t modelUniformAlignment;

	//Model* modelTransferSpace;   //We use this pointer with the right offsets when we are copying data to the Device

	//Vulkan Functions
	//-->Create Functions
	void createSwapChain();
	void createInstance();
	void CreateLogicalDevice();
	void createDebugMessenger();
	void createSurface();
	void createRenderPass();
	void createDescriptorSetLayout();
	void createPushConstantRange();
	void createGraphicsPipeline();
	void createColourBufferImages();
	void createDepthBufferImage();
	void createFramebuffers();
	void createCommandPool();
	void createCommandBuffers();
	void createSynchronisation();
	void createTextureSampler();

	void createUniformBuffers();
	void createDescriptorPool();
	void createDescriptorSets();
	void createInputDescriptorSets();

	void updateUniformBuffers(uint32_t imageIndex);

	//-->Get Functions
	void getPhysicalDevice();
	QueueFamilyIndices getQueueFamilyIndices(VkPhysicalDevice physicalDevice);
	SwapChainDetails getSwapChainDetails(VkPhysicalDevice device);

	//-->Allocate functions
	/*void allocateDynamicBufferTransferSpace();*/

	//-->Helper Functions
	bool checkValidationLayerSupport();
	bool checkDeviceExtensionSuppport(VkPhysicalDevice device);
	bool checkInstanceExtensionSupport(const char** extensions, uint32_t extension_count);
	bool checkPhysicalDeviceSuitable(VkPhysicalDevice physicalDevice);
	void populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& createInfo);

	//-->Choose functions
	VkSurfaceFormatKHR chooseBestSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& formats);
	VkPresentModeKHR chooseBestPresentMode(const std::vector<VkPresentModeKHR>& presentationModes);
	VkExtent2D chooseSwapImageExtent(const VkSurfaceCapabilitiesKHR& surfaceCapabilities);
	VkFormat chooseSupportedFormat(const std::vector<VkFormat>& formats, VkImageTiling tiling, VkFormatFeatureFlags featureFlags);

	// -- Hepler Create Functions
	VkImage createImage(uint32_t width, uint32_t height, VkFormat format, VkImageTiling tiling, VkImageUsageFlags useFlags, VkMemoryPropertyFlags propFlags, VkDeviceMemory* imageMemory);
	VkImageView createImageView(VkImage image, VkFormat format, VkImageAspectFlags aspectFlags);
	VkShaderModule createShaderModule(const  std::vector<char>& code);
	int createTextureImage(std::string fileName, GeometryPass& texGeoPass);  //Returns int because we put the texture image into the texture image array and then return the index
	int createTexture(std::string fileName, GeometryPass& texGeoPass);
	int createTextureDescriptor(VkImageView textureImage);

	//-- Record Functions
	void recordCommands(uint32_t currentImage);

	//-- Loader Functions
	stbi_uc* loadTextureFile(std::string fileName, int* width, int* height, GeometryPass& geoPass, VkDeviceSize* imageSize);
};
