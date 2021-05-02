#include "VulkanRenderer.h"
#include <iostream>

VulkanRenderer::VulkanRenderer()
{
}

int VulkanRenderer::init(GLFWwindow* newWindow)
{
	window = newWindow;
	if (enableValidationLayers) {
		printf("Started in Debug mode\n\n");
	}
	else
	{
		printf("Start in Debug mode for validation layers\n\n");
	}
	try
	{
		createInstance();
		createDebugMessenger();
		createSurface();
		getPhysicalDevice();
		CreateLogicalDevice();
		createSwapChain();
		createColourBufferImages();
		createDepthBufferImage();
		createRenderPass();
		createDescriptorSetLayout();
		createPushConstantRange();
		createGraphicsPipeline();
		createFramebuffers();
		createCommandPool();
		createCommandBuffers();
		//allocateDynamicBufferTransferSpace();   //Not needed because we aren't using dynamic uniform buffers anymore, instead we are using push constants
		createTextureSampler();
		createUniformBuffers();
		createDescriptorPool();
		createDescriptorSets();
		createInputDescriptorSets();
		createSynchronisation();

		uboViewProjection.projection = glm::perspective(glm::radians(45.0f), (float)swapChainExtent.width / (float)swapChainExtent.height, 0.1f, 100.0f);
		cameraPosition = glm::vec3(0.0f, 0.0f, 10.0f);
		uboViewProjection.view = glm::lookAt(cameraPosition, glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));

		//Inverting the y co-ordinate for vulkan
		uboViewProjection.projection[1][1] *= -1;

		//Scene objects
		GeometryPass defPass;
		createTexture("default.jpg", defPass);
	}
	catch (const std::runtime_error& e)
	{
		printf("ERROR: %s\n", e.what());
		return EXIT_FAILURE;
	}
	return 0;
}

void VulkanRenderer::draw()
{
	//In the init function we are allocating command buffers and, recording commands, but we don't submit them to the graphics queue, which is what we need to do
	//First we get the next available image to draw to and set a semaphore to signmal when we are done with an image
	//Then we submit the command buffer to the queue makign sure that the image we want to draw to is set as available by the semaphore
	//Signal when image has finished rendering
	//Then present image to screen when it has signalled fininshed rendering

	//Check if the current frame is still on the queue and hasnt been executed, if it is , wait untill it is free, before submitting again
	vkWaitForFences(mainDevice.logicalDevice, 1, &drawFences[currentFrame], VK_TRUE, std::numeric_limits<uint64_t>::max());
	vkResetFences(mainDevice.logicalDevice, 1, &drawFences[currentFrame]); //Close the fence again so that the other frames wait

	//Get the image
	uint32_t imageIndex = 0;
	vkAcquireNextImageKHR(mainDevice.logicalDevice, swapChain, std::numeric_limits<uint64_t>::max(), imageAvailable[currentFrame], VK_NULL_HANDLE, &imageIndex);


	//TODO: from the NVIDIA vulkan tips page --> recording command buffers is a CPU intensive task and no driver threads come to the rescue, so try to mutlithread it.
	//TODO:Use a separate command pool for each thread which records command buffers, for each frame. Before starting work on this part, please first check the vulkan do's and dont's section on this.
	//
	//Update the uniform buffers
	recordCommands(imageIndex);  //The reason we send imageIndex to the record function is because if we try to re record all the command buffers(like we originally were) in the record function, we might be rtrying to re record on command buffers that are still in use by the queue
	updateUniformBuffers(imageIndex);

	//Submit command buffer to render
	//create submit inifo
	VkSubmitInfo submitInfo = {};
	submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	submitInfo.waitSemaphoreCount = 1; // number of semaphores to wait on
	submitInfo.pWaitSemaphores = &imageAvailable[currentFrame]; //list of semaphores to wait on
	//Semaphores are used at some point in the pipeline, where the pipeline has to pause because the raesource is being used
	//Here we stop before the fragment shader, because everything before that doesn't require the actual image at all
	//So we need masks again
	VkPipelineStageFlags waitStages[] = {
		VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT   //Stages to check semaphore at
	};
	submitInfo.pWaitDstStageMask = waitStages;
	submitInfo.commandBufferCount = 1; //How many command buffers do we want ot submit
	submitInfo.pCommandBuffers = &commandBuffers[imageIndex]; //The command buffer to submit
	submitInfo.signalSemaphoreCount = 1;  //Number of semaphores to signal after the command buffer finishes
	submitInfo.pSignalSemaphores = &renderFinished[currentFrame]; //List of semaphores to signal after the command buffer fininshes

	//Now we submit the command buffers, execution is automatic.
	VkResult result = vkQueueSubmit(graphicsQueue, 1, &submitInfo, drawFences[currentFrame]);  //Fence is opened when the commands in the buffer have finished executing, wether or not the image has been presented is irrelavant
	if (result != VK_SUCCESS) {
		throw std::runtime_error("Error submitting command buffer to graphics queue");
	}

	//Now present the image to the screen
	VkPresentInfoKHR presentInfo = {};
	presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
	presentInfo.waitSemaphoreCount = 1;
	presentInfo.pWaitSemaphores = &renderFinished[currentFrame]; // We wait untill render is finished to present
	presentInfo.swapchainCount = 1;  //Number of swapchains to present to
	presentInfo.pSwapchains = &swapChain;  //SwapChain to present images to
	presentInfo.pImageIndices = &imageIndex; //Index of image in swapchains to present to

	//Now present image to swapchain
	result = vkQueuePresentKHR(presentationQueue, &presentInfo);
	if (result != VK_SUCCESS) {
		throw std::runtime_error("Error presenting image to swapchain");
	}
	//Increment the currentFrame number
	currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
}

void VulkanRenderer::cleanup()
{
	vkDeviceWaitIdle(mainDevice.logicalDevice);   //Dont call cleanup function untill device is idle, because we dont want to destroy semaphores and command pools while the command buffers are still on the graphics queue

	//	_aligned_free(modelTransferSpace);       //This is a C function, not Vulkan, no longer in use

	for (size_t i = 0; i < modelList.size(); i++) {
		modelList[i].destroyMeshModel();
	}

	//Destroying the desccriptor pool destroys all the descriptor sets allocated in it
	vkDestroyDescriptorPool(mainDevice.logicalDevice, inputDescriptorPool, nullptr);
	vkDestroyDescriptorSetLayout(mainDevice.logicalDevice, inputSetLayout, nullptr);

	vkDestroyDescriptorPool(mainDevice.logicalDevice, samplerDescriptorPool, nullptr);
	vkDestroyDescriptorSetLayout(mainDevice.logicalDevice, samplerSetLayout, nullptr);
	vkDestroySampler(mainDevice.logicalDevice, textureSampler, nullptr);
	for (size_t i = 0; i < textureImages.size(); i++) {
		vkDestroyImageView(mainDevice.logicalDevice, textureImageViews[i], nullptr);
		vkDestroyImage(mainDevice.logicalDevice, textureImages[i], nullptr);
		vkFreeMemory(mainDevice.logicalDevice, textureImageMemory[i], nullptr);
	}
	for (size_t i = 0; i < revealageColourBufferImage.size(); i++) {
		vkDestroyImageView(mainDevice.logicalDevice, revealageColourBufferImageView[i], nullptr);
		vkDestroyImage(mainDevice.logicalDevice, revealageColourBufferImage[i], nullptr);
		vkFreeMemory(mainDevice.logicalDevice, revealageColourBufferImageMemory[i], nullptr);
	}
	for (size_t i = 0; i < opaqueColorBufferImage.size(); i++) {
		vkDestroyImageView(mainDevice.logicalDevice, opaqueColourBufferImageView[i], nullptr);
		vkDestroyImage(mainDevice.logicalDevice, opaqueColorBufferImage[i], nullptr);
		vkFreeMemory(mainDevice.logicalDevice, opaqueColourBufferImageMemory[i], nullptr);
	}
	for (size_t i = 0; i < accumulationColourBufferImage.size(); i++) {
		vkDestroyImageView(mainDevice.logicalDevice, accumulationColourBufferImageView[i], nullptr);
		vkDestroyImage(mainDevice.logicalDevice, accumulationColourBufferImage[i], nullptr);
		vkFreeMemory(mainDevice.logicalDevice, accumulationColourBufferImageMemory[i], nullptr);
	}
	for (size_t i = 0; i < depthBufferImage.size(); i++) {
		vkDestroyImageView(mainDevice.logicalDevice, depthBufferImageView[i], nullptr);
		vkDestroyImage(mainDevice.logicalDevice, depthBufferImage[i], nullptr);
		vkFreeMemory(mainDevice.logicalDevice, depthBufferImageMemory[i], nullptr);
	}
	vkDestroyDescriptorPool(mainDevice.logicalDevice, descriptorPool, nullptr);
	vkDestroyDescriptorSetLayout(mainDevice.logicalDevice, descriptorSetLayout, nullptr);
	for (size_t i = 0; i < swapChainImages.size(); i++)
	{
		vkDestroyBuffer(mainDevice.logicalDevice, vpUniformBuffer[i], nullptr);
		vkFreeMemory(mainDevice.logicalDevice, vpUniformBufferMemory[i], nullptr);
		/*vkDestroyBuffer(mainDevice.logicalDevice, modelDUniformBuffer[i], nullptr);
		vkFreeMemory(mainDevice.logicalDevice, modelDUniformBufferMemory[i], nullptr);*/ //No buffers for model data(no dynamic buffers)
	}
	for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
	{
		vkDestroySemaphore(mainDevice.logicalDevice, renderFinished[i], nullptr);
		vkDestroySemaphore(mainDevice.logicalDevice, imageAvailable[i], nullptr);
		vkDestroyFence(mainDevice.logicalDevice, drawFences[i], nullptr);
	}
	vkDestroyCommandPool(mainDevice.logicalDevice, graphicsCommandPool, nullptr);
	for (auto framebuffer : swapChainFrameBuffers) {
		vkDestroyFramebuffer(mainDevice.logicalDevice, framebuffer, nullptr);
	}
	vkDestroyPipeline(mainDevice.logicalDevice, compositionPipeline, nullptr);
	vkDestroyPipelineLayout(mainDevice.logicalDevice, compositionPipelineLayout, nullptr);
	vkDestroyPipeline(mainDevice.logicalDevice, translucentGeometryPipeline, nullptr);
	vkDestroyPipelineLayout(mainDevice.logicalDevice, translucentGeometryPipelineLayout, nullptr);
	vkDestroyPipeline(mainDevice.logicalDevice, graphicsPipeline, nullptr);
	vkDestroyPipelineLayout(mainDevice.logicalDevice, pipelineLayout, nullptr);
	vkDestroyRenderPass(mainDevice.logicalDevice, renderPass, nullptr);
	for (auto image : swapChainImages)
	{
		vkDestroyImageView(mainDevice.logicalDevice, image.imageView, nullptr);
	}
	vkDestroySwapchainKHR(mainDevice.logicalDevice, swapChain, nullptr);
	if (enableValidationLayers) {
		DestroyDebugUtilsMessengerEXT(instance, debugMessenger, nullptr);
	}
	vkDestroySurfaceKHR(instance, surface, nullptr);
	vkDestroyDevice(mainDevice.logicalDevice, nullptr);
	vkDestroyInstance(instance, nullptr);
}

void VulkanRenderer::updateCamera(glm::mat4 view, glm::mat4 projection)
{
	uboViewProjection.view = view;
	uboViewProjection.projection = projection;
	uboViewProjection.projection[1][1] *= -1;
}

void VulkanRenderer::updateModel(int modelId, glm::mat4 newModel)
{
	if ((size_t)modelId >= modelList.size()) return;

	modelList[modelId].setModel(newModel);
}

VulkanRenderer::~VulkanRenderer()
{
}

void VulkanRenderer::createSwapChain()
{
	//Get the swapchain details, so we can pick best settings
	SwapChainDetails swapChainDetails = getSwapChainDetails(mainDevice.physicalDevice);

	//Choose best surface format
	VkSurfaceFormatKHR surfaceFormat = chooseBestSurfaceFormat(swapChainDetails.formats);

	//Choose best presentation mode
	VkPresentModeKHR presentMode = chooseBestPresentMode(swapChainDetails.presentationModes);

	//Choose swapchain image resolution.
	VkExtent2D imageExtent = chooseSwapImageExtent(swapChainDetails.surfaceCapabilities);

	//Choose the number of images in the swapChain --> one more than the minimum
	uint32_t imageCount = swapChainDetails.surfaceCapabilities.minImageCount + 1;
	//Make sure image count is less than maxImageCount(surface)
	//if maxImageCount == 0 then it actually means that the max is limitless
	if (swapChainDetails.surfaceCapabilities.minImageCount > 0 && swapChainDetails.surfaceCapabilities.maxImageCount < imageCount) {
		imageCount = swapChainDetails.surfaceCapabilities.maxImageCount;
	}

	//Build the swapchainCreateInfo struct
	VkSwapchainCreateInfoKHR swapChainCreateInfo = {};
	swapChainCreateInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
	swapChainCreateInfo.surface = surface;                                //SwapChain surface
	swapChainCreateInfo.imageFormat = surfaceFormat.format;               //SwapChain image format
	swapChainCreateInfo.imageColorSpace = surfaceFormat.colorSpace;       //SwapChain image colorspace
	swapChainCreateInfo.presentMode = presentMode;
	swapChainCreateInfo.imageExtent = imageExtent;						  //SwapChain image extent
	swapChainCreateInfo.minImageCount = imageCount;                       //minimum number of images(for triple buffer)
	swapChainCreateInfo.imageArrayLayers = 1;							  //Number of layers for each image
	swapChainCreateInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT; //What attachment images will be used as
	swapChainCreateInfo.preTransform = swapChainDetails.surfaceCapabilities.currentTransform;  //Transform to perform on the swapChain images
	swapChainCreateInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR; //How to handle blending images with external graphics(ex: if it was on another window)
	swapChainCreateInfo.clipped = VK_TRUE;                                  //When there is another window on top, we want to clip

	//If we have separate presentation, and graphics queues, we have to share images between the two
	QueueFamilyIndices indices = getQueueFamilyIndices(mainDevice.physicalDevice);

	if (indices.graphicsFamily != indices.presentationFamily) {
		uint32_t queueFamilyIndices[] = { (uint32_t)indices.graphicsFamily, (uint32_t)indices.presentationFamily };

		//In case they are separate, then sharing mode is set to concurrent
		swapChainCreateInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
		swapChainCreateInfo.queueFamilyIndexCount = 2;
		swapChainCreateInfo.pQueueFamilyIndices = queueFamilyIndices;
	}
	else
	{
		swapChainCreateInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
		swapChainCreateInfo.queueFamilyIndexCount = 0;
		swapChainCreateInfo.pQueueFamilyIndices = nullptr;
	}

	//useful when we want to resize the window or something(because old swapchain needs to be destroyed)
	swapChainCreateInfo.oldSwapchain = VK_NULL_HANDLE;

	//Now create the SwapChain
	VkResult result = vkCreateSwapchainKHR(mainDevice.logicalDevice, &swapChainCreateInfo, nullptr, &swapChain);
	if (result != VK_SUCCESS) {
		throw std::runtime_error("\nError creating swapchain\n");
	}

	// Store for later reference
	swapChainImageFormat = surfaceFormat.format;
	swapChainExtent = imageExtent;

	//Now get the images, and create the image views
	uint32_t SwapChainImageCount = 0;
	vkGetSwapchainImagesKHR(mainDevice.logicalDevice, swapChain, &SwapChainImageCount, nullptr);
	std::vector<VkImage> images(SwapChainImageCount);
	vkGetSwapchainImagesKHR(mainDevice.logicalDevice, swapChain, &SwapChainImageCount, images.data());

	for (VkImage image : images) {
		//store the image handle
		//Create a vkimageview, add it to a SwapChainImage struct, and push onto swapChainImages
		SwapchainImage	swapChainImage = {};
		swapChainImage.image = image;
		swapChainImage.imageView = createImageView(image, swapChainImageFormat, VK_IMAGE_ASPECT_COLOR_BIT);

		// Add to swapchain image list
		swapChainImages.push_back(swapChainImage);
	}
}

void VulkanRenderer::createInstance()
{
	// Information about the application itself
	// Most data here doesn't affect the program and is for developer convenience
	VkApplicationInfo appInfo = {};
	appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
	appInfo.pApplicationName = "Vulkan App";					// Custom name of the application
	appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);		// Custom version of the application
	appInfo.pEngineName = "No Engine";							// Custom engine name
	appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);			// Custom engine version
	appInfo.apiVersion = VK_API_VERSION_1_2;					// The Vulkan Version

	// Creation information for a VkInstance (Vulkan Instance)
	VkInstanceCreateInfo createInfo = {};
	createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
	createInfo.pApplicationInfo = &appInfo;

	// Create list to hold instance extensions
	std::vector<const char*> instanceExtensions = std::vector<const char*>();

	// Set up extensions Instance will use
	uint32_t glfwExtensionCount = 0;				// GLFW may require multiple extensions
	const char** glfwExtensions;					// Extensions passed as array of cstrings, so need pointer (the array) to pointer (the cstring)

	// Get GLFW extensions
	glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

	// Add GLFW extensions to list of extensions
	for (size_t i = 0; i < glfwExtensionCount; i++)
	{
		instanceExtensions.push_back(glfwExtensions[i]);
	}

	//Checking if the extensions required by GLFW are actually supported
	checkInstanceExtensionSupport(glfwExtensions, glfwExtensionCount);

	//if validation layers are enabled also adding the debug utils extension to the instance extensions, this is needed for message callbacks
	if (enableValidationLayers) {
		instanceExtensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
		checkInstanceExtensionSupport(&instanceExtensions[instanceExtensions.size() - 1], 1);
	}

	createInfo.enabledExtensionCount = static_cast<uint32_t>(instanceExtensions.size());
	createInfo.ppEnabledExtensionNames = instanceExtensions.data();


	if (!checkValidationLayerSupport() && enableValidationLayers) {
		throw std::runtime_error("validation layers requested, but not available!");
	}

	//We also send a the debug messenger create info to the pnext field, this allows us to debug the creation, and destruction of the instance.
	VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo;
	if (enableValidationLayers) {
		createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
		createInfo.ppEnabledLayerNames = validationLayers.data();

		//The next two lines allow us to debug the creation/destruction of the instance(before validation layers are even created)
		populateDebugMessengerCreateInfo(debugCreateInfo);
		createInfo.pNext = (VkDebugUtilsMessengerCreateInfoEXT*)&debugCreateInfo;
	}
	else {
		createInfo.enabledLayerCount = 0;
		createInfo.ppEnabledLayerNames = nullptr;
	}

	// Create instance
	VkResult result = vkCreateInstance(&createInfo, nullptr, &instance);

	if (result != VK_SUCCESS)
	{
		throw std::runtime_error("Failed to create a Vulkan Instance!");
	}
}

void VulkanRenderer::CreateLogicalDevice()
{
	//The queue family inidces on the logical device and the physical device need to be the same
	//ONce again we print the chosen devices queue indices
	printf("\nretrieving the queue family indices of the chosen physical device:\n");
	QueueFamilyIndices indices = getQueueFamilyIndices(mainDevice.physicalDevice);

	//Now we create a  vector of the Queue create infos of all the required queues
	std::vector< VkDeviceQueueCreateInfo> queueCreateInfos;
	std::set<int> queueFamilyIndices = { indices.graphicsFamily, indices.presentationFamily };

	//Since the indices of  the graphics queue and the presentation queue are generally the same, we use set to avoid duplicate create infos
	for (int queueFamilyIndex : queueFamilyIndices) {
		VkDeviceQueueCreateInfo queueCreateInfo = {};
		queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
		queueCreateInfo.queueFamilyIndex = queueFamilyIndex;
		queueCreateInfo.queueCount = 1;
		float priority = 1.f;
		queueCreateInfo.pQueuePriorities = &priority;

		//Now push the createInfo struct onto the vector
		queueCreateInfos.push_back(queueCreateInfo);
	}

	//initialize the deviceCreateInfo struct
	VkDeviceCreateInfo deviceCreateInfo = {};
	deviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
	deviceCreateInfo.queueCreateInfoCount = static_cast<uint32_t> (queueCreateInfos.size());              //this is the number of queue create info structs, not queues
	deviceCreateInfo.pQueueCreateInfos = queueCreateInfos.data();  //this has to be the list of queue create infos
	deviceCreateInfo.enabledExtensionCount = static_cast<uint32_t> (deviceExtensions.size());             //These are device extensions, not instance extensions
	deviceCreateInfo.ppEnabledExtensionNames = deviceExtensions.data();     //list of enable logical extensions

	//Physical device deviceFeatures that the logicval device will be using need to be specified in a struct, and passed to deviceCreateInfo
	VkPhysicalDeviceFeatures deviceFeatures = {};
	deviceFeatures.samplerAnisotropy = VK_TRUE;
	deviceFeatures.independentBlend = VK_TRUE;

	deviceCreateInfo.pEnabledFeatures = &deviceFeatures;

	//Finally create the logical device
	VkResult result = vkCreateDevice(mainDevice.physicalDevice, &deviceCreateInfo, nullptr, &mainDevice.logicalDevice);
	if (result != VK_SUCCESS) {
		std::cout << result << std::endl;
		throw std::runtime_error("\nError creating logical device\n");
	}
	else
	{
		printf("\nsuccessfully created logical device\n");
	}

	//Now we get handles to the queues from the logical device
	vkGetDeviceQueue(mainDevice.logicalDevice, indices.graphicsFamily, 0, &graphicsQueue);
	vkGetDeviceQueue(mainDevice.logicalDevice, indices.presentationFamily, 0, &presentationQueue);
}

void VulkanRenderer::createDebugMessenger()
{
	//Note: This Debug messenger requires an instance to exist, and thus can't debug creation and destruction of instances.

	//skip step if in release configuration
	if (!enableValidationLayers) return;

	//Build the create info struct for the debug messenger
	VkDebugUtilsMessengerCreateInfoEXT createInfo;
	populateDebugMessengerCreateInfo(createInfo);

	//Since we have the create info struct, we can create the debug messenger with the custom callback.
	if (CreateDebugUtilsMessengerEXT(instance, &createInfo, nullptr, &debugMessenger) != VK_SUCCESS) {
		throw std::runtime_error("failed to set up debug messenger!");
	}
}

void VulkanRenderer::createSurface()
{
	//create surface (GLFW function that creates a createInfo struct, then runs the vkCreateSurface function, and returns the vkResult)
	//I guess this is why GLFW requires the VK_Surface_KHR, and the other Win32 Extensions.
	VkResult result = glfwCreateWindowSurface(instance, window, nullptr, &surface);
	if (result != VK_SUCCESS) {
		throw std::runtime_error("Error creating surface");
	}
}

void VulkanRenderer::createRenderPass()
{
	//Multiple subpasses array
	std::array<VkSubpassDescription, 3> subpasses{};

	//ATTACHMENTS
	/////////////--SUBPASS 1 (opaque geometry pass) ATTACHMENTS and REFERENCES--//////////////////////////////////////////////////////

	//colour attachment 
	VkAttachmentDescription colourAttachment = {};
	colourAttachment.format = chooseSupportedFormat(
		{ VK_FORMAT_R8G8B8A8_UNORM }, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL  //Like the swapchain images earlier
	);
	colourAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
	colourAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;			//Clear at the start of the render pass
	colourAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;    //This store op is how to store after the renderpass has finished not the subpass
	colourAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
	colourAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
	colourAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	colourAttachment.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

	//Depth attachment of subpass 1
	VkAttachmentDescription depthAttachment = {};
	depthAttachment.format = depthBufferImageFormat;
	depthAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
	depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
	depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
	depthAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
	depthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
	depthAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	depthAttachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;  //We DONT want the layout to change when the render pass ends

	//Colour attachment  reference
	VkAttachmentReference colourAttachmentReference = {};
	colourAttachmentReference.attachment = 1;  //Order matters (should be the same as the frame buffer attachmnet index)
	colourAttachmentReference.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

	//Depth attachment  reference
	VkAttachmentReference depthAttachmentReference = {};
	depthAttachmentReference.attachment = 2;  //In the framebuffer
	depthAttachmentReference.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

	// Set up Subpass 1
	subpasses[0].pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
	subpasses[0].colorAttachmentCount = 1;
	subpasses[0].pColorAttachments = &colourAttachmentReference;
	subpasses[0].pDepthStencilAttachment = &depthAttachmentReference;





	///////------SUBPASS 2 (translucent geometry) ATTACHMENTS AND REFERENCES -----------/////////////////////////////
	

	//ATTACHMENTS

	// 1. Accumulation buffer output attachment
	
	VkAttachmentDescription accumAttachment = {};
	accumAttachment.format = chooseSupportedFormat(
		{ VK_FORMAT_R16G16B16A16_SFLOAT }, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL  //Like the swapchain images earlier
	);  ///choosing an image format with 16bit channels
	accumAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
	accumAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
	accumAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE; //cuz it specifies how to store the image after the render pass, not the subpass
	accumAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
	accumAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
	accumAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED; //doesn't matter what format the accum tex is in before the subpass begins
	accumAttachment.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL; //cuz at the end of this subpasses pipeline we write to this color attachment (I say end of the subpass, but i mean the fragment shader stage)
	//this attachment needs to be the input attachment to the shader of the next subpass, so we need to transition the layout to shader read only optimal later


	// 2. The depth attachment, is the same Image which was used as the depth attachment of the first subpass, but we will have depth write disabled
	//The attachment structures are only needed by the render pass create info, so we don't need to create the same attachment again, 

	// 3. The revealage buffer image output attachment

	VkAttachmentDescription revealAttachment = {};
	revealAttachment.format = chooseSupportedFormat(
		{ VK_FORMAT_R16_SFLOAT }, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL  //Like the swapchain images earlier
	);  /// There is no VK_FORMAT_R8_SFLOAT, i mean it's not even in the spec
	revealAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
	revealAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
	revealAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
	revealAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
	revealAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
	revealAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	revealAttachment.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;  //since at the end of subpass 2 we output to the revealage buffer too from the translucent geometry shader


	//REFERENCES
	
	//1. the attachment reference for the accumulation texture
	VkAttachmentReference accumAttachmentReference = {};
	accumAttachmentReference.attachment = 3; //the accum texture is the attachment at index 3 of the framebuffer
	accumAttachmentReference.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL; //the layout the image will be in during the subpass

	//2. Depth attachment reference: we can use the reference from earlier

	//3. Reveal attachment reference
	VkAttachmentReference revealAttachmentReference = {};
	revealAttachmentReference.attachment = 4; //it's the 4th index attachment of the framebuffer
	revealAttachmentReference.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL; // layout during the subpass

	std::array<VkAttachmentReference, 2> subpass2ColorAttachmentReferences = {accumAttachmentReference, revealAttachmentReference};

	// Set up subpass 2 (The transparent geometry subpass)
	subpasses[1].pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
	subpasses[1].colorAttachmentCount = static_cast<uint32_t> (subpass2ColorAttachmentReferences.size());
	subpasses[1].pColorAttachments = subpass2ColorAttachmentReferences.data();
	subpasses[1].pDepthStencilAttachment = &depthAttachmentReference;  //reusing the same attachment reference created for subpass 1





	///////---SUBPASS 3 (composition) ATTACHMENTS and REFERENCES ---///////////////////////////////////////////////////////
	//swapchain colour attachment
	VkAttachmentDescription SwapChainColourAttachment = {};
	SwapChainColourAttachment.format = swapChainImageFormat;                  //Format to use for attachment
	SwapChainColourAttachment.samples = VK_SAMPLE_COUNT_1_BIT;                //Number ofsamples for multisampling
	SwapChainColourAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;           //Describes what to do with the attchment before rendering
	SwapChainColourAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;         //What to do after rendering to framebuffer(store because we want to present it later)
	SwapChainColourAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;//	What to do with the stencil before rendering
	SwapChainColourAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;//What to do with the stencil after rendering

	//Framebuffer data will be stored as an image, but images can be given different layouts
	//To give optimal use for certain operations we have to do transitions.
	SwapChainColourAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;         //Image data layout before render pass starts
	SwapChainColourAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;     //Image data layout after render pass (this is the one we want to change to)

	//Attachment reference uses an index to refer to the attachment in the attachment list we pass to the render pass create info
	//And subpasses use these attachment references.
	VkAttachmentReference SwapChainColourAttachmentReference = {};
	SwapChainColourAttachmentReference.attachment = 0;
	SwapChainColourAttachmentReference.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

	std::array<VkAttachmentReference, 4> inputReferences;

	//References to the inputs the next subpass will need

	inputReferences[0].attachment = 1;   //The opaque color attachment
	inputReferences[0].layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
	inputReferences[1].attachment = 2; //The depth attachnent
	inputReferences[1].layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
	inputReferences[2].attachment = 3; // The accumulation color attachment
	inputReferences[2].layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
	inputReferences[3].attachment = 4; // The revealage color attachment
	inputReferences[3].layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

	// Set up Subpass 3 (The composition pass)
	subpasses[2].pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
	subpasses[2].colorAttachmentCount = 1;  // just the swapchain image
	subpasses[2].pColorAttachments = &SwapChainColourAttachmentReference; 
	subpasses[2].inputAttachmentCount = static_cast<uint32_t> (inputReferences.size());  //to pass in as inputs to subpass 2
	subpasses[2].pInputAttachments = inputReferences.data();

	

	//Need to determine when layout transitions occur using subpass dependencies
	//Basically say between which two events we want the transition occurs
	//Transitions can haappen by default but we have to specify when.


	//Now since we have 3 subpasses, we will have  4 subpass dependencies

	std::array<VkSubpassDependency, 4> subpassDependencies;



	//Layout transition from subpass ext to subpass 1 (opaque)
	//First convert from  VK_IMAGE_LAYOUT_UNDEFINED to VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
	subpassDependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;  //source subpass
	subpassDependencies[0].srcStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;  //After which stage in the src subpass should the transition occur-->for the external subpass this is the last stage
	subpassDependencies[0].srcAccessMask = VK_ACCESS_MEMORY_READ_BIT;     //Basically saying that the read operation in srcStageMask stage needs to finish before transition

	subpassDependencies[0].dstSubpass = 0;                    //We pass an array to the subpasses in the renderPassCreate info, this is the index of the subpass in the array
	subpassDependencies[0].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT; //We want the transition to occur BEFORE the colour attachment output
	subpassDependencies[0].dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT; //Convert before reading and writing operations dstStageMask stage.
	subpassDependencies[0].dependencyFlags = 0;
	//access masks are almost like substages (they are not stages they're memory accesses (memory here meaning the image in the memory )) in the stagemask stages


	//Subpass 1 to 2 layout transition (opaque subpass to translucent subpass, so the accum image is being transitioned from undefined to color attachment optimal)
	//the reveal texture is also transitioned in a similar way to the accumulation buffer image
	subpassDependencies[1].srcSubpass = 0;
	subpassDependencies[1].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT; //before the layout transition, we want the color attachment output stage of the opaque subpass to have finished
	subpassDependencies[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
	subpassDependencies[1].dstSubpass = 1;
	subpassDependencies[1].dstStageMask = VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;  //Settting it to this stage becuase if we are using late fragments tests, the depth tests happen only after the fragment shader has completed execution
	subpassDependencies[1].dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT; //and the layout transition needs to happen before the translucent subpass tries to read from its depth buffer (depth stencil attachment)
	subpassDependencies[1].dependencyFlags = 0;
	//TODO: if the above destination stage mask, and access mask don't work, try making the stage = fragment shader bit, and the access flag = shader read bit. Doing this will force the layout transition before the frag shader tries to read from uniform buffers, or samplers etc. By doing this we are not delaying the layout transition as much as possible, so prolly it's not as optimal as we'd like

	//subpass 2 to subpass 3 layout transitions (the input attachments get converted from colour attachment optimal to shader read only optimal, and the output attachment which is the swapchain image is transitioned from undefined to color attachment optimal)
	subpassDependencies[2].srcSubpass = 1;
	subpassDependencies[2].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT; //we want the color attachment output stage of subpass 1 to be finished before the layout transition
	subpassDependencies[2].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT; 
	subpassDependencies[2].dstSubpass = 2;
	subpassDependencies[2].dstStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT; //we want to finish the layout transition before the fragment shader stage of subpass 2(well, the frag shader read stage to be specific, and that is specified in the next parameter : dstAccessMask)
	subpassDependencies[2].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
	subpassDependencies[2].dependencyFlags = 0;

	//subpass 3 to external subpass transition, just the swap chain image is transitioned from color attachment optimal to present optimal
	//Now convert from VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL to VK_IMAGE_LAYOUT_PRESENT_SRC_KHR
	subpassDependencies[3].srcSubpass = 0;
	subpassDependencies[3].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;  //Here we want the transition after the color attchment output
	subpassDependencies[3].srcAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
	subpassDependencies[3].dstSubpass = VK_SUBPASS_EXTERNAL;
	subpassDependencies[3].dstStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
	subpassDependencies[3].dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;
	subpassDependencies[3].dependencyFlags = 0;

	//Vulkan will implictly do the transitions using the dependencies we just provided

	//add all the framebuffer attachments while creating the renderpass
	std::array<VkAttachmentDescription, 5> renderPassAttachments = { SwapChainColourAttachment ,colourAttachment, depthAttachment, accumAttachment, revealAttachment }; //Order is very important (same as in framebuffer)

	//The render pass create info struct
	VkRenderPassCreateInfo renderPassCreateInfo = {};
	renderPassCreateInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
	renderPassCreateInfo.attachmentCount = static_cast<uint32_t>(renderPassAttachments.size());
	renderPassCreateInfo.pAttachments = renderPassAttachments.data();
	renderPassCreateInfo.subpassCount = 3;
	renderPassCreateInfo.pSubpasses = subpasses.data();
	renderPassCreateInfo.dependencyCount = static_cast<uint32_t>(subpassDependencies.size());
	renderPassCreateInfo.pDependencies = subpassDependencies.data();

	VkResult result = vkCreateRenderPass(mainDevice.logicalDevice, &renderPassCreateInfo, nullptr, &renderPass);
	if (result != VK_SUCCESS) {
		throw std::runtime_error("Error creating render pass");
	}
}

void VulkanRenderer::createDescriptorSetLayout()
{

	//---------------UNIFORM BUFFERS----------------------//


	//--Create descriptor set layouts for the vp uniform buffers --//
	// UboViewProjection Binding Info
	VkDescriptorSetLayoutBinding vpLayoutBinding = {};
	vpLayoutBinding.binding = 0;											// Binding point in shader (designated by binding number in shader)
	vpLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;	// Type of descriptor (uniform, dynamic uniform, image sampler, etc)
	vpLayoutBinding.descriptorCount = 1;									// Number of descriptors for binding
	vpLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;				// Shader stage to bind to
	vpLayoutBinding.pImmutableSamplers = nullptr;							// For Texture: Can make sampler data unchangeable (immutable) by specifying in layout

	//// Model Binding Info
	//VkDescriptorSetLayoutBinding modelLayoutBinding = {};
	//modelLayoutBinding.binding = 1;											// Binding point in shader (designated by binding number in shader)
	//modelLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;	// Type of descriptor (uniform, dynamic uniform, image sampler, etc)
	//modelLayoutBinding.descriptorCount = 1;									// Number of descriptors for binding
	//modelLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;				// Shader stage to bind to
	//modelLayoutBinding.pImmutableSamplers = nullptr;							// For Texture: Can make sampler data unchangeable (immutable) by specifying in layout

	std::vector<VkDescriptorSetLayoutBinding> layoutBidnings = { vpLayoutBinding }; //Removed the model descriptor

	// Create Descriptor Set Layout with given bindings
	VkDescriptorSetLayoutCreateInfo layoutCreateInfo = {};
	layoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	layoutCreateInfo.bindingCount = static_cast<uint32_t>(layoutBidnings.size());		// Number of binding infos
	layoutCreateInfo.pBindings = layoutBidnings.data();		// Array of binding infos

	// Create Descriptor Set Layout
	VkResult result = vkCreateDescriptorSetLayout(mainDevice.logicalDevice, &layoutCreateInfo, nullptr, &descriptorSetLayout);
	if (result != VK_SUCCESS)
	{
		throw std::runtime_error("Error creating a Descriptor Set Layout!");
	}

	
	//----------------SAMPLERS-----------------------//


	//--Create descriptor set layouts for the image sampler --//
	//Texture binding info
	VkDescriptorSetLayoutBinding samplerLayoutBinding = {};
	samplerLayoutBinding.binding = 0;    //We will be using a different set so we can just reuse the binding 0 of the shader
	samplerLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; //The type o descriptor
	samplerLayoutBinding.descriptorCount = 1;
	samplerLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;  //The texture sampler is sent to the fragment shader not the vertex
	samplerLayoutBinding.pImmutableSamplers = nullptr;

	//Create a descriptor set layout with thte given bindings for the texture sampler
	VkDescriptorSetLayoutCreateInfo textureLayoutCreateinfo = {};
	textureLayoutCreateinfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	textureLayoutCreateinfo.bindingCount = 1; //Just the combined image sampler
	textureLayoutCreateinfo.pBindings = &samplerLayoutBinding;  //Create using the given binding

	result = vkCreateDescriptorSetLayout(mainDevice.logicalDevice, &textureLayoutCreateinfo, nullptr, &samplerSetLayout);
	if (result != VK_SUCCESS)
	{
		throw std::runtime_error("Error creating  sampler descriptor Set Layout!");
	}

	

	//-----------------INPUT ATTACHMENTS------------------------//


	//Create input attachments descriptor set layouts
	//colour input binding (opaque image)
	VkDescriptorSetLayoutBinding colourInputLayoutBinding = {};
	colourInputLayoutBinding.binding = 0; //Because it's a separate set
	colourInputLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT;
	colourInputLayoutBinding.descriptorCount = 1;
	colourInputLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

	//Depth input binding (maybe the depth image will be useful in the composition pass lol)
	VkDescriptorSetLayoutBinding depthInputLayoutBinding = {};
	depthInputLayoutBinding.binding = 1; //Because it's a separate set(set 0 of a different pipeline)
	depthInputLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT;
	depthInputLayoutBinding.descriptorCount = 1;
	depthInputLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

	//Another color input binding (accumulation image)
	VkDescriptorSetLayoutBinding accumColorInputLayoutBinding = {};
	accumColorInputLayoutBinding.binding = 2;
	accumColorInputLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT;
	accumColorInputLayoutBinding.descriptorCount = 1; 
	accumColorInputLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

	//Another color input binding (accumulation image)
	VkDescriptorSetLayoutBinding revealColorInputLayoutBinding = {};
	revealColorInputLayoutBinding.binding = 3;
	revealColorInputLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT;
	revealColorInputLayoutBinding.descriptorCount = 1;
	revealColorInputLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

	std::vector< VkDescriptorSetLayoutBinding> inputBindings = { colourInputLayoutBinding, depthInputLayoutBinding, accumColorInputLayoutBinding, revealColorInputLayoutBinding };

	//Create the set layout for the input attachments
	VkDescriptorSetLayoutCreateInfo inputLayoutCreateInfo = {};
	inputLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	inputLayoutCreateInfo.bindingCount = static_cast<uint32_t> (inputBindings.size());
	inputLayoutCreateInfo.pBindings = inputBindings.data();

	result = vkCreateDescriptorSetLayout(mainDevice.logicalDevice, &inputLayoutCreateInfo, nullptr, &inputSetLayout);
	if (result != VK_SUCCESS)
	{
		throw std::runtime_error("Error creating  input attachment descriptor set Layout!");
	}
}

void VulkanRenderer::createPushConstantRange()
{
	//Define push constant values, no createInfo needed
	pushConstantRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;  //Which stage of shader to send the data to
	pushConstantRange.offset = 0; //Offset into given data to pass into push constant
	pushConstantRange.size = sizeof(Model);  //Number of bytes to pass from the offset
}

void VulkanRenderer::createGraphicsPipeline()
{
	//Read the SPIR-V code of shaders
	auto vertexShaderCode = readFile("Shaders/vert.spv");
	auto fragmentShaderCode = readFile("Shaders/frag.spv");

	//Now Build shader modules for the graphics pipeline
	VkShaderModule vertexShaderModule = createShaderModule(vertexShaderCode);
	VkShaderModule fragmentShaderModule = createShaderModule(fragmentShaderCode);

	//--SHADER STAGE CREATE INFOS
	//---VERTEX SHADER STAGE
	VkPipelineShaderStageCreateInfo vertexShaderCreateInfo = {};
	vertexShaderCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;  //Just the stype
	vertexShaderCreateInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;							 //flag to indicate the shader type
	vertexShaderCreateInfo.module = vertexShaderModule;                                  //Shader module to be used
	vertexShaderCreateInfo.pName = "main";											     //entry point

	//--FRAGMENT SHADER STAGE
	VkPipelineShaderStageCreateInfo fragmentShaderCreateInfo = {};
	fragmentShaderCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;  //Just the stype
	fragmentShaderCreateInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;							 //flag to indicate the shader type
	fragmentShaderCreateInfo.module = fragmentShaderModule;                                  //Shader module to be used
	fragmentShaderCreateInfo.pName = "main";											     //entry point

	//Create array of shader stages because thats required by the pipeline create function
	VkPipelineShaderStageCreateInfo shaderStages[] = { vertexShaderCreateInfo ,fragmentShaderCreateInfo };										     //entry point

	//First we describe how the Vertex data (including normals, tex coordinates, and colors) is as a whole
	VkVertexInputBindingDescription bindingDescription = {};
	bindingDescription.binding = 0;  //Can bind multiple streams of data, here we specify which one
	bindingDescription.stride = sizeof(Vertex);  //The stride (the size of the vertex(not just the position))
	bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;  //How to move between data after each vertex
																 //VK_VERTEX_INPUT_RATE_VERTEX means that we want to move on to the next vertex
																 //VK_VERTEX_INPUT_RATE_INSTANCE means that we want to move on to the same vertex in the next instance(object instance, just like in unity)

	//Now we describe how an attribute of a vertex is defined within the vertex (for each attribute)
	std::array<VkVertexInputAttributeDescription, 3> attributeDescriptions;   //Only one vertex attribute for now

	//Position attribute
	attributeDescriptions[0].binding = 0;    //The binding (should be same as above)
	attributeDescriptions[0].location = 0;   //Because we send in the position at location = 0 in the vertex shader
	attributeDescriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT; //Because that's what a vec3 is made out of
	attributeDescriptions[0].offset = offsetof(Vertex, pos);   //Offset of the attribute in the vertex, here we can use offsetof because the vertex is a struct

	// Colour Attribute
	attributeDescriptions[1].binding = 0;
	attributeDescriptions[1].location = 1;
	attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
	attributeDescriptions[1].offset = offsetof(Vertex, col);

	// Texture Attribute
	attributeDescriptions[2].binding = 0;
	attributeDescriptions[2].location = 2;
	attributeDescriptions[2].format = VK_FORMAT_R32G32_SFLOAT;
	attributeDescriptions[2].offset = offsetof(Vertex, tex);

	VkPipelineVertexInputStateCreateInfo vertexInputCreateInfo = {};
	vertexInputCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
	vertexInputCreateInfo.vertexBindingDescriptionCount = 1;
	vertexInputCreateInfo.pVertexBindingDescriptions = &bindingDescription; //Data spacing, stride etc
	vertexInputCreateInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());
	vertexInputCreateInfo.pVertexAttributeDescriptions = attributeDescriptions.data(); //Data format and where to bind in the shader etc

	//--INPUT ASSEMBLY-- //
	VkPipelineInputAssemblyStateCreateInfo inputAssemblyCreateInfo = {};
	inputAssemblyCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
	inputAssemblyCreateInfo.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;  //primimtive type to assemble the vertices as
	inputAssemblyCreateInfo.primitiveRestartEnable = VK_FALSE;               //Setting this to true will allow overriding of strip topology to start new primitives

	//--VIEWPORT and SCISSOR-- //
	VkViewport viewPort = {};
	viewPort.x = 0.f;                                           //X start co ordinate
	viewPort.y = 0.f;                                           //Y start co oridinate
	viewPort.width = (float)swapChainExtent.width;              //width of viewport
	viewPort.height = (float)swapChainExtent.height;            //Height of viewport
	viewPort.minDepth = 0.f;                                    //minframebuffer depth
	viewPort.maxDepth = 1.f;                                    //Max framebuffer depth

	//create a sciccor to cut off parts of the viewport
	VkRect2D scissor = {};
	scissor.offset = { 0, 0 };                                  //Offset to use region from
	scissor.extent = swapChainExtent;                           //Extent to describe region to use, starting at offset

	VkPipelineViewportStateCreateInfo viewportStateCreateInfo = {};
	viewportStateCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
	viewportStateCreateInfo.viewportCount = 1;
	viewportStateCreateInfo.pViewports = &viewPort;
	viewportStateCreateInfo.scissorCount = 1;
	viewportStateCreateInfo.pScissors = &scissor;

	// -- DYNAMIC STATES -- (Leaving out for now)
	// Dynamic states to enable
	//std::vector<VkDynamicState> dynamicStateEnables;
	//dynamicStateEnables.push_back(VK_DYNAMIC_STATE_VIEWPORT);	// Dynamic Viewport : Can resize in command buffer with vkCmdSetViewport(commandbuffer, 0, 1, &viewport);
	//dynamicStateEnables.push_back(VK_DYNAMIC_STATE_SCISSOR);	// Dynamic Scissor	: Can resize in command buffer with vkCmdSetScissor(commandbuffer, 0, 1, &scissor);

	//// Dynamic State creation info
	//VkPipelineDynamicStateCreateInfo dynamicStateCreateInfo = {};
	//dynamicStateCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
	//dynamicStateCreateInfo.dynamicStateCount = static_cast<uint32_t>(dynamicStateEnables.size());
	//dynamicStateCreateInfo.pDynamicStates = dynamicStateEnables.data();

	//-- RASTERIZER -- //
	VkPipelineRasterizationStateCreateInfo rasterizerCreateInfo = {};
	rasterizerCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
	rasterizerCreateInfo.depthClampEnable = VK_FALSE; //Needs additional GPU feature to be enabled   //Change if fragments beyond far plane are clipped, or clamped to plane
	rasterizerCreateInfo.rasterizerDiscardEnable = VK_FALSE;  //If this is set to true, nothing will be rasterized, useful when you want to use shaders for toher stuff than drawing to the framebuffer
	rasterizerCreateInfo.polygonMode = VK_POLYGON_MODE_FILL;//Needs additional GPU feature to be enabled if anything other than FILL is used  //Because we want to color in the interior of the triangle too
	rasterizerCreateInfo.lineWidth = 1.0f;                  //Line thickness when drawn
	rasterizerCreateInfo.cullMode = VK_CULL_MODE_BACK_BIT; //Which face of a tri to color
	rasterizerCreateInfo.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE; //Winding to determine which side is front
	rasterizerCreateInfo.depthBiasEnable = VK_FALSE;  //Whether to add depth bias to fragments to fragments(good for shadowMapping, to stop acne)

	//-- MULTISAMPLING --//
	VkPipelineMultisampleStateCreateInfo multisamplingCreateInfo = {};
	multisamplingCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
	multisamplingCreateInfo.sampleShadingEnable = VK_FALSE;    //Multisampling disabled(multisampling is like supersamplping but for edges only, so textures get just one sample)
	multisamplingCreateInfo.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;  //Number of samples to use per fragment

	// -- BLENDING -- //
	// Blending decides how to blend a new colour being written to a fragment, with the old value

	// Blend Attachment State (how blending is handled)
	VkPipelineColorBlendAttachmentState colourState = {};
	colourState.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT	// Colours to apply blending to
		| VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
	colourState.blendEnable = VK_TRUE;													// Enable blending

	// Blending uses equation: (srcColorBlendFactor * new colour) colorBlendOp (dstColorBlendFactor * old colour)
	colourState.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
	colourState.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
	colourState.colorBlendOp = VK_BLEND_OP_ADD;

	// Summarised: (VK_BLEND_FACTOR_SRC_ALPHA * new colour) + (VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA * old colour)
	//			   (new colour alpha * new colour) + ((1 - new colour alpha) * old colour)

	colourState.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
	colourState.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
	colourState.alphaBlendOp = VK_BLEND_OP_ADD;
	// Summarised: (1 * new alpha) + (0 * old alpha) = new alpha

	VkPipelineColorBlendStateCreateInfo colourBlendingCreateInfo = {};
	colourBlendingCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
	colourBlendingCreateInfo.logicOpEnable = VK_FALSE;				// Alternative to calculations is to use logical operations
	colourBlendingCreateInfo.attachmentCount = 1;
	colourBlendingCreateInfo.pAttachments = &colourState;

	// -- PIPELINE LAYOUT --//   -->The data we want to pass into the pipeline
	std::array<VkDescriptorSetLayout, 2> descriptorSetLayouts = { descriptorSetLayout , samplerSetLayout };

	VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = {};
	pipelineLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
	pipelineLayoutCreateInfo.setLayoutCount = static_cast<uint32_t>(descriptorSetLayouts.size());
	pipelineLayoutCreateInfo.pSetLayouts = descriptorSetLayouts.data();   //Array of layoutas of all the descriptor sets we want to send to the shaders
	pipelineLayoutCreateInfo.pushConstantRangeCount = 1;
	pipelineLayoutCreateInfo.pPushConstantRanges = &pushConstantRange; //Array of all the push constant ranges(i think only one push constant block(can contain multiple constants) can be sent into one shader anyway)

	//pipeline layout object needs to be created and passed, it's not just a struct
	//Create Pipeline layout
	VkResult result = vkCreatePipelineLayout(mainDevice.logicalDevice, &pipelineLayoutCreateInfo, nullptr, &pipelineLayout);
	if (result != VK_SUCCESS) {
		throw std::runtime_error("Error creating pipeline layout");
	}

	//--DEPTH STENCIL TESTING --//
	VkPipelineDepthStencilStateCreateInfo depthStencilCreateInfo = {};
	depthStencilCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
	depthStencilCreateInfo.depthTestEnable = VK_TRUE;   //We want to test if the current fragment should be drawn
	depthStencilCreateInfo.depthWriteEnable = VK_TRUE;  //We want to update the depth image if we find a new closest object for the fragment
	depthStencilCreateInfo.depthCompareOp = VK_COMPARE_OP_LESS;   //We want to draw if depth is LESS than the closest value
	depthStencilCreateInfo.depthBoundsTestEnable = VK_FALSE;   //We can use bounds for depth testing (does the depth exist between two bounds? is yes then draw)
	depthStencilCreateInfo.stencilTestEnable = VK_FALSE;		//Not using depth stencil

	///--FINALLY CREATE GRAPHICS PIPELINE--///
	VkGraphicsPipelineCreateInfo pipelineCreateInfo = {};
	pipelineCreateInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
	pipelineCreateInfo.stageCount = 2;   //The number od shader stages
	pipelineCreateInfo.pStages = shaderStages;    //The shader stages array
	pipelineCreateInfo.pVertexInputState = &vertexInputCreateInfo;
	pipelineCreateInfo.pInputAssemblyState = &inputAssemblyCreateInfo;
	pipelineCreateInfo.pViewportState = &viewportStateCreateInfo;
	pipelineCreateInfo.pDynamicState = nullptr;
	pipelineCreateInfo.pRasterizationState = &rasterizerCreateInfo;
	pipelineCreateInfo.pMultisampleState = &multisamplingCreateInfo;
	pipelineCreateInfo.pColorBlendState = &colourBlendingCreateInfo;
	pipelineCreateInfo.pDepthStencilState = &depthStencilCreateInfo;
	pipelineCreateInfo.layout = pipelineLayout;
	pipelineCreateInfo.renderPass = renderPass;    //This specifies which render pass the pipeline will be used by
	pipelineCreateInfo.subpass = 0;                //Index of the subpass of the render pass to use with the pipeline

	pipelineCreateInfo.basePipelineHandle = VK_NULL_HANDLE;  //Used when creating based off another pieline
	pipelineCreateInfo.basePipelineIndex = -1;               //Index of the pipeline being created to derive from (when we create multiple at once)

	result = vkCreateGraphicsPipelines(mainDevice.logicalDevice, VK_NULL_HANDLE, 1, &pipelineCreateInfo, nullptr, &graphicsPipeline);
	if (result != VK_SUCCESS) {
		throw std::runtime_error("Error creating opaque graphics pipeline");
	}

	//Destroy shader modules. because they are no longer needed after the pipeline is created
	vkDestroyShaderModule(mainDevice.logicalDevice, fragmentShaderModule, nullptr);
	vkDestroyShaderModule(mainDevice.logicalDevice, vertexShaderModule, nullptr);




	//CREATE TRANSLUCENT PASS PIPELINE
	//translucent geometry pipeline

	//disabling backface culling for the translucency subpass
	rasterizerCreateInfo.cullMode = VK_CULL_MODE_NONE;

	auto translucentVertexShaderCode = readFile("Shaders/translucent_vert.spv");
	auto translucentFragmentShaderCode = readFile("Shaders/translucent_frag.spv");

	//Build shaders
	VkShaderModule translucentVertexShaderModule = createShaderModule(translucentVertexShaderCode);
	VkShaderModule translucentFragmentShaderModule = createShaderModule(translucentFragmentShaderCode);

	//set the new shaders in the pipeline shader stages
	vertexShaderCreateInfo.module = translucentVertexShaderModule;
	fragmentShaderCreateInfo.module = translucentFragmentShaderModule;

	VkPipelineShaderStageCreateInfo translucentShaderStages[] = { vertexShaderCreateInfo, fragmentShaderCreateInfo };

	// Don't want to write to depth buffer
	depthStencilCreateInfo.depthWriteEnable = VK_FALSE;
	
	
	//and there are 2 color attachments in this subpass the accumulation color buffer image, and the revealage color buffer image
	colourBlendingCreateInfo.attachmentCount = 2;
	//we need different blend states for each of the 2 attachments
	//the first one can be reused, but we need a new one for the revealage anyway
	//the accumulation image needs to be set to GL_ONE, GL_ONE and the revealage needs GL_ZERO and GL_ONE_MINUS_SRC_ALPHA
	
	VkPipelineColorBlendAttachmentState colourStateAccumulation= {};
	colourStateAccumulation.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT
		| VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
	colourStateAccumulation.blendEnable = VK_TRUE;
	colourStateAccumulation.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
	colourStateAccumulation.dstColorBlendFactor = VK_BLEND_FACTOR_ONE;
	colourStateAccumulation.colorBlendOp = VK_BLEND_OP_ADD;
	//T
	colourStateAccumulation.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
	colourStateAccumulation.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
	colourStateAccumulation.alphaBlendOp = VK_BLEND_OP_ADD;


	VkPipelineColorBlendAttachmentState colourStateRevealage = {};
	colourStateRevealage.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT
		| VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
	colourStateRevealage.blendEnable = VK_TRUE;
	colourStateRevealage.srcColorBlendFactor = VK_BLEND_FACTOR_ZERO;
	colourStateRevealage.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR;
	colourStateRevealage.colorBlendOp = VK_BLEND_OP_ADD;
	//The alpha values can't really blend because there is no alpha channel here
	colourStateRevealage.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
	colourStateRevealage.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
	colourStateRevealage.alphaBlendOp = VK_BLEND_OP_ADD;


	std::vector<VkPipelineColorBlendAttachmentState> colourStates = { colourStateAccumulation, colourStateRevealage };
	colourBlendingCreateInfo.pAttachments = colourStates.data();


	// Create new pipeline layout
	VkPipelineLayoutCreateInfo translucentPipelineLayoutCreateInfo = {};
	translucentPipelineLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
	translucentPipelineLayoutCreateInfo.setLayoutCount = static_cast<uint32_t>(descriptorSetLayouts.size());
	translucentPipelineLayoutCreateInfo.pSetLayouts = descriptorSetLayouts.data();
	translucentPipelineLayoutCreateInfo.pushConstantRangeCount = 1;
	translucentPipelineLayoutCreateInfo.pPushConstantRanges = &pushConstantRange;

	result = vkCreatePipelineLayout(mainDevice.logicalDevice, &translucentPipelineLayoutCreateInfo, nullptr, &translucentGeometryPipelineLayout);
	if (result != VK_SUCCESS)
	{
		throw std::runtime_error("Failed to create translucent Pipeline Layout!");
	}

	pipelineCreateInfo.pStages = translucentShaderStages;	// Update second shader stage list
	pipelineCreateInfo.layout = translucentGeometryPipelineLayout;	// Change pipeline layout for input attachment descriptor sets
	pipelineCreateInfo.subpass = 1; //the second subpass is the one where the translucent geometry is drawn to the accumulation texture

	// Create composition pipeline
	result = vkCreateGraphicsPipelines(mainDevice.logicalDevice, VK_NULL_HANDLE, 1, &pipelineCreateInfo, nullptr, &translucentGeometryPipeline);
	if (result != VK_SUCCESS)
	{
		throw std::runtime_error("Failed to create translucent Graphics Pipeline!");
	}

	// Destroy translucency shader modules, cuz the pipeline has already been created
	vkDestroyShaderModule(mainDevice.logicalDevice, translucentFragmentShaderModule, nullptr);
	vkDestroyShaderModule(mainDevice.logicalDevice, translucentVertexShaderModule, nullptr);




	// CREATE COMPOSITION PASS PIPELINE
	// composition pass shaders
	auto compositionVertexShaderCode = readFile("Shaders/second_vert.spv");
	auto compositionFragmentShaderCode = readFile("Shaders/second_frag.spv");

	// Build shaders
	VkShaderModule compositionVertexShaderModule = createShaderModule(compositionVertexShaderCode);
	VkShaderModule compositionFragmentShaderModule = createShaderModule(compositionFragmentShaderCode);

	// Set new shaders
	vertexShaderCreateInfo.module = compositionVertexShaderModule;
	fragmentShaderCreateInfo.module = compositionFragmentShaderModule;

	VkPipelineShaderStageCreateInfo compositionShaderStages[] = { vertexShaderCreateInfo, fragmentShaderCreateInfo };

	// No vertex data for composition pass
	vertexInputCreateInfo.vertexBindingDescriptionCount = 0;
	vertexInputCreateInfo.pVertexBindingDescriptions = nullptr;
	vertexInputCreateInfo.vertexAttributeDescriptionCount = 0;
	vertexInputCreateInfo.pVertexAttributeDescriptions = nullptr;

	// Don't want to write to depth buffer
	depthStencilCreateInfo.depthWriteEnable = VK_FALSE;
	//there is just on ecolor attachment in this subpass and that is the swapchain image
	colourBlendingCreateInfo.attachmentCount = 1;
	colourBlendingCreateInfo.pAttachments = &colourState;

	// Create new pipeline layout
	VkPipelineLayoutCreateInfo compositionPipelineLayoutCreateInfo = {};
	compositionPipelineLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
	compositionPipelineLayoutCreateInfo.setLayoutCount = 1;
	compositionPipelineLayoutCreateInfo.pSetLayouts = &inputSetLayout;
	compositionPipelineLayoutCreateInfo.pushConstantRangeCount = 0;
	compositionPipelineLayoutCreateInfo.pPushConstantRanges = nullptr;



	result = vkCreatePipelineLayout(mainDevice.logicalDevice, &compositionPipelineLayoutCreateInfo, nullptr, &compositionPipelineLayout);
	if (result != VK_SUCCESS)
	{
		throw std::runtime_error("Failed to create composition Pipeline Layout!");
	}

	pipelineCreateInfo.pStages = compositionShaderStages;	// Update second shader stage list
	pipelineCreateInfo.layout = compositionPipelineLayout;	// Change pipeline layout for input attachment descriptor sets
	pipelineCreateInfo.subpass = 2;						// Use third subpass

	// Create composition pipeline
	result = vkCreateGraphicsPipelines(mainDevice.logicalDevice, VK_NULL_HANDLE, 1, &pipelineCreateInfo, nullptr, &compositionPipeline);
	if (result != VK_SUCCESS)
	{
		throw std::runtime_error("Failed to create composition Graphics Pipeline!");
	}

	// Destroy composition shader modules
	vkDestroyShaderModule(mainDevice.logicalDevice, compositionFragmentShaderModule, nullptr);
	vkDestroyShaderModule(mainDevice.logicalDevice, compositionVertexShaderModule, nullptr);
}

void VulkanRenderer::createColourBufferImages()
{
	//Create the colour buffer images that the first subpass will output to

	//-------------OPAQUE COLOR IMAGE-------------------------------------//

	//First resize the vectors
	opaqueColorBufferImage.resize(swapChainImages.size());
	opaqueColourBufferImageMemory.resize(swapChainImages.size());
	opaqueColourBufferImageView.resize(swapChainImages.size());

	//Now get the supported format for the colour attachment
	VkFormat colourFormat = chooseSupportedFormat(
		{ VK_FORMAT_R8G8B8A8_UNORM }, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL  //Like the swapchain images earlier
	);

	//Now loop through the vector and create the images
	for (size_t i = 0; i < opaqueColorBufferImage.size(); i++) {
		//Create the image
		//in the usage flags we put colour attachmnet bit(subpass 1) and input attachment bit(subpass 3)
		opaqueColorBufferImage[i] = createImage(swapChainExtent.width, swapChainExtent.height, colourFormat, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &opaqueColourBufferImageMemory[i]);

		//create the image views
		opaqueColourBufferImageView[i] = createImageView(opaqueColorBufferImage[i], colourFormat, VK_IMAGE_ASPECT_COLOR_BIT);
	}

	//---------------------ACCUMULATION COLOUR IMAGE------------------------------------------//

	accumulationColourBufferImage.resize(swapChainImages.size());
	accumulationColourBufferImageView.resize(swapChainImages.size());
	accumulationColourBufferImageMemory.resize(swapChainImages.size());

	//Now get the supported format for the colour attachment
	colourFormat = chooseSupportedFormat(
		{ VK_FORMAT_R16G16B16A16_SFLOAT }, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL  //Like the swapchain images earlier
	);  //as can be seen we need atleats 16 bits for each of the accumulation texture channels

	//Now create the accum color images for each swapchain image
	for (size_t i = 0; i < accumulationColourBufferImage.size(); i++) {
		//create the image
		//the usage flags are again output attachment for subpass 2 and inupt for subpass 3
		accumulationColourBufferImage[i] = createImage(swapChainExtent.width, swapChainExtent.height, colourFormat, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &accumulationColourBufferImageMemory[i]);
		//now create the image views
		accumulationColourBufferImageView[i] = createImageView(accumulationColourBufferImage[i], colourFormat, VK_IMAGE_ASPECT_COLOR_BIT);
	}


	//---------------------REVEALAGE COLOUR IMAGE--------------------------------------------//

	revealageColourBufferImage.resize(swapChainImages.size());
	revealageColourBufferImageView.resize(swapChainImages.size());
	revealageColourBufferImageMemory.resize(swapChainImages.size());

	//Now get the supported format for the colour attachment
	colourFormat = chooseSupportedFormat(
		{ VK_FORMAT_R16_SFLOAT }, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL  //Like the swapchain images earlier
	);  //as can be seen we need atleats 16 bits for each of the accumulation texture channels

	//Now create the reveal color images for each swapchain image
	for (size_t i = 0; i < revealageColourBufferImage.size(); i++) {
		//create the image
		//the usage flags are again output attachment for subpass 2 and inupt for subpass 3
		revealageColourBufferImage[i] = createImage(swapChainExtent.width, swapChainExtent.height, colourFormat, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &revealageColourBufferImageMemory[i]);
		//now create the image views
		revealageColourBufferImageView[i] = createImageView(revealageColourBufferImage[i], colourFormat, VK_IMAGE_ASPECT_COLOR_BIT);
	}
}

void VulkanRenderer::createDepthBufferImage()
{

	//Resize the depthbuffer images vector to the number of swapchain images
	depthBufferImage.resize(swapChainImages.size());
	depthBufferImageView.resize(swapChainImages.size());
	depthBufferImageMemory.resize(swapChainImages.size());

	//First get the format
	VkFormat depthFormat = chooseSupportedFormat(
		{ VK_FORMAT_D24_UNORM_S8_UINT,  VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D32_SFLOAT },
		VK_IMAGE_TILING_OPTIMAL,
		VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT);

	//Storing the format for the render pass attachment struct
	depthBufferImageFormat = depthFormat;

	//Now create all the depth buffer images(one for each swapchain image)
	for (size_t i = 0; i < swapChainImages.size(); i++) {
		//Create the image
		depthBufferImage[i] = createImage(swapChainExtent.width, swapChainExtent.height, depthFormat, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &depthBufferImageMemory[i]); //We wont be modifying the depth buffer from the CPU

		//Also create an Image view for the depth image
		depthBufferImageView[i] = createImageView(depthBufferImage[i], depthFormat, VK_IMAGE_ASPECT_DEPTH_BIT); //We will only be looking at the depth bit of the image
	}
}

void VulkanRenderer::createFramebuffers()
{
	swapChainFrameBuffers.resize(swapChainImages.size());

	//Now we create a framebuffer for each swapchain image (we could just use one , but we will have to rebind --> we have to wait)
	for (size_t i = 0; i < swapChainFrameBuffers.size(); i++) {
		//we set the attachments(output) of the framebuffers to the imageViews we created

		//Keep in mind the render pass and the framebuffers have a 1:1 mapping of attachments i.e the first attachment in the renderPass outputs to the first attachmnet of the frameBuffer, the second to the second etc.

		//We add both the color, and the depth attachments (which are image views in this case)
		//The order is very important
		//We don't need another framebuffer for a different subpass, because we can just specify in the subpass rreferences which attachment ot output to
		std::array<VkImageView, 5> attachments = {
			swapChainImages[i].imageView,  //0   swapchain image
			opaqueColourBufferImageView[i], //1  opaque texture
			depthBufferImageView[i],  // 2  Now we use a separate depth buffer image for each framebuffer
			accumulationColourBufferImageView[i], //3   accumulation buffer image
			revealageColourBufferImageView[i] //4        revealage buffer image
		};

		VkFramebufferCreateInfo framebufferCreateInfo = {};
		framebufferCreateInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
		framebufferCreateInfo.renderPass = renderPass;   //Render pass the framebuffer  will be used with
		framebufferCreateInfo.attachmentCount = static_cast<uint32_t>(attachments.size());  //The number of attachments to the framebuffer
		framebufferCreateInfo.pAttachments = attachments.data();  //List of attachments (these are image views)
		framebufferCreateInfo.width = swapChainExtent.width;   //width of framebuffer
		framebufferCreateInfo.height = swapChainExtent.height; //Height of framebuffer
		framebufferCreateInfo.layers = 1; //The image view can look at multiple layers of an image, but we've got only one.

		//Now create the framebuffer
		VkResult result = vkCreateFramebuffer(mainDevice.logicalDevice, &framebufferCreateInfo, nullptr, &swapChainFrameBuffers[i]);
		if (result != VK_SUCCESS) {
			throw std::runtime_error("Error creating framebuffer");
		}
	}
}

void VulkanRenderer::createCommandPool()
{
	//Get indices of queue families
	QueueFamilyIndices indices = getQueueFamilyIndices(mainDevice.physicalDevice);

	VkCommandPoolCreateInfo poolInfo = {};
	poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
	poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT; //This means that the command buffers can now be reset, because we want to re-record commands(for push constants this is necessary)
	poolInfo.queueFamilyIndex = indices.graphicsFamily; //Queue family that buffers from this command pool will use

	//Create a command pool for the graphics queue family
	VkResult result = vkCreateCommandPool(mainDevice.logicalDevice, &poolInfo, nullptr, &graphicsCommandPool);
	if (result != VK_SUCCESS) {
		throw std::runtime_error("Error creating command pool");
	}
}

void VulkanRenderer::createCommandBuffers()
{
	//We want one command buffer for each frame buffer
	commandBuffers.resize(swapChainImages.size());

	//We build the create info structs
	VkCommandBufferAllocateInfo cbAllocInfo = {};
	cbAllocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
	cbAllocInfo.commandPool = graphicsCommandPool;   //Which pool to use
	cbAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;  //Primaries are executed by the queue, secondaries, are executed by other command buffers
	cbAllocInfo.commandBufferCount = static_cast<uint32_t> (commandBuffers.size()); //The number of ca=ommand buffers

	//Now we allocate the command buffers from the pool, and place handles in our array
	VkResult result = vkAllocateCommandBuffers(mainDevice.logicalDevice, &cbAllocInfo, commandBuffers.data());
	if (result != VK_SUCCESS) {
		throw std::runtime_error("Error allocating command buffers");
	}
	//We dont need to destroy command  buffers, they get deallocated when command pool is destroyed
}

void VulkanRenderer::createSynchronisation()
{
	imageAvailable.resize(MAX_FRAMES_IN_FLIGHT);
	renderFinished.resize(MAX_FRAMES_IN_FLIGHT);
	drawFences.resize(MAX_FRAMES_IN_FLIGHT);

	vkDeviceWaitIdle(mainDevice.logicalDevice);

	//Semaphore create infos
	VkSemaphoreCreateInfo semaphoreCreateInfo = {};
	semaphoreCreateInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

	//Fence create Info
	VkFenceCreateInfo fenceCreateInfo = {};
	fenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
	fenceCreateInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT; //Start with the fence open

	//We actually need no details for the semaphore lol, so we can just reuse the same struct for all semaphores
	for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
	{
		if (vkCreateSemaphore(mainDevice.logicalDevice, &semaphoreCreateInfo, nullptr, &imageAvailable[i]) != VK_SUCCESS ||
			vkCreateSemaphore(mainDevice.logicalDevice, &semaphoreCreateInfo, nullptr, &renderFinished[i]) != VK_SUCCESS ||
			vkCreateFence(mainDevice.logicalDevice, &fenceCreateInfo, nullptr, &drawFences[i]) != VK_SUCCESS) {
			throw std::runtime_error("Error creating a semaphore or Fence");
		}
	}
}

void VulkanRenderer::createTextureSampler()
{
	//Sampler create info
	VkSamplerCreateInfo samplerCreateInfo = {};
	samplerCreateInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
	samplerCreateInfo.magFilter = VK_FILTER_LINEAR;    //How to filter texture when it is magnified
	samplerCreateInfo.minFilter = VK_FILTER_LINEAR;    //How to filter texture when it is minified on screen
	samplerCreateInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;   //When the u value goes outside range(0 to 1) , just wrap
	samplerCreateInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;   //When the v value goes out of range --> wrap
	samplerCreateInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;   //When the w value goes out of range --> wrap
	samplerCreateInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;  //Wont actually be used because we just repeat the texture anyway, but we set it anyway as a formality
	samplerCreateInfo.unnormalizedCoordinates = VK_FALSE;			   //Normlized co-ordinates
	samplerCreateInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;      //As we move away the blending between two mip maps is linear
	samplerCreateInfo.mipLodBias = 0.f;								   //Level of detail bias for the mip level
	samplerCreateInfo.maxLod = 0.f;									   //Maximumn detail level to pick mip level
	samplerCreateInfo.minLod = 0.f;									   //Minimum LOD to pick mip level
	samplerCreateInfo.anisotropyEnable = VK_TRUE;					   //Enable anisotropic filtering for viewing at oblique angles
	samplerCreateInfo.maxAnisotropy = 16;                              //Anisotropy sample level

	//Now create the sampler
	VkResult result = vkCreateSampler(mainDevice.logicalDevice, &samplerCreateInfo, nullptr, &textureSampler);
	if (result != VK_SUCCESS) {
		throw std::runtime_error("Error creating texture sampler");
	}
}

void VulkanRenderer::createUniformBuffers()
{
	// ViewProjection buffer size
	VkDeviceSize vpBufferSize = sizeof(UboViewProjection);

	// Model buffer size
	//VkDeviceSize modelBufferSize = modelUniformAlignment * MAX_OBJECTS;  //No longer using a buffer for the model matrix

	// One uniform buffer for each image (and by extension, command buffer)
	vpUniformBuffer.resize(swapChainImages.size());
	vpUniformBufferMemory.resize(swapChainImages.size());
	//modelDUniformBuffer.resize(swapChainImages.size());
	//modelDUniformBufferMemory.resize(swapChainImages.size());

	// Create Uniform buffers
	for (size_t i = 0; i < swapChainImages.size(); i++)
	{
		createBuffer(mainDevice.physicalDevice, mainDevice.logicalDevice, vpBufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &vpUniformBuffer[i], &vpUniformBufferMemory[i]);

		//	createBuffer(mainDevice.physicalDevice, mainDevice.logicalDevice, modelBufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
		//		VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &modelDUniformBuffer[i], &modelDUniformBufferMemory[i]);
		//Now we are creating only static uniform buffers (for view and projection descriptors)
	}
}

void VulkanRenderer::createDescriptorPool()
{
	//---Create uniform descriptor pool ---//
	//Pool size type (one for vp and one for Model)
	VkDescriptorPoolSize vpPoolSize = {};
	vpPoolSize.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
	vpPoolSize.descriptorCount = static_cast<uint32_t>(vpUniformBuffer.size());  //One descriptor for each frame

	//VkDescriptorPoolSize modelPoolSize = {};   //Removed beacuse now model data is sent in via push constants
	//modelPoolSize.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
	//modelPoolSize.descriptorCount = static_cast<uint32_t>(modelDUniformBuffer.size());  //One descriptor for each frame

	//The descriptors will be put in the descriptor sets(not sure) so in the case of just one descriptor, we need to set both the descriptor sets and the descriptors to the number of frames
	//If we were sending two descriptors per set then we would have to have decsriptorSets = number of frames, and descriptors = twice the number of frames

	std::vector<VkDescriptorPoolSize> descriptorPoolSizes = { vpPoolSize }; //Model pool has been removed because the data isn't being sent by uniform buffers

	//The create Info
	VkDescriptorPoolCreateInfo poolCreateInfo = {};
	poolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
	poolCreateInfo.poolSizeCount = static_cast<uint32_t>(descriptorPoolSizes.size());   //Because we have only one poolSize struct
	poolCreateInfo.maxSets = static_cast<uint32_t>(swapChainImages.size());     //Specify the max number of descriptor sets in the pool --> one set for each image
	poolCreateInfo.pPoolSizes = descriptorPoolSizes.data();  //The pool sizes array

	VkResult result = vkCreateDescriptorPool(mainDevice.logicalDevice, &poolCreateInfo, nullptr, &descriptorPool);
	if (result != VK_SUCCESS) {
		throw std::runtime_error("Error creating descriptor pool");
	}

	//---Create sampler descriptor pool ---//
	//Texture sampler pool
	VkDescriptorPoolSize samplerPoolSize = {};
	samplerPoolSize.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;  //Both a sampler and an image
	samplerPoolSize.descriptorCount = MAX_OBJECTS;    //Onbe descriptor for each object (because we just assume that one object has just one texture)
	//The reason we do above step is because we create the images and their descriptor sets at the same time --

	VkDescriptorPoolCreateInfo samplerPoolCreateInfo = {};
	samplerPoolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
	samplerPoolCreateInfo.maxSets = MAX_OBJECTS;	//One set for each oject, and one set contains just one descriptor (the small limit was on the numer of uniform buffer descriptor sets not sampler image descriptor sets)
	samplerPoolCreateInfo.poolSizeCount = 1;        //We pass only one pool size
	samplerPoolCreateInfo.pPoolSizes = &samplerPoolSize;  //The pool size we want to use

	result = vkCreateDescriptorPool(mainDevice.logicalDevice, &samplerPoolCreateInfo, nullptr, &samplerDescriptorPool);
	if (result != VK_SUCCESS) {
		throw std::runtime_error("Error creating sampler descriptor pool");
	}

	//Create the descriptor pools for the input attachments

	//First for the colour input attachment (opaque image)
	VkDescriptorPoolSize colourInputPoolSize = {};
	colourInputPoolSize.type = VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT;
	colourInputPoolSize.descriptorCount = static_cast<uint32_t> (opaqueColourBufferImageView.size());

	//And now the depth input attachment
	VkDescriptorPoolSize depthPoolSize = {};
	depthPoolSize.type = VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT;
	depthPoolSize.descriptorCount = static_cast<uint32_t> (depthBufferImageView.size());

	//Another color attachment for the accumulation buffer input attachment
	VkDescriptorPoolSize accumColorInputPoolSize = {};
	accumColorInputPoolSize.type = VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT;
	accumColorInputPoolSize.descriptorCount = static_cast<uint32_t> (accumulationColourBufferImageView.size());  //one descriptor for each image

	//Another color attachment for the reveal color buffer input attachment
	VkDescriptorPoolSize revealColourPoolSize = {};
	revealColourPoolSize.type = VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT;
	revealColourPoolSize.descriptorCount = static_cast<uint32_t> (revealageColourBufferImageView.size());

	//Create an array of pool sizes
	std::vector<VkDescriptorPoolSize> inputPoolSizes = { colourInputPoolSize , depthPoolSize, accumColorInputPoolSize, revealColourPoolSize };

	//Create a descriptor pool for the attachments using the above sizes
	VkDescriptorPoolCreateInfo inputPoolCreateInfo = {};
	inputPoolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
	inputPoolCreateInfo.maxSets = swapChainImages.size();
	inputPoolCreateInfo.poolSizeCount = static_cast<uint32_t>(inputPoolSizes.size());
	inputPoolCreateInfo.pPoolSizes = inputPoolSizes.data();

	result = vkCreateDescriptorPool(mainDevice.logicalDevice, &inputPoolCreateInfo, nullptr, &inputDescriptorPool);
	if (result != VK_SUCCESS) {
		throw std::runtime_error("Error creating input attachment descriptor pool");
	}
}

void VulkanRenderer::createDescriptorSets()
{
	// Resize Descriptor Set list so one for every buffer
	descriptorSets.resize(swapChainImages.size());

	std::vector<VkDescriptorSetLayout> setLayouts(swapChainImages.size(), descriptorSetLayout);

	// Descriptor Set Allocation Info
	VkDescriptorSetAllocateInfo setAllocInfo = {};
	setAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
	setAllocInfo.descriptorPool = descriptorPool;									// Pool to allocate Descriptor Set from
	setAllocInfo.descriptorSetCount = static_cast<uint32_t>(swapChainImages.size());	// Number of sets to allocate
	setAllocInfo.pSetLayouts = setLayouts.data();									// Layouts to use to allocate sets (1:1 relationship)

	// Allocate descriptor sets (multiple)
	VkResult result = vkAllocateDescriptorSets(mainDevice.logicalDevice, &setAllocInfo, descriptorSets.data());
	if (result != VK_SUCCESS)
	{
		throw std::runtime_error("Error allocating Descriptor Sets!");
	}

	// Update all of descriptor set buffer bindings
	for (size_t i = 0; i < swapChainImages.size(); i++)
	{
		//VP DESCRIPTOR
		// Buffer info and data offset info
		VkDescriptorBufferInfo vpBufferInfo = {};
		vpBufferInfo.buffer = vpUniformBuffer[i];		// Buffer to get data from
		vpBufferInfo.offset = 0;						// Position of start of data
		vpBufferInfo.range = sizeof(UboViewProjection);				// Size of data

		// Data about connection between binding and buffer
		VkWriteDescriptorSet vpSetWrite = {};
		vpSetWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		vpSetWrite.dstSet = descriptorSets[i];								// Descriptor Set to update
		vpSetWrite.dstBinding = 0;											// Binding to update (matches with binding on layout/shader)
		vpSetWrite.dstArrayElement = 0;									// Index in array to update, because multiple descriptors can be allocated at once, from index dstArrayElement to dstArrayElement + descriptorCount
		vpSetWrite.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;		// Type of descriptor
		vpSetWrite.descriptorCount = 1;									// numnber of descriptors to update
		vpSetWrite.pBufferInfo = &vpBufferInfo;							// Information about buffer data to bind

		////MODEL DESCRIPTOR
		////Model buffer binding info
		//VkDescriptorBufferInfo modelBufferInfo = {};
		//modelBufferInfo.buffer = modelDUniformBuffer[i];
		//modelBufferInfo.offset = 0;						   //Position of start of data
		//modelBufferInfo.range = modelUniformAlignment;     // Size of data, remember this has to be a multiple of the minimum

		//// Data about connection between binding and buffer
		//VkWriteDescriptorSet modelSetWrite = {};
		//modelSetWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		//modelSetWrite.dstSet = descriptorSets[i];								// Descriptor Set to update
		//modelSetWrite.dstBinding = 1;											// Binding to update (matches with binding on layout/shader)
		//modelSetWrite.dstArrayElement = 0;										// Index in array to update, because multiple descriptors can be allocated at once, from index dstArrayElement to dstArrayElement + descriptorCount
		//modelSetWrite.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;		// Type of descriptor
		//modelSetWrite.descriptorCount = 1;										// numnber of descriptors to update
		//modelSetWrite.pBufferInfo = &modelBufferInfo;

		std::vector<VkWriteDescriptorSet> setWrites = { vpSetWrite };    //Now we are only linking the view, projection descriptors with the appropriate uniform buffers

		// Update the descriptor sets with new buffer/binding info
		vkUpdateDescriptorSets(mainDevice.logicalDevice, static_cast<uint32_t>(setWrites.size()), setWrites.data(), 0, nullptr);
	}
}

void VulkanRenderer::createInputDescriptorSets()
{
	//Resize our array to hold descriptors for each swapchain image
	inputDescriptorSets.resize(swapChainImages.size());

	//Fill array of layouts ready for set creation
	std::vector<VkDescriptorSetLayout> setLayouts(swapChainImages.size(), inputSetLayout); //Actually all are the same layout

	VkDescriptorSetAllocateInfo setAllocInfo = {};
	setAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
	setAllocInfo.descriptorPool = inputDescriptorPool;
	setAllocInfo.descriptorSetCount = static_cast<uint32_t>(swapChainImages.size());
	setAllocInfo.pSetLayouts = setLayouts.data();

	//Now allocate the descriptor sets in the pool
	VkResult result = vkAllocateDescriptorSets(mainDevice.logicalDevice, &setAllocInfo, inputDescriptorSets.data());
	if (result != VK_SUCCESS)
	{
		throw std::runtime_error("Error allocating  input attachment descriptor sets!");
	}

	//Now update the descriptor sets with the input attachments
	for (size_t i = 0; i < swapChainImages.size(); i++) {
		//Colour attachment (opaque) descriptor
		VkDescriptorImageInfo colourAttachmentDescriptor = {};
		colourAttachmentDescriptor.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;  //The format when it's read from
		colourAttachmentDescriptor.imageView = opaqueColourBufferImageView[i];
		colourAttachmentDescriptor.sampler = VK_NULL_HANDLE;		//Cant use sampler because the subpass with the same fragment

		//Colour attachment descriptor write
		VkWriteDescriptorSet colourWrite = {};
		colourWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		colourWrite.dstSet = inputDescriptorSets[i];
		colourWrite.dstBinding = 0;
		colourWrite.dstArrayElement = 0;
		colourWrite.descriptorType = VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT;
		colourWrite.descriptorCount = 1;
		colourWrite.pImageInfo = &colourAttachmentDescriptor;

		// Depth Attachment Descriptor
		VkDescriptorImageInfo depthAttachmentDescriptor = {};
		depthAttachmentDescriptor.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		depthAttachmentDescriptor.imageView = depthBufferImageView[i];
		depthAttachmentDescriptor.sampler = VK_NULL_HANDLE;

		// Depth Attachment Descriptor Write
		VkWriteDescriptorSet depthWrite = {};
		depthWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		depthWrite.dstSet = inputDescriptorSets[i];
		depthWrite.dstBinding = 1;  //Which binding in the set to write the descriptor to (1 for the depth attachment)
		depthWrite.dstArrayElement = 0;
		depthWrite.descriptorType = VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT;
		depthWrite.descriptorCount = 1;
		depthWrite.pImageInfo = &depthAttachmentDescriptor;

		//Another color attachment descriptor (accum Image)
		VkDescriptorImageInfo accumAttachmentDescriptor = {};
		accumAttachmentDescriptor.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;  //The format when it's read from
		accumAttachmentDescriptor.imageView = accumulationColourBufferImageView[i];
		accumAttachmentDescriptor.sampler = VK_NULL_HANDLE;

		//accum color attachment descriptor write
		VkWriteDescriptorSet accumDescriptorWrite = {};
		accumDescriptorWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		accumDescriptorWrite.dstSet = inputDescriptorSets[i];
		accumDescriptorWrite.dstBinding = 2; //binding 2 of the input descriptor set
		accumDescriptorWrite.dstArrayElement = 0;
		accumDescriptorWrite.descriptorType = VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT;
		accumDescriptorWrite.descriptorCount = 1;
		accumDescriptorWrite.pImageInfo = &accumAttachmentDescriptor;

		//and finally the revealage colour buffer image descriptpr write
		//first the descriptor 
		VkDescriptorImageInfo revealAttachmentDescriptor = {};
		revealAttachmentDescriptor.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;  //The format when it's read from
		revealAttachmentDescriptor.imageView = revealageColourBufferImageView[i];
		revealAttachmentDescriptor.sampler = VK_NULL_HANDLE;

		//then the descriptor write
		VkWriteDescriptorSet revealDescriptorWrite = {};
		revealDescriptorWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		revealDescriptorWrite.dstSet = inputDescriptorSets[i];
		revealDescriptorWrite.dstBinding = 3; //binding 3 of the input descriptor set is the revealage image, check the set layouts (of the input descriptor set --> inputDesciptorSetLayout) if confused
		revealDescriptorWrite.dstArrayElement = 0;
		revealDescriptorWrite.descriptorType = VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT;
		revealDescriptorWrite.descriptorCount = 1;
		revealDescriptorWrite.pImageInfo = &revealAttachmentDescriptor;


		// List of input descriptor set writes
		std::vector<VkWriteDescriptorSet> setWrites = { colourWrite, depthWrite, accumDescriptorWrite, revealDescriptorWrite };
		// Update descriptor sets
		vkUpdateDescriptorSets(mainDevice.logicalDevice, static_cast<uint32_t>(setWrites.size()), setWrites.data(), 0, nullptr);
	}
}

void VulkanRenderer::updateUniformBuffers(uint32_t imageIndex)
{
	//Copy vp data
	void* data;
	vkMapMemory(mainDevice.logicalDevice, vpUniformBufferMemory[imageIndex], 0, sizeof(UboViewProjection), 0, &data);
	memcpy(data, &uboViewProjection, sizeof(UboViewProjection));
	vkUnmapMemory(mainDevice.logicalDevice, vpUniformBufferMemory[imageIndex]);

	//Copy Model data, this will be different for each object
	//for (size_t i = 0; i < opaqueMeshes.size(); i++)
	//{
	//	Model* thisModel = (Model*)((uint64_t)modelTransferSpace + (i * modelUniformAlignment));  //Get pointer to the Model matrix of the patricular object from the Model transfer space
	//	*thisModel = opaqueMeshes[i].getModel();     //Copying over the mesh Model data to the Model transfer space
	//}

	//// Map the list of Model data
	////copy from Model transfer space to the data pointer, which will be mapped to the modelDUniformBufferMemory on the device
	//vkMapMemory(mainDevice.logicalDevice, modelDUniformBufferMemory[imageIndex], 0, modelUniformAlignment * opaqueMeshes.size(), 0, &data);
	//memcpy(data, modelTransferSpace, modelUniformAlignment * opaqueMeshes.size());
	//vkUnmapMemory(mainDevice.logicalDevice, modelDUniformBufferMemory[imageIndex]);
}

void VulkanRenderer::getPhysicalDevice()
{
	//first get the number of physical devices
	uint32_t physicalDeviceCount;
	vkEnumeratePhysicalDevices(instance, &physicalDeviceCount, nullptr);

	//Now get the list of physical devices
	std::vector<VkPhysicalDevice> physicalDevices(physicalDeviceCount);
	vkEnumeratePhysicalDevices(instance, &physicalDeviceCount, physicalDevices.data());
	printf("\n");
	for (size_t i = 0; i < physicalDeviceCount; i++) {
		if (checkPhysicalDeviceSuitable(physicalDevices[i])) {
			mainDevice.physicalDevice = physicalDevices[i];
			//picking the last physical device as of now.
		}
	}

	//We also need to save the minimum offset alignment for the dynamic uniform buffer, which is a property of the physical device
	VkPhysicalDeviceProperties physicalDeviceProperties;
	vkGetPhysicalDeviceProperties(mainDevice.physicalDevice, &physicalDeviceProperties);

	//minUniformBufferOffset = physicalDeviceProperties.limits.minUniformBufferOffsetAlignment;  //not needed anymore
}

bool VulkanRenderer::checkValidationLayerSupport()
{
	//First get the number of available layers in the instace
	uint32_t layerCount;
	vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

	//Now the layer properties(including layername) of all the available layers(These are just all the layers the instance supports, not just the ones we want to use)
	std::vector<VkLayerProperties> availableLayers(layerCount);
	vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

	//Output the layernames
	printf("\nThe following layers are supprted by the instance : \n");
	for (size_t i = 0; i < layerCount; i++) {
		printf("%s\n", availableLayers[i].layerName);
	}

	//Check if all the required layers are in the available layers
	for (const char* layerName : validationLayers) {
		bool layerFound = false;

		for (const auto& layerProperties : availableLayers) {
			if (strcmp(layerName, layerProperties.layerName) == 0) {
				layerFound = true;
				break;
			}
		}

		if (!layerFound) {
			return false;
		}
	}

	return true;
}

bool VulkanRenderer::checkDeviceExtensionSuppport(VkPhysicalDevice device)
{
	uint32_t extensionCount = 0;
	vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);

	//if no extensions found return failure
	if (extensionCount == 0) {
		return false;
	}

	//Retrieve supported device extensions
	std::vector< VkExtensionProperties> extensions(extensionCount);
	VkPhysicalDeviceProperties props;
	vkGetPhysicalDeviceProperties(device, &props);
	vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, extensions.data());

	//printing the  supported device extensions
	printf("\n%s supported Device Extensions :\n", props.deviceName);
	for (VkExtensionProperties supportedExtension : extensions) {
		printf("%s\n", supportedExtension.extensionName);
	}

	//checking if the required extensions are supported
	for (const char* deviceExtension : deviceExtensions) {
		bool hasExtension = false;
		for (VkExtensionProperties supportedExtension : extensions) {
			if (strcmp(deviceExtension, supportedExtension.extensionName) == 0) {
				hasExtension = true;
			}
		}

		if (!hasExtension) {
			return false;
		}
	}
	return true;
}

bool VulkanRenderer::checkInstanceExtensionSupport(const char** extensions, uint32_t extension_count)
{
	//get the number of supported extensions
	uint32_t supportedExtensionCount;
	vkEnumerateInstanceExtensionProperties(nullptr, &supportedExtensionCount, nullptr);
	printf("%d supported extensions : \n", supportedExtensionCount);

	//now get the list of the supported extensions
	std::vector<VkExtensionProperties> supportedExtensions(supportedExtensionCount);
	vkEnumerateInstanceExtensionProperties(nullptr, &supportedExtensionCount, supportedExtensions.data());
	for (size_t i = 0; i < supportedExtensionCount; i++) {
		printf("%s\n", supportedExtensions[i].extensionName);
	}

	//Now printing the required extensions
	printf("\nRequired extensions : \n");
	for (size_t i = 0; i < extension_count; i++) {
		printf("%s\n\n", extensions[i]);
	}

	//

	return false;
}

bool VulkanRenderer::checkPhysicalDeviceSuitable(VkPhysicalDevice physicalDevice)
{
	// Information about what the device can do (geo shader, tess shader, wide lines, etc)
	VkPhysicalDeviceFeatures deviceFeatures;
	vkGetPhysicalDeviceFeatures(physicalDevice, &deviceFeatures);

	//Check if required queue families are on the physical device
	QueueFamilyIndices indices = getQueueFamilyIndices(physicalDevice);

	//Check if the physical device supports the required extensions
	bool extensionSupported = checkDeviceExtensionSuppport(physicalDevice);

	//Check if alteast one format and presentaion mode is supported by the physical device surface
	bool swapChainValid = false;

	if (extensionSupported) {
		//Checki validity of the swapchain
		SwapChainDetails swapChainDetails = getSwapChainDetails(physicalDevice);
		swapChainValid = !swapChainDetails.formats.empty() && !swapChainDetails.presentationModes.empty();
	}

	//return if the device satisfies all the requirements
	return indices.isValid() && extensionSupported && swapChainValid && deviceFeatures.samplerAnisotropy && deviceFeatures.independentBlend;
}

void VulkanRenderer::populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& createInfo)
{
	createInfo = {};
	createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT; //just the structur type lol
	createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;  //General debug info is turned off
	createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT; //all messga types are enabled
	createInfo.pfnUserCallback = debugCallback; //our custom callback function we defined in VulkanValidation.h
	createInfo.pUserData = nullptr;  //allows user data to be passed to the callback in the 4th parameter i guess.
}

VkSurfaceFormatKHR VulkanRenderer::chooseBestSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& formats)
{
	//We are simply gonna use VK_FORMAT_R8G8B8A8_UNORM for the surface format
	//We choose the colorspace VK_COLOR_SPACE_SRGB_NONLINEAR_KHR (standard dynamic range)

	if (formats.size() == 1 && formats[0].format == VK_FORMAT_UNDEFINED) {
		//This actually means that all the surface formats are supported by the surface
		return { VK_FORMAT_R8G8B8A8_UNORM , VK_COLOR_SPACE_SRGB_NONLINEAR_KHR };
	}

	for (const auto& format : formats) {
		if (format.format == VK_FORMAT_R8G8B8A8_UNORM || format.format == VK_FORMAT_B8G8R8A8_UNORM && format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
			//In case all formats are not available, just return as soon as we find the one we want
			return format;
		}
	}

	return formats[0];
}

VkPresentModeKHR VulkanRenderer::chooseBestPresentMode(const std::vector<VkPresentModeKHR>& presentationModes)
{
	//We try to select mailbox presentation mode
	for (const auto& presentationMode : presentationModes) {
		if (presentationMode == VK_PRESENT_MODE_FIFO_KHR) {
			printf("\nChoosing presentaion mode : FIFO because without vsync the GPU heats up\n");
			return presentationMode;
		}
	}

	//If we couldn't find mailbox, we just return FIFO because the Vulkan Spec guarantees that FIFO is supported.
}

VkExtent2D VulkanRenderer::chooseSwapImageExtent(const VkSurfaceCapabilitiesKHR& surfaceCapabilities)
{
	if (surfaceCapabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
		//If the width is not set to the max value of uint32_t then we can just return the currentExtent(Because GLFW sets the size for us)
		return surfaceCapabilities.currentExtent;
	}
	else
	{
		//Get window size
		int width, height;
		glfwGetFramebufferSize(window, &width, &height);

		//Set new extent
		VkExtent2D newExtent = {};
		newExtent.width = static_cast<uint32_t> (width);
		newExtent.height = static_cast<uint32_t> (height);

		//Now we make sure the currentExtent sits in between the max and min Extents specified in the surface capabilities
		newExtent.width = std::max(surfaceCapabilities.minImageExtent.width, std::min(surfaceCapabilities.maxImageExtent.width, newExtent.width));
		newExtent.height = std::max(surfaceCapabilities.minImageExtent.height, std::min(surfaceCapabilities.maxImageExtent.height, newExtent.height));

		//After setting the appropriate Extent, we just return the new extent
		return newExtent;
	}
}

VkFormat VulkanRenderer::chooseSupportedFormat(const std::vector<VkFormat>& formats, VkImageTiling tiling, VkFormatFeatureFlags featureFlags)
{
	//Loop through all the available image formats and pick a compatible one

	for (VkFormat format : formats) {
		//Get the properties of the current format
		VkFormatProperties properties;
		vkGetPhysicalDeviceFormatProperties(mainDevice.physicalDevice, format, &properties);

		//Now we pick the optimal format for the given tiling
		//Check if the current formats properties have the required features for the given tiling
		if (tiling == VK_IMAGE_TILING_LINEAR && (properties.linearTilingFeatures & featureFlags) == featureFlags) {
			return format;
		}
		else if (tiling == VK_IMAGE_TILING_OPTIMAL && (properties.optimalTilingFeatures & featureFlags) == featureFlags) {
			return format;
		}
	}

	throw std::runtime_error("Format with required tiling and features could not be found");
}

VkImage VulkanRenderer::createImage(uint32_t width, uint32_t height, VkFormat format, VkImageTiling tiling, VkImageUsageFlags useFlags, VkMemoryPropertyFlags propFlags, VkDeviceMemory* imageMemory)
{
	// CREATE IMAGE
	// Image Creation Info
	VkImageCreateInfo imageCreateInfo = {};
	imageCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
	imageCreateInfo.imageType = VK_IMAGE_TYPE_2D;						// Type of image (1D, 2D, or 3D)
	imageCreateInfo.extent.width = width;								// Width of image extent
	imageCreateInfo.extent.height = height;								// Height of image extent
	imageCreateInfo.extent.depth = 1;									// Depth of image (just 1, no 3D aspect)
	imageCreateInfo.mipLevels = 1;										// Number of mipmap levels
	imageCreateInfo.arrayLayers = 1;									// Number of levels in image array
	imageCreateInfo.format = format;									// Format type of image
	imageCreateInfo.tiling = tiling;									// How image data should be "tiled" (arranged for optimal reading)
	imageCreateInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;			// Layout of image data on creation
	imageCreateInfo.usage = useFlags;									// Bit flags defining what image will be used for
	imageCreateInfo.samples = VK_SAMPLE_COUNT_1_BIT;					// Number of samples for multi-sampling
	imageCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;			// Whether image can be shared between queues

	// Create image
	VkImage image;
	VkResult result = vkCreateImage(mainDevice.logicalDevice, &imageCreateInfo, nullptr, &image);
	if (result != VK_SUCCESS)
	{
		throw std::runtime_error("Failed to create an Image!");
	}

	// CREATE MEMORY FOR IMAGE

	// Get memory requirements for a type of image
	VkMemoryRequirements memoryRequirements;
	vkGetImageMemoryRequirements(mainDevice.logicalDevice, image, &memoryRequirements);

	// Allocate memory using image requirements and user defined properties
	VkMemoryAllocateInfo memoryAllocInfo = {};
	memoryAllocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	memoryAllocInfo.allocationSize = memoryRequirements.size;
	memoryAllocInfo.memoryTypeIndex = findMemoryTypeIndex(mainDevice.physicalDevice, memoryRequirements.memoryTypeBits, propFlags);

	result = vkAllocateMemory(mainDevice.logicalDevice, &memoryAllocInfo, nullptr, imageMemory);
	if (result != VK_SUCCESS)
	{
		throw std::runtime_error("Failed to allocate memory for image!");
	}

	// Connect memory to image
	vkBindImageMemory(mainDevice.logicalDevice, image, *imageMemory, 0);

	return image;
}

VkImageView VulkanRenderer::createImageView(VkImage image, VkFormat format, VkImageAspectFlags aspectFlags)
{
	VkImageViewCreateInfo viewCreateInfo = {};
	viewCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
	viewCreateInfo.image = image;											// Image to create view for
	viewCreateInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;						// Type of image (1D, 2D, 3D, Cube, etc)
	viewCreateInfo.format = format;											// Format of image data
	viewCreateInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;			// Allows remapping of rgba components to other rgba values
	viewCreateInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
	viewCreateInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
	viewCreateInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;

	// Subresources allow the view to view only a part of an image
	viewCreateInfo.subresourceRange.aspectMask = aspectFlags;				// Which aspect of image to view (e.g. COLOR_BIT for viewing colour)
	viewCreateInfo.subresourceRange.baseMipLevel = 0;						// Start mipmap level to view from
	viewCreateInfo.subresourceRange.levelCount = 1;							// Number of mipmap levels to view
	viewCreateInfo.subresourceRange.baseArrayLayer = 0;						// Start array level to view from
	viewCreateInfo.subresourceRange.layerCount = 1;							// Number of array levels to view

	// Create image view and return it
	VkImageView imageView;
	VkResult result = vkCreateImageView(mainDevice.logicalDevice, &viewCreateInfo, nullptr, &imageView);
	if (result != VK_SUCCESS)
	{
		throw std::runtime_error("\nError creating  an Image View!\n");
	}

	return imageView;
}

VkShaderModule VulkanRenderer::createShaderModule(const std::vector<char>& code)
{
	//First build the create info struct
	VkShaderModuleCreateInfo shaderModuleCreateInfo = {};
	shaderModuleCreateInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
	shaderModuleCreateInfo.codeSize = code.size();										//size of code
	shaderModuleCreateInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());		//code body

	//Now create the shader module
	VkShaderModule shaderModule;
	VkResult result = vkCreateShaderModule(mainDevice.logicalDevice, &shaderModuleCreateInfo, nullptr, &shaderModule);
	if (result != VK_SUCCESS) {
		throw std::runtime_error("Error creating shader module");
	}

	//Finally return the shader module
	return shaderModule;
}

int VulkanRenderer::createTextureImage(std::string fileName, GeometryPass& texGeoPass)
{
	int width, height;
	VkDeviceSize imageSize;
	int nrChannels;

	stbi_uc* imageData = loadTextureFile(fileName, &width, &height, texGeoPass, &imageSize); //Load from the texture file

	//Now we have the data on the host memory, but we want it on the device memory, so we create a  staging buffer, and use transfer commands
	VkBuffer imageStagingBuffer;
	VkDeviceMemory imageStagingBufferMemory;
	createBuffer(mainDevice.physicalDevice, mainDevice.logicalDevice, imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &imageStagingBuffer, &imageStagingBufferMemory);

	//Copy over the data to the staging buffer
	void* data;
	vkMapMemory(mainDevice.logicalDevice, imageStagingBufferMemory, 0, imageSize, 0, &data);
	memcpy(data, imageData, static_cast<size_t>(imageSize));
	vkUnmapMemory(mainDevice.logicalDevice, imageStagingBufferMemory);

	//Free the host memory
	stbi_image_free(imageData);

	//Now create image to hold final texture
	VkImage texImage;    //Not a buffer, an image
	VkDeviceMemory texImageMemory;
	//Now we create the memory with the usage types specifying that it's the transfer destination(from the staging buffer), and also that it will be used for sampling
	texImage = createImage(width, height, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &texImageMemory);

	//Transition image layout so that it can be the destination of imageCopy
	transitionImageLayout(mainDevice.logicalDevice, graphicsQueue, graphicsCommandPool, texImage, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

	//Copy data to the image in deviceMemory
	copyImageBuffer(mainDevice.logicalDevice, graphicsQueue, graphicsCommandPool, imageStagingBuffer, texImage, width, height);

	// Transition image to be shader readable for shader usage
	transitionImageLayout(mainDevice.logicalDevice, graphicsQueue, graphicsCommandPool,
		texImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

	//Now that the VkImage actually has the texture data, we can add it to the vector for referencing later
	textureImages.push_back(texImage);
	textureImageMemory.push_back(texImageMemory);

	//Destroy staging buffers now that we have the texture data in the device local memory
	vkDestroyBuffer(mainDevice.logicalDevice, imageStagingBuffer, nullptr);
	vkFreeMemory(mainDevice.logicalDevice, imageStagingBufferMemory, nullptr);

	//Return the indeex to the texture, since its the last on ewe added we can just return size - 1; this is what we willl use to refer to the images
	return textureImages.size() - 1;
}

int VulkanRenderer::createTexture(std::string fileName, GeometryPass& texGeoPass)
{
	//First create and get a handle to the texture image, int the texture image array
	int textureImageLoc = createTextureImage(fileName, texGeoPass);

	//testing geometry pass assignment logic
	if (texGeoPass == GeometryPass::OPAQUE_PASS) {
		std::cout << "The mesh with texture : " << fileName << " will be drawn in the opaque pass" << std::endl;
	}
	else {
		std::cout << "The mesh with texture : " << fileName << " will be drawn in the translucency pass" << std::endl;
	}

	//Create an image view for the texture image
	VkImageView imageView = createImageView(textureImages[textureImageLoc], VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_ASPECT_COLOR_BIT);
	textureImageViews.push_back(imageView);

	//The texture images and their respective image views need to be aligned

	//Now we create the textures descriptor set, and get the index
	int descriptorLoc = createTextureDescriptor(imageView);

	return descriptorLoc;
}

int VulkanRenderer::createTextureDescriptor(VkImageView textureImage)
{
	VkDescriptorSet descriptorSet;

	//Descriptor set  allocation info
	VkDescriptorSetAllocateInfo setAllocInfo = {};
	setAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
	setAllocInfo.descriptorPool = samplerDescriptorPool;
	setAllocInfo.descriptorSetCount = 1;  //Just for one texture at a time, we will be calling this inside the createTexture itself
	setAllocInfo.pSetLayouts = &samplerSetLayout;  //Which layout to use while creating the descriptor sets

	//Allocate the descriptor sets
	VkResult result = vkAllocateDescriptorSets(mainDevice.logicalDevice, &setAllocInfo, &descriptorSet);
	if (result != VK_SUCCESS) {
		throw std::runtime_error("Error allocating texture descriptor sets");
	}

	//Texture image info
	VkDescriptorImageInfo imageInfo = {};
	imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;  //Image layout when in use
	imageInfo.imageView = textureImage;								   //Image to bind to the set
	imageInfo.sampler = textureSampler;                                //All the descriptor sets (one for each texture/object in this case) can use the same sampler, because nothing really changes

	//Descriptor write info
	VkWriteDescriptorSet descriptorWrite = {};
	descriptorWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	descriptorWrite.dstSet = descriptorSet;
	descriptorWrite.dstBinding = 0;  //Remember we used zero because it was a completely different set
	descriptorWrite.dstArrayElement = 0;
	descriptorWrite.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
	descriptorWrite.descriptorCount = 1;  //Just writing one descriptor to the set
	descriptorWrite.pImageInfo = &imageInfo;  //Bind the descriptor with the image and sampler information

	//Now update the new descriptor set because we have the write info
	vkUpdateDescriptorSets(mainDevice.logicalDevice, 1, &descriptorWrite, 0, nullptr);

	//add the descriptor set to the list
	samplerDescriptorSets.push_back(descriptorSet);

	//Finally return the location of the descriptor setr in the array of sampler descriptor sets
	return samplerDescriptorSets.size() - 1;
}

void VulkanRenderer::createMeshModel(std::string modelFile)
{
	Assimp::Importer importer;
	const aiScene* scene = importer.ReadFile(modelFile, aiProcess_Triangulate | aiProcess_FlipUVs | aiProcess_JoinIdenticalVertices);
	if (!scene) {
		throw std::runtime_error("Error loading a model" + modelFile + " ");
	}

	//Get vector of all materials with one-one ID placement
	std::vector<std::string> textureNames = MeshModel::LoadMaterials(scene);

	//Convert from our material list Id's to the descriptor array Id's
	std::vector<int> matToTex(textureNames.size());
	std::vector<GeometryPass> geoPasses(textureNames.size());
	// Loop over textureNames and create textures for them
	for (size_t i = 0; i < textureNames.size(); i++)
	{
		// If material had no texture, set '0' to indicate no texture, texture 0 will be reserved for a default texture
		if (textureNames[i].empty())
		{
			matToTex[i] = 0;
		}
		else
		{
			// Otherwise, create texture and set value to index of new texture
			//The matToTex vector has the locations of all the actual descriptors for this model
			matToTex[i] = createTexture(textureNames[i], geoPasses[i]);
		}

		//Now load in the meshes (if we load in the root node, then all the children will be loaded inn recursively)
	}

	std::vector<Mesh> modelMeshes = MeshModel::LoadNode(mainDevice.physicalDevice, mainDevice.logicalDevice, graphicsQueue, graphicsCommandPool,
		scene->mRootNode, scene, matToTex, geoPasses);

	// Create mesh model and add to list
	MeshModel meshModel = MeshModel(modelMeshes);//automatically separates the translucent and opaque meshes but it takes time
	modelList.push_back(meshModel);
}

void VulkanRenderer::recordCommands(uint32_t currentImage)
{
	//To record we first need to begin the buffer(begin means start recording)
	//Information about how to begin each command buffer
	VkCommandBufferBeginInfo bufferBeginInfo = {};
	bufferBeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	//bufferBeginInfo.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT; //Can two of the same command buffer be submitted to the queue simultaneously? here we set it to yes
	//The above flag was removed because , now that we are using fences the above scenario has been avoided anyway

	//Information about how to begin the render pass(only needed in graphical applicaations)
	VkRenderPassBeginInfo renderPassBeginInfo = {};
	renderPassBeginInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
	renderPassBeginInfo.renderPass = renderPass;
	renderPassBeginInfo.renderArea.offset = { 0, 0 };  //Start point of render pass(which part of image to run render pass on)
	renderPassBeginInfo.renderArea.extent = swapChainExtent;

	//We build the clear values (an array of clearValues)
	std::array<VkClearValue, 5> clearValues = {};
	clearValues[0].color = { 0.f, 0.f, 0.f, 0.0f };  //swapchain image clear color
	clearValues[1].color = { 0.f, 0.f, 0.f, 0.0f };  //opaque image clear color
	clearValues[2].depthStencil.depth = 1.0f;  //depth image clear color
	clearValues[3].color = {0.f, 0.f, 0.f, 0.f}; //accumulation buffer image clear color
	clearValues[4].color = { 1.f, 0.f, 0.f, 0.f }; //revealage buffer image clear color
	//Order is really important for the clear values

	renderPassBeginInfo.pClearValues = clearValues.data();    //List of clear values
	renderPassBeginInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());

	renderPassBeginInfo.framebuffer = swapChainFrameBuffers[currentImage];  //This is not a create info, it's a begin info, so we just change the value everytime

	//Start recording commands to the command buffer
	VkResult result = vkBeginCommandBuffer(commandBuffers[currentImage], &bufferBeginInfo); //All the command buffers use the same flag here
	if (result != VK_SUCCESS) {
		throw std::runtime_error("Error starting recordin to command buffer");
	}

	//Begin renderPass
	vkCmdBeginRenderPass(commandBuffers[currentImage], &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE); //The last argument specifies that all the commands will be primary commands

		//Bind compatible pipeline to be used in render pass
	vkCmdBindPipeline(commandBuffers[currentImage], VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);

	for (size_t j = 0; j < modelList.size(); j++) {
		MeshModel thisModel = modelList[j];
		vkCmdPushConstants(commandBuffers[currentImage], pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(Model), &thisModel.model);

		for (size_t k = 0; k < thisModel.getOpaqueMeshCount(); k++) {
			VkBuffer vertexBuffers[] = { thisModel.getOpaqueMesh(k)->getVertexBuffer() };					// Buffers to bind
			VkDeviceSize offsets[] = { 0 };												// Offsets into buffers being bound
			vkCmdBindVertexBuffers(commandBuffers[currentImage], 0, 1, vertexBuffers, offsets);	// Command to bind vertex buffer before drawing with them

			vkCmdBindIndexBuffer(commandBuffers[currentImage], thisModel.getOpaqueMesh(k)->getIndexBuffer(), 0, VK_INDEX_TYPE_UINT32);  //Only one index buffer can be bound at a time (even in openGl remember we had to bind vertex array and index buffer separately)

			//Dynamic offset amount
			//uint32_t dynamicOffset = static_cast<uint16_t>(modelUniformAlignment) * j; //Since j is the mesh number the first mesh will get offset = 0, the next one modelUniformAlignment, the next one 2*modelUniformAlignment and so on

			//Push constants are pushed not bound
			//Since we re record the commands for every mesh anyways
			//Since push constants do not use buffers they are technically slower, but we avoid the overhead from memory alloc and dealloc(in dynamic uniform buffers), which actually leads to better performance

			std::array<VkDescriptorSet, 2> descriptorSetGroup = { descriptorSets[currentImage],
					samplerDescriptorSets[thisModel.getOpaqueMesh(k)->getTexId()] };

			// Bind Descriptor Sets
			vkCmdBindDescriptorSets(commandBuffers[currentImage], VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout,
				0, static_cast<uint32_t>(descriptorSetGroup.size()), descriptorSetGroup.data(), 0, nullptr);
			//Remember when we created the pipeline we also specified which subpass will use it(basically one pipeline only works with one subpass)
			//Now execute the pipeline
			vkCmdDrawIndexed(commandBuffers[currentImage], thisModel.getOpaqueMesh(k)->getIndexCount(), 1, 0, 0, 0);
			//Above command will call pipeline for each vertex i guess(not sure), becasue Vertex shader needs to be called once for each Vertex
		}
	}

	//Start second subpass (translucent geometry)

	vkCmdNextSubpass(commandBuffers[currentImage], VK_SUBPASS_CONTENTS_INLINE);
	vkCmdBindPipeline(commandBuffers[currentImage], VK_PIPELINE_BIND_POINT_GRAPHICS, translucentGeometryPipeline);
	for (size_t j = 0; j < modelList.size(); j++) {
		
		
		MeshModel thisModel = modelList[j];
		vkCmdPushConstants(commandBuffers[currentImage], pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(Model), &thisModel.model);
		
		
		for (size_t k = 0; k < thisModel.getTranslucentMeshCount(); k++) {
			VkBuffer vertexBuffers[] = { thisModel.getTranslucentMesh(k)->getVertexBuffer() };					// Buffers to bind
			VkDeviceSize offsets[] = { 0 };												// Offsets into buffers being bound
			vkCmdBindVertexBuffers(commandBuffers[currentImage], 0, 1, vertexBuffers, offsets);	// Command to bind vertex buffer before drawing with them
			vkCmdBindIndexBuffer(commandBuffers[currentImage], thisModel.getTranslucentMesh(k)->getIndexBuffer(), 0, VK_INDEX_TYPE_UINT32);  //Only one index buffer can be bound at a time (even in openGl remember we had to bind vertex array and index buffer separately)
			std::array<VkDescriptorSet, 2> descriptorSetGroup = { descriptorSets[currentImage],
					samplerDescriptorSets[thisModel.getTranslucentMesh(k)->getTexId()] };
			vkCmdBindDescriptorSets(commandBuffers[currentImage], VK_PIPELINE_BIND_POINT_GRAPHICS, translucentGeometryPipelineLayout,
				0, static_cast<uint32_t>(descriptorSetGroup.size()), descriptorSetGroup.data(), 0, nullptr);
			vkCmdDrawIndexed(commandBuffers[currentImage], thisModel.getTranslucentMesh(k)->getIndexCount(), 1, 0, 0, 0);
			
		}
	}
	


	//Start third subpass
	vkCmdNextSubpass(commandBuffers[currentImage], VK_SUBPASS_CONTENTS_INLINE);   //No secondary command buffers
	vkCmdBindPipeline(commandBuffers[currentImage], VK_PIPELINE_BIND_POINT_GRAPHICS, compositionPipeline);
	vkCmdBindDescriptorSets(commandBuffers[currentImage], VK_PIPELINE_BIND_POINT_GRAPHICS, compositionPipelineLayout,
		0, 1, &inputDescriptorSets[currentImage], 0, nullptr);
	vkCmdDraw(commandBuffers[currentImage], 3, 1, 0, 0);
	//End renderPass
	vkCmdEndRenderPass(commandBuffers[currentImage]);

	//Stop recording commands
	result = vkEndCommandBuffer(commandBuffers[currentImage]);
	if (result != VK_SUCCESS) {
		throw std::runtime_error("Error stopping recording to command buffer");
	}
}

stbi_uc* VulkanRenderer::loadTextureFile(std::string fileName, int* width, int* height, GeometryPass& geoPass, VkDeviceSize* imageSize)
{
	// Number of channels image uses
	int channels;

	// Load pixel data for image
	std::string fileLoc = "Textures/" + fileName;
	stbi_uc* image = stbi_load(fileLoc.c_str(), width, height, &channels, STBI_rgb_alpha);

	if (!image)
	{
		throw std::runtime_error("Error loading a Texture file! (" + fileName + ")");
	}

	if (channels == 3)
	{
		geoPass = GeometryPass::OPAQUE_PASS;
	}
	else
	{
		geoPass = GeometryPass::OPAQUE_PASS;

		for (int i = 0; i < (*width) * (*height); i++)
		{
			int alphaIndex = 4 * i + 3;
			unsigned char currentPixelAlpha = image[alphaIndex];

			if ((currentPixelAlpha > 12) && (currentPixelAlpha < 242))
			{
				//std::cout << "here" <<std::endl;

				geoPass = GeometryPass::TRANSLUCENCY_PASS;

				break;
			}
		}
	}
	// Calculate image size using given and known data
	*imageSize = *width * *height * 4;

	return image;
}

QueueFamilyIndices VulkanRenderer::getQueueFamilyIndices(VkPhysicalDevice physicalDevice)
{
	QueueFamilyIndices indices;

	///first get the number of queue families for the given physical device
	uint32_t queueFamilyCount;
	vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, nullptr);

	///Now get the list of all the available queue families. Note: it's not guaranteed that a queue family conatains atleast one physical queue, strange huh?
	std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
	vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, queueFamilies.data());

	//now go through each family, check tyhe flags if it is the family we need, using flags
	int j = 0;
	for (VkQueueFamilyProperties i : queueFamilies) {
		if (i.queueCount > 0 && i.queueFlags & VK_QUEUE_GRAPHICS_BIT) {
			VkPhysicalDeviceProperties props;
			vkGetPhysicalDeviceProperties(physicalDevice, &props);
			printf("%s has a graphics queue, at index = %d\n", props.deviceName, j);
			indices.graphicsFamily = j;
		}

		VkBool32 presentationSupport = false;
		vkGetPhysicalDeviceSurfaceSupportKHR(physicalDevice, j, surface, &presentationSupport);
		if (i.queueCount > 0 && presentationSupport) {
			VkPhysicalDeviceProperties props;
			vkGetPhysicalDeviceProperties(physicalDevice, &props);
			printf("%s has a presentation queue, at index = %d\n", props.deviceName, j);
			indices.presentationFamily = j;
		}

		//if queue family indices are in a valid state, quit the loop
		if (indices.isValid()) {
			break;
		}
		j++;
	}

	return indices;
}

SwapChainDetails VulkanRenderer::getSwapChainDetails(VkPhysicalDevice device)
{
	SwapChainDetails swapChainDetails = {};

	//Retrieve the surface capabilities for the device
	vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &swapChainDetails.surfaceCapabilities);

	//retrieve the formats suppported by the surface
	uint32_t formatsCount;
	vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatsCount, nullptr);
	if (formatsCount != 0) {
		swapChainDetails.formats.resize(formatsCount);
		vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatsCount, swapChainDetails.formats.data());
	}

	//retrieve the presentation modes
	uint32_t presentationCount;
	vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentationCount, nullptr);
	if (presentationCount != 0) {
		swapChainDetails.presentationModes.resize(presentationCount);
	}
	vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentationCount, swapChainDetails.presentationModes.data());

	return swapChainDetails;
}

//void VulkanRenderer::allocateDynamicBufferTransferSpace()
//{
//	//Explaining with an example
//	//Lets say the minimum alignment is 32 i.e 00100000
//	//Now all the multiples of 32 have the last 5 bits as zeros, and the rest of the bits should bnot be all zeros. ex: 64 = 01000000 and 96 = 01100000 (notice how the last five bits are always zeros)
//	//So to check if the Model alignemt(or any descriptor we want to put in a dynamic uniform buffer for that matter) is we use the mask 11100000 (All ones above the index of 32)
//	//Then we perform an 'and' operation between the mask and the size of the ubomodel struct, to get the correct alignment (only if the ubomodel is larger than the minimum alignment, and is a multiple of the min value)
//	//By ubomodel alignment i mean the alignment we want to choose (it needs to fit the ubomodel), which is always a multiple of the min alignment that the device permits
//	//Ex: if the size of the ubomodel was 64 then 11100000 & 0100000 =  01000000 --> 64
//	//And if its smaller than the minimum alignment then we just get zero, so choose 32
//	//But what if its not a multiple of the min value(32 in this case)
//	//Ex: if it was 66 --> 01000010 & 1110000  = 01000000 --> we get back 64 which is less than what we need
//	//So instead we add 32 to the ubomodel size, then subtract 1, and then do the and operation
//	//Ex: 66 --> 01000010 after adding 32 -->01100010  now subtracting 1 we get  01100000, 'and' operation with mask gives 01100000 which is 96 which is what we want
//	//Ex 165 --> 10100101 after adding 32 --> 11000101, now subtracting 1 we get 11000100, 'and' opertaion with mask gives 11000000 which is 192 which is what we want
//	//To get the mask we just do ~(min -1)  this is obvious, just try it out
//	//The intuition between why we need to add 32 and subtract 1 --> if the size wasn't a multiple of 32 we can just add 32 and do the and opertaion, and we would have still gotten the same answer
//	//But if it was a multiple of 32 we would have and extra of 32 but we get rid of it by subtracting the 1(which will remove the bit with least index whose value is 1 )
//
//	// Calculate alignment of Model data
//	modelUniformAlignment = (sizeof(Model) + minUniformBufferOffset - 1)
//		& ~(minUniformBufferOffset - 1);
//
//	//Now we allocate memory for the ubo Model structs, but we allocate for the maximum number of objects we want to allow, because if we keeps reallocating everytime we want a new object, we will be losing performance(allocation is very slow)
//	modelTransferSpace = (Model*)_aligned_malloc(modelUniformAlignment * MAX_OBJECTS, modelUniformAlignment);
//}