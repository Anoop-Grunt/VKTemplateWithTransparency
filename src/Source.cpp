#define STB_IMAGE_IMPLEMENTATION
#define GLM_FROCE_DEPTH_ZERO_TO_ONE

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <vector>
#include<iostream>
#include "VulkanRenderer.h"
#include "freecam.h"


GLFWwindow* window;
VulkanRenderer vulkanRenderer;

freecam primary_cam;

void MouseControlWrapper(GLFWwindow* window, double mouse_x, double mouse_y) {
	primary_cam.mouse_handler(window, mouse_x, mouse_y);
}

void ScrollControlWrapper(GLFWwindow* window, double x_disp, double y_disp) {
	primary_cam.scroll_handler(window, x_disp, y_disp);
}


void initWindow(std::string wName = "Main Window", const int width = 1920, const int height  = 1080) {
	glfwInit();
	glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
	glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
	window = glfwCreateWindow(width, height, wName.c_str(), nullptr, nullptr);
	glfwSetCursorPosCallback(window, MouseControlWrapper);
	glfwSetScrollCallback(window, ScrollControlWrapper);
	glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
}

int main(void)
{
	//create window
	initWindow();

	//Create Vulkan renderer instance
	if (vulkanRenderer.init(window) == EXIT_FAILURE) {
		return EXIT_FAILURE;
	}

	float angle = 0.0f;
	float deltaTime = 0.0f;
	float lastTime = 0.0f;
	
	
	
	vulkanRenderer.createMeshModel("Models/super_meatboy_free.obj");
	vulkanRenderer.createMeshModel("Models/pose.obj");
	vulkanRenderer.createMeshModel("Models/hollow kinght.obj");


	//loop until closed
	while (!glfwWindowShouldClose(window)) {
		glfwPollEvents();

		float now = glfwGetTime();
		deltaTime = now - lastTime;
		lastTime = now;

		angle += 10.0f * deltaTime;
		if (angle > 360.0f) { angle -= 360.0f; }

		glm::mat4 model1 = glm::mat4(1.f);
		model1 = glm::translate(model1, glm::vec3(-4.f, -2.f, 0.f));
		model1 = glm::rotate(model1, glm::radians(angle*10),glm::vec3(0.f, 1.f, 0.f));

		glm::mat4 model2 = glm::mat4(1.f);
		model2 = glm::translate(model2, glm::vec3(2.75f, -2.f, 0.f));
		model2 = glm::scale(model2, glm::vec3(0.25, 0.25, 0.25));
		model2 = glm::rotate(model2, glm::radians(angle * -10), glm::vec3(0.f, 1.f, 0.f));

		glm::mat4 model3 = glm::mat4(1.f);
		model3 = glm::translate(model3, glm::vec3(0.f, -2.f, -5.f));
		model3 = glm::rotate(model3, glm::radians(angle * -10), glm::vec3(0.f, 1.f, 0.f));

		primary_cam.input_handler(window);
		vulkanRenderer.updateCamera(primary_cam.view(), primary_cam.projection());

		vulkanRenderer.updateModel(2, model1);
		vulkanRenderer.updateModel(1, model2);
		vulkanRenderer.updateModel(0, model3);

		vulkanRenderer.draw();
	}
	glfwDestroyWindow(window);
	glfwTerminate();
	vulkanRenderer.cleanup();
	return 0;
}