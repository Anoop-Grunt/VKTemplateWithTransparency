#pragma once
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <GLFW/glfw3.h>


class  freecam
{
public:
	 freecam();
	~ freecam();
	void input_handler(GLFWwindow* window);
	void mouse_handler(GLFWwindow* window, double mouse_x, double mouse_y);
	void scroll_handler(GLFWwindow* window, double x_disp, double y_disp);
	
	glm::mat4 view();
	glm::mat4 projection();
	glm::vec3 position();


private:
	glm::vec3 worldUP = glm::vec3(0.f, 1.f, 0.f);
	glm::vec3 CamPos = glm::vec3(0.0f, 0.0f, 12.0f);
	glm::vec3 CamFront = glm::vec3(0.0f, 0.0f, -1.0f);
	glm::vec3 Target = glm::vec3(0.0f, 0.0f, 0.0f);
	

	float deltaTime = 0.0f;
	float prevFrameTime = 0.0f;
	float win_x = 960;
	float win_y = 540;
	float yaw = -90.0f;
	float pitch = 0.f;
	float field_of_view = 45.f;
	glm::vec3 cf_init;
	bool firstMouse = true;


	

};

