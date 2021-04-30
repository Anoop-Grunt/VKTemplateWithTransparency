#include "freecam.h"

freecam::freecam()
	:cf_init(glm::vec3(0.f, 0.f, 0.f))
{
	cf_init.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
	cf_init.y = sin(glm::radians(pitch));
	cf_init.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch));
	CamFront = glm::normalize(cf_init);
}

freecam::~freecam()
{
}

void freecam::input_handler(GLFWwindow* window)
{
	float currentFrameTime = glfwGetTime();
	deltaTime = currentFrameTime - prevFrameTime;
	prevFrameTime = currentFrameTime;
	float speed = 5.f;
	glm::vec3 velocity_cr = speed * CamFront;
	glm::vec3 velocity_strafe_right = speed * (glm::cross(CamFront, worldUP));
	if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
		CamPos += velocity_cr * deltaTime;
	}
	if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
		CamPos -= velocity_cr * deltaTime;
	}
	if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
		CamPos += velocity_strafe_right * deltaTime;
	}
	if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
		CamPos -= velocity_strafe_right * deltaTime;
	}
	if (glfwGetKey(window, GLFW_KEY_1) == GLFW_PRESS) {
		glm::mat4 test = glm::mat4(1.f);
		test = glm::rotate(test, glm::radians(5.f), CamFront);
		glm::vec4 _ = glm::vec4(worldUP.x, worldUP.y, worldUP.z, 1.f);
		_ = test * _;
		worldUP = glm::vec3(_.x, _.y, _.z);
	}
	if (glfwGetKey(window, GLFW_KEY_3) == GLFW_PRESS) {
		glm::mat4 test = glm::mat4(1.f);
		test = glm::rotate(test, glm::radians(-5.f), CamFront);
		glm::vec4 _ = glm::vec4(worldUP.x, worldUP.y, worldUP.z, 1.f);
		_ = test * _;
		worldUP = glm::vec3(_.x, _.y, _.z);
	}
	if (glfwGetKey(window, GLFW_KEY_2) == GLFW_PRESS) {

		worldUP = glm::vec3(0.f, 1.f, 0.f);
		CamPos = glm::vec3(0.0f, 0.0f, 3.0f);
		CamFront = glm::vec3(0.0f, 0.0f, -1.0f);
		Target = glm::vec3(0.0f, 0.0f, 0.0f);
	}

	if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
		glfwSetWindowShouldClose(window, true);
}

void freecam::mouse_handler(GLFWwindow* window, double mouse_x, double mouse_y)
{
	{
		if (firstMouse)
		{
			win_x = mouse_x;
			win_y = mouse_y;
			firstMouse = false;
		}

		float mouse_x_disp = win_x - mouse_x;
		win_x = mouse_x;

		float mouse_y_disp = mouse_y - win_y;
		win_y = mouse_y;

		const float mouse_sensitivity = 0.015f;
		mouse_x_disp *= mouse_sensitivity;
		mouse_y_disp *= mouse_sensitivity;

		yaw -= mouse_x_disp;
		pitch -= mouse_y_disp;

		if (pitch > 85.0f)
			pitch = 85.0f;
		if (pitch < -85.0f)
			pitch = -85.0f;

		glm::vec3 cf;
		cf.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
		cf.y = sin(glm::radians(pitch));
		cf.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch));

		CamFront = glm::normalize(cf);
	}
}

void freecam::scroll_handler(GLFWwindow* window, double x_disp, double y_disp)
{
		if (field_of_view >= 1.0f && field_of_view <= 45.0f)
			field_of_view -= y_disp;
		if (field_of_view <= 1.0f)
			field_of_view = 1.0f;
		if (field_of_view >= 45.0f)
			field_of_view = 45.0f;
}



glm::mat4 freecam::view()
{
	glm::mat4 view = glm::mat4(1.0f);
	view = glm::lookAt(CamPos, CamFront + CamPos, worldUP);
	return view;

}

glm::mat4 freecam::projection()
{
	glm::mat4 projection = glm::mat4(1.0f);
	projection = glm::perspective(glm::radians(field_of_view), (float)1920.f / (float)1080.f, 0.1f, 100.0f);
	glm::mat4 model = glm::mat4(1.0f);
	model = glm::rotate(model, glm::radians(-55.0f), glm::vec3(1.0f, 0.0f, 0.0f));
	return projection;
}

glm::vec3 freecam::position()
{
	return CamPos;
}
