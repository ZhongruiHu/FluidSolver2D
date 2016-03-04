#include <iostream>
#include <fstream>
#include <CL/cl.hpp>
#include <SFML/Graphics.hpp>
#include <math.h>

#include "FluidSolver2D.h"
#include "FluidSolver.h"
#include "Config.h"

using namespace std;

/** Entry point of the application*/
int main() {
	// constants:
	constexpr float initial_radius = 10.0f;
	constexpr float velocity_add = 0.01f;
	constexpr float mouse_wheel_increment = 1.0f;
	constexpr float mouse_pressure_increment = 10.0f;

	// sfml init
	sf::RenderWindow window(sf::VideoMode(WIDTH, HEIGHT, 32), "SFML Graphics", sf::Style::Default);
	//sf::RenderWindow window(sf::VideoMode(WIDTH, HEIGHT, 32), "SFML Graphics", sf::Style::Fullscreen);
	
	sf::Image image;
	sf::Texture texture;
	sf::Sprite sprite;
	image.create(WIDTH, HEIGHT);
	texture.loadFromImage(image);
	sprite.setTexture(texture);
	sprite.setPosition(0, 0);

	// fluid simulation solver init
	FluidSolver fluid;
	fluid.initialization();
	cl_uint8* pixelData = (cl_uint8*)image.getPixelsPtr();
	fluid.set_data_image(pixelData);

	// other variables
	sf::Clock deltaClock;
	sf::Vector2f pos0; // for the mouse
	float radius = initial_radius; // mouse radius

	// main loop
	while (window.isOpen()){
		sf::Time deltaTime = deltaClock.restart();
		const float dt = deltaTime.asSeconds();

		// Manage input
		sf::Event event;
		while (window.pollEvent(event)){
			if (event.type == sf::Event::Closed) {
				window.close();
			}
			if (event.type == sf::Event::KeyReleased) {
				if (event.key.code == sf::Keyboard::Escape) {
					window.close();
				}
				if (event.key.code == sf::Keyboard::Space) {
					fluid.reset();
				}
			}
			if (event.type == sf::Event::MouseWheelMoved) {
				radius += mouse_wheel_increment*event.mouseWheel.delta;
			}
			if (sf::Mouse::isButtonPressed(sf::Mouse::Right)) {
				sf::Vector2f pos = window.mapPixelToCoords(sf::Mouse::getPosition(window));
				sf::Vector2f delta = pos - pos0;
				if(delta.x < 100.f && delta.y < 100.f)
					fluid.add_velocity((int)pos.x, (int)pos.y, delta.x, delta.y, velocity_add, (int)radius);
				pos0 = window.mapPixelToCoords(sf::Mouse::getPosition(window));
			}
		}
		if (sf::Mouse::isButtonPressed(sf::Mouse::Left)) {
			sf::Vector2f pos = window.mapPixelToCoords(sf::Mouse::getPosition(window));
			fluid.add_pressure((int)pos.x, (int)pos.y, (int)radius, mouse_pressure_increment*dt);
		}
		// update the simulation
		fluid.update(dt);
		fluid.update_image();
		// display 
		texture.update(image);
		window.clear(sf::Color::Black);
		window.draw(sprite);
		window.display();
	}
	return 0;
}