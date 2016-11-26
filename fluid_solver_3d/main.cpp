#include <iostream>
#include <SFML/Graphics.hpp>

#include "OpenCLFactory.hpp"
#include "Fluid3D.h"
#include "main.h"

const bool FULLSCREEN = false;

using namespace std;

/** Entry point of the application*/
int main() {
	
	auto device_context = OpenCLFactory::createContext();
	cl::Device & device = device_context.first;
	cl::Context & context = device_context.second;
	Fluid3D fluid(context, device);
	
	
	// constants:
	constexpr float initial_radius = 10.0f;
	constexpr float velocity_add = 0.01f;
	constexpr float mouse_wheel_increment = 1.0f;
	constexpr float mouse_pressure_increment = 10.0f;

	// sfml init
	constexpr auto window_style = (!FULLSCREEN) ? sf::Style::Default : sf::Style::Fullscreen;
	sf::RenderWindow window(sf::VideoMode(fluid.getWidth(), fluid.getHeight(), 32), "Fluid Solver", window_style);
	window.setFramerateLimit(60); // framerate limit 60Hz
	
	sf::Image image;
	sf::Texture texture;
	sf::Sprite sprite;
	image.create(fluid.getWidth(), fluid.getHeight());
	texture.loadFromImage(image);
	sprite.setTexture(texture);
	sprite.setPosition(0, 0);

	// fluid simulation solver init
	fluid.initialization();
	cl_uint8* pixelData = (cl_uint8*)image.getPixelsPtr();
	fluid.setDataImage(pixelData);

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
				if (event.key.code == sf::Keyboard::D) {
					fluid.save();
				}
			}
			if (event.type == sf::Event::MouseWheelMoved) {
				radius += mouse_wheel_increment*event.mouseWheel.delta;
			}
			if (sf::Mouse::isButtonPressed(sf::Mouse::Right)) {
				sf::Vector2f pos = window.mapPixelToCoords(sf::Mouse::getPosition(window));
				sf::Vector2f delta = pos - pos0;
				if(delta.x < 100.f && delta.y < 100.f) {
					fluid.addVelocity((int)pos.x, (int)pos.y, (int)delta.x, (int)delta.y, velocity_add, (int)radius);
				}
				pos0 = window.mapPixelToCoords(sf::Mouse::getPosition(window));
			}
		}
		if (sf::Mouse::isButtonPressed(sf::Mouse::Left)) {
			sf::Vector2f pos = window.mapPixelToCoords(sf::Mouse::getPosition(window));
			fluid.addPressure((int)pos.x, (int)pos.y, (int)radius, mouse_pressure_increment*dt);
		}
		// update the simulation
		fluid.update(dt);
		fluid.updateImage();
		// display 
		texture.update(image);
		window.clear(sf::Color::Black);
		window.draw(sprite);
		window.display();
	}
	return 0;
}