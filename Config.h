#ifndef CONFIG_H
#define CONFIG_H

// OpenCL platform
constexpr unsigned int PLATEFORM = 1;
// Screen size:
constexpr int WIDTH = 1280;
constexpr int HEIGHT = 720;
// Fluid properties
constexpr auto DIFF_DENSITY = 0.000001f;
constexpr auto VISCO = 0.00001f;

// ---
constexpr int MEM_SIZE = WIDTH*HEIGHT;

#endif
