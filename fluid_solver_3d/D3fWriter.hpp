#ifndef D3F_WRITER_H
#define D3F_WRITER_H

#include <fstream>
#include <cstdint>
#include <iostream>
#include <limits>

/** Tell if we are on a little endian architecture */
bool is_little_endian()
{
	union {
		uint16_t i;
		char c[2];
	} const bint = {0x0100};
	return !(bint.c[0] == 1); 
}

/** Write in the stream T the value in binary */
template<typename T>
std::ostream& binary_write(std::ostream& stream, const T& value){
	return stream.write(reinterpret_cast<const char*>(&value), sizeof(T));
}

/** Write an uint16_t in big endian */
std::ostream& binary_write16big(std::ostream& stream, uint16_t v){
	if(is_little_endian()){
		v = ((v>>8) & 0xFF) | ((v & 0xFF) << 8);
	}
	return stream.write(reinterpret_cast<const char*>(&v), sizeof(v));
}

/** Write an uint32_t in big endian */
std::ostream& binary_write32big(std::ostream& stream, uint32_t v){
	if(is_little_endian()){
		v = (((v>>24) & 0xFF))  | (((v>>16) & 0xFF) << 8)  | (((v>>8) & 0xFF) << 16) | ((v & 0xFF) << 24);
	}
	return stream.write(reinterpret_cast<const char*>(&v), sizeof(v));
}

/** Write an uint32_t in big endian if the input is in little indian */
std::ostream& binary_write32big_unsafe(std::ostream& stream, uint32_t v){
	v = (((v>>24) & 0xFF))  | (((v>>16) & 0xFF) << 8)  | (((v>>8) & 0xFF) << 16) | ((v & 0xFF) << 24);
	return stream.write(reinterpret_cast<const char*>(&v), sizeof(v));
}

namespace D3fWriter
{
	/** Write a df3 file (density map) 
	* constitued by a header of three int16 (width x height x depth) 
	* followed by the density for each cell in 8, 16 or 32 bits in the (x,y,z) order */
	void exportdf3(const std::string & filename, const float* data, const unsigned int width, const unsigned int height, const unsigned int depth)
	{
		const unsigned int wh = width*height;
		const unsigned int volume = width*height*depth;

		std::ofstream out(filename.c_str(), std::ofstream::binary);
		if (!out.good()) {
			std::cout<<"cannot open "<<filename<<" =( "<<std::endl;
			delete [] data;
			out.close();
		}
		// Write the header
		uint16_t w = width, h=height, d=depth;
		binary_write16big(out,w);
		binary_write16big(out,h);
		binary_write16big(out,d);

		// Write the content
		if(is_little_endian()){
			for (unsigned int e = 0; e < volume; ++e) {
				const uint32_t val = (uint32_t)(data[e]/255.0f*std::numeric_limits<uint32_t>::max());
				binary_write32big_unsafe(out,val);
			}
		} else {
			for (unsigned int e = 0; e < volume; ++e) {
				const uint32_t val = (uint32_t)(data[e]/255.0f*std::numeric_limits<uint32_t>::max());
				binary_write(out,val);
			}
		}
		out.close();

		delete [] data;
	}
}

#endif // !D3F_WRITER_H