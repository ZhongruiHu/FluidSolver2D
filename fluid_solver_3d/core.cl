__kernel void diffuse(__global float* dest, __global float* source, float a, float div, int width, int height, int depth)
{
	const int x = get_global_id(0);
	const int y = get_global_id(1);
	const int z = get_global_id(2);
	int wh = width*height;
	int index = x + y*width + z*wh;
	//if(x > 0 && x+1 < width && y > 0 && y+1 < height && z > 0 && z+1 < depth){
		float val = (source[index] 
		        + a*(dest[index-1]+dest[index+1]
					+dest[index-width]+dest[index+width]
					+dest[index-wh]+dest[index+wh]))/div;
		dest[index] = val;
	//}
}

__kernel void diffuse3D(__global float* field, __global float* source, float a, float div, int width, int height, int depth)
{
	const int x = get_global_id(0);
	const int y = get_global_id(1);
	const int z = get_global_id(2);
	int wh = width*height;
	int vindex = x + y*width + z*wh;
	//if(x > 0 && x+1 < width && y > 0 && y+1 < height && z > 0 && z+1 < depth){
		for(int i=0;i<3;++i){
			int index = 3*vindex+i;
			float val = (source[index] + 
					a*(  field[3*(vindex-1)+i]+field[3*(vindex+1)+i]
						+field[3*(vindex-width)+i]+field[3*(vindex+width)+i]
						+field[3*(vindex-wh)+i]+field[3*(vindex+wh)+i]))/div;
			field[index] = val;
		}
	//}
}

__kernel void drawScreen(__global float* field, __write_only image2d_t img_out, int width, int height)
{
	const int2 ipos = (int2)(get_global_id(0), get_global_id(1));
	const int4 pos = (int4)(get_global_id(0), get_global_id(1), 1,0);
	const int4 pos2 = (int4)(get_global_id(0), get_global_id(1), 1,0);
	float v = field[pos.x+pos.y*width+pos.z*width*height];
	float v2 = field[pos.x+pos.y*width+pos2.z*width*height];
	int r = (int)(v*200.0f);
	int g = (int)(v*56.0f);
	int b = (int)(v*10.f);
	if(r>255.0f) r = 255.0f;
	if(g>255.0f) g = 255.0f;
	if(b>255.0f) b = 255.0f;
	write_imageui(img_out, ipos, (uint4)(r, g, b, 255));
}

__kernel void drawScreen2(__global float* field, __write_only image2d_t img_out, int width, int height)
{
	const int2 ipos = (int2)(get_global_id(0), get_global_id(1));
	const int4 pos = (int4)(get_global_id(0), get_global_id(1), 1,0);
	float v1 = field[3*(pos.x+pos.y*width+pos.z*width*height)];
	float v2 = field[3*(pos.x+pos.y*width+pos.z*width*height)+1];
	float v3 = field[3*(pos.x+pos.y*width+pos.z*width*height)+2];
	//float4 v2 = read_imagef(img_in, samplerA, pos2);
	int r = (int)(fabs(v1)*100.0f);
	int g = (int)(fabs(v2)*100.0f);
	int b = (int)(fabs(v3)*100.0f);
	if(r>=255.0f) r = 255.0f;
	if(g>=255.0f) g = 255.0f;
	if(b>=255.0f) b = 255.0f;
	write_imageui(img_out, ipos, (uint4)(r, g, b, 255));
}

__kernel void addSource(__global float* field, int px, int py, int pz, float add, float radius, int width, int height)
{
	const int xpos = get_global_id(0);
	const int ypos = get_global_id(1);
	const int zpos = get_global_id(2);
	const float dx = (float)xpos - px;
	const float dy = (float)ypos - py;
	const float dz = (float)zpos - pz;
	const float d_sq = dx*dx+dy*dy;//+dz*dz;
	if (d_sq <= radius*radius) {
		float value = add;//*(1.0-sqrt(d_sq)/radius);
		int index = xpos+ypos*width+zpos*width*height;
		field[index] += value;
	}
}

__kernel void addSource3D(__global float* field, int px, int py, int pz, float add_x, float add_y, float radius, int width, int height)
{
	const int xpos = get_global_id(0);
	const int ypos = get_global_id(1);
	const int zpos = get_global_id(2);
	const float dx = (float)xpos - px;
	const float dy = (float)ypos - py;
	const float dz = (float)zpos - pz;
	const float d_sq = dx*dx+dy*dy;//+dz*dz;
	if (d_sq <= radius*radius) {
		float3 value = (float3)(add_x,add_y,0);//*(1.0-sqrt(d_sq)/radius);
		int index = xpos+ypos*width+zpos*width*height;
		field[3*index  ] += value.x;
		field[3*index+1] += value.y;
		field[3*index+2] = 0;
	}
}

__kernel void resetBuffer(__global float* field, int width, int height)
{
	const int x = get_global_id(0);
	const int y = get_global_id(1);
	const int z = get_global_id(2);
	int index = x + y*width + z*width*height;
	field[index] = 0.0f;
}

__kernel void resetBuffer3D(__global float* field, int width, int height)
{
	const int x = get_global_id(0);
	const int y = get_global_id(1);
	const int z = get_global_id(2);
	int index = x + y*width + z*width*height;
	for(int i=0;i<3;++i){
		field[3*index+i] = 0.0f;
	}
}

__kernel void advect(__global float* density_out, __global float* density, __global float* velocity,
		int width, int height, int depth, float dt)
{
	const int3 pos = (int3)(get_global_id(0), get_global_id(1), get_global_id(2));
	const float3 dt0 = dt*(float3)(width, height, depth);
	const int wh = width*height;
	const int index = pos.x + pos.y*width + pos.z*wh;
	float3 vvv = (float3)(velocity[3*index],velocity[3*index+1],velocity[3*index+2]);
	float3 dpos = (float3)(pos.x, pos.y, pos.z) - dt0*vvv;
	dpos.x = clamp(dpos.x, 0.5f, width + 0.5f);
	dpos.y = clamp(dpos.y, 0.5f, height + 0.5f);
	dpos.z = clamp(dpos.z, 0.5f, depth + 0.5f);
	int3 vi = (int3)(dpos.x, dpos.y, dpos.z);// integer final position
	float input000 = density[vi.x+ vi.y*width+ vi.z*wh];
	float input010 = density[vi.x+ (vi.y+1)*width+ vi.z*wh];
	float input100 = density[vi.x+1 + vi.y*width+ vi.z*wh];
	float input110 = density[vi.x+1 + (vi.y+1)*width+ vi.z*wh];
	float input001 = density[vi.x+ vi.y*width+ (vi.z+1)*wh];
	float input011 = density[vi.x+ (vi.y+1)*width+ (vi.z+1)*wh];
	float input101 = density[vi.x+1 + vi.y*width+ (vi.z+1)*wh];
	float input111 = density[vi.x+1 + (vi.y+1)*width+ (vi.z+1)*wh];
	
	float3 rest = dpos - (float3)(vi.x, vi.y, vi.z);
	float3 org = (float3)(1.0f,1.0f,1.0f) - rest; 
	float value = 
		  org.x *org.y *org.z *input000
		+ org.x *rest.y*org.z *input010
		+ rest.x*org.y *org.z *input100
		+ rest.x*rest.y*org.z *input110
		+ org.x *org.y *rest.z*input001
		+ org.x *rest.y*rest.z*input011
		+ rest.x*org.y *rest.z*input101
		+ rest.x*rest.y*rest.z*input111;
	density_out[pos.x+pos.y*width+pos.z*wh] = value;
}

__kernel void advect3D(__global float* velocity_out, __global float* velocity,
		int width, int height, int depth, float dt)
{
	const int3 pos = (int3)(get_global_id(0), get_global_id(1), get_global_id(2));
	const float3 dt0 = dt*(float3)(width, height, depth);
	const int wh = width*height;
	const int index = pos.x + pos.y*width + pos.z*wh;
	float3 vvv = (float3)(velocity[3*index],velocity[3*index+1],velocity[3*index+2]);
	float3 dpos = (float3)(pos.x, pos.y, pos.z) - dt0*vvv;
	dpos.x = clamp(dpos.x, 0.5f, width + 0.5f);
	dpos.y = clamp(dpos.y, 0.5f, height + 0.5f);
	dpos.z = clamp(dpos.z, 0.5f, depth + 0.5f);
	int3 vi = (int3)(dpos.x, dpos.y, dpos.z);// integer final position
	
	float3 rest = dpos - (float3)(vi.x, vi.y, vi.z);
	float3 org = (float3)(1.0f,1.0f,1.0f) - rest;
	
	for(int i=0;i<3;++i){
		float input000 = velocity[3*(vi.x   + vi.y*width     + vi.z*wh)+i];
		float input010 = velocity[3*(vi.x   + (vi.y+1)*width + vi.z*wh)+i];
		float input100 = velocity[3*(vi.x+1 + vi.y*width     + vi.z*wh)+i];
		float input110 = velocity[3*(vi.x+1 + (vi.y+1)*width + vi.z*wh)+i];
		float input001 = velocity[3*(vi.x   + vi.y*width     + (vi.z+1)*wh)+i];
		float input011 = velocity[3*(vi.x   + (vi.y+1)*width + (vi.z+1)*wh)+i];
		float input101 = velocity[3*(vi.x+1 + vi.y*width     + (vi.z+1)*wh)+i];
		float input111 = velocity[3*(vi.x+1 + (vi.y+1)*width + (vi.z+1)*wh)+i];
		

		float value = 
			  org.x *org.y *org.z *input000
			+ org.x *rest.y*org.z *input010
			+ rest.x*org.y *org.z *input100
			+ rest.x*rest.y*org.z *input110
			+ org.x *org.y *rest.z*input001
			+ org.x *rest.y*rest.z*input011
			+ rest.x*org.y *rest.z*input101
			+ rest.x*rest.y*rest.z*input111;
		velocity_out[3*index+i] = value;
	}
}

__kernel void project1(__global float* out,
	__global float* velocity, int width, int height, int depth) {

	const int xpos = get_global_id(0);
	const int ypos = get_global_id(1);
	const int zpos = get_global_id(2);
	const int wh = width*height;
	const int index = xpos + ypos*width + zpos*wh;
	const float hx = 1.0f/width;
	const float hy = 1.0f/height;
	const float hz = 1.0f/depth;
	
	float dr = velocity[3*(index+1)];
	float dl = velocity[3*(index-1)];
	float dd = velocity[3*(index+width)+1];
	float du = velocity[3*(index-width)+1];
	float dt = velocity[3*(index+wh)+2];
	float db = velocity[3*(index-wh)+2];

	float value = -0.5f*(hx*(dr - dl) + hy*(dd - du) + hz*(dt - db));

	out[xpos + ypos*width + zpos*wh] = value;
}

__kernel void project2(__global float* in,
	__global float* velocity, int width, int height, int depth)
{
	const int xpos = get_global_id(0);
	const int ypos = get_global_id(1);
	const int zpos = get_global_id(2);

	const int wh = width*height;
	const int index = xpos + ypos*width + zpos*wh;
	
	float dr = in[index+1];
	float dl = in[index-1];
	float dd = in[index+width];
	float du = in[index-width];
	float dt = in[index+wh];
	float db = in[index-wh];

	velocity[3*index+0] -= 0.5f*(dr - dl) * width;
	velocity[3*index+1] -= 0.5f*(dd - du) * height;
	velocity[3*index+2] -= 0.5f*(dt - db) * depth;
}
