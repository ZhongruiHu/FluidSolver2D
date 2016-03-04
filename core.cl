
const sampler_t samplerA = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
const sampler_t samplerB = CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_REPEAT | CLK_FILTER_LINEAR;

__kernel void diffuse(	__read_only image2d_t img_in,
						__write_only image2d_t img_out,
						__read_only image2d_t previous_in,
						float a, float div) {
	const int xpos = get_global_id(0);
	const int ypos = get_global_id(1);
	float4 dprev = read_imagef(previous_in, samplerA, (int2)(xpos,ypos));
	float4 dl = read_imagef(img_in, samplerA, (int2)(xpos-1, ypos));
	float4 dr = read_imagef(img_in, samplerA, (int2)(xpos+1, ypos));
	float4 du = read_imagef(img_in, samplerA, (int2)(xpos, ypos-1));
	float4 dd = read_imagef(img_in, samplerA, (int2)(xpos, ypos+1));
	float val = (dprev.x + a*(dl.x + dr.x + du.x + dd.x))/div;
	write_imagef(img_out, (int2)(xpos, ypos), (float4)(val,0,0,0));
}

__kernel void advect(__read_only image2d_t img_in,
	__write_only image2d_t img_out,
	__read_only image2d_t u,
	__read_only image2d_t v,
	float dt, int w, int h) {
	const int2 pos = (int2)(get_global_id(0), get_global_id(1));

	const float2 dt0 = dt*(float2)(w, h);
	float4 inputU = read_imagef(u, samplerA, pos);
	float4 inputV = read_imagef(v, samplerA, pos);
	float2 velocity = (float2)(inputU.x, inputV.x);
	float2 dpos = (float2)(pos.x, pos.y) - dt0*velocity.xy;
	clamp(dpos.x, 0.5f, w + 0.5f);
	clamp(dpos.y, 0.5f, h + 0.5f);
	int2 vi = (int2)(dpos.x, dpos.y);// cast to int
	float4 input00 = read_imagef(img_in, samplerA, (int2)(vi.x, vi.y));
	float4 input01 = read_imagef(img_in, samplerA, (int2)(vi.x, vi.y + 1));
	float4 input10 = read_imagef(img_in, samplerA, (int2)(vi.x + 1, vi.y));
	float4 input11 = read_imagef(img_in, samplerA, (int2)(vi.x + 1, vi.y + 1));
	float4 s;
	s.zw = dpos - (float2)(vi.x, vi.y); // s0 = s.x, t0 = s.y
	s.xy = 1 - s.zw;  // s1 = s.z, t1 = s.w
	float value = s.x*(s.y*input00.x + s.w*input01.x)
		+ s.z*(s.y*input10.x + s.w*input11.x);
	write_imagef(img_out, pos, (float4)(value, 0, 0, 0));
}

__kernel void advect_circular(__read_only image2d_t img_in,
	__write_only image2d_t img_out,
	__read_only image2d_t u,
	__read_only image2d_t v,
	float dt, int w, int h) {
	const int2 pos = (int2)(get_global_id(0), get_global_id(1));
	const float2 size = (float2)((float)w, (float)h);
	const float2 dt0 = dt*(float2)(w, h);
	const float2 pos_norm = (float2)(1.0f+pos.x, 1.0f+pos.y) / size;
	float4 inputU = read_imagef(u, samplerB, pos_norm);
	float4 inputV = read_imagef(v, samplerB, pos_norm);
	float2 velocity = (float2)(inputU.x, inputV.x);
	float2 dpos = (float2)(pos.x, pos.y) - dt0*velocity.xy;

	int2 vi = (int2)(dpos.x, dpos.y);// cast to int
	float2 vf = dpos / size;// cast to int
	float4 input00 = read_imagef(img_in, samplerB, (float2)(vf.x, vf.y));
	float4 input01 = read_imagef(img_in, samplerB, (float2)(vf.x, vf.y + 1.0f));
	float4 input10 = read_imagef(img_in, samplerB, (float2)(vf.x + 1.0f, vf.y));
	float4 input11 = read_imagef(img_in, samplerB, (float2)(vf.x + 1.0f, vf.y + 1.0f));
	float4 s;
	s.zw = dpos - (float2)(vi.x, vi.y); // s0 = s.x, t0 = s.y
	s.xy = 1 - s.zw;  // s1 = s.z, t1 = s.w
	float value = s.x*(s.y*input00.x + s.w*input01.x)
		+ s.z*(s.y*input10.x + s.w*input11.x);
	write_imagef(img_out, pos, (float4)(value, 0, 0, 0));
}


__kernel void project1(__write_only image2d_t img_out,
	__read_only image2d_t u,
	__read_only image2d_t v,
	float hx, float hy) {

	const int xpos = get_global_id(0);
	const int ypos = get_global_id(1);

	float dl = read_imagef(u, samplerA, (int2)(xpos - 1, ypos)).x;
	float dr = read_imagef(u, samplerA, (int2)(xpos + 1, ypos)).x;
	float du = read_imagef(v, samplerA, (int2)(xpos, ypos - 1)).x;
	float dd = read_imagef(v, samplerA, (int2)(xpos, ypos + 1)).x;

	float value = -0.5f*(hx*(dr - dl) + hy*(dd - du));

	write_imagef(img_out, (int2)(xpos,ypos), (float4)(value, 0, 0, 0));
}

__kernel void project2(__read_only image2d_t img_in,
	__global float* u,
	__global float* v,
	int width, int height)
{

	const int xpos = get_global_id(0);
	const int ypos = get_global_id(1);

	float dr = read_imagef(img_in, samplerA, (int2)(xpos + 1, ypos)).x;
	float dl = read_imagef(img_in, samplerA, (int2)(xpos - 1, ypos)).x;
	float dd = read_imagef(img_in, samplerA, (int2)(xpos, ypos + 1)).x;
	float du = read_imagef(img_in, samplerA, (int2)(xpos, ypos - 1)).x;

	float u_val = 0.5f*(dr - dl) * width;
	float v_val = 0.5f*(dd - du) * height;
	u[xpos + width*ypos] -= u_val;
	v[xpos + width*ypos] -= v_val;
}

__kernel void reset(__write_only image2d_t img_out) {
	write_imagef(img_out, (int2)(get_global_id(0), get_global_id(1)), (float4)(0, 0, 0, 0));
}

__kernel void floatToR(__read_only image2d_t img_in, __write_only image2d_t img_out) {
	const int2 pos = (int2)(get_global_id(0), get_global_id(1));
	float4 v = read_imagef(img_in, samplerA, pos);
	int r = (int)(v.x*200.0f);
	int g = (int)(v.x*56.0f);
	int b = (int)(v.x*10.f);
	write_imageui(img_out, pos, (uint4)(r, g, b, 255));
}

__kernel void addCircleValue(__read_only image2d_t img_in, 
						__write_only image2d_t img_out,
					int px, int py, float add, float radius) {
	const int xpos = get_global_id(0);
	const int ypos = get_global_id(1);
	const float dx = (float)xpos - px;
	const float dy = (float)ypos - py;
	
	if (dx*dx+dy*dy <= radius*radius) {
		float4 value = read_imagef(img_in, samplerA, (int2)(xpos, ypos));
		value.x += add;
		write_imagef(img_out, (int2)(xpos, ypos), value);
	}
}