__kernel void hello_world (__global uint *a, __global uint *b, __global uint *c)
{
	uint id = get_global_id(0);
	c[id] = 0;
	for (uint i = 0; i < 4294967295U; ++i)
	{
		for (uint k = 0; k < 4294967295U; ++k)
			c[id] += exp((float)1.0 + (float)i + (float)k);
	}
}