//fixed 4 step reduce

__kernel void minimum(global const int* A, global int* B, local int* scratch) {
	int id = get_global_id(0); // returns unique global work-item value.
	int lid = get_local_id(0); // local work item id - used for faster access of computer systems.
	int N = get_local_size(0); // total number of work items.

	//cache all N values from global memory to local memory
	scratch[lid] = A[id];

	barrier(CLK_LOCAL_MEM_FENCE);//wait for all local threads to finish copying from global to local memory

	for (int i = 1; i < N; i *= 2) {
		if (!(lid % (i * 2)) && ((lid + i) < N))
			if (scratch[lid] > scratch[lid + i]) { // fi the id values are higher than the iterating values;
				scratch[lid] = scratch[lid + i]; // equal the values to the that lower value, until the lowest value is found.
			}

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	//we add results from all local groups to the first element of the array
	//serial operation! but works for any group size
	//copy the cache to output array
	if (!lid) { // if there are no more local ids, perform the atomic-min function with the work items contained within the cache. 
		atomic_min(&B[0], scratch[lid]);
	}
}

kernel void reduce_Maximum(__global const int* A, global int* B, local int* localScratch) {
	int id = get_global_id(0); // returns unique global work-item.
	int lid = get_local_id(0); // id of work item within work group. (local)
	int N = get_local_size(0); // total number of work-items

	// for improved access to the data, it is transfered from global to local memory.
	localScratch[lid] = A[id];

	barrier(CLK_LOCAL_MEM_FENCE); // Waiting until all data is transferred..

	for (int i = 1; i < N; i *= 2) { 
		if (!(lid % (i * 2)) && ((lid + i) < N)) // if the local ids cannot go through any more even values, and there are no more work items available;
			if (localScratch[lid] < localScratch[lid + i]) { // if the local values are lesser than the value it is iterating through;
				localScratch[lid] = localScratch[lid + i]; // the value is then equalled to that value, finding the highest value.
			}

		barrier(CLK_LOCAL_MEM_FENCE); // will wait until all calculations are complete.
	}


	if (!lid) { // if there are no more local ids, perform the atomic-max function with the work items contained within the cache. 
		atomic_max(&B[0], localScratch[lid]);
	
	}

}

kernel void get_StandardDeviation(global const int* A, global int* B, float meanValue, int vectorSize ) {
	int id = get_global_id(0);
	// global work-items for this function in kernel.
	B[id] = A[id];

	barrier(CLK_LOCAL_MEM_FENCE);

	// vectorSize - size of temp vector.
	// while the vector size is more than the id, calculate the variance 
	// this will continue until the there are no more available ids.
	if (vectorSize > id) {
		B[id] = (A[id] - meanValue) * (A[id] - meanValue); // taking away the mean value from each of the elements in A.
		// then equalling to B, which is the output.

		barrier(CLK_LOCAL_MEM_FENCE); // barrier to make sure each sum is completed.
	}
}

//a very simple histogram implementation
kernel void hist_simple(global const int* A, global int* H) {
	int id = get_global_id(0);

	//assumes that H has been initialised to 0
	int bin_index = A[id];//take value as a bin index
	// bin = binary.
	atomic_inc(&H[bin_index]);//serial operation, not very efficient!
}

/*
kernel void reduce_Median(global const int* A, global int* B, local int* localScratch) {

}
*/

//Hillis-Steele basic inclusive scan
//requires additional buffer B to avoid data overwrite 
kernel void scan_hs(global int* A, global int* B) {
	int id = get_global_id(0);
	int N = get_global_size(0);
	global int* C;

	for (int stride = 1; stride < N; stride *= 2) {
		B[id] = A[id];
		if (id >= stride)
			B[id] += A[id - stride];

		barrier(CLK_GLOBAL_MEM_FENCE); //sync the step

		C = A; A = B; B = C; //swap A & B between steps
	}
}

kernel void reduce_add_1(global const int* A, global int* B) {
	int id = get_global_id(0);
	int N = get_global_size(0);

	B[id] = A[id]; //copy input to output

	barrier(CLK_GLOBAL_MEM_FENCE); //wait for all threads to finish copying
	 
	//perform reduce on the output array
	//modulo operator is used to skip a set of values (e.g. 2 in the next line)
	//we also check if the added element is within bounds (i.e. < N)
	if (((id % 2) == 0) && ((id + 1) < N)) 
		B[id] += B[id + 1];

	barrier(CLK_GLOBAL_MEM_FENCE);

	if (((id % 4) == 0) && ((id + 2) < N)) 
		B[id] += B[id + 2];

	barrier(CLK_GLOBAL_MEM_FENCE);

	if (((id % 8) == 0) && ((id + 4) < N)) 
		B[id] += B[id + 4];

	barrier(CLK_GLOBAL_MEM_FENCE);

	if (((id % 16) == 0) && ((id + 8) < N)) 
		B[id] += B[id + 8];
}

//flexible step reduce 
kernel void reduce_add_2(global const int* A, global int* B) {
	int id = get_global_id(0);
	int N = get_global_size(0);

	B[id] = A[id];

	barrier(CLK_GLOBAL_MEM_FENCE);

	for (int i = 1; i < N; i *= 2) { //i is a stride
		if (!(id % (i * 2)) && ((id + i) < N)) 
			B[id] += B[id + i];

		barrier(CLK_GLOBAL_MEM_FENCE);
	}
}

//reduce using local memory (so called privatisation)
kernel void reduce_add_3(global const int* A, global int* B, local int* scratch) {
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);

	//cache all N values from global memory to local memory
	scratch[lid] = A[id];

	barrier(CLK_LOCAL_MEM_FENCE);//wait for all local threads to finish copying from global to local memory

	for (int i = 1; i < N; i *= 2) {
		if (!(lid % (i * 2)) && ((lid + i) < N)) 
			scratch[lid] += scratch[lid + i];

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	//copy the cache to output array
	B[id] = scratch[lid];
}

//reduce using local memory + accumulation of local sums into a single location
//works with any number of groups - not optimal!
// although end value is found, it is 
kernel void reduce_add_4(global const int* A, global int* B, local int* scratch) {
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);

	//cache all N values from global memory to local memory
	scratch[lid] = A[id];

	barrier(CLK_LOCAL_MEM_FENCE);//wait for all local threads to finish copying from global to local memory

	for (int i = 1; i < N; i *= 2) {
		if (!(lid % (i * 2)) && ((lid + i) < N)) 
			scratch[lid] += scratch[lid + i];

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	//we add results from all local groups to the first element of the array
	//serial operation! but works for any group size
	//copy the cache to output array
	if (!lid) {
		atomic_add(&B[0],scratch[lid]);
	}
}


/*
// sequential addressing
		// uses local cache for quicker getting.
		// parallelly calculates reductions not serially like reduce_add_1/2.
kernel void reduce_add_4(global const int* A, global int* B, local int* scratch) { 
	// insert sequential addressing code to this kernel method.
}
*/

//a double-buffered version of the Hillis-Steele inclusive scan
//requires two additional input arguments which correspond to two local buffers
kernel void scan_add(__global const int* A, global int* B, local int* scratch_1, local int* scratch_2) {
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);
	local int *scratch_3;//used for buffer swap

	//cache all N values from global memory to local memory
	scratch_1[lid] = A[id];

	barrier(CLK_LOCAL_MEM_FENCE);//wait for all local threads to finish copying from global to local memory

	for (int i = 1; i < N; i *= 2) {
		if (lid >= i)
			scratch_2[lid] = scratch_1[lid] + scratch_1[lid - i];
		else
			scratch_2[lid] = scratch_1[lid];

		barrier(CLK_LOCAL_MEM_FENCE);

		//buffer swap
		scratch_3 = scratch_2;
		scratch_2 = scratch_1;
		scratch_1 = scratch_3;
	}

	//copy the cache to output array
	B[id] = scratch_1[lid];
}

//Blelloch basic exclusive scan
kernel void scan_bl(global int* A) {
	int id = get_global_id(0);
	int N = get_global_size(0);
	int t;

	//up-sweep
	for (int stride = 1; stride < N; stride *= 2) {
		if (((id + 1) % (stride * 2)) == 0)
			A[id] += A[id - stride];

		barrier(CLK_GLOBAL_MEM_FENCE); //sync the step
	}

	//down-sweep
	if (id == 0)
		A[N - 1] = 0;//exclusive scan

	barrier(CLK_GLOBAL_MEM_FENCE); //sync the step

	for (int stride = N / 2; stride > 0; stride /= 2) {
		if (((id + 1) % (stride * 2)) == 0) {
			t = A[id];
			A[id] += A[id - stride]; //reduce 
			A[id - stride] = t;		 //move
		}

		barrier(CLK_GLOBAL_MEM_FENCE); //sync the step
	}
}

//calculates the block sums
kernel void block_sum(global const int* A, global int* B, int local_size) {
	int id = get_global_id(0);
	B[id] = A[(id+1)*local_size-1];
}

//simple exclusive serial scan based on atomic operations - sufficient for small number of elements
kernel void scan_add_atomic(global int* A, global int* B) {
	int id = get_global_id(0);
	int N = get_global_size(0);
	for (int i = id+1; i < N; i++)
		atomic_add(&B[i], A[id]);
}

//adjust the values stored in partial scans by adding block sums to corresponding blocks
kernel void scan_add_adjust(global int* A, global const int* B) {
	int id = get_global_id(0);
	int gid = get_group_id(0);
	A[id] += B[gid];
}
