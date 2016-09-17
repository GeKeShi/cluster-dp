


#include "iostream"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
//#include "math.h"
#include "cuda_runtime.h"
#include "Header.h"
#include "device_launch_parameters.h"
#include "device_functions.h"
using namespace std;

#ifndef bool
#define bool int
#define false ((bool)0)
#define true  ((bool)1)
#endif

#define NEIGHBORRATE 0.020  //每个点周围平均2%的点在dc范围内
#define RHO_RATE 0.6//暂时不用
#define DELTA_RATE 0.2//no use
#define DATASIZE 788//点的数目
#define FEATURE_DIM 2//点特征维度
#define BLOCK_DIM 32
#define GRID_DIM ((DATASIZE+BLOCK_DIM-1)/BLOCK_DIM)

typedef struct Point_ {
	float feature_data[FEATURE_DIM];
}Point;//数据点

__host__ void cuda_kernel_check(const char *file, int line){
	cudaDeviceSynchronize();
	if (cudaPeekAtLastError() != cudaSuccess)
	{
		printf("\n%s at line %d in %s", cudaGetErrorString(cudaPeekAtLastError()), line, file);
		cudaDeviceReset();
		exit(EXIT_FAILURE);
	}

}//kernel错误检查
__global__ void distance_kernel(Point *dev_data, float *dev_distance,size_t pitch){
	int tid_row = blockIdx.y*blockDim.y + threadIdx.y;
	int tid_col = blockIdx.x*blockDim.x + threadIdx.x;
	if (tid_row<DATASIZE&&tid_col<DATASIZE)
	{
		float *address = (float *)((char *)dev_distance + tid_row*pitch);
		float point_distance = 0;
		for (int i = 0; i < FEATURE_DIM; i++)
		{
			point_distance += powf(dev_data[tid_row].feature_data[i] - dev_data[tid_col].feature_data[i], 2);
		}
		address[tid_col] = sqrtf(point_distance);
	}

}//计算距离核函数
void get_distance_gpu(Point *dev_data_ptr, float *dev_distance_ptr,size_t pitch){

	dim3 dimBlock(16,16);
	dim3 dimGrid((DATASIZE + 15) / 16, (DATASIZE + 15) / 16);
	distance_kernel <<<dimGrid, dimBlock>>>(dev_data_ptr, dev_distance_ptr,pitch);
	cuda_kernel_check(__FILE__, __LINE__);
}//计算距离



__global__ void dc_kernel(float *dev_distance, float dc, size_t pitch,int *result){
	__shared__ int distance_tmp[16][16];
	int tid_row = blockIdx.x * 16 + threadIdx.y;
	int tid_col = threadIdx.x;
	distance_tmp[threadIdx.y][threadIdx.x] = 0;
	if (tid_row<DATASIZE)
	{
		float *address = (float *)((char *)dev_distance + tid_row*pitch);
		while (tid_col<DATASIZE)
		{
			if (dc>address[tid_col])
			{
				distance_tmp[threadIdx.y][threadIdx.x] += 1;
			}
			tid_col += 16;
		}
	}
	__syncthreads();
	if (threadIdx.x == 0 && tid_row<DATASIZE)
	{
		for (int i = 1; i < 16; i++)
		{
			distance_tmp[threadIdx.y][0] += distance_tmp[threadIdx.y][i];
		}
	}
	__syncthreads();
	if (threadIdx.y == 0)
	{
		for (int i = 1; i < 16; i++)
		{
			distance_tmp[0][0] += distance_tmp[i][0];
		}
	}
	if (threadIdx.x == 0 && threadIdx.y == 0){
		atomicAdd(result, distance_tmp[0][0]);
	}
}
__global__ void reduction_min(float *min_distance){
	int tid = threadIdx.x;
	int index = tid;
	while (index<(DATASIZE + 15) / 16)
	{
		if (min_distance[tid]>min_distance[index])
		{
			min_distance[tid] = min_distance[index];
		}
		index += blockDim.x;
	}
	__syncthreads();
	if (tid==0)
	{
		for (int i = 0; i < blockDim.x*gridDim.x; i++)
		{
			if (min_distance[0]>min_distance[i])
			{
				min_distance[0] = min_distance[i];
			}
		}
	}

}
__global__ void reduction_max(float *max_distance){
	int tid = threadIdx.x;
	int index = tid;
	while (index<(DATASIZE + 15) / 16)
	{
		if (max_distance[tid]<max_distance[index])
		{
			max_distance[tid] = max_distance[index];
		}
		index += blockDim.x;
	}
	__syncthreads();
	if (tid == 0)
	{
		for (int i = 0; i < blockDim.x*gridDim.x; i++)
		{
			if (max_distance[0]<max_distance[i])
			{
				max_distance[0] = max_distance[i];
			}
		}
	}

}
__global__ void min_distance_kernel(float *dev_distance, float *min_distance, size_t pitch){
	__shared__ float distance_tmp[16][16];
	int tid_row = blockIdx.x * 16 + threadIdx.y;
	int tid_col = threadIdx.x;
	if (tid_row<DATASIZE)
	{
		float *address = (float *)((char *)dev_distance + tid_row*pitch);
		distance_tmp[threadIdx.y][threadIdx.x] = address[tid_col];
		while (tid_col<DATASIZE)
		{
			tid_col += 16;
			if (distance_tmp[threadIdx.y][threadIdx.x]>address[tid_col])
			{
				distance_tmp[threadIdx.y][threadIdx.x] = address[tid_col];
			}
		}
	}

	__syncthreads();
	if (threadIdx.x==0&&tid_row<DATASIZE)
	{
		for (int i = 1; i < 16; i++)
		{
			if (distance_tmp[threadIdx.x][0]>distance_tmp[threadIdx.x][i])
			{
				distance_tmp[threadIdx.x][0] = distance_tmp[threadIdx.x][i];
			}
		}
	}
	__syncthreads();
	if (threadIdx.y == 0)
	{
		for (int i = 1; i < 16; i++)
		{
			if ((blockIdx.x*blockDim.y + i)<DATASIZE)
			{
				if (distance_tmp[0][0]>distance_tmp[i][0])
					distance_tmp[0][0] = distance_tmp[i][0];
			}
		}
	}
	if (threadIdx.x == 0 && threadIdx.y == 0){
		min_distance[blockIdx.x] = distance_tmp[threadIdx.x][threadIdx.y];
	}
}
__global__ void max_distance_kernel(float *dev_distance, float *max_distance, size_t pitch){
	__shared__ float distance_tmp[16][16];
	int tid_row = blockIdx.x * 16 + threadIdx.y;
	int tid_col = threadIdx.x;
	if (tid_row<DATASIZE)
	{
		float *address = (float *)((char *)dev_distance + tid_row*pitch);
		distance_tmp[threadIdx.y][threadIdx.x] = address[tid_col];
		while (tid_col<DATASIZE)
		{
			tid_col += 16;
			if (distance_tmp[threadIdx.y][threadIdx.x]<address[tid_col])
			{
				distance_tmp[threadIdx.y][threadIdx.x] = address[tid_col];
			}
		}
	}

	__syncthreads();
	if (threadIdx.x == 0 && tid_row<DATASIZE)
	{
		for (int i = 1; i < 16; i++)
		{
			if (distance_tmp[threadIdx.x][0]<distance_tmp[threadIdx.x][i])
			{
				distance_tmp[threadIdx.x][0] = distance_tmp[threadIdx.x][i];
			}
		}
	}
	__syncthreads();
	if (threadIdx.y == 0)
	{
		for (int i = 1; i < 16; i++)
		{
			if ((blockIdx.x*blockDim.y + i)<DATASIZE)
			{
				if (distance_tmp[0][0]<distance_tmp[i][0])
					distance_tmp[0][0] = distance_tmp[i][0];
			}
		}
	}
	if (threadIdx.x == 0 && threadIdx.y == 0){
		max_distance[blockIdx.x] = distance_tmp[threadIdx.x][threadIdx.y];
	}
}
float getdc_gpu(float *dev_distance, float neighborRate,size_t pitch){

	cudaEvent_t start, stop;
	HANDLE_ERROR(cudaEventCreate(&start));
	HANDLE_ERROR(cudaEventCreate(&stop));
	HANDLE_ERROR(cudaEventRecord(start, 0));
	long data_size = DATASIZE*DATASIZE;
	int nSamples_rate = DATASIZE*DATASIZE*neighborRate;
	int dc_num=0;
	dim3 dimBlock(16, 16);
	dim3 dimGrid((DATASIZE + 15) / 16);

	float *dev_min_distance, *dev_max_distance;
	HANDLE_ERROR(cudaMalloc(&dev_min_distance, sizeof(float)*((int)((DATASIZE + 15) / 16))));
	HANDLE_ERROR(cudaMalloc(&dev_max_distance, sizeof(float)*((int)((DATASIZE + 15) / 16))));

	min_distance_kernel <<<dimGrid, dimBlock>>>(dev_distance, dev_min_distance, pitch);
	cuda_kernel_check(__FILE__, __LINE__);
	max_distance_kernel <<<dimGrid, dimBlock>>>(dev_distance, dev_max_distance, pitch);
	cuda_kernel_check(__FILE__, __LINE__);

	reduction_min <<<1,BLOCK_DIM>>>(dev_min_distance);
	cuda_kernel_check(__FILE__, __LINE__);
	reduction_max <<<1,BLOCK_DIM>>>(dev_max_distance);
	cuda_kernel_check(__FILE__, __LINE__);

	float host_min_dis, host_max_dis;
	HANDLE_ERROR(cudaMemcpy(&host_min_dis, dev_min_distance, sizeof(float), cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy(&host_max_dis, dev_max_distance, sizeof(float), cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaFree(dev_max_distance));
	HANDLE_ERROR(cudaFree(dev_min_distance));
	printf("host_min_dis=%f,host_max_dis=%f\n", host_min_dis, host_max_dis);
	float dc = host_min_dis;
	while (dc_num<nSamples_rate)
	{
		dc += (host_max_dis - host_min_dis) / 500;
		int *dev_result;
		HANDLE_ERROR(cudaMalloc(&dev_result, sizeof(int)));
		HANDLE_ERROR(cudaMemset(dev_result, 0, sizeof(int)));
		dc_kernel <<<dimGrid, dimBlock>>>(dev_distance, dc, pitch, dev_result);
		cuda_kernel_check(__FILE__, __LINE__);
		HANDLE_ERROR(cudaMemcpy(&dc_num, dev_result, sizeof(int), cudaMemcpyDeviceToHost));
		HANDLE_ERROR(cudaFree(dev_result));
		printf("dc=%f\t", dc);
	}



	HANDLE_ERROR(cudaEventRecord(stop, 0));
	HANDLE_ERROR(cudaEventSynchronize(stop));
	float elapsedtime;
	HANDLE_ERROR(cudaEventElapsedTime(&elapsedtime, start, stop));
	printf("getdc_gpu time:%3.1f ms\n", elapsedtime);
	HANDLE_ERROR(cudaEventDestroy(start));
	HANDLE_ERROR(cudaEventDestroy(stop));
	return dc;
}
__global__ void Gussian_kernel(double *dev_rho, float *dev_distance, float dc,size_t pitch){
	int tid_row= blockIdx.x*blockDim.x + threadIdx.x;

	if (tid_row<DATASIZE)
	{
		double tmprho = 0;
		float *address = (float *)((char *)dev_distance + tid_row*pitch);
		for (int i = 0; i < DATASIZE; i++)

		{
			tmprho += exp(-pow((address[i] / dc), 2));
		}
		dev_rho[tid_row] = tmprho;
	}
}
void getLocalDensity_gpu(float *dev_distance, float dc, double *dev_rho,size_t pitch){
	dim3 dimBlock(BLOCK_DIM);
	dim3 dimGrid(GRID_DIM);
	Gussian_kernel <<<dimGrid, dimBlock>>>(dev_rho, dev_distance, dc,pitch);
	cuda_kernel_check(__FILE__, __LINE__);
}
__global__ void get_delta_kernel(float *dev_distance, double *dev_rho, int *near_cluster_lable, float *delta,size_t pitch){
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	if (tid<DATASIZE)
	{
		float dist = 0.0;
		bool flag = false;
		float *address = (float *)((char *)dev_distance + tid*pitch);
		float tmp = 0.0;
		for (int i = 0; i < DATASIZE; i++)
		{
			if (tid == i)
			{
				continue;
			}
			if (dev_rho[tid]<dev_rho[i])
			{
				tmp = address[i];
				if (!flag)
				{
					dist = tmp;
					near_cluster_lable[tid] = i;
					flag = true;
				}
				else if (tmp<dist)
				{
					dist = tmp;
					near_cluster_lable[tid] = i;
				}
			}
		}
		if (!flag)
		{

			for (int i = 0; i < DATASIZE; i++)
			{
				if (tid == i)
				{
					continue;
				}
				tmp = address[i];
				dist = tmp > dist ? tmp : dist;
			}
			near_cluster_lable[tid] = 0;
		}
		delta[tid] = dist;
	}
}
void getDistanceToHigherDensity_gpu(float *dev_distance, double *dev_rho, int *near_cluster_lable, float *delta,size_t pitch){
	dim3 dimBlock(BLOCK_DIM);
	dim3 dimGrid(GRID_DIM);
	get_delta_kernel <<<dimGrid, dimBlock>>>(dev_distance, dev_rho, near_cluster_lable, delta, pitch);
	cuda_kernel_check(__FILE__, __LINE__);
}

__global__ void getDecisionvalue_kernel(float *delta, double *dev_rho, double *dev_decision_value){
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	if (tid<DATASIZE)
	{
		dev_decision_value[tid] = (double)delta[tid] * dev_rho[tid];
	}

}
void getdecisionvalue_gpu(double* decision_value_ptr, float *delta, double *dev_rho){
	cudaEvent_t start, stop;
	HANDLE_ERROR(cudaEventCreate(&start));
	HANDLE_ERROR(cudaEventCreate(&stop));
	HANDLE_ERROR(cudaEventRecord(start, 0));

	double *dev_decision_value;
	HANDLE_ERROR(cudaMalloc(&dev_decision_value, sizeof(double) * DATASIZE));
	dim3 dimBlock(BLOCK_DIM);
	dim3 dimGrid(GRID_DIM);
	getDecisionvalue_kernel <<<dimGrid, dimBlock>>>(delta, dev_rho, dev_decision_value);
	cuda_kernel_check(__FILE__, __LINE__);
	HANDLE_ERROR(cudaMemcpy(decision_value_ptr, dev_decision_value, sizeof(double) * DATASIZE, cudaMemcpyDeviceToHost));


	HANDLE_ERROR(cudaFree(dev_decision_value));
	HANDLE_ERROR(cudaEventRecord(stop, 0));
	HANDLE_ERROR(cudaEventSynchronize(stop));
	float elapsedtime;
	HANDLE_ERROR(cudaEventElapsedTime(&elapsedtime, start, stop));
	printf("getdecisionvalue_gpu time:%3.1f ms\n", elapsedtime);
	HANDLE_ERROR(cudaEventDestroy(start));
	HANDLE_ERROR(cudaEventDestroy(stop));
	HANDLE_ERROR(cudaDeviceSynchronize());
	for (int i = 0; i < 100; i++)
	{
		printf("decision_value:%lf\t", decision_value_ptr[i]);

	}
	printf("\n");
}
void get_cluster_center_auto(int *decision, double *decision_value){

}
void set_cluster_center(int *decision, double *decision_value, int cluster_num){
	printf("startset %d center", cluster_num);
	for (int i = 0; i < cluster_num; i++)
	{
		int tmp = 0;
		double tmp_value = 0;
		for (int j = 0; j < DATASIZE; j++)
		{
			if (tmp_value<decision_value[j])
			{
				tmp = j;
				tmp_value = decision_value[j];
			}
		}
		printf("the %dst cluster center is point %d", i, tmp);
		decision[tmp] = i;
		decision_value[tmp] = 0;
	}
}
__global__ void even_sort_kernel(double *dev_rho, int *dev_rho_order){
	int data_size = DATASIZE;
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	while (tid + 1 < data_size && (tid % 2 == 1))
	{
		if (dev_rho[dev_rho_order[tid]] < dev_rho[dev_rho_order[tid + 1]])
		{
			float tmp = dev_rho_order[tid];
			dev_rho_order[tid] = dev_rho_order[tid + 1];
			dev_rho_order[tid + 1] = tmp;
		}
		tid += blockDim.x*gridDim.x;
	}

}
__global__ void odd_sort_kernel(double *dev_rho, int *dev_rho_order){
	int data_size = DATASIZE;
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	while (tid + 1 < data_size && (tid % 2 == 0))
	{

		if (dev_rho[dev_rho_order[tid]] < dev_rho[dev_rho_order[tid + 1]])
		{
			float tmp = dev_rho_order[tid];
			dev_rho_order[tid] = dev_rho_order[tid + 1];
			dev_rho_order[tid + 1] = tmp;
		}
		tid += blockDim.x*gridDim.x;
	}
}
void assign_cluster_gpu(double *dev_rho, int* decision, int *Host_near_cluster_label){
	cudaEvent_t start, stop;
	HANDLE_ERROR(cudaEventCreate(&start));
	HANDLE_ERROR(cudaEventCreate(&stop));
	HANDLE_ERROR(cudaEventRecord(start, 0));

	int *rho_order = (int *)malloc(sizeof(int)*DATASIZE);
	for (int i = 0; i < DATASIZE; ++i)
	{
		/* code */
		rho_order[i] = i;
	}
	int *dev_rho_order;
	HANDLE_ERROR(cudaMalloc(&dev_rho_order, sizeof(int) * DATASIZE));
	HANDLE_ERROR(cudaMemcpy(dev_rho_order, rho_order, sizeof(int)*DATASIZE, cudaMemcpyHostToDevice));
	dim3 dimBlock(BLOCK_DIM);
	dim3 dimGrid(GRID_DIM);
	for (int i = 0; i < DATASIZE; i++)
	{
		even_sort_kernel << <dimGrid, dimBlock >> >(dev_rho, dev_rho_order);
		cuda_kernel_check(__FILE__, __LINE__);
		odd_sort_kernel << <dimGrid, dimBlock >> >(dev_rho, dev_rho_order);
		cuda_kernel_check(__FILE__, __LINE__);
	}
	HANDLE_ERROR(cudaMemcpy(rho_order, dev_rho_order, sizeof(int) * DATASIZE, cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaFree(dev_rho_order));
	HANDLE_ERROR(cudaEventRecord(stop, 0));
	HANDLE_ERROR(cudaEventSynchronize(stop));
	float elapsedtime;
	HANDLE_ERROR(cudaEventElapsedTime(&elapsedtime, start, stop));
	printf("assign_cluster_gpu time:%3.1f ms\n", elapsedtime);
	HANDLE_ERROR(cudaEventDestroy(start));
	HANDLE_ERROR(cudaEventDestroy(stop));
	//cudaDeviceSynchronize();
	for (int i = 0; i < DATASIZE; ++i)
	{
		/* code */
		if (decision[rho_order[i]] == -1)
		{
			/* code */
			decision[rho_order[i]] = decision[Host_near_cluster_label[rho_order[i]]];
		}
	}
	free(rho_order);

}
int main(int argc, char **argv)
{

	//errno_t err;
	//打开文件
	FILE *input = fopen("dataset_2D.txt", "r");
	if (input == NULL)
		printf("data file not found\n");
	else

	{
		printf("data file was opened\n");
	}


	//读取数据
	Point *data = (Point *)calloc(DATASIZE, sizeof(Point));//最后打印结果还要用到，届时free
	int counter = 0;
	for (int i = 0; i < DATASIZE; i++)
	{
		Point tmppoint;
		int tmp;
		for (int j = 0; j < FEATURE_DIM; j++)
		{
			fscanf(input, "%f,", &(tmppoint.feature_data[j]));
		}
		fscanf(input, "%d", &tmp);
		memcpy((void *)&(data[i]), (void *)&tmppoint, sizeof(Point));
		counter++;
	}
		printf("read %d samples,datafile closed\n", counter);



	/*for (size_t i = 0; i < DATASIZE; i++)
	{
		printf("%f,%f\t", data[i].feature_data[0], data[i].feature_data[1]);
	}*/
	Point *dev_data;
	float *dev_distance;
	size_t pitch;
	//计算距离矩阵并计时
	cudaEvent_t start_getdistance, stop_getdistance;
	HANDLE_ERROR(cudaEventCreate(&start_getdistance));
	HANDLE_ERROR(cudaEventCreate(&stop_getdistance));
	HANDLE_ERROR(cudaEventRecord(start_getdistance, 0));
	HANDLE_ERROR(cudaMalloc(&dev_data, sizeof(Point) * DATASIZE));
	HANDLE_ERROR(cudaMallocPitch(&dev_distance,&pitch,sizeof(float)*DATASIZE,DATASIZE));//分配距离矩阵显存
	HANDLE_ERROR(cudaMemcpy(dev_data, data, sizeof(Point) * DATASIZE, cudaMemcpyHostToDevice));
	get_distance_gpu(dev_data, dev_distance,pitch);
	HANDLE_ERROR(cudaEventRecord(stop_getdistance, 0));
	HANDLE_ERROR(cudaEventSynchronize(stop_getdistance));
	float get_distanceelapsedtime;
	HANDLE_ERROR(cudaEventElapsedTime(&get_distanceelapsedtime, start_getdistance, stop_getdistance));
	printf("getdistance time:%3.1f ms\n", get_distanceelapsedtime);
	HANDLE_ERROR(cudaEventDestroy(start_getdistance));
	HANDLE_ERROR(cudaEventDestroy(stop_getdistance));
	HANDLE_ERROR(cudaFree(dev_data));

	//计算dc
	float dc = 0;
	dc=getdc_gpu(dev_distance, NEIGHBORRATE, pitch);

	//局部密度
	double *dev_rho;
	cudaEvent_t start_getLocalDensity, stop_getLocalDensity;
	HANDLE_ERROR(cudaEventCreate(&start_getLocalDensity));
	HANDLE_ERROR(cudaEventCreate(&stop_getLocalDensity));
	HANDLE_ERROR(cudaEventRecord(start_getLocalDensity, 0));
	HANDLE_ERROR(cudaMalloc(&dev_rho, sizeof(double) * DATASIZE));

	getLocalDensity_gpu(dev_distance, dc, dev_rho,pitch);

	HANDLE_ERROR(cudaEventRecord(stop_getLocalDensity, 0));
	HANDLE_ERROR(cudaEventSynchronize(stop_getLocalDensity));
	float getLocalDensityelapsedtime;
	HANDLE_ERROR(cudaEventElapsedTime(&getLocalDensityelapsedtime, start_getLocalDensity, stop_getLocalDensity));
	printf("getLocalDensity_gpu time:%3.1f ms\n", getLocalDensityelapsedtime);
	HANDLE_ERROR(cudaEventDestroy(start_getLocalDensity));
	HANDLE_ERROR(cudaEventDestroy(stop_getLocalDensity));

	//计算距离最近的高密度点的标号和及其距离
	int *near_cluster_lable;
	float *delta;
	int *Host_near_Cluster_lable = (int *)malloc(sizeof(int) * DATASIZE);
	cudaEvent_t start, stop;
	HANDLE_ERROR(cudaEventCreate(&start));
	HANDLE_ERROR(cudaEventCreate(&stop));
	HANDLE_ERROR(cudaEventRecord(start, 0));

	HANDLE_ERROR(cudaMalloc(&near_cluster_lable, sizeof(int) * DATASIZE));
	HANDLE_ERROR(cudaMemset(near_cluster_lable, -1, sizeof(int) * DATASIZE));
	HANDLE_ERROR(cudaMalloc(&delta, sizeof(float) * DATASIZE));
	getDistanceToHigherDensity_gpu(dev_distance, dev_rho, near_cluster_lable, delta, pitch);
	HANDLE_ERROR(cudaMemcpy(Host_near_Cluster_lable, near_cluster_lable, sizeof(int) * DATASIZE, cudaMemcpyDeviceToHost));

	HANDLE_ERROR(cudaEventRecord(stop, 0));
	HANDLE_ERROR(cudaEventSynchronize(stop));
	float elapsedtime;
	HANDLE_ERROR(cudaEventElapsedTime(&elapsedtime, start, stop));
	printf("getDelta_gpu time:%3.1f ms\n", elapsedtime);
	HANDLE_ERROR(cudaEventDestroy(start));
	HANDLE_ERROR(cudaEventDestroy(stop));

	HANDLE_ERROR(cudaFree(dev_distance));
	HANDLE_ERROR(cudaFree(near_cluster_lable));

	float *host_delta = (float *)malloc(sizeof(float)*DATASIZE);
	HANDLE_ERROR(cudaMemcpy(host_delta, delta, sizeof(float) * DATASIZE, cudaMemcpyDeviceToHost));
	for (int i = 0; i < 100; i++)
	{
		printf("delta %d:%f\t near label:%d\t", i,host_delta[i],Host_near_Cluster_lable[i]);

	}
	printf("\n");
	free(host_delta);
	double *decision_value = (double *)malloc(sizeof(double)*DATASIZE);
	//计算决策值
	getdecisionvalue_gpu(decision_value, delta, dev_rho);




	//设定聚类中心，如果命令行给出聚类个数则按照设定的聚类个数，否则自动获取聚类个数
	HANDLE_ERROR(cudaFree(delta));
	int *decision = (int *)malloc(sizeof(int) * DATASIZE);
	memset(decision, -1, sizeof(int) * DATASIZE);
	//cudaDeviceSynchronize();
	int cluster_num = 0;
	if (argc == 1)
	{
		printf("set cluster number:Y/N\n");
		char tmp;
		scanf("%c", &tmp);
		if (tmp=='Y')
		{
			printf("cluster number is:");
			scanf("%d", &cluster_num);
			set_cluster_center(decision, decision_value, cluster_num);
		}
		else if (tmp=='N')
		{
			get_cluster_center_auto(decision, decision_value);
		}

	}
	if (argc>1)
	{
		cluster_num = atoi(argv[1]);
		set_cluster_center(decision, decision_value, cluster_num);
	}

	free(decision_value);


	//指定每个点所属的类簇
	assign_cluster_gpu(dev_rho, decision, Host_near_Cluster_lable);
	free(Host_near_Cluster_lable);
	HANDLE_ERROR(cudaFree(dev_rho));

//print results by the "%f,%f,%d"format
	FILE *output = fopen("result.txt", "w");
	if (output != NULL)
		printf("result file open");
	for (int i = 0; i < counter; ++i)
	{
		/* code */
		for(int j=0;j<FEATURE_DIM;j++){
		fprintf(output,"%f,",data[i].feature_data[j]);
		}
		fprintf(output, "%d", decision[i]);
		fprintf(output, "\n");
	}
	fclose(input);
	fclose(output);
	free(data);
	free(decision);
	return 0;
}

