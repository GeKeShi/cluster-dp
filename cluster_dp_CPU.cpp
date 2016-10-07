
// Clustering by fast search and find of density peaks
// Science 27 June 2014:
// Vol. 344 no. 6191 pp. 1492-1496
// DOI: 10.1126/science.1242072
// http://www.sciencemag.org/content/344/6191/1492.full//


#include "iostream"
#include <stdio.h>
#include <string.h>
//#include <ctime>
#include "vector"
#include "math.h"
#include "algorithm"
using namespace std;

#define DIM 3
#define elif else if

#ifndef bool
#define bool int
#define false ((bool)0)
#define true  ((bool)1)
#endif

#define NEIGHBORRATE 0.020

#define RHO_RATE 0.6
#define DELTA_RATE 0.2

vector<vector<float> > data;
vector< vector<float> > data_distance;
vector<int> near_cluster_label;
    //vector<bool> cluster_halo;
vector<float> rho;
vector<float> delta;
vector<int> decision;
int nSamples;
struct Point3d {
    float x;
    float y;
    Point3d(float xin, float yin) : x(xin), y(yin) {}
};

int dataPro(vector< vector<float> > &src, vector<Point3d> &dst){
    for (int i = 0; i < src.size(); i++){
        Point3d pt(src[i][0], src[i][1]);
        dst.push_back(pt);
    }
    return dst.size();
}

float get_point_Distance(Point3d &pt1, Point3d &pt2){
    float tmp = pow(pt1.x - pt2.x, 2) + pow(pt1.y - pt2.y, 2);
    return pow(tmp, 0.5);
}

void get_distanc(vector< vector<float> > &data_distance, vector<Point3d> &data){
    int data_size = data.size();
    for (int i = 0; i < data_size; ++i)
    {
        /* code */
        vector<float> tmp(data_size, 0.0);
        for (int j = 0; j < data_size; ++j)
        {
            /* code */
            if (i != j)
            {
                /* code */
                tmp[j] = get_point_Distance(data[i], data[j]);
            }
        }
        data_distance.push_back(tmp);
    }
}

void selfdef_sort(vector<float> &v, long left, long right){
    if (left < right){
        float key = v[left];
        long low = left;
        long high = right;
        while (low < high) {
            // high下标位置开始，向左边遍历，查找不大于基准数的元素
            while (low < high && v[high] >= key) {
                high--;
            }
            if (low < high) {// 找到小于准基数key的元素
                v[low] = v[high];// 赋值给low下标位置，low下标位置元素已经与基准数对比过了
                low++;// low下标后移
            }
            else {// 没有找到比准基数小的元素
                // 说明high位置右边元素都不小于准基数
                break;
            }
            // low下标位置开始，向右边遍历，查找不小于基准数的元素
            while (low < high && v[low] <= key) {
                low++;
            }
            if (low < high) {// 找到比基准数大的元素
                v[high] = v[low];// 赋值给high下标位置，high下标位置元素已经与基准数对比过了
                high--;// high下标前移，
            }
            else {// 没有找到比基准数小的元素
                // 说明low位置左边元素都不大于基准数
                break;
            }
        }
        v[low] = key;// low下标赋值基准数
        selfdef_sort(v, left, low - 1);
        selfdef_sort(v, low + 1, right);
    }
}
float getdc(vector< vector<float> > &data_distance, float neighborRate,int nSamples){
    int nSamples_rate = round(nSamples*(nSamples - 1)*neighborRate / 2);
    float dc = 0.0;
    vector<float> distance_tmp;

    for (int i = 0; i <nSamples; ++i)
    {
        /* code */
        for (int j = i + 1; j < nSamples; j++)
        {
            /* code */
            distance_tmp.push_back(data_distance[i][j]);
        }
    }

    selfdef_sort(distance_tmp, 0, distance_tmp.size()-1);//sort

    dc = distance_tmp.at(nSamples_rate);
    // cout<<"dc:"<<dc<<endl;
    return dc;
}
float get_potential(float thigma,  int row_num){
    float phi=0;
    for (int i = 0; i < nSamples; ++i)
    {
        /* code */
        phi+=exp(-pow(data_distance[row_num][i]/thigma,2));
    }
    return phi;
}
float get_entropy(float thigma){
    float total_potential=0,entropy=0;
    for (int i = 0; i < nSamples; ++i)
    {
         /* code */
        total_potential+=get_potential(thigma,i);
    }
    for (int i = 0; i < nSamples; ++i)
      {
          /* code */
        tmp=get_potential(thigma,i)/total_potential;
        entropy+=-(tmp*log2f(tmp));
      }  
}
float get_dc_entropy(){
    float thigma_max=0,thigma_min=0.001;
    for (int i = 0; i < nSamples; ++i)
    {
        /* code */
        if (thigma_max<data_distance[0][i])
        {
            /* code */
            thigma_max=data_distance[0][i];
        }
    }
    float x1=thigma_min+0.382*(thigma_max-thigma_min);
    float x2=thigma_min+0.618*(thigma_max-thigma_min);
    while(thigma_max-thigma_min<0.001){
        tmp=get_entropy(x1)-get_entropy(x2);
        if (tmp<0||tmp==0)
        {
            /* code */
            thigma_max=x2;
            x2=x1;
            x1=thigma_max-0.618*(thigma_max-thigma_min);
        }
        else{
            thigma_min=x1;
            x1=x2;
            x2=thigma_min+0.618*(thigma_max-thigma_min);
        }
    }
    return ((thigma_max+thigma_min)/2)*(sqrt(3)/2);
    
}
//cut-off kernel
vector<float> getLocalDensity(vector< vector<float> > &data_distance, float dc,int nSamples){
    vector<float> rho(nSamples, 0.0);
    for (int i = 0; i < nSamples - 1; i++){
        for (int j = i + 1; j < nSamples; j++){
            if (data_distance[i][j] < dc){
                ++rho[i];
                ++rho[j];
            }
        }
        //cout<<"getting rho. Processing point No."<<i<<endl;
    }
    return rho;
}
//gussian kernel
vector<float> getLocalDensity_gussian(vector< vector<float> > &data_distance, float dc, int nSamples){
    // dc=1.9;
    vector<float> rho(nSamples, 0.0);
    for (int i = 0; i < nSamples; i++){
        for (int j = 0; j < nSamples; j++){
            rho[i] = rho[i] + exp(-pow((data_distance[i][j] / dc), 2));
        }
        // cout<<"getting rho. Processing point No."<<i<<rho[i]<<endl;
    }
    return rho;
}
/**
 * 
 */
vector<float> getDistanceToHigherDensity(vector< vector<float> > &data_distance, vector<float> &rho){
    int nSamples = data_distance[0].size();
    vector<float> delta(nSamples, 0.0);
    for (int i = 0; i < nSamples; i++){
        float dist = 0.0;
        bool flag = false;
        near_cluster_label.push_back(-1);
        for (int j = 0; j < nSamples; j++){
            if (i == j) continue;
            if (rho[j] > rho[i]){
                float tmp = data_distance[i][j];
                if (!flag){
                    dist = tmp;
                    near_cluster_label.back() = j;
                    flag = true;
                }
                else if (tmp<dist)
                {
                    dist = tmp;
                    near_cluster_label.back() = j;
                }
            }
        }
        if (!flag){
            for (int j = 0; j < nSamples; j++){
                if(i==j)
                    continue;
                float tmp = data_distance[i][j];
                dist = tmp > dist ? tmp : dist;
            }
            near_cluster_label.back() = 0;//the bigger data's lable will step over later
        }
        delta[i] = dist;
        // cout<<"delta"<<i<<":"<<delta[i]<<endl;
    }
    return delta;
}
//应该讲rho进行排序，避免有相同最大密度的点，也可以通过高斯核计算密度来最大程度避免这个问题

/*
vector<int> decidegragh(vector<float> &delta, vector<float> &rho){
int nSamples = rho.size();
vector<int> decision(nSamples, -1);
vector<float> multiple(nSamples, 0.0);
for (int i = 0; i < nSamples; ++i)
{

multiple[i] = delta[i] * rho[i];
    }
    for (int i = 0; i < CLUSTER_NUM; ++i)
    {

        float tmp_max = 0.0;
        int tmp_lable = 0;
        for (int j = 0; j < nSamples; ++j)
        {

            if (tmp_max <= multiple[j])
            {

                tmp_max = multiple[j];
                tmp_lable = j;
            }

        }
        multiple[tmp_lable] = 0.0;
        decision[tmp_lable] = i;
    }
    return decision;
}

*/

vector<int> decidegragh(vector<float> &delta, vector<float> &rho,int &cluster_num){
    int nSamples = rho.size();
    int counter = 0;
    vector<int> decision(nSamples, -1);
    float max_rho = 0.0, min_rho = 0.0, max_delta = 0.0, min_delta = 0.0,rho_bound=0.0,delta_bound=0.0;
    for (int i = 0; i < nSamples; ++i)
    {
        /* code */
        if (max_rho <= rho[i])
        {
            max_rho = rho[i];
        }
        if (min_rho>=rho[i])
        {
            min_rho = rho[i];
        }
        if (max_delta <= delta[i])
        {
            max_delta = delta[i];
        }
        if (min_delta >= delta[i])
        {
            min_delta = delta[i];
        }
    }
    rho_bound = RHO_RATE*(max_rho - min_rho) + min_rho;
    delta_bound = DELTA_RATE*(max_delta - min_delta) + min_delta;
    for (int i = 0; i < nSamples; ++i)
    {
        /* code */
        if (rho[i]>rho_bound && delta[i]>delta_bound)
        {
            decision[i] = counter;
            counter++;
        }
    }
    cluster_num = counter;
    // cout<<"cluster_num:"<<cluster_num<<endl;
    return decision;
}
struct decision_pair
{
    float value;
    long order;
    decision_pair(float value,long order):value(value),order(order){}
};
bool comp(const decision_pair &a,const decision_pair &b)
{
    return a.value>b.value;
}
vector<int> decide_value(vector<float> &delta, vector<float> &rho,int &cluster_num){
    int nSamples = rho.size();
    // int counter = 0;
    vector<int> decision(nSamples, -1);
    vector<decision_pair> decision_value;
    for (int i = 0; i < nSamples; ++i)
    {
        /* code */
        decision_pair tmp(delta[i]*rho[i],i);
        decision_value.push_back(tmp);
    }
   
    sort(decision_value.begin(), decision_value.end(),comp);
    for (int i = 1; i < nSamples; ++i)
    {
        /* code */
        float meandif=((decision_value[i].value-decision_value[i+1].value)+(decision_value[i+1].value-decision_value[i+2].value)+(decision_value[i+2].value-decision_value[i+3].value))/3;

        if (-(decision_value[i].value-decision_value[i-1].value)/decision_value[i].value>0.5&&meandif/decision_value[i].value<0.1)
        {
            /* code */
            cluster_num=i;
            break;
        }
    }
    //  for (int i = 1; i < 20; ++i)
    // {
    //     /* code */
    //     float meandif=((decision_value[i].value-decision_value[i+1].value)+(decision_value[i+1].value-decision_value[i+2].value)+(decision_value[i+2].value-decision_value[i+3].value))/3;
    //     cout<<i<<":"<<decision_value[i].value<<"    "<<meandif<<"    "<<-(decision_value[i].value-decision_value[i-1].value)/decision_value[i].value<<"   "<<meandif/decision_value[i].value<<endl;
    //     // if ((decision_value[i].value-decision_value[i-1].value)/decision_value[i].value>0.5&&meandif/decision_value[i].value<0.1)
    //     // {
    //     //     /* code */
    //     //     cluster_num=i;
    //     //     break;
    //     // }
    // }
    for (int i = 0; i < cluster_num-1; ++i)
    {
        /* code */
        decision[decision_value[i].order]=i;
    }
    // cout<<"cluster_num:"<<cluster_num<<endl;
    return decision;
}
void quicksort(vector<float> &rho, vector<int> &rho_order, long left, long right){
    if (left < right){
        long key = rho_order[left];
        long low = left;
        long high = right;
        while (low < high){
            while (low < high && rho[rho_order[high]] <= rho[key]){
                high--;
            }
            if (low<high)
            {
                rho_order[low] = rho_order[high];
                low++;
            }
            else
            {
                break;
            }
            while (low < high && rho[rho_order[low]] >= rho[key]){
                low++;
            }
            if (low<high)
            {
                rho_order[high] = rho_order[low];
                high--;
            }
            else
            {
                break;
            }
        }
        rho_order[low] = key;
        quicksort(rho, rho_order, left, low - 1);
        quicksort(rho, rho_order, low + 1, right);
    }
}
void assign_cluster(vector<float> &rho, vector<int> &decision, vector<int> &near_cluster_label){
    vector<int> rho_order(rho.size(), -1);
    for (int i = 0; i < rho.size(); ++i)
    {
        /* code */
        rho_order[i] = i;
    }
    quicksort(rho, rho_order, 0, rho.size()-1);
    // for (int i = 0; i < rho.size(); ++i)
    // {
    //     /* code */
    //     printf("rho_order:%d:%f  ",rho_order[i],rho[rho_order[i]] );
    // }
    for (int i = 0; i < rho_order.size(); ++i)
    {
        /* code */
        if (decision[rho_order[i]] == -1)
        {
            /* code */
            decision[rho_order[i]] = decision[near_cluster_label[rho_order[i]]];
        }
    }


}
int assign_cluster_recursive(int index){
    float min_dist=10000;
    bool flag=true;
    int neighbor=-1;
    // int MAX=10000;
    for(int i=0;i<nSamples;i++){
        
        if(min_dist>data_distance[index][i]&&rho[index]<rho[i]){
            min_dist=data_distance[index][i];
            neighbor=i;
            flag=false;
        }

    }
    if(decision[neighbor]==-1&&flag==false)
        decision[neighbor]=assign_cluster_recursive(neighbor);
    if(decision[neighbor]!=-1&&flag==false)
        return decision[neighbor];
    // if(flag==true)
    //     {
    //         cout<<"the first center is"<<index<<":"<<decision[index]<<endl;
    //         return decision[index];
    //     }
}
void get_halo(vector<int> &decision, vector< vector<float> > &data_distance, vector<bool> &cluster_halo, vector<float> &rho, float dc,int cluster_num){
    vector<float> density_bound(cluster_num, 0.0);
    int nSamples = decision.size();
    for (int i = 0; i < nSamples - 1; ++i)
    {
        /* code */
        float avrg_rho;
        for (int j = i+1; j < nSamples; ++j)
        {
            /* code */
            if (decision[i] != decision[j] && data_distance[i][j]<dc)
            {
                /* code */
                avrg_rho = (rho[i] + rho[j]) / 2;
                if (avrg_rho>density_bound[decision[i]])
                {
                    /* code */
                    density_bound[decision[i]] = avrg_rho;
                }
                if (avrg_rho>density_bound[decision[j]])
                {
                    /* code */
                    density_bound[decision[j]] = avrg_rho;
                }
            }
        }
    }
    for (int i = 0; i < nSamples; ++i)
    {
        /* code */
        if (rho[i] <= density_bound[decision[i]])
        {
            /* code */
            cluster_halo.push_back(false);
        }
        else cluster_halo.push_back(true);
    }
}
int main(int argc, char** argv)
{
    long start, end;
    //errno_t err;
    FILE *input;
    char inputfile[100];
    char prefix[100]="dataset/";
    printf("inputfile:");
    scanf("%s",inputfile);
    strcat(prefix,inputfile);
    printf("%s\n",prefix );  
    if((input=fopen(prefix, "r"))==NULL)
        printf("data file not found\n");
    else
    {
        printf("data file was opened\n");
    }

    float point_x, point_y;
    int point_lable;
    int counter = 0,cluster_num=0;


    while (1){
        if (fscanf(input, "%lf,%lf", &point_x, &point_y) == EOF) break;

        vector<float> tpvec;
        data.push_back(tpvec);

        data[counter].push_back(point_x/10000);
        data[counter].push_back(point_y/10000);

        ++counter;
    }
    if (fclose(input) == 0)
        printf("read %d samples,datafile closed\n", counter);
    else
    {
        printf("datafile closed failed\n");
    }

    start = clock();
    // cout << "********" << endl;
    vector<Point3d> points;
    nSamples=dataPro(data, points);
    start = clock();
    get_distanc(data_distance, points);
    long distance_clock=clock();
    cout<<"get disance using time:"<<(float(distance_clock-start))/CLOCKS_PER_SEC<<endl;
    float dc = getdc(data_distance, NEIGHBORRATE,nSamples);
    float dc_entropy=get_dc_entropy();
    cout<<"dc: "<<dc<<"  "<<"dc_entropy："<<dc_entropy<<endl;
    long dc_clock=clock();
    cout<<"get dc using time:"<<(float(dc_clock-distance_clock))/CLOCKS_PER_SEC<<endl;
    rho = getLocalDensity_gussian(data_distance, dc, nSamples);
    long rho_clock=clock();
    cout<<"get density using time:"<<(float(rho_clock-start))/CLOCKS_PER_SEC<<endl;
    delta = getDistanceToHigherDensity(data_distance, rho);
    long delta_clock=clock();
    cout<<"get delta using time:"<<(float(delta_clock-start))/CLOCKS_PER_SEC<<endl;
    // decision = decidegragh(delta, rho,cluster_num);
    decision=decide_value(delta,rho,cluster_num);
    assign_cluster(rho, decision, near_cluster_label);
    // for(int i=0;i<nSamples;i++){
    //     decision[i]=assign_cluster_recursive(i);
    //     cout<<i<<":"<<decision[i]<<endl;
    // }
    //get_halo(decision, data_distance, cluster_halo, rho, dc, cluster_num);

    end = clock();
    cout << "used time: " << ((float)(end - start)) / CLOCKS_PER_SEC << endl;

    FILE *output;
    if((output=fopen("result_CPU.txt", "w"))!=NULL)
        printf("result file open");
    for (int i = 0; i < counter; ++i)
    {
        /* code */
        fprintf(output, "%4.2f,%4.2f,%d\n", data[i][0], data[i][1], decision[i]);
    }
    fclose(output);

    return 0;
}


