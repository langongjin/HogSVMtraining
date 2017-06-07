#include <iostream>
#include <fstream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/ml/ml.hpp>
#include <sys/time.h>
#include <dirent.h>

using namespace std;
using namespace cv;

string detector_file;
int folder_num;
HOGDescriptor Hog;
char filePath[3][200];

int type[3]={1,-1,-1}; //the array containing types for three sample list 样本类别数组
//检测窗口(48,112),块尺寸(32,32),块步长(16,16),cell尺寸(8,8),计算检测窗口(16,16),(winSize-blockSize) % blockStride == 0
//No.Block=(WinSize-blockSize)/blockstride+1=（(112-16)/16+1）* 2=14block/win,4cells/block,9bins/cell, Demension/block=13*4*9=468
Size winSize(112,24), blockSize(16,16),blockStride(16,8),cellSize(8,8),computeSize(8,8); //computeSize=winStride
//Size winSize(112,24), blockSize(24,24),blockStride(22,24),cellSize(12,12),computeSize(12,12); //5*4*9=180

void config(){

    int nbin=9;//orientation bins
    folder_num=3;//number of folders containing samples/100_0428hards10_50_261
    snprintf(filePath[0],150,"/Users/lan/Desktop/TarReg/svm/crop_samples/pos_samples/totestsamples/"); //path of folder containing positive samples 正样本来源
    snprintf(filePath[1],150,"/Users/lan/Desktop/TarReg/svm/crop_samples/neg_samples/totestsamples/"); //path of folder containing negative samples 负样本来源
    snprintf(filePath[2],150,"/Users/lan/Desktop/TarReg/svm/crop_samples/hard_samples/"); //path of folder containing hard and negative samples 负难例来源
    detector_file.assign("/Users/lan/Desktop/TarReg/svm/svmrobot/training/HOGDetector0502robot.txt"); //txt file preserving detectors检测子保存文件

    HOGDescriptor hog(winSize,blockSize,blockStride,cellSize,nbin);//define the parameters of HOG descriptors
    Hog=hog;
}
//继承自CvSVM的类，因为生成setSVMDetector()中用到的检测子参数时，需要用到训练好的SVM的decision_func参数，
//但通过查看CvSVM源码可知decision_func参数是protected类型变量，无法直接访问到，只能继承之后通过函数访问
class MySVM : public CvSVM
{
public:
    //get the alpha array of decision_func in the SVM model
    double * get_alpha_vector()
    {
        return this->decision_func->alpha;
    }
    //get the rho parameter of decision_func in the SVM model, means 获得SVM的决策函数中的rho参数,即偏移量
    float get_rho()
    {
        return this->decision_func->rho;
    }
};

//获取path文件夹下的所有文件的文件名 // get all the file names in the folder, and push them into a vectors
void getFiles( string path, vector<string>& files )
{
    DIR  *dir;
    struct dirent  *ptr;
    dir = opendir(path.c_str());
    string pathName;
    while((ptr = readdir(dir)) != NULL){ //遍历目录中的文件
        if(ptr->d_name[0]!='.'){
            files.push_back(pathName.assign(path).append("/").append(string(ptr->d_name)));
        }
    }
}

int main()
{
    config();
    int DescriptorDim = Hog.getDescriptorSize(); //dimensions of Hog descriptor HOG描述子的维数，由图片大小、检测窗口大小、块大小、细胞单元中直方图bin个数决定

    MySVM svm;//定义SVM类的分类器MySVM, 开始训练MySVM
    //定义三个vector,每个vector存储一个文件夹下的所有文件的文件名,define three vector and push all of files names to the the storage of vectors.
    vector<string> file_name_array[3];
    vector<string> logs;    //logs
    logs.push_back(string("reading positive samples,file num:"));  //constructing the contents of logs.
    logs.push_back(string("reading negative samples,file num:"));
    logs.push_back(string("reading hard negative samples,file num:"));

    int sample_num=0; //the total number of samples including positive and negative and hard ones.

    for(int i=0;i<folder_num;++i){

        vector<string> file_names;//the temperate container for preserving file names.
        getFiles(filePath[i], file_names);
        file_name_array[i]=file_names;
        sample_num+=file_names.size();  //add the number of samples in different files
        //以下三行代码都在构造日志内容 all the three lines of codes are constructing contents of logs
        char*file_names_size_str=(char*)malloc(20*sizeof(char));
        snprintf(file_names_size_str,19,"%d",file_names.size());//每个文件夹下的图片数目，把整型转成字符串
        logs[i].append(file_names_size_str);//把上面生成的字符串拼接进对应的日志
    }
    //Mat类是opencv中表示一个n维的稠密数值型的单通道或多通道数组的类。用于存储实数或复数值的向量和矩阵、灰度或彩色图像、体素、向量场、点云、张量、直方图
    Mat sampleFeatureMat= Mat::zeros(sample_num,DescriptorDim, CV_32FC1); //所有训练样本的特征向量组成的矩阵，行数等于所有样本的个数，列数等于HOG描述子维数
    //所有训练样本的正负类型组成的矩阵，行数等于所有样本的个数，只有一列存+1或者-1代表正样本或者是负样本
    Mat sampleLabelMat= Mat::zeros(sample_num,1, CV_32FC1);
    int offset_sample_index=0; //the offset for every sample list

    for(int i=0;i<folder_num;++i){
        cout<<logs[i]<<endl;
        for(int j=0;j<file_name_array[i].size();++j){
            Mat src = imread(file_name_array[i][j]); //load and read image 读取图片
            vector<float> descriptors;//HOG描述子向量
            Hog.compute(src,descriptors,computeSize); //compute the hog descriptors计算HOG描述子

            for(int k=0; k<DescriptorDim; k++)
                sampleFeatureMat.at<float>(j+offset_sample_index,k) = descriptors[k]; //第i个文件中第j个样本的特征向量中的第k个元素
            sampleLabelMat.at<float>(j+offset_sample_index,0) = type[i]; //样本类别
            src.release(); //释放内存
        }
        offset_sample_index+=file_name_array[i].size();
    }
    //迭代终止条件，当迭代满1000次或误差小于FLT_EPSILON时停止迭代
    CvTermCriteria criteria = cvTermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, 1000, FLT_EPSILON);
    //SVM参数：SVM类型为C_SVC；线性核函数；松弛因子C=0.01
    CvSVMParams param(CvSVM::C_SVC, CvSVM::LINEAR, 0, 1, 0, 0.01, 0, 0, 0, criteria); //compare parameters for paper
    // CvSVMParams param( CvSVM::C_SVC, CvSVM::RBF, 10.0, 0.09, 1.0, 7.0, 0.5, 1.0, NULL, criteria );
    // SVM种类：CvSVM::C_SVC
    // Kernel的种类：CvSVM::RBF
    // degree：10.0（此次不使用）
    // gamma：8.0
    // coef0：1.0（此次不使用）
    // C：10.0 7 4 2
    // nu：0.5（此次不使用）
    // p：0.1（此次不使用）

    cout<<"start to train the SVM classifier model ......"<<endl;
    struct timeval tpstart,tpend; //the two struct values about time
    double timeuse; //running time of training function
    gettimeofday(&tpstart,NULL); //get the starting time
    svm.train(sampleFeatureMat, sampleLabelMat, Mat(), Mat(), param);//训练分类器

    gettimeofday(&tpend,NULL); //get the time of end
    timeuse=1000000*(tpend.tv_sec-tpstart.tv_sec)+tpend.tv_usec-tpstart.tv_usec; //calculations for running time.
    timeuse/=1000;
    cout<<"finish training，training time cost："<<timeuse<<"ms"<<endl;

    //save the trained SVM to be xml file
    svm.save("/Users/lan/Desktop/TarReg/svm/svmrobot/training/SVM_HOG_robot0501.xml");

    /*************************************************************************************************
    线性SVM训练完成后得到的XML文件里面，有一个数组，叫做support vector，还有一个数组，叫做alpha,有一个浮点数，叫做rho;
    将alpha矩阵同support vector相乘，注意，alpha*supportVector,将得到一个列向量。之后，再该列向量的最后添加一个元素rho。
    如此，变得到了一个分类器，利用该分类器，直接替换opencv中行人检测默认的那个分类器（cv::HOGDescriptor::setSVMDetector()），
    就可以利用你的训练样本训练出来的分类器进行行人检测了。
    ***************************************************************************************************/
    DescriptorDim = svm.get_var_count();//特征向量的维数，即HOG描述子的维数
    int supportVectorNum = svm.get_support_vector_count();//支持向量的个数

    Mat alphaMat = Mat::zeros(1, supportVectorNum, CV_32FC1);//alpha向量，长度等于支持向量个数
    Mat supportVectorMat = Mat::zeros(supportVectorNum, DescriptorDim, CV_32FC1);//支持向量矩阵
    Mat resultMat = Mat::zeros(1, DescriptorDim, CV_32FC1);//alpha向量乘以支持向量矩阵的结果

    // 将支持向量的数据复制到supportVectorMat矩阵中
    for(int i=0; i<supportVectorNum; i++)
    {
        const float * pSVData = svm.get_support_vector(i);//返回第i个支持向量的数据指针
        for(int j=0; j<DescriptorDim; j++)
        {
            supportVectorMat.at<float>(i,j) = pSVData[j];
        }
    }

    // 将alpha向量的数据复制到alphaMat中
    double * pAlphaData = svm.get_alpha_vector();//返回SVM的决策函数中的alpha向量
    for(int i=0; i<supportVectorNum; i++)
    {
        alphaMat.at<float>(0,i) = pAlphaData[i];
    }

    // 计算-(alphaMat * supportVectorMat),结果放到resultMat中
    resultMat = -1 * alphaMat * supportVectorMat;

    // 得到最终的setSVMDetector(const vector<float>& detector)参数中可用的检测子
    vector<float> myDetector;
    // 将resultMat中的数据复制到数组myDetector中
    for(int i=0; i<DescriptorDim; i++)
    {
        myDetector.push_back(resultMat.at<float>(0,i));
    }
    // 最后添加偏移量rho，得到检测子
    myDetector.push_back(svm.get_rho());
    // cout<<"检测子维数："<<myDetector.size()<<endl;
    // 设置HOGDescriptor的检测子

    // 保存检测子参数到文件
    ofstream fout(detector_file);
    for(int i=0; i<myDetector.size(); i++)
    {
        fout<<myDetector[i]<<endl;
    }

    fout.close();
}