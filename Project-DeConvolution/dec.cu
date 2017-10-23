#include"library.h"

using namespace std;

cv::Mat dec(cv::Mat f, float lambda, cv::Mat u, cv::Mat k, int scale, int iter){
    cout<<endl<<"Inside dec"<<endl;
   	 
	cv::Mat u_clone=u.clone();
	cv::Mat k_clone=k.clone();
	cv::Mat f_clone=f.clone();
		
	for (int it=0;it<iter;it++){

	u_clone=NonConvexOptimization(u_clone,k_clone,f_clone, lambda).clone();	
	
  	}

	cout<<endl<<"Returning dec"<<endl;
	return u_clone;
}





