#include"library.h"

using namespace std;

BlindReturn blind(cv::Mat f, float lambda, cv::Mat u, cv::Mat k, int scale, int iter){
    cout<<endl<<"Inside Blind"<<endl;

	cv::Mat u_clone=u.clone();
	cv::Mat k_clone=k.clone();
	cv::Mat f_clone=f.clone();

	cv::Mat gradk(k.size(),k.type());
	cv::Mat gradk_cloned,gradk_abs;

	float sh;

	for (int it=0;it<iter;it++){


		u_clone=NonConvexOptimization(u_clone,k_clone,f_clone, lambda).clone();	
		
    	gradk  = GetGradK(u_clone,k_clone,f_clone).clone();
            
        gradk_cloned=gradk.clone();
	
		gradk_abs=cv::abs(gradk_cloned);

		sh= 1e-3*MaxFinder(k_clone)/max(1e-31,MaxFinder(gradk_abs));
    	
		k_clone = k_clone - scalarMult(gradk,sh);// Maybe can do

		k_clone=KernelProjection(k_clone).clone();

  	}

	cout<<endl<<"Returning Blind"<<endl;
	
	BlindReturn br;
	
	br.k=k_clone;
	br.u=u_clone;
	
	return br;
}

