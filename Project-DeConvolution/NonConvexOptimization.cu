#include"library.h"
#include<iostream>
// u gets optimized with given k and given f

using namespace std;

cv::Mat NonConvexOptimization(cv::Mat u,cv::Mat k,cv::Mat f, float lambda){
	cv::Mat u2sub;
	cv::Mat diff;
	cv::Mat k_flipped;

	cv::Mat gradudata(u.size(),u.channels());
	cv::Mat lambdaGradTVccU(gradudata.size(), gradudata.type());
	cv::Mat gradu(gradudata.size(), gradudata.type());
	cv::Mat sfGradu(gradu.size(), gradu.type());
	cv::Mat gradTVccU(u.size(),u.type());
		cv::Mat gradu_abs,gradu_cloned;
	
	gradudata=0.0;	
	gradu=0.0;
	sfGradu=0.0;
	gradTVccU=0.0;
	lambdaGradTVccU=0.0;

	conv2(u,k,CONVOLUTION_VALID,u2sub);

	diff= u2sub-f;

	cv::flip(k,k_flipped,-1);
	conv2(diff,k_flipped,CONVOLUTION_FULL,gradudata);

	gradTVccU=gradTVcc(u);
 
	lambdaGradTVccU=scalarMult(gradTVccU,lambda);

	//cout<<lambdaGradTVccU<<endl;
	gradu = gradudata - lambdaGradTVccU;

	gradu_cloned=gradu.clone();
	
	gradu_abs=cv::abs(gradu_cloned);

	float sf = 5e-3*MaxFinder(u)/max(1e-31,MaxFinder(gradu_abs));
		
	sfGradu=scalarMult(gradu,sf);
    u   = u - sfGradu;

	return u;
}
