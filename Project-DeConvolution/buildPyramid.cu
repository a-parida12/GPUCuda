#include "helper.h"
#include "library.h"
#include <stdio.h>	
#include <vector>
#include <cmath>
using namespace std;

//void buildPyramid (int hK, int wK, float lambda, float lambdaMultiplier, int w, int h, int nc, float scaleMultiplier, float largestLambda) //TEST


ReturnBuildPyramid buildPyramid (int hK, int wK, float lambda, float lambdaMultiplier, int w, int h, int nc, float scaleMultiplier, float largestLambda)
{
//	printf("\nInside buildPyramid");
//	cout<<"scaleMultiplier: "<<scaleMultiplier<<endl;
    int smallestScale = 3;  //Smallest Scale is 3X3
    int scales = 0;	//Changed from 1 to 0 (As C++ array starts at 0)
  
    std::vector<int> hp(61);	//initial Image Size
	hp[0]=h;
    std::vector<int> wp(61);	//initial Image Size
	wp[0] =w;
    std::vector<int> hKp(61);	//initial Kernel Size
	hKp[0] =hK;
    std::vector<int> wKp(61);	//initial Kernel Size
	wKp[0] =wK;
    std::vector<float> lambdas(61); 
	lambdas[0]=lambda;

    while (hKp[scales] > smallestScale && wKp[scales] > smallestScale && lambdas[scales] * lambdaMultiplier < largestLambda)
    {
		scales = scales + 1;
	    
		// Compute lambda value for the current scale
		lambdas[scales] = lambdas[scales - 1] * lambdaMultiplier;

		hKp[scales] = (round(hKp[scales - 1] / scaleMultiplier));
		wKp[scales] = (round(wKp[scales - 1] / scaleMultiplier));
   	    
		// Makes kernel dimension odd
		if (hKp[scales]%2 == 0)
		{hKp[scales] = hKp[scales] - 1;}

		if (wKp[scales]%2 == 0)
		{wKp[scales] = wKp[scales] - 1;}
		    
		if (hKp[scales] == hKp[scales-1])
		{hKp[scales] = hKp[scales] - 2;}

		if (wKp[scales] == wKp[scales-1])
		{wKp[scales] = wKp[scales] - 2;}

		if (hKp[scales] < smallestScale)
		{hKp[scales] = smallestScale;}
	    
		if (wKp[scales] < smallestScale)
		{wKp[scales] = smallestScale;}
		    		    
		// Correct scaleFactor for kernel dimension correction
		
		float factorH = float (hKp[scales - 1])/hKp[scales];
		float factorW = float (wKp[scales - 1])/wKp[scales];
		    
		hp[scales] = round(hp[scales - 1] / factorH);
		wp[scales] = round(wp[scales - 1] / factorW);
//		cout<<"factorH: "<<factorH<<endl;    

		//**Makes image dimension odd
		if (hp[scales]%2 == 0)
		{hp[scales] = hp[scales] - 1;}  
		if (wp[scales]%2 == 0)
		{wp[scales] = wp[scales] - 1;}

//		cout<<"hp: "<<hp[scales]<<endl;
//		cout<<"wp: "<<wp[scales]<<endl;
//		cout<<"hKp[scales]"<<hKp[scales]<<endl;
//		cout<<"wKp[scales]"<<wKp[scales]<<endl;		
//		cout<<"lambdas[scales]"<<(lambdas[scales]* lambdaMultiplier)<<endl;	
	}
//	printf("\nExiting buildPyramid\n");

		//**Returning multiple values
		ReturnBuildPyramid rbp;		
		rbp.hp = hp;
		rbp.wp = wp;
		rbp.hKp = hKp;
		rbp.wKp = wKp;
		rbp.lambdas = lambdas;
		rbp.scales = scales;	
	return (rbp); 
}
