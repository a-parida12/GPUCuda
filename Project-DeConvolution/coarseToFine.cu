#include "library.h"
#include <cmath>
#include <stdio.h>
#include <string>
using namespace std;
//Uncomment to TEST
//#define TEST_CTF
//#define TEST_PYRAMID
void coarseToFine(cv::Mat f, int heightK, int widthK, ParamGammaCor blind_params, Param params, int w, int h, int nc)
{
//	printf("\nInside coarseToFine\n");
//****PADDING
//**Required dimensions of output image - Padded	
	int top = std::floor(heightK/2);
	int bottom = std::floor(heightK/2);
    int left = std::floor(widthK/2);
	int right = std::floor(widthK/2);

//**Define: Output image - Padded 
	cv::Mat u(f.rows+top+bottom, f.cols+left+right, f.type());
	cv::copyMakeBorder( f, u, top, bottom, left, right, cv::BORDER_REPLICATE, 0 );
//  cout << "Padded image dimensions: w * h: " << u.size().width << " x " << u.size().height; 
//	cout<< " and No. of Channels " <<u.channels()<<endl;
	
//**Define: Kernel	
	float kValue = 1.0/(heightK * widthK);
//	float kValue = 1.0;
	cv::Mat k(heightK, widthK,  CV_32FC1); 	
	for (int y = 0; y<k.rows; y++)
	{
		for(int x = 0; x<(k.cols); x++)
		{
			k.at<float>(y, x) = kValue;
		}
	}
//	cout<<k<<endl;

#ifdef TEST_PYRAMID
	buildPyramid(heightK, widthK, params.finalLambda, params.lambdahultiplier, w, h, nc, params.kernelSizehultiplier, params.maxLambda); 
#else
	ReturnBuildPyramid rbp = buildPyramid(heightK, widthK, params.finalLambda, params.lambdaMultiplier, w, h, nc, params.kernelSizeMultiplier, params.maxLambda); 
#endif
//	cout<<"Back to ctf"<<endl;

//**Multiscale Processing
	int scales = rbp.scales;
	int hs, ws; //Ms = hs
	int hKs,wKs;  //Mks = hKs
	int hu, wu, scale; //Mu = hu; wu = wu
	float lambda;
//For Output image
	string a, c, d, eK, fK, gK, aI, cI, dI;
	char b;
	aI = "Scaled Input";
	a = "Deblurred Output";
	eK = "Kernel";
	cv::Mat uCloned2;
	cv::Mat kCloned2;
	Timer mytimer;
	for (scale=scales; scale>= 1; scale--) 
	{
	mytimer.start();	
	//**Required dimensions of image and kernel (from Pyramid function)
		hs = rbp.hp[scale]; //Ms = hs
		ws = rbp.wp[scale]; //Ns = ws

		hKs = rbp.hKp[scale];
		wKs = rbp.wKp[scale];
		lambda = rbp.lambdas[scale];

	//**Defining Image from (pyramid scheme) 
		cv::Mat fs(hs, ws, f.type());
		cv::Mat fCloned = f.clone(); 
		cv::resize(fCloned, fs, cv::Size(ws, hs), 0, 0, CV_INTER_CUBIC);

	//**Required dimensions of u
		hu = (hs + hKs - 1);
		wu = (ws + wKs - 1);

	//**Resizing u and k
		cv::Mat uDest; 
		cv::Mat kDest; 				
 				
		cv::resize(u, uDest, cv::Size(wu, hu), 0, 0, CV_INTER_CUBIC);
		cv::resize(k, kDest, cv::Size(wKs, hKs), 0, 0, CV_INTER_CUBIC);

	//**Making negative vales as 0 and normalization
		//cout<<kDest<<endl;
		kDest=KernelProjection(kDest).clone();
		//cout<<kDest<<endl;
		/*float sum = 0;

		for (int y = 0; y<kDest.rows; y++)
		{
			for(int x = 0; x<kDest.cols; x++)
			{
			if (kDest.at<float>(y,x)<0)
				{
					kDest.at<float>(y,x) = 0;
				}					
				sum = sum + kDest.at<float>(y,x);
			}
		}
		for (int y = 0; y<kDest.rows; y++)
		{
			for(int x = 0; x<kDest.cols; x++)
			{
				kDest.at<float>(y,x) = 	kDest.at<float>(y,x) / sum;
			}			
		}*/
	cv::Mat fsOut = fs.clone();
	
	cout<<"Scaled Number: "<<scale<<endl<<endl<<endl<<endl<<endl<<endl; 	
	cout<<"Scaled Input size: "<<fsOut.rows<<" x "<<fsOut.cols<<endl<<endl<<endl<<endl<<endl<<endl;
	cout<<"k Before Blind size: "<<kDest.rows<<" x "<<kDest.cols<<endl<<endl<<endl<<endl<<endl<<endl;

	//**blind function
	cv::Mat fsCloned = fs.clone();
	cv::Mat uDestCloned = uDest.clone();
	BlindReturn rb = blind(fsCloned, lambda, uDestCloned, kDest, scale, 1000);
	k = rb.k.clone();
	
	cout<<"kernel after Blind size: "<<k.rows<<" x "<<k.cols<<endl<<endl<<endl<<endl<<endl<<endl;
	
	//**dec function
	cout<<"u sent to dec size: "<<uDestCloned.rows<<" x "<<uDestCloned.cols<<endl<<endl<<endl<<endl<<endl<<endl;

	kCloned2 = k.clone();	
	u = dec(fsCloned, lambda, uDestCloned, kCloned2, scale, 1000);
	cout<<"Image deblurred size: "<<u.rows<<" x "<<u.cols<<endl<<endl<<endl<<endl<<endl<<endl;
	uCloned2 = u.clone();	
	
	mytimer.end();

	cout<<"Time for scale number"<<scale<<" : "<<mytimer.get()<<"s"<<endl;
	//**SHOW IMAGE	

	int xImage = 40*scale;
	int yImage = 40;
	
// Printing Scaled Input		
	showImage("fs", fsOut, 40, 40);
	b = static_cast<char> (scale+48);
	cI =  aI + b;	
	showImage(cI, fsOut, xImage, yImage);
	dI = cI + ".png";
	cv::imwrite(dI, fsOut*255.f);

// Printing Kernel
	double minVal,maxVal;
	cv::minMaxLoc(kCloned2,&minVal,&maxVal);
	cv::Mat KOut=kCloned2/maxVal;
	fK = eK + b;	
	showImage(fK, KOut, xImage, yImage);
	gK = fK + ".png";
    cv::imwrite(gK, KOut*255.f); 
// Printing Sharpened Image
	c =  a + b;	
	cv::Mat uOut = u.clone();	
	showImage(c, uOut, xImage, yImage);
	d = c + ".png";
    cv::imwrite(d, uOut*255.f); 

}
//	cout<<endl<<"Exiting coarseToFine"<<endl; 
	double minVal,maxVal;	
	cv::minMaxLoc(kCloned2,&minVal,&maxVal);
	cv::Mat KOut=kCloned2/maxVal;
	showImage("Final Kernel", KOut, 40, 40);
    cv::imwrite("FinalKernel.png", KOut*255.f); 


	showImage("Deblurred Image", uCloned2, 240,40);
    cv::imwrite("DeblurredImage.png", uCloned2*255.f); 
	return;
}

