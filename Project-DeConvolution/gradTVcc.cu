 // Total Variation gradient
 #include "library.h"
 #include <cmath>
 using namespace std;
 
cv::Mat fxforwFunc(cv::Mat fCloned);
cv::Mat fyforwFunc(cv::Mat fCloned);
cv::Mat fxbackFunc(cv::Mat fCloned);
cv::Mat fybackFunc(cv::Mat fCloned);
cv::Mat fxmixdFunc(cv::Mat fCloned);
cv::Mat fymixdFunc(cv::Mat fCloned);
cv::Mat customOp(cv::Mat fCloned, cv::Mat g, float eps);

cv::Mat gradTVcc(cv::Mat f, float eps)
{
cv::Mat fCloned = f.clone();
cv::Mat fxforw = fxforwFunc(fCloned);
cv::Mat fyforw = fyforwFunc(fCloned);
cv::Mat fxback = fxbackFunc(fCloned);
cv::Mat fyback = fybackFunc(fCloned);
cv::Mat fxmixd = fxmixdFunc(fCloned);
cv::Mat fymixd = fymixdFunc(fCloned);

cv::Mat fxforwCloned = fxforw.clone();
cv::Mat fyforwCloned = fyforw.clone();
cv::Mat fxbackCloned = fxback.clone();
cv::Mat fybackCloned = fyback.clone();
cv::Mat fxmixdCloned = fxmixd.clone();
cv::Mat fymixdCloned = fymixd.clone();
 
 cv::Mat divTV(f.cols, f.rows,CV_32FC3);
 divTV = 0.0;
divTV  = (fyforwCloned + fxforwCloned)/customOp(fxforwCloned, fyforwCloned, eps) - fybackCloned/customOp(fybackCloned, fymixdCloned, eps) - fxbackCloned/customOp(fxmixdCloned, fxbackCloned, eps);
 
 return (divTV);
 }
 // Square, add, sqrt, max
 cv::Mat customOp(cv::Mat f, cv::Mat g, float eps)
 {
 	cv::Mat Output (f.rows, f.cols, f.type());
 	Output = 0.0;
 	cv::Mat fSquare (f.rows, f.cols, f.type());
 	fSquare = 0.0;
 	cv::Mat gSquare (f.rows, f.cols, f.type());
 	gSquare = 0.0;
 
 	cv::pow(f,2,fSquare);
 	cv::pow(g,2,gSquare);
 	
 	Output = fSquare + gSquare;
 
 	cv::sqrt(Output, Output);
 		
 	for(int y = 0; y<Output.rows; y++)
 	{
 		for (int x = 0; x<Output.cols; x++)
 		{
 			cv::Vec3f intensity = Output.at<cv::Vec3f>(y, x);
 			if (intensity.val[0]<eps)
 				intensity.val[0]=eps;
 			if (intensity.val[1]<eps)
 				intensity.val[1]=eps;
 			if (intensity.val[2]<eps)
 				intensity.val[2]=eps;
 			Output.at<cv::Vec3f>(y, x)=intensity;		
 		}
 	}	
 	return Output;
 }	   			
 
 //fxforw = f([2:end end],:,:)-f; //2nd to last row then last row is repeated - original array
 cv::Mat fxforwFunc(cv::Mat f)
 {
 	cv::Mat fxforw(f.rows, f.cols, f.type());
 	fxforw = 0.0;
 	for (int i = 0; i<f.rows; i++)
 	{
 		for(int j = 0; j<(f.cols-1); j++)
 		{
 			cv::Vec3f intensity2 = f.at<cv::Vec3f>(i, j + 1);
 			cv::Vec3f intensity1 = f.at<cv::Vec3f>(i, j);
 			fxforw.at<cv::Vec3f>(i, j) = intensity2 - intensity1;  
 		}
 	}
 return fxforw;
 }
 //fxback = f-f([1 1:end-1],:,:);
 cv::Mat fxbackFunc(cv::Mat f)
 {
 	cv::Mat fxback(f.rows, f.cols, f.type());
 	fxback = 0.0;
 	for (int i = 0; i<f.rows; i++)
 	{
 		for(int j = 1; j<(f.cols); j++)
 		{
 			cv::Vec3f intensity2 = f.at<cv::Vec3f>(i, j);
 			cv::Vec3f intensity1 = f.at<cv::Vec3f>(i, j - 1);
 			fxback.at<cv::Vec3f>(i, j) = intensity2 - intensity1;  
 		}
 	}
 return fxback;
 }
 //fyforw = f(:,[2:end end],:)-f;
 cv::Mat fyforwFunc(cv::Mat f)
 {
 	cv::Mat fyforw(f.rows, f.cols, f.type());
 	fyforw = 0.0;
 	for (int i = 0; i<(f.rows - 1); i++)
 	{
 		for(int j = 0; j<(f.cols); j++)
 		{
 			cv::Vec3f intensity2 = f.at<cv::Vec3f>(i + 1, j);
 			cv::Vec3f intensity1 = f.at<cv::Vec3f>(i, j);
 			fyforw.at<cv::Vec3f>(i, j) = intensity2 - intensity1;  
 		}
 	}
 return fyforw;
 }
 //fyback = f-f(:,[1 1:end-1],:);
 cv::Mat fybackFunc(cv::Mat f)
 {
 	cv::Mat fyback(f.rows, f.cols, f.type());
 	fyback = 0.0;
 	for (int i = 1; i<(f.rows); i++)
 	{
 		for(int j = 0; j<(f.cols); j++)
 		{
 			cv::Vec3f intensity2 = f.at<cv::Vec3f>(i, j);
 			cv::Vec3f intensity1 = f.at<cv::Vec3f>(i - 1, j);
 			fyback.at<cv::Vec3f>(i, j) = intensity2 - intensity1;  
 		}
 	}
 return fyback;
 }
 /////////////////////////////////// fxmixd f([1 1:end-1],[2:end end],:)-f([1 1:end-1],:,:)
 cv::Mat fxmixdFunc(cv::Mat f)
 {
 	// x and y are original image column and row index;
 	// iN and jN are new image column and row index
 
 	// DEFINING fxmixd2 :  f([2:end end],[1 1:end-1],:)
 	// DEFINING fxmixd2a : f([2:end end],:,:)
 //	cv::Mat fxmixd2a
 	cv::Mat fxmixd2a(f.rows, f.cols, f.type());
 	fxmixd2a = 0.0;
 	int y = 0;
 	int x;		
 		for(x = 0; x<(f.cols);x++)
 		{
 			int yN = y;
 			int xN = x;
 			cv::Vec3f intensityA = f.at<cv::Vec3f>(y, x);				
 			fxmixd2a.at<cv::Vec3f>(yN, xN) = intensityA;
 		}
 	
 	for (y = 0; y<(f.rows - 1); y++) 
 	{							
 		for(x = 0; x<(f.cols);x++)
 		{
 			int yN = y + 1;
 			int xN = x;
 			cv::Vec3f intensityA = f.at<cv::Vec3f>(y, x);				
 			fxmixd2a.at<cv::Vec3f>(yN, xN) = intensityA;
 		}
 	}
 	
 //  fxmixd f([1 1:end-1],[2:end end],:)-f([1 1:end-1],:,:)
 	cv::Mat fxmixd2b(f.rows, f.cols, f.type());
 	fxmixd2b = 0.0;
 	for (y = 0; y<(f.rows); y++) 
 	{					
 		for(x = 1; x<(f.cols);x++)
 		{
 			int yN = y;
 			int xN = x - 1;
 			cv::Vec3f intensityA = fxmixd2a.at<cv::Vec3f>(y, x);
 			fxmixd2b.at<cv::Vec3f>(yN, xN) = intensityA;
 		}
 	}
 	x--;
 	for(y = 0; y<(f.rows);y++)
 		{
 			int yN = y;
 			int xN = x;
 			cv::Vec3f intensityA = fxmixd2a.at<cv::Vec3f>(y, x);
 			fxmixd2b.at<cv::Vec3f>(yN, xN) = intensityA;
 		}	
 
 return (fxmixd2b-fxmixd2a);
 }
 
 //////////////////////////////// fymixdFunc
 cv::Mat fymixdFunc(cv::Mat f)
 {
 	// x and y are original image column and row index;
 	// iN and jN are new image column and row index
 
 	// DEFINING fymixd2a : f([2:end end],:,:)
 	cv::Mat fymixd2a(f.rows, f.cols, f.type());
 	fymixd2a = 0.0;
 	int y;
 	for (y = 1; y<(f.rows); y++) 
 	{					
 		int x;		
 		for(x = 0; x<(f.cols);x++)
 		{
 			int yN = y - 1;
 			int xN = x;
 			cv::Vec3f intensityA = f.at<cv::Vec3f>(y, x);				
 			fymixd2a.at<cv::Vec3f>(yN, xN) = intensityA;
 		}
 	}
 	y = f.rows - 1; // For the last row
 	int x;	
 	for(x = 0; x<(f.cols);x++)
 		{
 			int yN = y;
 			int xN = x;
 			cv::Vec3f intensityA = f.at<cv::Vec3f>(y, x);				
 			fymixd2a.at<cv::Vec3f>(yN, xN) = intensityA;
 		}
 	
 // for cv::Mat fymixd2b(f.rows, f.cols, f.type()); f([2:end end],[1 1:end-1],:)
 	cv::Mat fymixd2b(f.rows, f.cols, f.type());
 	fymixd2b = 0.0;
 	x = 0;
 	for(y = 0; y<(f.rows);y++)
 		{
 			int yN = y;
 			int xN = x;
 			cv::Vec3f intensityA = fymixd2a.at<cv::Vec3f>(y, x);
 			fymixd2b.at<cv::Vec3f>(yN, xN) = intensityA;
 		}	
 	for (y = 0; y<(f.rows); y++) 
 	{					
 		for(x = 0; x<(f.cols-1);x++)
 		{
 			int yN = y;
 			int xN = x + 1;
 			cv::Vec3f intensityA = fymixd2a.at<cv::Vec3f>(y, x);
 			fymixd2b.at<cv::Vec3f>(yN, xN) = intensityA;
 		}
 	}
 
 // for f(:,[1 1:end-1],:)
 	cv::Mat fymixd1(f.rows, f.cols, f.type());
 	fymixd1 = 0.0;
 	x = 0;
 	for(y = 0; y<(f.rows);y++)
 		{
 			int yN = y;
 			int xN = x;
 
 			cv::Vec3f intensityA = f.at<cv::Vec3f>(y, x);			
 			fymixd1.at<cv::Vec3f>(yN, xN) = intensityA;
 		}	
 	for (y = 0; y<(f.rows); y++) 
 	{					
 		for(x = 0; x<(f.cols-1);x++)
 		{
 			int yN = y;
 			int xN = x + 1;
 		cv::Vec3f intensityA = f.at<cv::Vec3f>(y, x);			
 			fymixd1.at<cv::Vec3f>(yN, xN) = intensityA;
 		}
 	}
 return (fymixd2b-fymixd1);
 }
