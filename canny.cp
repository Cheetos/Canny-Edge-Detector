/*****************************************
 Author: David Esparza Alba
 Date: Someday in 2011
 Ritsumeikan University.
 *****************************************/

#include <cv.h>
#include <highgui.h>
#include <string.h>
#include <math.h>
#include <list>

#define EPS 1e-3
#define PI acos(-1.0)
#define WHITE 255
#define GRAY 128
#define BLACK 0

int LOW_THRESHOLD = 16;
int HIGH_THRESHOLD = 32;

using namespace std;

list <int> L;
IplImage *imgSrc;
IplImage *imgDst;
int nRows;
int nColumns;

void Sobel(double **, double **);
void Float_to_Char(double **);
double Max_Value(double **);
double Min_Value(double **);
double ** ConvImgToDbl(IplImage *);
void ConvDblToImg(double **, IplImage *);

void Smoothing(double **, double **);
void Non_Maximum_Supression(double **, double **);
void Double_Tresholding(double **, double **);
void Edge_Tracking(double **, double **);
void DFS(double **);

void Trackbar_Function(int);

int main()
{
    int prevHthreshold = -1;
    int prevLthreshold = -1;
    double **CannyA;
    double **CannyB;
    double **CannyC;
    double **CannyD;
    double **CannyE;
    double **CannyF;
    
    cvNamedWindow("Canny A");
    cvNamedWindow("Canny B");
    cvNamedWindow("Canny C");
    cvNamedWindow("Canny D");
    cvNamedWindow("Canny E");
    cvNamedWindow("Canny F");
    
    IplImage *imgTemp = cvLoadImage("/Users/MacBook/Desktop/David/PHOTOS/Angel.jpg");
    imgSrc = cvCreateImage(cvSize(imgTemp->width, imgTemp->height), IPL_DEPTH_8U, 1);
    imgDst = cvCreateImage(cvSize(imgTemp->width, imgTemp->height), IPL_DEPTH_8U, 1);
    cvCvtColor(imgTemp, imgSrc, CV_RGB2GRAY );
    
    nRows = imgSrc->height;
    nColumns = imgSrc->width;
    
    cvShowImage("Canny A", imgSrc);
    //cvSaveImage("cannyA.jpg", imgSrc);
    
    CannyA = ConvImgToDbl(imgSrc);
    CannyB = ConvImgToDbl(imgDst);
    CannyC = ConvImgToDbl(imgDst);
    CannyD = ConvImgToDbl(imgDst);
    CannyE = ConvImgToDbl(imgDst);
    CannyF = ConvImgToDbl(imgDst);
    
    Smoothing(CannyA, CannyB);
    ConvDblToImg(CannyB, imgDst);
    cvShowImage("Canny B", imgDst);
    //cvSaveImage("cannyB.jpg", imgDst);
    
    Sobel(CannyB, CannyC);
    ConvDblToImg(CannyC, imgDst);
    cvShowImage("Canny C", imgDst);
    //cvSaveImage("cannyC.jpg", imgDst);
    
    Non_Maximum_Supression(CannyC, CannyD);
    ConvDblToImg(CannyD, imgDst);
    cvShowImage("Canny D", imgDst);
    //cvSaveImage("cannyD.jpg", imgDst);
    
    cvCreateTrackbar("Low Threshold:", "Canny E", &LOW_THRESHOLD, 255, Trackbar_Function);
    cvCreateTrackbar("High Threshold:", "Canny E", &HIGH_THRESHOLD, 255, Trackbar_Function);
    
    while(1)
    {
        if(HIGH_THRESHOLD != prevHthreshold || LOW_THRESHOLD != prevLthreshold)
        {
            Double_Tresholding(CannyD, CannyE);
            ConvDblToImg(CannyE, imgDst);
            cvShowImage("Canny E", imgDst);
            //cvSaveImage("cannyE.jpg", imgDst);
            
            Edge_Tracking(CannyE, CannyF);
            ConvDblToImg(CannyF, imgDst);
            cvShowImage("Canny F", imgDst);
            cvSaveImage("cannyF.jpg", imgDst);
            
            prevHthreshold =  HIGH_THRESHOLD;
            prevLthreshold = LOW_THRESHOLD;
        } 
        
        char c = cvWaitKey(33);
        if(c == 27)
            break;
    }
    
    cvDestroyWindow("CannyA");
    cvDestroyWindow("CannyB");
    cvDestroyWindow("CannyC");
    cvDestroyWindow("CannyD");
    cvDestroyWindow("CannyE");
    cvDestroyWindow("CannyF");
    
    cvReleaseImage(&imgSrc);
    cvReleaseImage(&imgDst);
    
    return 0;
}

void Trackbar_Function(int)
{
}

void Smoothing(double **S, double **D)
{
    int i, j;
    int r, c;
    
    double M[5][5] = {
        {2, 4, 5, 4, 2}, 
        {4, 9, 12, 9, 4},
        {5, 12, 15, 12, 5},
        {4, 9, 12, 9, 4},
        {2, 4, 5, 4, 2}
    };
    
    for(i=0; i<nRows; i++)
    {
        for(j=0; j<nColumns; j++)
        {
            for(r=i-2; r<=i+2; r++)
            {
                for(c=j-2; c<=j+2; c++)
                {
                    if(r < 0 || r >= nRows || c < 0 || c >= nColumns)
                        continue;
                    
                    D[i][j] += S[r][c]*M[r-i+2][c-j+2];
                }
            }
            
            D[i][j] /= 159.0;
        }
    }
}

void Non_Maximum_Supression(double **S, double **D)
{
    int i, j;
    double refVal;
    double gx, gy;
    double theta;
    
    for(i=1; i<nRows-1; i++)
    {
        for(j=1; j<nColumns-1; j++)
        {
            gx = (S[i-1][j+1] + 2*S[i][j+1] + S[i+1][j+1]) - (S[i-1][j-1] + 2*S[i][j-1] + S[i+1][j-1]);
            gy = (S[i-1][j-1] + 2*S[i-1][j] + S[i-1][j+1]) - (S[i+1][j-1] + 2*S[i+1][j] + S[i+1][j+1]);
            
            if(fabs(gy) < EPS)
                theta = PI/2.0;
            else 
                theta = atan(fabs(gy)/fabs(gx));
            
            if(theta < PI/8.0)
                refVal = S[i][j-1] > S[i][j+1] ? S[i][j-1] : S[i][j+1];
            else if(theta < 3.0*PI/8.0)
                refVal = S[i+1][j-1] > S[i-1][j+1] ? S[i+1][j-1] : S[i-1][j+1];
            else if(theta < 5.0*PI/8.0)
                refVal = S[i+1][j] > S[i-1][j] ? S[i+1][j] : S[i-1][j];
            else if(theta < 7.0*PI/8.0)
                refVal = S[i+1][j+1] > S[i-1][j-1] ? S[i+1][j+1] : S[i-1][j-1];
            else 
                refVal = S[i][j-1] > S[i][j+1] ? S[i][j-1] : S[i][j+1];

            if(S[i][j] > refVal)
                D[i][j] = S[i][j];
            else 
                D[i][j] = 0.0;
        }
    }
}

void Double_Tresholding(double **S, double **D)
{
    int i, j;
    
    for(i=0; i<nRows; i++)
    {
        for(j=0; j<nColumns; j++)
        {
            if(S[i][j] >= HIGH_THRESHOLD)
                D[i][j] = WHITE;
            else if(S[i][j] >= LOW_THRESHOLD)
                D[i][j] = GRAY;
            else 
                D[i][j] = BLACK;
        }
    }
}

void Edge_Tracking(double **S, double **D)
{
    int i, j;
    
    for(i=0; i<nRows; i++)
        for(j=0; j<nColumns; j++)
            D[i][j] = S[i][j];
    
    for(i=0; i<nRows; i++)
    {
        for(j=0; j<nColumns; j++)
        {
            if(D[i][j] == WHITE)
            {
                L.push_front(i*nColumns + j);
                DFS(D);
            }
        }
    }
    
    for(i=0; i<nRows; i++)
        for(j=0; j<nColumns; j++)
            if(D[i][j] == GRAY)
                D[i][j] = BLACK;
}

void DFS(double **D)
{
    int i, j, k;
    int row, col;
    
    while(!L.empty())
    {
        k = L.front();
        L.pop_front();
        
        row = k/nColumns;
        col = k%nColumns;
        
        for(i=row-1; i<=row+1; i++)
        {
            for(j=col-1; j<=col+1; j++)
            {
                if(i < 0 || i >= nRows || j < 0 || j >= nColumns)
                    continue;
                
                if(D[i][j] == GRAY)
                {
                    D[i][j] = WHITE;
                    L.push_front(i*nColumns + j);
                }
            }
        }
    }
}

double ** ConvImgToDbl(IplImage *img)
{
    CvScalar color;
    double **T;
    int i, j;
    
    T = new double *[img->height];
    for(i=0; i<img->height; i++)
    {
        T[i] = new double[img->width];
        for(j=0; j<img->width; j++)
        {
            color = cvGet2D(img, i, j);
            T[i][j] = color.val[0];
        }
    }
    
    return T;
}

void ConvDblToImg(double **I, IplImage *img)
{
    int i, j;
    int r, g, b;
    
    for(i=0; i<nRows; i++)
    {
        for(j=0; j<nColumns; j++)
        {
            r = g = b = (int)I[i][j];
            cvSet2D(img, i, j, cvScalar(r, g, b));
        }
    }
}

void Sobel(double **S, double **D)
{
    int i, j;
    double gx, gy;
    
    for(i=1; i<nRows-1; i++)
    {
        for(j=1; j<nColumns-1; j++)
        {
            gx = (S[i-1][j+1] + 2*S[i][j+1] + S[i+1][j+1]) - (S[i-1][j-1] + 2*S[i][j-1] + S[i+1][j-1]);
            gy = (S[i-1][j-1] + 2*S[i-1][j] + S[i-1][j+1]) - (S[i+1][j-1] + 2*S[i+1][j] + S[i+1][j+1]);
            
            D[i][j] = sqrt(gx*gx + gy*gy);
        }
    }
    
    Float_to_Char(D);
}

void Float_to_Char(double **I)
{
    int i, j;
    double maxVal = Max_Value(I);
    double minVal = Min_Value(I);
    
    for(i=0; i<nRows; i++)
        for(j=0; j<nColumns; j++)
            I[i][j] = 255.0*((I[i][j] - minVal)/(maxVal - minVal));
}

double Max_Value(double **I)
{
    int i, j;
    double maxVal = I[0][0];
    
    for(i=0; i<nRows; i++)
        for(j=0; j<nColumns; j++)
            if(I[i][j] > maxVal)
                maxVal = I[i][j];
    
    return maxVal;
}

double Min_Value(double **I)
{
    int i, j;
    double minVal = I[0][0];
    
    for(i=0; i<nRows; i++)
        for(j=0; j<nColumns; j++)
            if(I[i][j] < minVal)
                minVal = I[i][j];
    
    return minVal;
}