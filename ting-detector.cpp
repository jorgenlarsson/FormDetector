/* TING detector
 * J¿rgen Larsson 2014
 *
 */
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/core.hpp"
#include "oscpack/osc/OscOutboundPacketStream.h"
#include "oscpack/ip/UdpSocket.h"

#include <iostream>
#include <math.h>
#include <string.h>

#define PI 3.14159
#define ADDRESS "127.0.0.1"
#define PORT 7000
#define OUTPUT_BUFFER_SIZE 1024

using namespace cv;
using namespace std;

int thresh = 50, N = 11, ssize = 100, cosslider = 300, cannythresh = 50, ksize = 3, cycles = 100;
int ksizeslider, brightness, contrast;
float centerdistf = 20;
										// minimum square size
										// maxCosine * 1000
const char* wndname3 = "Adjusted";
const char* wndname2 = "Lines";


// helper function:
// finds a cosine of angle between vectors
// from pt0->pt1 and from pt0->pt2
static double angle( Point pt1, Point pt2, Point pt0 )
{
    double dx1 = pt1.x - pt0.x;
    double dy1 = pt1.y - pt0.y;
    double dx2 = pt2.x - pt0.x;
    double dy2 = pt2.y - pt0.y;
    return (dx1*dx2 + dy1*dy2)/sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2) + 1e-10);
}

// helper function:
// distance between center of squares
static float PDistance(Point pt0, Point pt1)
{
    return sqrt((pt1.x - pt0.x)*(pt1.x - pt0.x) + (pt1.y - pt0.y)*(pt1.y - pt0.y));
}


// returns sequence of squares detected on the image.
// the sequence is stored in the specified memory storage
static void findSquares( const Mat& adjusted, vector<vector<Point> >& squares )
{
    squares.clear();

    Mat gray0(Size(640, 480), CV_8UC1);
    Mat gray(Size(640, 480), CV_8UC1);
    // Mat adjusted(Size(640, 480), CV_8UC3);


    vector<vector<Point> > contours;

        //Apply blur to smooth edges and use adapative thresholding
         cvtColor(adjusted, gray0, CV_RGB2GRAY); //convert to gray
         ksize = (ksizeslider * 2) + 1; // ensure the ksize is odd and > 0
         Size size(ksize,ksize);
         GaussianBlur(gray0, gray0, size, 0);

                // apply Canny. Take the upper threshold from slider
                // and set the lower to 0 (which forces edges merging)
                Canny(gray0, gray0, 0, cannythresh, 3);

                // dilate canny output to remove potential
                // holes between edge segments
                dilate(gray0, gray, Mat(), Point(-1,-1));
                imshow(wndname2, gray);

            // find contours and store them all as a list
            findContours(gray, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);
            vector<Point> approx;

            // test each contour
            for( size_t i = 0; i < contours.size(); i++ )
            {
                // approximate contour with accuracy proportional
                // to the contour perimeter
                approxPolyDP(Mat(contours[i]), approx, arcLength(Mat(contours[i]), true)*0.02, true);

                // square contours should have 4 vertices after approximation
                // relatively large area (to filter out noisy contours)
                // and be convex.
                // Note: absolute value of an area is used because
                // area may be positive or negative - in accordance with the
                // contour orientation
                if( approx.size() == 4 &&
                    fabs(contourArea(Mat(approx))) > ssize &&
                    isContourConvex(Mat(approx)) )
                {
                    double maxCosine = cosslider/1000;

                    for( int j = 2; j < 5; j++ )
                    {
                        // find the maximum cosine of the angle between joint edges
                        double cosine = fabs(angle(approx[j%4], approx[j-2], approx[j-1]));
                        maxCosine = MAX(maxCosine, cosine);
                    }

                    // if cosines of all angles are small
                    // (all angles are ~90 degree) then write quandrange
                    // vertices to resultant sequence
                    float cossliderf;
                    cossliderf=cosslider;
                    if( maxCosine < cossliderf/1000 )
                        squares.push_back(approx);
                    // cout << "number of squares=" << squares.size() << "\n";
                }
            }
        }
//    }

// }
// the function draws all the squares in the image
static void drawSquares( Mat& image, const vector<vector<Point> >& squares )
	{
	Point null = Point(0, 0);
	vector<Point> sqrpos(20, null);
	Point icenter1 = Point(0, 0);
	size_t s = squares.size();
	int cyc = 0;
	vector<double> angles(20,0);
	double angle1;
	size_t numberOfSqrs = 0;
	int positions[4][20];
	if( s > 20 )
		{
		cout << "More than 20 squares - adjust filters." << "\n";
		}
	if( s < 21 )
		{
		for( size_t i = 0; i < squares.size(); i++ )
			{
			int xcenter = (squares[i][1].x + squares[i][3].x)/2;
			int ycenter = (squares[i][1].y + squares[i][3].y)/2;
			double x1=squares[i][0].x; double x2=squares[i][1].x; double y1=squares[i][0].y; double y2=squares[i][1].y;
			icenter1 = Point(xcenter, ycenter);
			angle1 = (atan2(y2-y1, x2-x1) * 180.0 / PI);
			if(angle1<0){angle1=angle1+360;}
			angle1=fmod(angle1, 90);
			int angle1i = angle1;
			if( s > numberOfSqrs )
				{
				cyc = 0;
				++numberOfSqrs;
				auto k = find(sqrpos.begin(), sqrpos.end(), null);
				auto ind = distance(sqrpos.begin(), k);
				sqrpos[ind] = icenter1;
				angles[ind] = angle1;
				}
				for( int j = 0; j < 20; j++ )	//check for distance of squares
					{
					if( PDistance(sqrpos[j],icenter1) < centerdistf )  // set to 20 pixels
						{
						sqrpos[j] = icenter1;
						angles[j] = angle1;						// if less than centerdist, update vector
						}
					}
			if( s == numberOfSqrs)
				{
				cyc = 0;
				for( int j = 0; j < 20; j++ )	//check for distance of squares
					{
					if( PDistance(sqrpos[j],icenter1) < centerdistf )  // set to 20 pixels
						{
						sqrpos[j] = icenter1;
						angles[j] = angle1;			// if less than centerdist, update vector
						}
					}
				}
				if( s < numberOfSqrs )
				{
				--numberOfSqrs;
				++cyc;							// add to cycles
				for( int j = 0; j < 20; j++ )	//check for distance of squares
					{
					if( PDistance(sqrpos[j],icenter1) < centerdistf )  // set to 20 pixels
						{
						sqrpos[j] = icenter1;
						angles[j] = angle1;		// if less than centerdist, update vector
						}
					if( (PDistance(sqrpos[j],icenter1) > centerdistf) && (cyc == cycles) )
						{
						sqrpos[j] = null;
						angles[j] = 0;			// updates empty point after number of cycles (100)
						}
					}
				}
			for( int j = 0; j < 20; j++ )
				{
				positions[0][j] = j;
				positions[1][j] = sqrpos[j].x;
				positions[2][j] = sqrpos[j].y;
				positions[3][j] = angles[j];
				}
				/*UdpTransmitSocket transmitSocket( IpEndpointName( ADDRESS, PORT ) );
				    char buffer[OUTPUT_BUFFER_SIZE];
				    osc::OutboundPacketStream p( buffer, OUTPUT_BUFFER_SIZE );

				    p << osc::BeginBundleImmediate
				        << osc::BeginMessage( "positions" )
				    << positions[0][0] << positions[1][0] << positions[2][0] << positions[3][0]
				    << positions[0][1] << positions[1][1] << positions[2][1] << positions[3][1]
				    << positions[0][2] << positions[1][2] << positions[2][2] << positions[3][2]
				    << positions[0][3] << positions[1][3] << positions[2][3] << positions[3][3]
				    << positions[0][4] << positions[1][4] << positions[2][4] << positions[3][4]
				    << positions[0][5] << positions[1][5] << positions[2][5] << positions[3][5]
				    << positions[0][6] << positions[1][6] << positions[2][6] << positions[3][6]
				    << positions[0][7] << positions[1][7] << positions[2][7] << positions[3][7]
				    << positions[0][8] << positions[1][8] << positions[2][8] << positions[3][8]
				    << positions[0][9] << positions[1][9] << positions[2][9] << positions[3][9]
				    << positions[0][10] << positions[1][10] << positions[2][10] << positions[3][10]
				    << positions[0][11] << positions[1][11] << positions[2][11] << positions[3][11]
				    << positions[0][12] << positions[1][12] << positions[2][12] << positions[3][12]
				    << positions[0][13] << positions[1][13] << positions[2][13] << positions[3][13]
				    << positions[0][14] << positions[1][14] << positions[2][14] << positions[3][14]
				    << positions[0][15] << positions[1][15] << positions[2][15] << positions[3][15]
				    << positions[0][16] << positions[1][16] << positions[2][16] << positions[3][16]
				    << positions[0][17] << positions[1][17] << positions[2][17] << positions[3][17]
				    << positions[0][18] << positions[1][18] << positions[2][18] << positions[3][18]
				    << positions[0][19] << positions[1][19] << positions[2][19] << positions[3][19]
				         << osc::EndMessage
				    	<< osc::EndBundle;

				    transmitSocket.Send( p.Data(), p.Size() );*/
				for(auto i=0; i<20; ++i)
				{
				cout << positions[0][i] << "\t" << positions[1][i] << "\t" << positions[2][i] << "\t" << positions[3][i] << "\t" << "\n";
				}

				// cout << j << "\t" << "x=" << sqrpos[j].x << "\t" << "y=" << sqrpos[j].y << "\t" << angles[j] << "\n";
				}
			}
		}

int main(int /*argc*/, char** /*argv*/)
{
	VideoCapture cap(0);
	    if(!cap.isOpened()) return -1;

	    Mat frame, adjusted(Size(640, 480), CV_8UC3), gray(Size(640, 480), CV_8UC1);

	int centerdist = centerdistf;
    namedWindow( wndname3, 1 );
    namedWindow( wndname2, 1);
    createTrackbar( "Threshold", wndname3, &thresh, 255);
    createTrackbar( "Brightness", wndname3, &brightness, 100);
    createTrackbar( "Contrast", wndname3, &contrast, 400);
    createTrackbar( "MinSqSize", wndname3, &ssize, 1000);
    createTrackbar( "MaxCosine", wndname3, &cosslider, 1000);
    createTrackbar( "KSize", wndname3, &ksizeslider, 40);
    createTrackbar( "CannyThreshold", wndname3, &cannythresh, 1000);
    createTrackbar( "Delay", wndname3, &cycles, 100);
    createTrackbar( "Distance", wndname3, &centerdist, 500);



    vector<vector<Point> > squares;
    vector<Point> sqrpos;
    for(;;)
    {
    	cap >> frame;
    	float contrastf = contrast;
    	frame.convertTo(adjusted, -1, (contrastf/100), brightness);
        findSquares(adjusted, squares);
        drawSquares(adjusted, squares);
        imshow(wndname3, adjusted);

                if(waitKey(10) >= 0) break;
    }

    return 0;
}



