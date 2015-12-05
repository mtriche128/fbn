/**
 * @file   proc_kernel.cpp
 * @author Matthew Triche
 * @brief  This source file contains the processing kernel.
 */

/* ------------------------------------------------------------------------- *
 * Include Headers and Namespaces                                            *
 * ------------------------------------------------------------------------- */

#include <iostream>
#include <vector>
#include <algorithm>
#include <time.h>
#include <stdio.h>
#include <math.h>
#include "opencv2/opencv_modules.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/gpu/gpu.hpp"
#include "opencv2/nonfree/gpu.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/flann/flann.hpp"

#include "proc_kernel.h"

using namespace std;
using namespace cv;
using namespace cv::gpu;

/* ------------------------------------------------------------------------- *
 * Declare Internal Functions                                                *
 * ------------------------------------------------------------------------- */

double homographyToHeading(const Mat&);
float angleBetweenRays(const Vec3f &a, const Vec3f &b);
void transPixelToCamRay(const cam_params_t &params, const Point2f &pix, Vec3f &ray);
Point3f transPixelToCamPoint(const cam_params_t &params, const Point2f &pix, float z);

/* ------------------------------------------------------------------------- *
 * Define Class Methods                                                      *
 * ------------------------------------------------------------------------- */

/**
 * @brief constructor
 *
 * @param objImg  object image
 * @param cparams camecra parameters
 * @param minHess minimum Hessian
 * @param ratio   matching ratio
 */

Kernel::Kernel(const Mat &objImg,
		       const cam_params_t &cparams,
		       int minHess,
		       float ratio)
{
	// make deep copy of camera parameters
	m_camParams.f_x = cparams.f_x;
	m_camParams.f_y = cparams.f_y;
	m_camParams.p_x = cparams.p_x;
	m_camParams.p_y = cparams.p_y;

	m_ratio = ratio;

#ifndef ENABLE_GPU

	m_detector  = new SurfFeatureDetector(minHess);
	m_extractor = new SurfDescriptorExtractor();

#else
	m_surfGPU = new SURF_GPU(minHess,
	                         SURF_N_OCT,
	                         SURF_N_OCT_LAYERS,
	                         SURF_EXTENDED,
	                         KP_RATIO);
#endif

	//-------------------------------------------------------------------------
	// process object image

	Mat procObjImg;

	equalizeHist(objImg, procObjImg); // normalize object image

#ifndef ENABLE_GPU

	m_detector->detect(procObjImg, m_objKpnts);
	m_extractor->compute(procObjImg, m_objKpnts, m_objDesc);

#else

	(*m_surfGPU)(procObjImg, GpuMat(), m_objKpGPU, m_objDescGPU);

#endif

	m_objImgVert.resize(4);
	m_objImgVert[0] = cvPoint(0,0);
	m_objImgVert[1] = cvPoint(objImg.cols, 0);
	m_objImgVert[2] = cvPoint(objImg.cols, objImg.rows);
	m_objImgVert[3] = cvPoint(0, objImg.rows);
}

/**
 * @brief destructor
 */

Kernel::~Kernel()
{
	m_homVert.clear();
	m_homVert.clear();
	m_objKpnts.clear();

#ifndef ENABLE_GPU

	delete m_detector;
	delete m_extractor;

#else

	m_objDescGPU.release();
	m_objKpGPU.release();
	m_surfGPU->releaseMemory();
	delete m_surfGPU;

#endif
}

/**
 * @brief Process the scene image.
 *
 * Upon successful execution of this method, scene features will be placed in
 * members 'm_sceneKpts' and 'm_sceneDesc' or 'm_sceneKpGPU' and
 * 'm_sceneDescGPU' in the case of GPU acceleration.
 */

void Kernel::ProcessSceneImage(const Mat &img)
{
	Mat imgBw, imgHist;

	cvtColor(img, imgBw, CV_BGR2GRAY);
	equalizeHist(imgBw, imgHist);

#ifndef ENABLE_GPU

	m_detector->detect(imgHist, m_sceneKpnts);
	m_extractor->compute(imgHist, m_sceneKpnts, m_sceneDesc);

#else

	(*m_surfGPU)(imgHist, GpuMat(), m_sceneKpGPU, m_sceneDescGPU);

#endif
}

/**
 * @brief Calculate the homography matrix.
 *
 * The homography matrix will be placed in member 'm_hm' upon successful
 * execution of this method.
 *
 * @return
 * True if a homography matrix was successfully executed. False otherwise.
 */

bool Kernel::CalculateHomography()
{
	vector<Point2f> objKeypointsHg;
	vector<Point2f> sceneKeypointsHg;

	cout << "ComputeHomography()" << endl;

	if(m_matches.size() < 4)
	{
		cout << "Warning: Not enough matches found to calculate homography: match count = "
		     << m_matches.size()
		     << endl;

		return false;
	}

	else
		cout << "Sufficient matches to compute homography found." << endl;

	for(int i = 0; i < m_matches.size(); i++)
	{
		objKeypointsHg.push_back(m_objKpnts[m_matches[i].queryIdx].pt);
		sceneKeypointsHg.push_back(m_sceneKpnts[m_matches[i].trainIdx].pt);
	}

	cout << "findHomography()...";
	m_hm = findHomography(objKeypointsHg, sceneKeypointsHg, CV_RANSAC );
	cout << "done" << endl;

	if(m_hm.empty())
	{
		cout << "Warning: An empty homography matrix was returned." << endl;
		return false;
	}

	// check if homography matrix is valid
	double det = determinant(m_hm(Rect(0,0,2,2)));
	cout << "det = " << det << endl;
	if( (det <= 0.01 ) || (det > 10.0) )
	{
		cout << "Warning: Homography matrix isn't valid." << endl;
		//return false;
	}

	cout << "perspectiveTransform()...";
	perspectiveTransform(m_objImgVert, m_homVert, m_hm);
	cout << "done" << endl;

	return true;
}

/**
 * @brief Find feature matches between object and scene images.
 *
 * Upon successful execution of this method, a list of feature matches will be
 * stored in member 'm_matches'.
 *
 * @return True of a list of matches was successful found. False otherwise.
 */

bool Kernel::MatchFeatures()
{
	vector< vector<DMatch> > init_matches;

	cout << "MatchFeatures()" << endl;

#ifndef ENABLE_GPU

	FlannBasedMatcher flann_matcher;

	if(m_objDesc.empty() || m_sceneDesc.empty())
	{
		cout << "Warning: Empty feature set." << endl;
		return false;
	}

	flann_matcher.knnMatch(m_objDesc, m_sceneDesc, init_matches, 2);

#else

	BFMatcher_GPU matcher(NORM_L2);

	if(m_objDescGPU.empty() || m_sceneDescGPU.empty())
	{
		cout << "Warning: Empty feature set." << endl;
		return false;
	}

	matcher.knnMatch(m_objDescGPU, m_sceneDescGPU, init_matches, 2);

#endif

	m_matches.clear();
	for (int i = 0; i < init_matches.size(); i++)
	{
		if(init_matches[i][0].distance <= (m_ratio*init_matches[i][1].distance))
		{
			m_matches.push_back(init_matches[i][0]);
		}
	}

	return true;
}


/**
 * @param frame input frame from camera
 *
 * @return True if the frame was successfully processed. False otherwise.
 */

bool Kernel::Input(const Mat &frame)
{
	ProcessSceneImage(frame); // extract features from frame

	if(!MatchFeatures())
	{
		cout << "Warning: Unable to match features." << endl;
		return false;
	}

	if(!CalculateHomography())
	{
		cout << "Warning: Unable to calculate homography." << endl;
		return false;
	}
	return true;
}

/* ------------------------------------------------------------------------- *
 * Define Internal Functions                                                 *
 * ------------------------------------------------------------------------- */

/**
 * @brief Find the angle between two rays.
 *
 * @param The first ray.
 * @param The second ray.
 *
 * @return
 * The angle (in radians) between the two rays.
 */

float angleBetweenRays(const Vec3f &a, const Vec3f &b)
{
	float mag_a = sqrt(a[0]*a[0] + a[1]*a[1] + a[2]*a[2]);
	float mag_b = sqrt(b[0]*b[0] + b[1]*b[1] + b[2]*b[2]);
	float dot   = a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
	return acos(dot / (mag_a*mag_b));
}

/* ----------------------------------------------------------------------------
 * @brief Pixel space to camera space conversion.
 *
 * Converts a pixel coordinate to it's corresponding ray in camera space.
 *
 * @param[in]  params camera parameters
 * @param[in]  pix    pixel coordinate
 * @param[out] ray    ray projected through image plane at pixel coordinate
 */

void transPixelToCamRay(const cam_params_t &params, const Point2f &pix, Vec3f &ray)
{
	ray[0] = (pix.x - params.p_x) / params.f_x;
	ray[1] = (pix.y - params.p_y) / params.f_y;
	ray[2] = 1.0;
}

/**
 * @brief Pixel space to camera space conversion, z-axis coordinate given.
 *
 * Converts a pixel coordinate to it's corresponding point in camera space
 * given a known z coordinate in camera space.
 *
 * @param[in] params camera parameters
 * @param[in] pix    pixel coordinate
 * @param[in] z      z axis coordinate of the desired camera space point
 *
 * @return
 * The transformed point in camera space.
 */

Point3f transPixelToCamPoint(const cam_params_t &params, const Point2f &pix, float z)
{
	Vec3f ray;
	transPixelToCamRay(params,pix, ray);
	return Point3f(ray[0]*z,ray[1]*z,z);
}

/**
 * @brief Convert homography matrix to heading.
 *
 * @param hm homography matrix
 */

double homographyToHeading(const Mat &hm)
{
	double mag, theda;
	Point2d vec, unit;
	vector<Point2d> in_vert(2), out_vert;
	in_vert[0].x = 0;
	in_vert[0].y = 0;
	in_vert[1].x = 0;
	in_vert[1].y = 1;

	perspectiveTransform(in_vert, out_vert, hm);

	vec = out_vert[0] - out_vert[1];

	mag = sqrt(vec.x*vec.x + vec.y*vec.y);

	unit.x = vec.x / mag;
	unit.y = vec.y / mag;

	theda = 180*acos(unit.y)/3.1415;

	if(unit.x < 0)
		theda = -theda;

	return theda;
}
