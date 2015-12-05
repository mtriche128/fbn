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

#include "proc_kernel.h"

using namespace std;
using namespace cv;
using namespace cv::gpu;

/* ------------------------------------------------------------------------- *
 * Declare Internal Functions                                                *
 * ------------------------------------------------------------------------- */

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
