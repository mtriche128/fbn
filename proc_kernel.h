/**
 * @file   proc_kernel.h
 * @author Matthew Triche
 * @brief  Header for the processing kernel.
 */

/* ------------------------------------------------------------------------- *
 * Include Headers and Namespaces                                            *
 * ------------------------------------------------------------------------- */

#include <iostream>
#include <vector>
#include <algorithm>
#include <time.h>
#include <stdio.h>
#include "opencv2/opencv_modules.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/gpu/gpu.hpp"
#include "opencv2/nonfree/gpu.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/calib3d/calib3d.hpp"

using namespace std;
using namespace cv;
using namespace cv::gpu;

/* ------------------------------------------------------------------------- *
 * Define Constants                                                          *
 * ------------------------------------------------------------------------- */

// uncomment to compile with GPU acceleration
// #define ENABLE_GPU

// define GPU feature extraction parameters
#define SURF_N_OCT        4
#define SURF_N_OCT_LAYERS 2
#define SURF_EXTENDED     false
#define KP_RATIO          0.01

/* ------------------------------------------------------------------------- *
 * Define Types                                                              *
 * ------------------------------------------------------------------------- */

/**
 * @brief Stores camera parameters.
 */

typedef struct CAM_PARAMS
{
	float f_x; // focal length, x axis
	float f_y; // focal length, y axis
	float p_x; // principle point, x axis
	float p_y; // principle point, y axis
} cam_params_t;

/**
 * @brief A container for navigation data output from Kernel.
 */

typedef struct NAV_DATA
{
	Point2f pos;  // position  (mm)
	float   head; // heading   (degrees)
	float   elev; // elevation (m)
} nav_data_t;

/**
 * @brief Process kernel's object.
 */

class Kernel
{
public:
	Kernel(const Mat&,
		   const cam_params_t&,
		   const int,
		   const float,
		   const float,
		   const Point2f&);
	~Kernel();

	bool Process(const Mat&,nav_data_t&);
	void DrawHomography(Mat&);

private:

	void ProcessSceneImage(const Mat&);
	bool CalculateHomography();
	bool MatchFeatures();
	float CalculateElevation();
	Point2f CalculatePostion(const float, const float);

	cam_params_t m_camParams;      // stores camera parameters
	nav_data_t m_navData;          // stores current navigation data
	vector<Point2f> m_objImgVert;  // object image vertices
	vector<Point2f> m_homVert;     // image vertices post-homography
	vector<DMatch> m_matches;      // feature matches
	vector<KeyPoint> m_objKpnts;   // object keypoints
	vector<KeyPoint> m_sceneKpnts; // scene keypoints
	Mat m_objDesc;                 // object descriptors
	Mat m_sceneDesc;               // scene descriptors
	Mat m_hm;                      // homography matrix
	float m_ratio;                 // matching ratio
	float m_dim;                   // target image dimension
	float m_sceneImgWidth;         // stores scene image width
	float m_sceneImgHeight;        // stores scene image height
	Point2f m_sceneImgPos;         // stores scene image's position

#ifndef ENABLE_GPU
	SurfFeatureDetector *m_detector;      // extracts keypoints
	SurfDescriptorExtractor *m_extractor; // extracts descriptors
#else
	SURF_GPU *m_surfGPU;    // feature extractor
	GpuMat m_objDescGPU;    // stores object descriptors on GPU
	GpuMat m_sceneDescGPU;  // stores scene descriptors on GPU
	GpuMat m_objKpGPU;      // stores object keypoint descriptors on GPU
	GpuMat m_sceneKpGPU;    // stores scene keypoint descriptors on GPU
#endif

};
