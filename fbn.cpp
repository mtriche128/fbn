/**
 * @file   fbn.cpp
 * @author Matthew Triche
 * @brief  This source file contains the main function.
 */

/* ------------------------------------------------------------------------- *
 * Include Headers and Namespaces                                            *
 * ------------------------------------------------------------------------- */

#include <iostream>
#include <string>
#include <fstream>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "proc_kernel.h"
using namespace std;

/* ------------------------------------------------------------------------- *
 * Define Constants                                                          *
 * ------------------------------------------------------------------------- */

#define MIN_HESS 800
#define RATIO    0.6

/* ------------------------------------------------------------------------- *
 * Define Types                                                              *
 * ------------------------------------------------------------------------- */

typedef struct ARG_INFO
{
	string target_img_path;  // path to the target image
	string output_img_path;  // path to the output image
	string output_json_path; // path to output data
	int pos_x;               // x coordinate position in mm
	int pos_y;               // y coordinate position in mm
	int target_dim;          // the target dimension in mm
	float fx;                // camera param, x focal length
	float fy;                // camera param, y focal length
	float px;                // camera param, principle x coord
	float py;                // camera param, principle y coord
} arg_info_t;

/* ------------------------------------------------------------------------- *
 * Declare Internal Functions.                                               *
 * ------------------------------------------------------------------------- */

void ParseArguments(int argc, char **argv, arg_info_t &arg_info);
void OutputJSON(const string &path, const nav_data_t &nav_data, float fps);

/* ------------------------------------------------------------------------- *
 * Define Functions.                                                         *
 * ------------------------------------------------------------------------- */

/* '{"dog": 2.3, "cat": 1}' */

/**
 * @brief main function
 */

int main(int argc, char **argv)
{
	cout << "===== fbn =====" << endl;
	int tout, tfps;
	float fps = 0.0;

	// ------------------------------------------------------------------------
	// initialize and parse arguments

	arg_info_t arg_info;
	arg_info.output_img_path  = "out.jpg";
	arg_info.output_json_path = "out.json";
	arg_info.target_img_path  = "target.jpg";
	arg_info.target_dim       = 300;
	arg_info.pos_x            = 0;
	arg_info.pos_y            = 0;
	arg_info.fx               = 1.0;
	arg_info.fy               = 1.0;
	arg_info.px               = 0;
	arg_info.py               = 0;

	ParseArguments(argc,argv,arg_info);

	// output arguments
	cout << "output_img_path  = " << arg_info.output_img_path << endl;
	cout << "output_json_path = " << arg_info.output_json_path << endl;
	cout << "target_img_path  = " << arg_info.target_img_path << endl;
	cout << "target_dim       = " << arg_info.target_dim << endl;
	cout << "x-axis position  = " << arg_info.pos_x << endl;
	cout << "y-axis position  = " << arg_info.pos_y << endl;
	cout << "x-axis focal len = " << arg_info.fx << endl;
	cout << "y-axis focal len = " << arg_info.fy << endl;
	cout << "principle x-axis = " << arg_info.px << endl;
	cout << "principle y-axis = " << arg_info.py << endl;

	// ------------------------------------------------------------------------
	// initialize capture device

	VideoCapture cap(0);
	if(!cap.isOpened())  // check if we succeeded
	{
		cout << "Error: Failed to initialize video capture device." << endl;
		exit(1);
	}

	// ------------------------------------------------------------------------
	// initialize processing kernel

	// set camera parameters
	cam_params_t cparams;
	cparams.f_x = arg_info.fx;
	cparams.f_y = arg_info.fy;
	cparams.p_x = arg_info.px;
	cparams.p_y = arg_info.py;

	// read target image
	Mat targetImg = imread(arg_info.target_img_path, CV_LOAD_IMAGE_GRAYSCALE);
	if(targetImg.empty())
	{
		cout << "Error: Unable to load target image." << endl;
		exit(1);
	}

	// create an initialze processing kernel
	Kernel k(targetImg,
			cparams,
			MIN_HESS,
			RATIO,
			arg_info.target_dim,
			Point2f(arg_info.pos_x, arg_info.pos_y));

	// ------------------------------------------------------------------------
	// run mainloop

    namedWindow("output",1);
    tout = clock();
    tfps = clock();
	while(1)
	{
		Mat frame;
		nav_data_t ndata;

		cap >> frame;

		if(k.Process(frame,ndata))
		{
			k.DrawHomography(frame);
		}

		fps = (float)CLOCKS_PER_SEC / (float)(clock() - tfps);
		tfps = clock();

		imshow("output", frame);

		if((cvWaitKey(1) & 0xFF) == 'q') break;

		if((clock() - tout) > CLOCKS_PER_SEC)
		{
			cout << "-------------TICK-------------" << endl;
			tout = clock();
			if(!imwrite(arg_info.output_img_path, frame))
			{
				cout << "Error: Unable to write output image." << endl;
				exit(1);
			}
			OutputJSON(arg_info.output_json_path,ndata,fps);
		}
	}

	return 0;
}

/**
 * @brief parse command-line arguments
 *
 * @param[in] argc The number of command-line arguments given.
 * @param[in] argv The string array containing command-line arguments.
 * @param[out] arg_info A struct containing parsed command-line arguments.
 */

void ParseArguments(int argc, char **argv, arg_info_t &arg_info)
{
	int i = 0;

	while(i < argc)
	{
		// output json path
		if(!strcmp("-o", argv[i]))
		{
			if(++i == argc)
			{
				cout << "Error: '-o' argument specified without path." << endl;
				exit(1);
			}

			arg_info.output_json_path.clear();
			arg_info.output_json_path.insert(0,argv[i]);
		}

		// target image path
		else if(!strcmp("-t", argv[i]))
		{
			if(++i == argc)
			{
				cout << "Error: '-t' argument specified without path." << endl;
				exit(1);
			}

			arg_info.target_img_path.clear();
			arg_info.target_img_path.insert(0,argv[i]);
		}

		// output image path
		else if(!strcmp("-v", argv[i]))
		{
			if(++i == argc)
			{
				cout << "Error: '-v' argument specified without path." << endl;
				exit(1);
			}

			arg_info.output_img_path.clear();
			arg_info.output_img_path.insert(0,argv[i]);
		}

		// target image dimension path
		else if(!strcmp("-d", argv[i]))
		{
			if(++i == argc)
			{
				cout << "Error: '-d' argument specified without value." << endl;
				exit(1);
			}

			arg_info.target_dim = atoi(argv[i]);
		}

		// x pos coordinate
		else if(!strcmp("-x", argv[i]))
		{
			if(++i == argc)
			{
				cout << "Error: '-x' argument specified without value." << endl;
				exit(1);
			}

			arg_info.pos_x = atoi(argv[i]);
		}

		// y pos coordinate
		else if(!strcmp("-y", argv[i]))
		{
			if(++i == argc)
			{
				cout << "Error: '-y' argument specified without value." << endl;
				exit(1);
			}

			arg_info.pos_y = atoi(argv[i]);
		}

		// focal x
		else if(!strcmp("-fx", argv[i]))
		{
			if(++i == argc)
			{
				cout << "Error: '-fx' argument specified without value." << endl;
				exit(1);
			}

			arg_info.fx = atof(argv[i]);
		}

		// focal y
		else if(!strcmp("-fy", argv[i]))
		{
			if(++i == argc)
			{
				cout << "Error: '-fy' argument specified without value." << endl;
				exit(1);
			}

			arg_info.fy = atof(argv[i]);
		}

		// principle x coord
		else if(!strcmp("-px", argv[i]))
		{
			if(++i == argc)
			{
				cout << "Error: '-px' argument specified without value." << endl;
				exit(1);
			}

			arg_info.px = atof(argv[i]);
		}

		// principle y coord
		else if(!strcmp("-py", argv[i]))
		{
			if(++i == argc)
			{
				cout << "Error: '-py' argument specified without value." << endl;
				exit(1);
			}

			arg_info.py = atof(argv[i]);
		}

		i++;
	}
}

void OutputJSON(const string &path, const nav_data_t &nav_data, float fps)
{
	fstream file;

	file.open(path.c_str(), fstream::out);
	if(!file.is_open())
	{
		cout << "Error: Unable to write JSON file." << endl;
		exit(1);
	}

	file << "{"
		 << "\"head\": "  << nav_data.head  << ", "
	     << "\"elev\": "  << nav_data.elev  << ", "
	     << "\"pos_x\": " << nav_data.pos.x << ", "
	     << "\"pos_y\": " << nav_data.pos.y << ", "
	     << "\"fps\": "   << fps
	     << "}";

	file.close();
}
