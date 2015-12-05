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
#include <stdlib.h>
#include <string.h>
using namespace std;

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
} arg_info_t;

/* ------------------------------------------------------------------------- *
 * Declare Internal Functions.                                               *
 * ------------------------------------------------------------------------- */

void ParseArguments(int argc, char **argv, arg_info_t &arg_info);

/* ------------------------------------------------------------------------- *
 * Define Functions.                                                         *
 * ------------------------------------------------------------------------- */

/**
 * @brief main function
 */

int main(int argc, char **argv)
{
	cout << "===== fbn =====" << endl;

	// ------------------------------------------------------------------------
	// initialize and parse arguments

	arg_info_t arg_info;
	arg_info.output_img_path  = "out.jpg";
	arg_info.output_json_path = "out.json";
	arg_info.target_img_path  = "target.jpg";
	arg_info.target_dim       = 300;
	arg_info.pos_x            = 0;
	arg_info.pos_y            = 0;

	ParseArguments(argc,argv,arg_info);

	// output arguments
	cout << "output_img_path  = " << arg_info.output_img_path << endl;
	cout << "output_json_path = " << arg_info.output_json_path << endl;
	cout << "target_img_path  = " << arg_info.target_img_path << endl;
	cout << "target_dim       = " << arg_info.target_dim << endl;
	cout << "pos_x            = " << arg_info.pos_x << endl;
	cout << "pos_y            = " << arg_info.pos_y << endl;

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

		// x coordinate
		else if(!strcmp("-x", argv[i]))
		{
			if(++i == argc)
			{
				cout << "Error: '-x' argument specified without value." << endl;
				exit(1);
			}

			arg_info.pos_x = atoi(argv[i]);
		}

		// y coordinate
		else if(!strcmp("-y", argv[i]))
		{
			if(++i == argc)
			{
				cout << "Error: '-y' argument specified without value." << endl;
				exit(1);
			}

			arg_info.pos_y = atoi(argv[i]);
		}

		i++;
	}
}

