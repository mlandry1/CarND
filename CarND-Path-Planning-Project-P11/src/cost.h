#ifndef COST_H
#define COST_H

#include "vehicle.h"
#include <iostream>
//#include <random>
//#include <sstream>
//#include <fstream>
#include <math.h>
#include <vector>
#include <map>
#include <string>
#include <iterator>


// helper defines
#define DEBUG_COST    true
#define DEBUG_VEHICLE true
#define DEBUG true

#define DELTA_T_PREDICTION 0.35  //timestep for trajectory predictions
#define DELTA_T 0.02

#define MAX_JERK     50.0   // m/s³
#define MAX_ACCEL    9.00   // m/s²
#define SPEED_LIMIT  22.21  // 22.352m/s  // 50 MPH in m/s

#define NUM_LANES    3
#define NUM_PATH_POINTS 50

#define TRACK_LENGTH 6946

// priority levels for costs
#define COLLISION  10000000
#define DANGER     1000000
#define COMFORT    10
#define EFFICIENCY 1000

#define PLANNING_HORIZON  2
#define DESIRED_TIME_BUFFER  1.0 //s

struct TrajectoryData{
  int               proposed_lane;
  double            avg_speed;
  double            max_acceleration;
  double            rms_acceleration;
  double            closest_approach;
  Vehicle::collider collides;
};

//double distance_from_goal_lane(Vehicle* vehicle_ptr, vector<Vehicle::snapshot> trajectory, map<int,vector < vector<double> > > predictions, TrajectoryData data);

double change_lane_cost(Vehicle* vehicle_ptr, vector<Vehicle::snapshot> trajectory, map<int,vector < vector<double> > > predictions, TrajectoryData data);

double inefficiency_cost(Vehicle* vehicle_ptr, vector<Vehicle::snapshot> trajectory, map<int,vector < vector<double> > > predictions, TrajectoryData data);

double collision_cost(Vehicle* vehicle_ptr, vector<Vehicle::snapshot> trajectory, map<int,vector < vector<double> > > predictions, TrajectoryData data);

double buffer_cost(Vehicle* vehicle_ptr, vector<Vehicle::snapshot> trajectory, map<int,vector < vector<double> > > predictions, TrajectoryData data);

double calculate_cost(Vehicle* vehicle_ptr, vector<Vehicle::snapshot> trajectory, map<int,vector < vector<double> > > predictions);

TrajectoryData get_helper_data(Vehicle* vehicle_ptr, vector<Vehicle::snapshot> trajectory, map<int,vector < vector<double> > > predictions);

bool check_collision(Vehicle::snapshot snapshot, double s_previous, double s_now);

map<int,vector < vector<double> > > filter_predictions_by_lane(map<int,vector < vector<double> > > predictions, int lane);

#endif
