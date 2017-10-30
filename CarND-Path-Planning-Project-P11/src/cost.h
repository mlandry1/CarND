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

struct TrajectoryData{
  int               proposed_lane;
  double            avg_speed;
  double            max_acceleration;
  double            rms_acceleration;
  double            closest_approach;
  Vehicle::collider collides;
};

//double change_lane_cost(Vehicle* vehicle_ptr, vector<Vehicle::snapshot> trajectory, map<int,vector < vector<double> > > predictions, TrajectoryData data);
//
//double distance_from_goal_lane(Vehicle* vehicle_ptr, vector<Vehicle::snapshot> trajectory, map<int,vector < vector<double> > > predictions, TrajectoryData data);

double inefficiency_cost(Vehicle* vehicle_ptr, vector<Vehicle::snapshot> trajectory, map<int,vector < vector<double> > > predictions, TrajectoryData data);

double collision_cost(Vehicle* vehicle_ptr, vector<Vehicle::snapshot> trajectory, map<int,vector < vector<double> > > predictions, TrajectoryData data);

double buffer_cost(Vehicle* vehicle_ptr, vector<Vehicle::snapshot> trajectory, map<int,vector < vector<double> > > predictions, TrajectoryData data);

double calculate_cost(Vehicle* vehicle_ptr, vector<Vehicle::snapshot> trajectory, map<int,vector < vector<double> > > predictions);

TrajectoryData get_helper_data(Vehicle* vehicle_ptr, vector<Vehicle::snapshot> trajectory, map<int,vector < vector<double> > > predictions);

bool check_collision(Vehicle::snapshot snapshot, double s_previous, double s_now);

map<int,vector < vector<double> > > filter_predictions_by_lane(map<int,vector < vector<double> > > predictions, int lane);

#endif
