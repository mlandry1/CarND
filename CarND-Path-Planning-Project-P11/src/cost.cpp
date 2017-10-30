#include "vehicle.h"
#include "cost.h"
#include <iostream>
//#include <random>
//#include <sstream>
//#include <fstream>
#include <math.h>
#include <vector>
#include <map>
#include <string>
#include <iterator>

// priority levels for costs
#define COLLISION  10000000
#define DANGER     1000000
#define REACH_GOAL 1000000
#define COMFORT    100000
#define EFFICIENCY 1000

#define PLANNING_HORIZON  2
#define DESIRED_BUFFER    1.5 // timestep

#define DEBUG_COST true

//double change_lane_cost(Vehicle* vehicle_ptr, vector<Vehicle::snapshot> trajectory, map<int,vector < vector<double> > > predictions, TrajectoryData data){
//
//  // Penalizes lane changes AWAY from the goal lane and rewards
//  // lane changes TOWARDS the goal lane.
//
//  int proposed_lanes = data.end_lanes_from_goal;
//  int cur_lanes = trajectory[0].lane;
//  double cost = 0.0;
//  if (proposed_lanes > cur_lanes)
//    cost = COMFORT;
//  if (proposed_lanes < cur_lanes)
//    cost = -COMFORT;
//
//  if(DEBUG_COST){
//    std::cout << __FUNCTION__ << " has cost " << cost << " for lane " << trajectory[0].lane <<std::endl<<std::endl;
//  }
//
//  return cost;
//}
//
//double distance_from_goal_lane(Vehicle* vehicle_ptr, vector<Vehicle::snapshot> trajectory, map<int,vector < vector<double> > > predictions, TrajectoryData data){
//
//  // Penalizes lane distance vs time to change lane
//
//  double distance = double(abs(data.end_distance_to_goal));
//  distance = max(distance, 1.0);
//  double time_to_goal = distance / data.avg_speed;
//  double lanes = double(data.end_lanes_from_goal);
//  double multiplier = 5 * lanes / time_to_goal;
//  double cost = multiplier * REACH_GOAL;
//
//  if(DEBUG_COST){
//    std::cout << __FUNCTION__ << " has cost " << cost << " for lane " << trajectory[0].lane <<std::endl;
//  }
//
//  return cost;
//}

double inefficiency_cost(Vehicle* vehicle_ptr, vector<Vehicle::snapshot> trajectory, map<int,vector < vector<double> > > predictions, TrajectoryData data){

  // Penalizes lower/higher speed than requested

  double speed = data.avg_speed;
  double target_speed = double(vehicle_ptr->target_speed);
  double diff = target_speed - speed;
  double pct = diff / target_speed;
  double multiplier = pow(pct, 2.0);
  double cost = multiplier * EFFICIENCY;

  return cost;
}

double collision_cost(Vehicle* vehicle_ptr, vector<Vehicle::snapshot> trajectory, map<int,vector < vector<double> > > predictions, TrajectoryData data){
  double cost;
  if (data.collides.collision){
    int time_til_collision = data.collides.time;
    double exponent = pow(double(time_til_collision), 2.0);
    double mult = exp(-exponent);

    cost = mult * COLLISION;
  }
  else
    cost = 0;

  return cost;
}

double buffer_cost(Vehicle* vehicle_ptr, vector<Vehicle::snapshot> trajectory, map<int,vector < vector<double> > > predictions, TrajectoryData data){
  double cost;
  double closest;
  closest = (double)data.closest_approach;
  if (closest == 0.0)
    cost = 10 * DANGER;
  else {
    double timesteps_away = closest / data.avg_speed;
    if(timesteps_away > DESIRED_BUFFER)
      cost = 0.0;
    else {
      double multiplier = 1.0 - pow(timesteps_away/DESIRED_BUFFER,2);
      cost = multiplier * DANGER;
    }
  }

  return cost;
}

double calculate_cost(Vehicle* vehicle_ptr, vector<Vehicle::snapshot> trajectory, map<int,vector < vector<double> > > predictions){

  // extra more data from the ego vehicle trajectory
  TrajectoryData trajectory_data = get_helper_data(vehicle_ptr, trajectory, predictions);

  vector <double> costs;

  // cost functions evaluation
//  costs.push_back(vehicle_ptr, distance_from_goal_lane(trajectory, predictions, trajectory_data));
//  costs.push_back(vehicle_ptr, change_lane_cost(trajectory, predictions, trajectory_data));
  costs.push_back(inefficiency_cost(vehicle_ptr, trajectory, predictions, trajectory_data));
  costs.push_back(collision_cost(vehicle_ptr, trajectory, predictions, trajectory_data));
  costs.push_back(buffer_cost(vehicle_ptr, trajectory, predictions, trajectory_data));

  if(DEBUG_COST){
    std::cout << "Inefficiency cost" << " is " << costs[0] << " for lane " << trajectory[1].lane <<std::endl;
    std::cout << "Collision cost" << " is " << costs[1] << " for lane " << trajectory[1].lane <<std::endl;
    std::cout << "Buffer cost" << " is " << costs[1] << " for lane " << trajectory[1].lane <<std::endl;
  }

  // Total cost of trajectory
  double cost = 0.0;
  for(int i=0; i<costs.size(); i++){
    cost += costs[i];
  }

  return cost;
}

TrajectoryData get_helper_data(Vehicle* vehicle_ptr, vector<Vehicle::snapshot> trajectory, map<int,vector < vector<double> > > predictions){

  // copy ego vehicle's trajectory
  vector<Vehicle::snapshot> t = trajectory;

  // current ego vehicle state
  Vehicle::snapshot current_snapshot = t[0];
  // first ego vehicle state in the trajectory
  Vehicle::snapshot first = t[1];
  // last ego vehicle state in the trajectory
  Vehicle::snapshot last = t.back();

  // delta t over the whole trajectory
  double dt = double(trajectory.size());

  int proposed_lane = first.lane;
  double avg_speed = (last.s - current_snapshot.s) / dt;

  // initialize a bunch of variables
  vector<int> accels;
  int closest_approach;
  closest_approach = 999999;
  Vehicle::collider collides;
  collides.collision = false;
  Vehicle::snapshot last_snap;
  last_snap = trajectory[0];

  // Extract only the predictions for vehicles in ego vehicle's lane
  map<int,vector < vector<double> > > filtered = filter_predictions_by_lane(predictions, proposed_lane);

  // for a number of future time steps
  for(int i=1; i < PLANNING_HORIZON+1; i++) {

    // ego vehicle state snapshot
    Vehicle::snapshot snapshot = trajectory[i];
    accels.push_back(snapshot.a);

    // for all other cars in our lane (not ego)
    map<int, vector<vector<double> > >::iterator it = filtered.begin();
    while(it != filtered.end())
    {
      // first item : vehicle id
      int v_id = it->first;
      // second item : vehicle predicted trajectory
      vector<vector<double> > predicted_traj = it->second;

      // vehicle_state[0] = lane, vehicle_state[1] = s
      // Extract both actual and last vehicle state from the predicted trajectory
      vector<double> vehicle_state = predicted_traj[i];
      vector<double> last_vehicle_state = predicted_traj[i-1];

      // Test to see if a collision happens..
      bool vehicle_collides = check_collision(snapshot, last_vehicle_state[1], vehicle_state[1]);

      if (vehicle_collides){
        collides.collision = true;
        collides.time = i;
      }
      // target vehicle's distance from ego vehicle
      int dist = abs(vehicle_state[1] - snapshot.s);

      // this target vehicle is the closest approach
      if (dist < closest_approach)
        closest_approach = dist;

      it++;
    }

    last_snap = snapshot;
  }

  // absolute max acceleration
  int max_accel = 0;
  for(int i=0; i<accels.size(); i++){
    if(abs(accels[i]) >= max_accel)
      max_accel = abs(accels[i]);
  }

  // rms accelerations
  vector<int> rms_accels;
  for(int i=0; i<accels.size(); i++){
    rms_accels.push_back(pow(accels[i],2));
  }

  double rms_acceleration=0.0;
  for(int i=0; i<rms_accels.size(); i++){
    rms_acceleration += double(rms_accels[i]);
  }
  rms_acceleration /= rms_accels.size();

  TrajectoryData traj_data;

  traj_data.proposed_lane = proposed_lane;
  traj_data.avg_speed = avg_speed;
  traj_data.max_acceleration = max_accel;
  traj_data.rms_acceleration = rms_acceleration;
  traj_data.closest_approach = closest_approach;
  traj_data.collides.collision =  collides.collision;
  traj_data.collides.time =  collides.time;

  return traj_data;
}

bool check_collision(Vehicle::snapshot snapshot, double s_previous, double s_now){
  double s_temp = snapshot.s;
  double v_temp = snapshot.v;
  double v_target = s_now - s_previous;

  // if target vehicle was behind ego vehicle
  if(s_previous < s_temp){
    // and is now either on or in front ego vehicle...
    if(s_now >= s_temp)
      return true;
    else
      return false;
  }

  // if target vehicle was in front ego vehicle
  if(s_previous > s_temp){
    // and is now either on or behind ego vehicle...
    if(s_now <= s_temp)
      return true;
    else
      return false;
  }
  // if target vehicle is on ego vehicle
  if(s_previous == s_temp){
    // and target vehicle speed is greater than ego vehicle's speed
    if(v_target > v_temp)
      return false;
    else
      return true;
  }
}

map<int,vector < vector<double> > > filter_predictions_by_lane(map<int,vector < vector<double> > > predictions, int lane){
  map<int,vector < vector<double> > > filtered;

  map<int, vector<vector<double> > >::iterator it = predictions.begin();
  while(it != predictions.end())
  {
    // first item : vehicle id
    int v_id = it->first;
    // second item :
    vector<vector<double> > predicted_traj = it->second;

    // If first prediction's lane is == lane... and v_id isn't ego vehicle
    if(predicted_traj[0][0] == double(lane) && v_id != -1)
      filtered[v_id] = predicted_traj;

    it++;
  }

  return filtered;
}

