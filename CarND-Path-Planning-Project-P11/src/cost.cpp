#include "vehicle.h"
#include "cost.h"
#include <iostream>
#include <math.h>
#include <vector>
#include <map>
#include <string>
#include <iterator>


double change_lane_cost(Vehicle* vehicle_ptr, vector<Vehicle::snapshot> trajectory, map<int,vector < vector<double> > > predictions, TrajectoryData data){

  // Penalizes lane changes and penalizes driving in any lane other than the center
  int proposed_lane = trajectory[1].laneSP;
  int cur_lane = trajectory[0].laneSP;
  double cost = 0.0;
  // If a change is proposed
  if (proposed_lane != cur_lane)
    cost = COMFORT;
  // if the proposed lane isn't the middle lane
  if (proposed_lane != 1)
    cost += COMFORT*6;

  return cost;
}

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
    double time_til_collision = data.collides.time;
    double exponent = pow(time_til_collision, 2.0);
    double mult = exp(-exponent);

    cost = mult * COLLISION;
  }
  else
    cost = 0;

  return cost;
}

double buffer_cost(Vehicle* vehicle_ptr, vector<Vehicle::snapshot> trajectory, map<int,vector < vector<double> > > predictions, TrajectoryData data){
  // Penalizes a short distance between ego vehicle and closest vehicle in front

  double cost;
  double closest = data.closest_approach;
  if ( 0 <= closest && closest <= vehicle_ptr->L)
    cost = 10 * DANGER;
  else {
    double timesteps_away = closest / abs(data.avg_speed);
    if(timesteps_away > DESIRED_TIME_BUFFER)
      cost = 0.0;
    else {
      double multiplier = 1.0 - pow(timesteps_away/DESIRED_TIME_BUFFER,2);
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
  costs.push_back(change_lane_cost(vehicle_ptr, trajectory, predictions, trajectory_data));
  costs.push_back(inefficiency_cost(vehicle_ptr, trajectory, predictions, trajectory_data));
  costs.push_back(collision_cost(vehicle_ptr, trajectory, predictions, trajectory_data));
  costs.push_back(buffer_cost(vehicle_ptr, trajectory, predictions, trajectory_data));

  // Total cost of trajectory
  double cost = 0.0;
  for(int i=0; i<costs.size(); i++){
    cost += costs[i];
  }

  if(DEBUG_COST){
    printf("\tChange Lane cost\t%+8.0f\tInefficiency cost\t%+8.0f"
        "\n\t\t\tCollision cost\t\t%+8.0f\tBuffer cost\t\t%+8.0f"
        "\n\t\t\t\t\t\t\t\tTOTAL ----------------->%+8.0f\n\n",
        costs[0],costs[1],costs[2],costs[3],cost);
  }

  return cost;
}

TrajectoryData get_helper_data(Vehicle* vehicle_ptr, vector<Vehicle::snapshot> trajectory, map<int,vector < vector<double> > > predictions){

  // current ego vehicle state
  Vehicle::snapshot current_snapshot = trajectory[0];

  // first ego vehicle state in the trajectory
  Vehicle::snapshot first = trajectory[1];

  // last ego vehicle state in the trajectory
  Vehicle::snapshot last = trajectory.back();

  int proposed_lane = first.laneSP;

  // delta t over the whole trajectory
  double dt = double(trajectory.size());

  // avg speed over the whole trajectory
  double avg_speed = (last.s - current_snapshot.s) / dt;

  // initialize a bunch of variables

  // acceleration over the whole trajectory
  vector<double> accels;

  // closest vehicle in proposed lane along s axis, either in front or behind, over the whole trajectory
  double closest_approach = 999999.0;

  // collision event initialisation
  Vehicle::collider collides;
  collides.collision = false;

  // Extract only the predictions for vehicles in ego vehicle's target lane
  map<int,vector < vector<double> > > filtered = filter_predictions_by_lane(predictions, proposed_lane);

  // for a number of future time steps
  for(int i=1; i < int(double(PLANNING_HORIZON)/DELTA_T_PREDICTION)+1; i++) {

    // ego vehicle state snapshot
    Vehicle::snapshot snapshot = trajectory[i];
    accels.push_back(snapshot.a);

    // for all other cars in ego vehicle's target lane
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
        collides.time = double(i);
      }
      // target vehicle's distance from ego vehicle
      double dist = abs(vehicle_state[1] - snapshot.s);

      // this target vehicle is the closest approach
      if (dist < closest_approach)
        closest_approach = dist;

      it++;
    }
  }

  // absolute max acceleration
  double max_accel = 0.0;
  for(int i=0; i<accels.size(); i++){
    if(abs(accels[i]) >= max_accel)
      max_accel = abs(accels[i]);
  }

  // rms accelerations
  vector<double> rms_accels;
  for(int i=0; i<accels.size(); i++){
    rms_accels.push_back(pow(accels[i],2.0));
  }

  double rms_acceleration=0.0;
  for(int i=0; i<rms_accels.size(); i++){
    rms_acceleration += rms_accels[i];
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

