#include <iostream>
#include "vehicle.h"
#include <math.h>
#include <map>
#include <string>
#include <iterator>
#include <vector>
#include <algorithm>
#include "assert.h"

/**
 * Initializes Vehicle
 */
Vehicle::Vehicle(int lane, int s, int v, int a) {

  this->lane = lane;
  this->s = s;
  this->v = v;
  this->a = a;
  this->state = "CS";
  this->max_acceleration = -1;

}

// priority levels for costs
#define COLLISION  10000000
#define DANGER     1000000
#define REACH_GOAL 1000000
#define COMFORT    100000
#define EFFICIENCY 1000

#define DESIRED_BUFFER    1.5 // timestep
#define PLANNING_HORIZON  2

#define DEBUG true

Vehicle::~Vehicle() {}

// TODO - Implement this method.
void Vehicle::update_state(map<int,vector < vector<int> > > predictions) {
	/*
  Updates the "state" of the vehicle by assigning one of the
  following values to 'self.state':

  "KL" - Keep Lane
   - The vehicle will attempt to drive its target speed, unless there is
     traffic in front of it, in which case it will slow down.

  "LCL" or "LCR" - Lane Change Left / Right
   - The vehicle will IMMEDIATELY change lanes and then follow longitudinal
     behavior for the "KL" state in the new lane.

  "PLCL" or "PLCR" - Prepare for Lane Change Left / Right
   - The vehicle will find the nearest vehicle in the adjacent lane which is
     BEHIND itself and will adjust speed to try to get behind that vehicle.

  INPUTS
  - predictions
  A dictionary. The keys are ids of other vehicles and the values are arrays
  where each entry corresponds to the vehicle's predicted location at the
  corresponding timestep. The FIRST element in the array gives the vehicle's
  current position. Example (showing a car with id 3 moving at 2 m/s):

  {
    3 : [
      {"s" : 4, "lane": 0},
      {"s" : 6, "lane": 0},
      {"s" : 8, "lane": 0},
      {"s" : 10, "lane": 0},
    ]
  }

  */

//  string _state = this->_get_next_state(predictions);
//
//  if(DEBUG){
//    std::cout << " Next state " << _state << std::endl;
//  }

  state = "KL"; // this is an example of how you change state.
}

//string Vehicle::_get_next_state(map<int,vector < vector<int> > > predictions) {
//
//  vector<string> states = {"KL", "LCL", "LCR", "PLCL", "PLCR"};
//
//  // Remove impossible states
//  if(this->lane == 0){
//    states.erase(std::remove(states.begin(), states.end(), "LCL"), states.end());
//  }
//  if(this->lane == this->lanes_available - 1){
//    states.erase(std::remove(states.begin(), states.end(), "LCR"), states.end());
//  }
//
//  vector<cost_t> costs;
//
//  map<int,vector < vector<int> > > predictions_copy;
//
//  // for each possible state
//  for(int i=0; i<states.size(); i++){
//    predictions_copy = predictions;
//    vector<Vehicle::snapshot> trajectory = this->_trajectory_for_state(state, predictions_copy, 5);
//
//    cost_t cost;
//    cost.cost = this->Vehicle::calculate_cost(this, trajectory, predictions);
//    cost.state = states[i];
//
//    costs.push_back(cost);
//  }
//
//  double best_cost = costs[0].cost;
//  string best_state = costs[0].state;
//
//  for(int i=1; i<costs.size(); i++){
//    if( costs[i].cost < best_cost){
//      best_cost = costs[i].cost;
//      best_state = costs[i].state;
//    }
//  }
//
//  return states[0];
//}
//
//vector<Vehicle::snapshot> Vehicle::_trajectory_for_state(string state, map<int,vector < vector<int> > > predictions, int horizon = 5){
//  Vehicle::snapshot snapshot_temp = get_snapshot();
//
//  // pretend to be in new  proposed state
//  this->state = state;
//  vector<Vehicle::snapshot> trajectory;
//  trajectory.push_back(snapshot_temp);
//
//  for(int i=0; i<horizon; i++){
//    this->restore_state_from_snapshot(snapshot_temp);
//    this->state = state;
//    this->realize_state(predictions);
//    assert(0 <= this->lane && this->lane < this->lanes_available);
//    this->increment(1);
//    trajectory.push_back(this->get_snapshot());
//
//    //need to remove first prediction for each vehicle
//    map<int, vector<vector<int> > >::iterator it = predictions.begin();
//    while(it != predictions.end())
//    {
//      int v_id = it->first;
//      vector<vector<int> > v = it->second;
//      v.erase(v.begin());
//      it++;
//    }
//  }
//
//  this->restore_state_from_snapshot(snapshot_temp);
//  return trajectory;
//}
//
//Vehicle::snapshot Vehicle::get_snapshot(){
//  Vehicle::snapshot snapshot_temp;
//  snapshot_temp.lane = this->lane;
//  snapshot_temp.s = this->s;
//  snapshot_temp.v = this->v;
//  snapshot_temp.a = this->a;
//  snapshot_temp.state = this->state;
//
//  return snapshot_temp;
//
//}
//
//void Vehicle::restore_state_from_snapshot(Vehicle::snapshot snapshot_temp){
//  this->lane = snapshot_temp.lane;
//  this->s = snapshot_temp.s;
//  this->v = snapshot_temp.v;
//  this->a = snapshot_temp.a;
//  this->state = snapshot_temp.state;
//}

void Vehicle::configure(vector<int> road_data) {
/*
  Called by simulator before simulation begins. Sets various
  parameters which will impact the ego vehicle.
  */
  target_speed = road_data[0];
  lanes_available = road_data[1];
  goal_s = road_data[2];
  goal_lane = road_data[3];
  max_acceleration = road_data[4];
}

string Vehicle::display() {

  ostringstream oss;

  oss << "s:    " << this->s << "\n";
  oss << "lane: " << this->lane << "\n";
  oss << "v:    " << this->v << "\n";
  oss << "a:    " << this->a << "\n";

  return oss.str();
}

void Vehicle::increment(int dt = 1) {

  this->s += this->v * dt;
  this->v += this->a * dt;
}

vector<int> Vehicle::state_at(int t) {

	/*
  Predicts state of vehicle in t seconds (assuming constant acceleration)
  */
  int s = this->s + this->v * t + this->a * t * t / 2;
  int v = this->v + this->a * t;
  return {this->lane, s, v, this->a};
}

bool Vehicle::collides_with(Vehicle other, int at_time) {

	/*
  Simple collision detection.
  */
  vector<int> check1 = state_at(at_time);
  vector<int> check2 = other.state_at(at_time);
  return (check1[0] == check2[0]) && (abs(check1[1]-check2[1]) <= L);
}

Vehicle::collider Vehicle::will_collide_with(Vehicle other, int timesteps) {

	Vehicle::collider collider_temp;
	collider_temp.collision = false;
	collider_temp.time = -1; 

	for (int t = 0; t < timesteps+1; t++)
	{
      	if( collides_with(other, t) )
      	{
			collider_temp.collision = true;
			collider_temp.time = t; 
        	return collider_temp;
    	}
	}

	return collider_temp;
}

void Vehicle::realize_state(map<int,vector < vector<int> > > predictions) {
   
	/*
  Given a state, realize it by adjusting acceleration and lane.
  Note - lane changes happen instantaneously.
  */
  string state = this->state;
  if(state.compare("CS") == 0)
  {
    realize_constant_speed();
  }
  else if(state.compare("KL") == 0)
  {
    realize_keep_lane(predictions);
  }
  else if(state.compare("LCL") == 0)
  {
    realize_lane_change(predictions, "L");
  }
  else if(state.compare("LCR") == 0)
  {
    realize_lane_change(predictions, "R");
  }
  else if(state.compare("PLCL") == 0)
  {
    realize_prep_lane_change(predictions, "L");
  }
  else if(state.compare("PLCR") == 0)
  {
    realize_prep_lane_change(predictions, "R");
  }

}

void Vehicle::realize_constant_speed() {
	a = 0;
}

int Vehicle::_max_accel_for_lane(map<int,vector<vector<int> > > predictions, int lane, int s) {

	int delta_v_til_target = target_speed - v;
  int max_acc = min(max_acceleration, delta_v_til_target);

  map<int, vector<vector<int> > >::iterator it = predictions.begin();
  vector<vector<vector<int> > > in_front;
  while(it != predictions.end())
  {

    int v_id = it->first;

      vector<vector<int> > v = it->second;

      if((v[0][0] == lane) && (v[0][1] > s))
      {
        in_front.push_back(v);

      }
      it++;
  }

  if(in_front.size() > 0)
  {
    int min_s = 1000;
    vector<vector<int>> leading = {};
    for(int i = 0; i < in_front.size(); i++)
    {
      if((in_front[i][0][1]-s) < min_s)
      {
        min_s = (in_front[i][0][1]-s);
        leading = in_front[i];
      }
    }
    
    int next_pos = leading[1][1];
    int my_next = s + this->v;
    int separation_next = next_pos - my_next;
    int available_room = separation_next - preferred_buffer;
    max_acc = min(max_acc, available_room);
  }

  return max_acc;

}

void Vehicle::realize_keep_lane(map<int,vector< vector<int> > > predictions) {
	this->a = _max_accel_for_lane(predictions, this->lane, this->s);
}

void Vehicle::realize_lane_change(map<int,vector< vector<int> > > predictions, string direction) {
	int delta = -1;
  if (direction.compare("L") == 0)
  {
    delta = 1;
  }
  this->lane += delta;
  int lane = this->lane;
  int s = this->s;
  this->a = _max_accel_for_lane(predictions, lane, s);
}

void Vehicle::realize_prep_lane_change(map<int,vector<vector<int> > > predictions, string direction) {
	int delta = -1;
  if (direction.compare("L") == 0)
  {
    delta = 1;
  }
  int lane = this->lane + delta;

  map<int, vector<vector<int> > >::iterator it = predictions.begin();
  vector<vector<vector<int> > > at_behind;
  while(it != predictions.end())
  {
    int v_id = it->first;
    vector<vector<int> > v = it->second;

    if((v[0][0] == lane) && (v[0][1] <= this->s))
    {
      at_behind.push_back(v);

    }
    it++;
  }
  if(at_behind.size() > 0)
  {

    int max_s = -1000;
    vector<vector<int> > nearest_behind = {};
    for(int i = 0; i < at_behind.size(); i++)
    {
      if((at_behind[i][0][1]) > max_s)
      {
        max_s = at_behind[i][0][1];
        nearest_behind = at_behind[i];
      }
    }
    int target_vel = nearest_behind[1][1] - nearest_behind[0][1];
    int delta_v = this->v - target_vel;
    int delta_s = this->s - nearest_behind[0][1];
    if(delta_v != 0)
    {

      int time = -2 * delta_s/delta_v;
      int a;
      if (time == 0)
      {
        a = this->a;
      }
      else
      {
        a = delta_v/time;
      }
      if(a > this->max_acceleration)
      {
        a = this->max_acceleration;
      }
      if(a < -this->max_acceleration)
      {
        a = -this->max_acceleration;
      }
      this->a = a;
    }
    else
    {
      int my_min_acc = max(-this->max_acceleration,-delta_s);
      this->a = my_min_acc;
    }
  }
}

vector<vector<int> > Vehicle::generate_predictions(int horizon = 10) {

  // prediction vector of vectors
  vector<vector<int> > predictions;
  for( int i = 0; i < horizon; i++)
  {
    vector<int> check1 = state_at(i);
    int lane = check1[0];
    int s    = check1[1];

    vector<int> lane_s = {lane, s};

    predictions.push_back(lane_s);
  }
  return predictions;

}

//double Vehicle::change_lane_cost(Vehicle* ptr_vehicle, vector<Vehicle::snapshot> trajectory, map<int,vector < vector<int> > > predictions, Vehicle::TrajectoryData data){
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
//  if (cost != 0.0)
//    std::cout<<"!! \n \ncost for lane change is "<< cost << "\n\n" <<std::endl;
//
//  if(DEBUG){
//    std::cout << __FUNCTION__ << " has cost " << cost << " for lane " << trajectory[0].lane <<std::endl;
//  }
//
//  return cost;
//}
//
//double Vehicle::distance_from_goal_lane(Vehicle* ptr_vehicle, vector<Vehicle::snapshot> trajectory, map<int,vector < vector<int> > > predictions, Vehicle::TrajectoryData data){
//
//  double distance = double(abs(data.end_distance_to_goal));
//  distance = max(distance,1.0);
//  double time_to_goal = distance / data.avg_speed;
//  double lanes = double(data.end_lanes_from_goal);
//  double multiplier = 5 * lanes / time_to_goal;
//  double cost = multiplier * REACH_GOAL;
//
//  if(DEBUG){
//    std::cout << __FUNCTION__ << " has cost " << cost << " for lane " << trajectory[0].lane <<std::endl;
//  }
//
//  return cost;
//}
//
//double Vehicle::inefficiency_cost(Vehicle* ptr_vehicle, vector<Vehicle::snapshot> trajectory, map<int,vector < vector<int> > > predictions, Vehicle::TrajectoryData data){
//  double speed = data.avg_speed;
//  double target_speed = double(ptr_vehicle->target_speed);
//  double diff = target_speed - speed;
//  double pct = diff / target_speed;
//  double multiplier = pow(pct, 2.0);
//  double cost = multiplier * EFFICIENCY;
//
//  if(DEBUG){
//    std::cout << __FUNCTION__ << " has cost " << cost << " for lane " << trajectory[0].lane <<std::endl;
//  }
//
//  return cost;
//}
//
//double Vehicle::collision_cost(Vehicle* ptr_vehicle, vector<Vehicle::snapshot> trajectory, map<int,vector < vector<int> > > predictions, Vehicle::TrajectoryData data){
//  double cost;
//  if (data.collides.collision){
//    int time_til_collision = data.collides.time;
//    double exponent = pow(double(time_til_collision), 2.0);
//    double mult = exp(-exponent);
//
//    cost = mult * COLLISION;
//  }
//  else
//    cost = 0;
//
//  if(DEBUG){
//    std::cout << __FUNCTION__ << " has cost " << cost << " for lane " << trajectory[0].lane <<std::endl;
//  }
//
//  return cost;
//}
//
//double Vehicle::buffer_cost(Vehicle* ptr_vehicle, vector<Vehicle::snapshot> trajectory, map<int,vector < vector<int> > > predictions, Vehicle::TrajectoryData data){
//  double cost;
//  double closest;
//  closest = (float)data.closest_approach;
//  if (closest == 0.0)
//    cost = 10 * DANGER;
//  else {
//    double timesteps_away = closest / data.avg_speed;
//    if(timesteps_away > DESIRED_BUFFER)
//      cost = 0.0;
//    else {
//      double multiplier = 1.0 - pow(timesteps_away/DESIRED_BUFFER,2);
//      cost = multiplier * DANGER;
//    }
//  }
//
//  if(DEBUG){
//    std::cout << __FUNCTION__ << " has cost " << cost << " for lane " << trajectory[0].lane <<std::endl;
//  }
//
//  return cost;
//}
//
//double Vehicle::calculate_cost(Vehicle* ptr_vehicle, vector<Vehicle::snapshot> trajectory, map<int,vector < vector<int> > > predictions){
//
//  Vehicle::TrajectoryData trajectory_data = get_helper_data(ptr_vehicle, trajectory, predictions);
//  double cost = 0.0;
//  vector <double> costs;
//
//  costs.push_back(distance_from_goal_lane(ptr_vehicle, trajectory, predictions, trajectory_data));
//  costs.push_back(inefficiency_cost(ptr_vehicle, trajectory, predictions, trajectory_data));
//  costs.push_back(collision_cost(ptr_vehicle, trajectory, predictions, trajectory_data));
//  costs.push_back(buffer_cost(ptr_vehicle, trajectory, predictions, trajectory_data));
//  costs.push_back(change_lane_cost(ptr_vehicle, trajectory, predictions, trajectory_data));
//
//  for(int i=0; i<costs.size(); i++){
//    cost += costs[i];
//  }
//
//  return cost;
//}
//
//Vehicle::TrajectoryData Vehicle::get_helper_data(Vehicle* ptr_vehicle, vector<Vehicle::snapshot> trajectory, map<int,vector < vector<int> > > predictions){
//  vector<Vehicle::snapshot> t;
//  t = trajectory;
//  Vehicle::snapshot current_snapshot = t[0];
//  Vehicle::snapshot first = t.front();
//  Vehicle::snapshot last = t.back();
//
//  int end_distance_to_goal = ptr_vehicle->goal_s - last.s;
//  int end_lanes_from_goal = abs(ptr_vehicle->goal_lane - last.lane);
//  double dt = double(trajectory.size());
//
//  int proposed_lane = first.lane;
//  double avg_speed = (last.s - current_snapshot.s) / dt;
//
//  // initialize a bunch of variables
//  vector<int> accels;
//  int closest_approach;
//  closest_approach = 999999;
//  collider collides;
//  collides.collision = false;
//  Vehicle::snapshot last_snap;
//  last_snap = trajectory[0];
//  map<int,vector < vector<int> > > filtered = filter_predictions_by_lane(predictions, proposed_lane);
//
//  for(int i=1; i < PLANNING_HORIZON+1; i++) {
//
//    Vehicle::snapshot snapshot = trajectory[i];
//
//    lane = snapshot.lane;
//    s = snapshot.s;
//    v = snapshot.v;
//    a = snapshot.a;
//
//    accels.push_back(a);
//
//    // for all cars in our lane
//    map<int, vector<vector<int> > >::iterator it = filtered.begin();
//    while(it != filtered.end())
//    {
//      // first item : vehicle id
//      int v_id = it->first;
//      // second item :
//      vector<vector<int> > predicted_traj = it->second;
//
//      // state[0] = lane, state[1] = s
//      vector<int> state = predicted_traj[i];
//      vector<int> last_state = predicted_traj[i-1];
//
////      try{
//      bool vehicle_collides = check_collision(snapshot, last_state['s'], state['s']);
////      }
////      catch(string cont& str) {
////        cerr << str << endl;
////      }
//
//      if (vehicle_collides){
//        collides.collision = true;
//        collides.time = i;
//      }
//      int dist = abs(state['s'] - s);
//
//      if (dist < closest_approach)
//        closest_approach = dist;
//
//      it++;
//    }
//
//    last_snap = snapshot;
//}
//
//  // absolute max acceleration
//  int max_accel = 0;
//  for(int i=0; i<accels.size(); i++){
//    if(abs(accels[i]) >= max_accel)
//      max_accel = abs(accels[i]);
//  }
//
//  // rms accelerations
//  vector<int> rms_accels;
//  for(int i=0; i<accels.size(); i++){
//    rms_accels.push_back(pow(accels[i],2));
//  }
//
//  double rms_acceleration=0.0;
//  for(int i=0; i<rms_accels.size(); i++){
//    rms_acceleration += double(rms_accels[i]);
//  }
//  rms_acceleration /= rms_accels.size();
//
//  Vehicle::TrajectoryData traj_data;
//
//  traj_data.proposed_lane = proposed_lane;
//  traj_data.avg_speed = avg_speed;
//  traj_data.max_acceleration = max_accel;
//  traj_data.rms_acceleration = rms_acceleration;
//  traj_data.closest_approach = closest_approach;
//  traj_data.end_distance_to_goal = end_distance_to_goal;
//  traj_data.end_lanes_from_goal = end_lanes_from_goal;
//  traj_data.collides.collision =  collides.collision;
//  traj_data.collides.time =  collides.time;
//
//  return traj_data;
//}
//
//bool Vehicle::check_collision(Vehicle::snapshot snapshot, double s_previous, double s_now){
//  double s = double(snapshot.s);
//  double v = double(snapshot.v);
//  double v_target = s_now - s_previous;
//
//  if(s_previous < s){
//    if(s_now >= s)
//      return true;
//    else
//      return false;
//  }
//
//  if(s_previous > s){
//    if(s_now <= s)
//      return true;
//    else
//      return false;
//  }
//
//  if(s_previous == s){
//    if(v_target > v)
//      return false;
//    else
//      return true;
//  }
//
////  throw string("no return value");
//}
//
//map<int,vector < vector<int> > > Vehicle::filter_predictions_by_lane(map<int,vector < vector<int> > > predictions, int lane){
//  map<int,vector < vector<int> > > filtered;
//
//  map<int, vector<vector<int> > >::iterator it = predictions.begin();
//  while(it != predictions.end())
//  {
//    // first item : vehicle id
//    int v_id = it->first;
//    // second item :
//    vector<vector<int> > predicted_traj = it->second;
//
//    // If first prediction's lane is == lane... and v_id isn't ego vehicle
//    if(predicted_traj[0][0] == lane && v_id != -1)
//      filtered[v_id] = predicted_traj;
//
//    it++;
//  }
//
//  return filtered;
//}
