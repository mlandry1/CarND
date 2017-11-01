#include <iostream>
#include "vehicle.h"
#include "cost.h"
#include <math.h>
#include <map>
#include <string>
#include <iterator>
#include <vector>
#include <algorithm>
#include "assert.h"

#define DEBUG_VEHICLE true
#define DELTA_T_PREDICTION 0.35  //timestep for trajectory predictions

/**
 * Initializes Vehicle
 */
Vehicle::Vehicle(int lane, double x, double y, double s, double d, double v, double yaw, double a) {

  this->lane = lane;
  this->x = x;
  this->y = y;
  this->s = s;
  this->d = d;
  this->v = v;
  this->yaw = yaw;
  this->a = a;
  this->FSM_state = "CS";

}

Vehicle::~Vehicle() {}

Vehicle::snapshot Vehicle::get_snapshot(){
  Vehicle::snapshot snapshot_temp;
  snapshot_temp.lane       = this->lane;
  snapshot_temp.s          = this->s;
  snapshot_temp.end_path_s = this->end_path_s;
  snapshot_temp.v          = this->v;
  snapshot_temp.a          = this->a;
  snapshot_temp.FSM_state  = this->FSM_state;

  return snapshot_temp;
}

void Vehicle::restore_state_from_snapshot(Vehicle::snapshot snapshot_temp){
  this->lane       = snapshot_temp.lane;
  this->s          = snapshot_temp.s;
  this->end_path_s = snapshot_temp.end_path_s;
  this->v          = snapshot_temp.v;
  this->a          = snapshot_temp.a;
  this->FSM_state  = snapshot_temp.FSM_state;
}

void Vehicle::configure(vector<double> road_data) {
/*
  Called by simulator before simulation begins. Sets various
  parameters which will impact the ego vehicle.
  */
  lanes_available  = int(road_data[0]);
  target_speed     = road_data[1];
  max_acceleration = road_data[2];
}

vector<vector<double> > Vehicle::generate_predictions(int horizon = 10) {

  // prediction vector of vectors
  vector<vector<double> > predictions;

  // for each second in the future
  for( int i = 0; i < int(double(horizon)/DELTA_T_PREDICTION); i++)
  {
    vector<double> check1 = state_at(i * DELTA_T_PREDICTION);
    double lane = check1[0];
    double s    = check1[1];

    vector<double> lane_s = {lane, s};

    predictions.push_back(lane_s);
  }
  return predictions;

}

void Vehicle::update_FSM_state(map<int,vector < vector<double> > > predictions) {
  /*
  Updates the "FSM_state" of the vehicle by assigning one of the
  following values to 'this->FSM_state':

  "KL" - Keep Lane
   - The vehicle will attempt to drive its target speed, unless there is
     traffic in front of it, in which case it will slow down.

  "LCL" or "LCR" - Lane Change Left / Right
   - The vehicle will IMMEDIATELY change lanes and then follow longitudinal
     behavior for the "KL" FSM_state in the new lane.

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
      {"s" : 10, "lane":
    ]
  }

  */

  string _FSM_state = this->_get_next_FSM_state(predictions);

  //TODO remove this
//  _FSM_state = "KL";

  this->FSM_state = _FSM_state;
}

string Vehicle::_get_next_FSM_state(map<int,vector < vector<double> > > predictions) {
//TODO remove
//  vector<string> FSM_states = {"KL", "LCL", "LCR", "PLCL", "PLCR"};
  vector<string> FSM_states = {"KL", "LCL", "LCR"};

  // Remove impossible/unuseful next FSM_states
  if(this->lane == 0){
    FSM_states.erase(std::remove(FSM_states.begin(), FSM_states.end(), "LCL"), FSM_states.end());
    //TODO remove
//    FSM_states.erase(std::remove(FSM_states.begin(), FSM_states.end(), "PLCL"), FSM_states.end());
  }
  if(this->lane == this->lanes_available - 1){
    FSM_states.erase(std::remove(FSM_states.begin(), FSM_states.end(), "LCR"), FSM_states.end());
    //TODO remove
//    FSM_states.erase(std::remove(FSM_states.begin(), FSM_states.end(), "PLCR"), FSM_states.end());
  }

  // initialize a bunch of variables
  cost_t cost;
  vector<cost_t> costs;
  map<int,vector < vector<double> > > predictions_copy;
  vector<Vehicle::snapshot> trajectory;

  if(DEBUG_VEHICLE){
    std::cout << std::string(100, '*') <<
        "\nGenerate trajectories for each possible FSM state\n" <<
        std::string(100, '*') << std::endl;
  }
  // for each possible FSM_state
  for(int i=0; i<FSM_states.size(); i++){
    // refresh the predictions
    predictions_copy = predictions;

    // compute the ego vehicle trajectory for a given FSM state for the 5 next seconds..
    trajectory = this->_trajectory_for_FSM_state(FSM_states[i], predictions_copy, 5);

    if(DEBUG_VEHICLE){
      std::cout << "--" << FSM_states[i] << "--\tLane\t" << trajectory[1].lane;
    }
    // compute the cost of the ego vehicle trajectory and note the associated FSM state
    cost.cost = calculate_cost(this, trajectory, predictions);
    cost.FSM_state = FSM_states[i];
    costs.push_back(cost);
  }

  // find the lowest cost and its associated FSM state
  double best_cost = costs[0].cost;
  string best_FSM_state = costs[0].FSM_state;

  for(int i=0; i<costs.size(); i++){
    if( costs[i].cost < best_cost){
      best_cost = costs[i].cost;
      best_FSM_state = costs[i].FSM_state;
    }
  }

  return best_FSM_state;
}

void Vehicle::realize_FSM_state(map<int,vector < vector<double> > > predictions, bool verbose) {

  /*
  Given a state, realize it by adjusting acceleration and lane.
  Note - lane changes happen instantaneously.
  */
  string FSM_state = this->FSM_state;
  if(FSM_state.compare("CS") == 0)
  {
    realize_constant_speed();
  }
  else if(FSM_state.compare("KL") == 0)
  {
    realize_keep_lane(predictions, verbose);
  }
  else if(FSM_state.compare("LCL") == 0)
  {
    realize_lane_change(predictions, "L", verbose);
  }
  else if(FSM_state.compare("LCR") == 0)
  {
    realize_lane_change(predictions, "R", verbose);
  }
  else if(FSM_state.compare("PLCL") == 0)
  {
    realize_prep_lane_change(predictions, "L",verbose);
  }
  else if(FSM_state.compare("PLCR") == 0)
  {
    realize_prep_lane_change(predictions, "R", verbose);
  }

}

void Vehicle::realize_constant_speed() {
  a = 0;
}

void Vehicle::realize_keep_lane(map<int,vector < vector<double> > > predictions, bool verbose) {
  this->a = _max_accel_for_lane(predictions, this->lane, this->s, this->end_path_s, verbose);
}

void Vehicle::realize_lane_change(map<int,vector < vector<double> > > predictions, string direction, bool verbose) {
  int delta = -1;
  if (direction.compare("R") == 0)
  {
    delta = 1;
  }
  this->lane += delta;
  this->a = _max_accel_for_lane(predictions, this->lane, this->s, this->end_path_s, verbose);
}

vector<double> Vehicle::state_at(double t) {

  /*
  Predicts state of vehicle in t seconds (assuming constant acceleration)
  */
  double s = this->s + this->v * t + this->a * t * t / 2;
  double v = this->v + this->a * t;
  return {double(this->lane), s, v, this->a};
}

void Vehicle::increment(double dt) {
//TODO is it ok?
  this->s += this->v * dt + this->a * dt * dt / 2;
  this->v += this->a * dt;
}

vector<Vehicle::snapshot> Vehicle::_trajectory_for_FSM_state(string FSM_state, map<int,vector < vector<double> > > predictions, int horizon = 5){
  //save ego vehicle state
  Vehicle::snapshot snapshot_temp = this->get_snapshot();

  // pretend to be in new proposed state
  this->FSM_state = FSM_state;

  // start trajectory with the original ego vehicle state
  vector<Vehicle::snapshot> trajectory;
  trajectory.push_back(snapshot_temp);

  // for each given time horizon
  for(int i=0; i<int(double(horizon)/DELTA_T_PREDICTION); i++){
    // restore to the original ego vehicle state
//    this->restore_state_from_snapshot(snapshot_temp);
    // restore to the original ego vehicle lane
    this->lane = snapshot_temp.lane;

    // FSM state
//    this->FSM_state = FSM_state;
    // realize FSM state (gives the lane and the acceleration for the timestep)
    this->realize_FSM_state(predictions, false);
    //ensure the vehicle is still in an existing lane..
    assert(0 <= this->lane && this->lane < this->lanes_available);
    // increment simulation
    this->increment(DELTA_T_PREDICTION);
    this->end_path_s = this->s + this->v * this->path_update_delay;
    // include incremented vehicle state in the trajectory..
    trajectory.push_back(this->get_snapshot());

    //need to remove first prediction for each vehicle on the road
    map<int, vector<vector<double> > >::iterator it = predictions.begin();
    while(it != predictions.end())
    {
      int v_id = it->first;
      vector<vector<double> > v = it->second;
      v.erase(v.begin());
      it->second = v;
      it++;
    }
  }

  // restore original vehicle state
  this->restore_state_from_snapshot(snapshot_temp);
  return trajectory;
}

// TODO: enlever end_path_s des arguments??
double Vehicle::_max_accel_for_lane(map<int,vector < vector<double> > > predictions, int lane, double s, double end_path_s, bool verbose) {

  double delta_v_til_target = this->target_speed - this->v;
  // either max_acceleration, or delta_v over 1s..
  double max_acc = min(this->max_acceleration, delta_v_til_target);

  map<int, vector<vector<double> > >::iterator it = predictions.begin();

  // creates a list of cars in front of the target s value in the target lane
  vector<vector<vector<double> > > in_front;
  while(it != predictions.end())
  {
    int v_id = it->first;

    vector<vector<double> > v = it->second;

    if((v[0][0] == double(lane)) && (v[0][1] > s) && v_id != -1)
    {
      in_front.push_back(v);
    }
    it++;
  }

  // if there is at least one vehicle in front of the target s value in the target lane
  if(in_front.size() > 0)
  {
    double min_s = 1000;
    vector<vector<double>> leading = {};
    // for all the vehicle in front of the target s value in the target lane
    for(int i = 0; i < in_front.size(); i++)
    {
      // find the car right in front of the target s value in the target lane
      if((in_front[i][0][1]-s) < min_s)
      {
        min_s = (in_front[i][0][1]-s);
        leading = in_front[i];
      }
    }

    // car in front next position
    double next_pos = leading[1][1];
    // ego next position
    double my_next = s + this->v * 1.0 + this->a * 1.0 * 1.0 / 2;
    // supposed to be positive !
    double separation_next = next_pos - my_next;
    // supposed to be positive !
    double available_room = separation_next - this->preferred_buffer - this->L;

    max_acc = min(max_acc, available_room);
    // negative saturation
    max_acc = max(max_acc, -this->max_acceleration);

    if(verbose){
      std::cout << "Looking in lane  : " << lane << std::endl;
      std::cout << "Available_room   : " << available_room << "m" << std::endl;
      std::cout << "Actual speed     : " << this->v << "m/s" << std::endl;
      std::cout << "Accel cmd        : " << max_acc << "m/s²" << std::endl<< std::endl<< std::endl;
    }
  }

  return max_acc;
}

void Vehicle::realize_prep_lane_change(map<int,vector < vector<double> > > predictions, string direction, bool verbose) {
  //TODO: inclure une référence à ça ? this->end_path_s

  int delta = -1;
  if (direction.compare("R") == 0)
  {
    delta = 1;
  }
  int lane = this->lane + delta;

  map<int, vector<vector<double> > >::iterator it = predictions.begin();

  // creates a list of cars behind the ego vehicle in target lane
  vector<vector<vector<double> > > at_behind;
  while(it != predictions.end())
  {
    int v_id = it->first;
    vector<vector<double> > v = it->second;

    if((v[0][0] == lane) && (v[0][1] <= this->s))
    {
      at_behind.push_back(v);
    }
    it++;
  }
  // if there is at least one vehicle behind ego vehicle in target lane
  if(at_behind.size() > 0)
  {

    double max_s = -1000;
    vector<vector<double> > nearest_behind = {};
    for(int i = 0; i < at_behind.size(); i++)
    {
      // find the nearest car behind ego in target lane
      if((at_behind[i][0][1]) > max_s)
      {
        max_s = at_behind[i][0][1];
        nearest_behind = at_behind[i];
      }
    }
    double target_vel = nearest_behind[1][1] - nearest_behind[0][1];
    double delta_v = this->v - target_vel;
    double delta_s = this->s - nearest_behind[0][1];
    // TODO : put a tolerance here ??
    if(delta_v != 0.0)
    {
      // time behind ego vehicle
      double t = -2 * delta_s/delta_v;
      // acceleration
      double a;
      // if nearest behind vehicle is too close don't change actual accel
      // TODO: put a tolerance here to take vehicle size into consideration??
      if (t == 0)
      {
        a = this->a;
      }
      else
      {
        //
        a = delta_v/t;
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
      // TODO: see if we need a factor on delta_s..
      // if delta_v==0 then slow down to close the gap?
      double my_min_acc = max(-this->max_acceleration,-delta_s);
      this->a = my_min_acc;
    }
  }
}

bool Vehicle::collides_with(Vehicle other, int at_time) {

  /*
  Simple collision detection.
  */
  vector<double> check1 = this->state_at(at_time);
  vector<double> check2 = other.state_at(at_time);
  return (check1[0] == check2[0]) && (abs(check1[1]-check2[1]) <= this->L);
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
