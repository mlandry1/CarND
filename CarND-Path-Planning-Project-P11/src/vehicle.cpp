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

/**
 * Initializes Vehicle
 */
Vehicle::Vehicle(int lane, int s, int v, int a) {

  this->lane = lane;
  this->s = s;
  this->v = v;
  this->a = a;
  this->FSM_state = "CS";
  this->max_acceleration = -1;

}

#define DEBUG true

Vehicle::~Vehicle() {}

void Vehicle::update_FSM_state(map<int,vector < vector<int> > > predictions) {
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

  if(DEBUG){
    std::cout << " Next FSM_state is " << _FSM_state << std::endl;
  }

  this->FSM_state = _FSM_state;
}

string Vehicle::_get_next_FSM_state(map<int,vector < vector<int> > > predictions) {

  vector<string> FSM_states = {"KL", "LCL", "LCR", "PLCL", "PLCR"};

  // Remove impossible/unuseful next FSM_states
  if(this->lane == 0){
    FSM_states.erase(std::remove(FSM_states.begin(), FSM_states.end(), "LCR"), FSM_states.end());
    FSM_states.erase(std::remove(FSM_states.begin(), FSM_states.end(), "PLCR"), FSM_states.end());
  }
  if(this->lane == this->lanes_available - 1){
    FSM_states.erase(std::remove(FSM_states.begin(), FSM_states.end(), "LCL"), FSM_states.end());
    FSM_states.erase(std::remove(FSM_states.begin(), FSM_states.end(), "PLCL"), FSM_states.end());
  }

  // initialize a bunch of variables
  cost_t cost;
  vector<cost_t> costs;
  map<int,vector < vector<int> > > predictions_copy;
  vector<Vehicle::snapshot> trajectory;
  // for each possible FSM_state
  for(int i=0; i<FSM_states.size(); i++){
    // refresh the predictions
    predictions_copy = predictions;

    // compute the ego vehicle trajectory for a given FSM state
    trajectory = this->_trajectory_for_FSM_state(FSM_states[i], predictions_copy, 5);

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

vector<Vehicle::snapshot> Vehicle::_trajectory_for_FSM_state(string FSM_state, map<int,vector < vector<int> > > predictions, int horizon = 5){
  //save ego vehicle state
  Vehicle::snapshot snapshot_temp = this->get_snapshot();

  // pretend to be in new proposed state
  this->FSM_state = FSM_state;

  // start trajectory with the original ego vehicle state
  vector<Vehicle::snapshot> trajectory;
  trajectory.push_back(snapshot_temp);

  // for each given time horizon
  for(int i=0; i<horizon; i++){
    // restore to the original ego vehicle state
    this->restore_state_from_snapshot(snapshot_temp);
    // FSM state
    this->FSM_state = FSM_state;
    // realize predicitons
    this->realize_FSM_state(predictions);
    //ensure the vehicle is still in an existing lane..
    assert(0 <= this->lane && this->lane < this->lanes_available);
    // increment simulation
    this->increment(1);
    // include incremented vehicle state in the trajectory..
    trajectory.push_back(this->get_snapshot());

    //need to remove first prediction for each vehicle on the road
    map<int, vector<vector<int> > >::iterator it = predictions.begin();
    while(it != predictions.end())
    {
      int v_id = it->first;
      vector<vector<int> > v = it->second;
      v.erase(v.begin());
      it->second = v;
      it++;
    }
  }

  // restore original vehicle state
  this->restore_state_from_snapshot(snapshot_temp);
  return trajectory;
}

Vehicle::snapshot Vehicle::get_snapshot(){
  Vehicle::snapshot snapshot_temp;
  snapshot_temp.lane = this->lane;
  snapshot_temp.s = this->s;
  snapshot_temp.v = this->v;
  snapshot_temp.a = this->a;
  snapshot_temp.FSM_state = this->FSM_state;

  return snapshot_temp;

}

void Vehicle::restore_state_from_snapshot(Vehicle::snapshot snapshot_temp){
  this->lane = snapshot_temp.lane;
  this->s = snapshot_temp.s;
  this->v = snapshot_temp.v;
  this->a = snapshot_temp.a;
  this->FSM_state = snapshot_temp.FSM_state;
}

void Vehicle::configure(vector<int> road_data) {
/*
  Called by simulator before simulation begins. Sets various
  parameters which will impact the ego vehicle.
  */
  lanes_available = road_data[0];
  target_speed = road_data[1];
  max_acceleration = road_data[2];
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

void Vehicle::realize_FSM_state(map<int,vector < vector<int> > > predictions) {

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
    realize_keep_lane(predictions);
  }
  else if(FSM_state.compare("LCL") == 0)
  {
    realize_lane_change(predictions, "L");
  }
  else if(FSM_state.compare("LCR") == 0)
  {
    realize_lane_change(predictions, "R");
  }
  else if(FSM_state.compare("PLCL") == 0)
  {
    realize_prep_lane_change(predictions, "L");
  }
  else if(FSM_state.compare("PLCR") == 0)
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

