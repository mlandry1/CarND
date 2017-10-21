#ifndef VEHICLE_H
#define VEHICLE_H
#include <iostream>
#include <random>
#include <sstream>
#include <fstream>
#include <math.h>
#include <vector>
#include <map>
#include <string>
#include <iterator>

using namespace std;

class Vehicle {
public:

  struct cost_t{
    double cost;
    string state;
  };

  struct collider{
    bool collision ;  // is there a collision?
    int  time;        // time collision happens
  };

  struct snapshot{

    int lane;       // lane number
    int s;          // distance (frenet coordinate)
    int v;          // actual speed
    int a;          // actual acceleration
    string state;   // actual FSM state (Keep Lane, Lane Change Right/Left, Prepare For Lane Change Right/Left)

  };

  struct TrajectoryData{
    int         proposed_lane;
    double      avg_speed;
    int         max_acceleration;
    double      rms_acceleration;
    double      closest_approach;
    int         end_distance_to_goal;
    int         end_lanes_from_goal;
    collider  collides;
  };



  int L = 1;

  int preferred_buffer = 6; // impacts "keep lane" behavior.

  int lane;

  int s;

  int v;

  int a;

  int target_speed;

  int lanes_available;

  int max_acceleration;

  int goal_lane;

  int goal_s;

  string state;

  /**
  * Constructor
  */
  Vehicle(int lane, int s, int v, int a);

  /**
  * Destructor
  */
  virtual ~Vehicle();

  void update_state(map<int, vector <vector<int> > > predictions);

  /*******/

  string _get_next_state(map<int,vector < vector<int> > > predictions);

  snapshot get_snapshot(void);

  vector<snapshot> _trajectory_for_state(string state, map<int,vector < vector<int> > > predictions, int horizon);

  void restore_state_from_snapshot(snapshot snapshot_temp);
  /*******/

  void configure(vector<int> road_data);

  string display();

  void increment(int dt);

  vector<int> state_at(int t);

  bool collides_with(Vehicle other, int at_time);

  collider will_collide_with(Vehicle other, int timesteps);

  void realize_state(map<int, vector < vector<int> > > predictions);

  void realize_constant_speed();

  int _max_accel_for_lane(map<int,vector<vector<int> > > predictions, int lane, int s);

  void realize_keep_lane(map<int, vector< vector<int> > > predictions);

  void realize_lane_change(map<int,vector< vector<int> > > predictions, string direction);

  void realize_prep_lane_change(map<int,vector< vector<int> > > predictions, string direction);

  vector<vector<int> > generate_predictions(int horizon);

  /******************/
  double change_lane_cost(Vehicle* ptr_vehicle, vector<Vehicle::snapshot> trajectory, map<int,vector < vector<int> > > predictions, Vehicle::TrajectoryData data);

  double distance_from_goal_lane(Vehicle* ptr_vehicle, vector<Vehicle::snapshot> trajectory, map<int,vector < vector<int> > > predictions, Vehicle::TrajectoryData data);

  double inefficiency_cost(Vehicle* ptr_vehicle, vector<Vehicle::snapshot> trajectory, map<int,vector < vector<int> > > predictions, Vehicle::TrajectoryData data);

  double collision_cost(Vehicle* ptr_vehicle, vector<Vehicle::snapshot> trajectory, map<int,vector < vector<int> > > predictions, Vehicle::TrajectoryData data);

  double buffer_cost(Vehicle* ptr_vehicle, vector<Vehicle::snapshot> trajectory, map<int,vector < vector<int> > > predictions, Vehicle::TrajectoryData data);

  double calculate_cost(Vehicle* ptr_vehicle, vector<Vehicle::snapshot> trajectory, map<int,vector < vector<int> > > predictions);

  TrajectoryData get_helper_data(Vehicle* ptr_vehicle, vector<Vehicle::snapshot> trajectory, map<int,vector < vector<int> > > predictions);

  bool check_collision(Vehicle::snapshot snapshot, double s_previous, double s_now);

  map<int,vector < vector<int> > > filter_predictions_by_lane(map<int,vector < vector<int> > > predictions, int lane);


};

#endif
