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
    string FSM_state;
  };

  struct collider{
    bool   collision ;  // is there a collision?
    double time;        // time collision happens
  };

  struct snapshot{
    int lane;          // lane number
    double s;          // distance (frenet coordinate)
    double end_path_s; // end_path_s
    double v;          // actual speed
    double a;          // actual acceleration
    string FSM_state;  // actual FSM state (Keep Lane, Lane Change Right/Left, Prepare For Lane Change Right/Left)
  };

  int lane;       // lane number

  // World coordinates
  double x;       // x world coordinate       in m
  double y;       // y world coordinate       in m

  // Frenet coordinates
  double s;       // distance along the road  in m
  double d;       // distance across the road in m

  double end_path_s = -1; // Frenet at the end of the path
  double end_path_d = -1; // Frenet at the end of the path

  double v;       // actual vehicle speed     in m/s
  double yaw;     // actual vehicle heading   in rad
  double a;       // actual acceleration      in m/s²

  // ego vehicle variables only..
  string FSM_state;              // actual FSM state (Keep Lane, Lane Change Right/Left, Prepare For Lane Change Right/Left)
  double L = 5.4;                // typical car length
  double preferred_buffer = 6;   // impacts "keep lane" behavior.
  double path_update_delay = 0.94; // in s

  double target_speed = -1;
  int lanes_available = -1;
  double max_acceleration = -1;
  int goal_lane = 1;

  /**
  * Constructor
  */
  Vehicle(int lane, double x, double y, double s, double d, double v, double yaw, double a);

  /**
  * Destructor
  */
  virtual ~Vehicle();

  void update_FSM_state(map<int, vector < vector<double> > > predictions);

  /*******/

  string _get_next_FSM_state(map<int,vector < vector<double> > > predictions);

  snapshot get_snapshot(void);

  vector<snapshot> _trajectory_for_FSM_state(string FSM_state, map<int,vector < vector<double> > > predictions, int horizon);

  void restore_state_from_snapshot(snapshot snapshot_temp);
  /*******/

  void configure(vector<double> road_data);

  void increment(double);

  vector<double> state_at(double t);

  bool collides_with(Vehicle other, int at_time);

  collider will_collide_with(Vehicle other, int timesteps);

  void realize_FSM_state(map<int, vector < vector<double> > > predictions);

  void realize_constant_speed();

  double _max_accel_for_lane(map<int,vector < vector<double> > > predictions, int lane, double s, double end_path_s);

  void realize_keep_lane(map<int,vector < vector<double> > > predictions);

  void realize_lane_change(map<int,vector < vector<double> > > predictions, string direction);

  void realize_prep_lane_change(map<int,vector < vector<double> > > predictions, string direction);

  vector<vector<double> > generate_predictions(int horizon);

};

#endif
