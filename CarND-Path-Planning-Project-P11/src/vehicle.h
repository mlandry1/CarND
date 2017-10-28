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
    bool collision ;  // is there a collision?
    int  time;        // time collision happens
  };

  struct snapshot{

    int lane;           // lane number
    int s;              // distance (frenet coordinate)
    int v;              // actual speed
    int a;              // actual acceleration
    string FSM_state;   // actual FSM state (Keep Lane, Lane Change Right/Left, Prepare For Lane Change Right/Left)

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

  string FSM_state;

  /**
  * Constructor
  */
  Vehicle(int lane, int s, int v, int a);

  /**
  * Destructor
  */
  virtual ~Vehicle();

  void update_FSM_state(map<int, vector <vector<int> > > predictions);

  /*******/

  string _get_next_FSM_state(map<int,vector < vector<int> > > predictions);

  snapshot get_snapshot(void);

  vector<snapshot> _trajectory_for_FSM_state(string FSM_state, map<int,vector < vector<int> > > predictions, int horizon);

  void restore_state_from_snapshot(snapshot snapshot_temp);
  /*******/

  void configure(vector<int> road_data);

  string display();

  void increment(int dt);

  vector<int> state_at(int t);

  bool collides_with(Vehicle other, int at_time);

  collider will_collide_with(Vehicle other, int timesteps);

  void realize_FSM_state(map<int, vector < vector<int> > > predictions);

  void realize_constant_speed();

  int _max_accel_for_lane(map<int,vector<vector<int> > > predictions, int lane, int s);

  void realize_keep_lane(map<int, vector< vector<int> > > predictions);

  void realize_lane_change(map<int,vector< vector<int> > > predictions, string direction);

  void realize_prep_lane_change(map<int,vector< vector<int> > > predictions, string direction);

  vector<vector<int> > generate_predictions(int horizon);

};

#endif
