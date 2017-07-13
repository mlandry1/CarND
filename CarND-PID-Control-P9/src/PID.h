#ifndef PID_H
#define PID_H

#include <vector>

#define I_TOL 0.0001
#define I_ERROR_MAX_SCALE 10
#define STEP_NUM_RUN 200//1450
#define TWIDDLE_TOL 0.05

class PID {
public:

  double p_error;
  double i_error;
  double d_error;

  double error;
  double error_1;

  double set_point;
  double max_set_point;

  double min_output;
  double max_output;

  bool is_initialized;

  // Twiddle
  std::vector<double> dp;
  std::vector<double> p;

  bool twiddle_enabled;

  double best_err;
  double err;
  int it, step;
  double twiddle_tol;
  int index;
  bool add;
  bool sub;


  // Constructor
  PID();

  // Destructor.
  virtual ~PID();

  // Initialize PID.
  void Init(double Kp, double Ki, double Kd, double min, double max, bool twiddle_enabled=false, double max_set_point=INFINITY);

  // Update the PID's computation
  double Update(double error);
};

#endif /* PID_H */
