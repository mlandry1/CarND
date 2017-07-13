#include <math.h>
#include <iostream>

#include "PID.h"

using namespace std;

/*
* TODO: Done!

Complete the PID class.
*/

PID::PID() {
  p_error = 0.0;
  i_error = 0.0;
  d_error = 0.0;

  // PID coefficients
  p = {0.0, 0.0, 0.0};

  error =   0.0;
  error_1 = 0.0;

  min_output = 0.0;
  max_output = 0.0;

  set_point = 0.0;
  max_set_point = INFINITY;

  is_initialized = false;

  // Twiddling parameters
  dp = {0.0, 0.0, 0.0};
  twiddle_enabled = false;

  best_err = INFINITY;
  err = 0.0;
  it = 0;
  step = 0;
  twiddle_tol = TWIDDLE_TOL;
  // parameter index
  index = 0;
  add = false;
  sub = false;
}

PID::~PID() {}

void PID::Init(double Kp, double Ki, double Kd, double min, double max, bool twiddle_enabled, double max_set_point) {

  PID::p = {Kp, Ki, Kd};
  PID::min_output = min;
  PID::max_output = max;
  PID::set_point = 0.0;
  PID::max_set_point = max_set_point;

  // Twiddling parameters
  PID::dp = {0.05*Kp,0.05*Ki,0.05*Kd};
  PID::twiddle_enabled = twiddle_enabled;
}

double PID::Update(double error) {
  double output = 0.0;

  if (is_initialized == false) {
    error_1 = error;
    is_initialized = true;
  }

  // delta_t is assumed to be included inside Kd and Ki parameters...
  p_error = error*p[0];
  d_error = (error-error_1)*p[2];

  // Update last cycle's error
  error_1 = error;

  // Reset integrator if Ki is close to 0.
  if (p[1] <= I_TOL)
    i_error = 0.0;

  // Limit integrator value according to a scale of Ki
  if ( error*p[1] + i_error > p[1] * I_ERROR_MAX_SCALE)
    i_error = p[1] * I_ERROR_MAX_SCALE;
  else if ( error*p[1] + i_error < -p[1] * I_ERROR_MAX_SCALE)
    i_error = -p[1] * I_ERROR_MAX_SCALE;

  // Anti-wind up
  if ((p_error + error*p[1] + i_error + d_error) <= max_output){
    i_error = error*p[1] + i_error;
  }
  else if ((p_error + error*p[1] + i_error + d_error) >= min_output){
    i_error = error*p[1] + i_error;
  }

  // Output computation
  output = p_error + i_error + d_error;

  //Ouput limiter
  if (output > max_output)
    output = max_output;
  else if (output < min_output)
    output = min_output;

  /*
   * Twiddle
   */
  if (twiddle_enabled && (dp[0]+dp[1]+dp[2]) > twiddle_tol){
    // update total error (settle time is ignored since the car is approximately
    // in the middle of the road at initialisation)
    err += error * error;

    // a "run" is considered complete every STEP_NUM_RUN steps..
    // the total error is then reseted and the twiddle algorithm is evaluated.
    if (step % STEP_NUM_RUN == 0 ){

      std::cout << "Iteration: " << it << std::endl;
      std::cout << "Total error: " << err << std::endl;
      std::cout << "Best error: " << best_err << std::endl;
      std::cout << "P= "<< p[0] << "  I= " << p[1] << "  D= " << p[2] << std::endl;

      // first try to add
      if (!add && !sub){
        p[index] += dp[index];
        add = true;
      }
      // then try to sub
      else if (add && !sub){
        p[index] -= 2 * dp[index];
        sub = true;
      }
      // else set it back the way it was, reduce dp, and move to the next param
      else{
        p[index] += dp[index];
        dp[index] *= 0.9;

        // Next parameter
        index++;
        if(index == 3){
          index = 0;
          it++;
          // Debug
          std::cout << "Iteration: " << it << std::endl;
          std::cout << "Total error: " << err << std::endl;
          std::cout << "Best error: " << best_err << std::endl;
          std::cout << "P= "<< p[0] << "  I= " << p[1] << "  D= " << p[2] << std::endl;

        }
        add = false;
        sub = false;
      }

      // Initialize the error after 2 lap
      if (step == 2*STEP_NUM_RUN){
        best_err = err;
        std::cout << "Total error initialized" << std::endl;
      }
      // If the error improved..
      if (err < best_err && step >= 3*STEP_NUM_RUN){
        best_err = err;
        dp[index] *= 1.1;

        std::cout << "Improved" << std::endl;
        std::cout << "Iteration: " << it << std::endl;
        std::cout << "Total error: " << err << std::endl;
        std::cout << "Best error: " << best_err << std::endl;
        std::cout << "P= "<< p[0] << "  I= " << p[1] << "  D= " << p[2] << std::endl;

        // Next parameter
        index++;
        if(index == 3){
          index = 0;
          it++;
          // Debug
          std::cout << "Iteration: " << it << std::endl;
          std::cout << "Total error: " << err << std::endl;
          std::cout << "Best error: " << best_err << std::endl;
          std::cout << "P= "<< p[0] << "  I= " << p[1] << "  D= " << p[2] << std::endl;
        }
        add = false;
        sub = false;
      }

      // Total error is reseted
      err = 0;
    }

    // Progress bar display in console
    float progress = float(step%STEP_NUM_RUN)/float(STEP_NUM_RUN);
    int barWidth = 70;

    std::cout << "[";
    int pos = barWidth * progress;
    for (int i = 0; i < barWidth; ++i) {
        if (i < pos) std::cout << "=";
        else if (i == pos) std::cout << ">";
        else std::cout << " ";
    }
    std::cout << "] " << int(progress * 100.0) << " %\r";
    std::cout.flush();
  }
  else if(twiddle_enabled && step % STEP_NUM_RUN == 0 && step > 0)
    std::cout << "Twiddle tolerance reached. P= "<< p[0] << "  I= " << p[1] << "  D= " << p[2] << std::endl;

  // Step counter
  step++;

  return output;
}
