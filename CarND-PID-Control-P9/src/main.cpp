#include <uWS/uWS.h>
#include <iostream>
#include "json.hpp"
#include "PID.h"
#include <math.h>

#define AUTO_TUNING 0
#define RACING_MODE 0
#define DEBUG 1

// for convenience
using json = nlohmann::json;

// For converting back and forth between radians and degrees.
constexpr double pi() { return M_PI; }
double deg2rad(double x) { return x * pi() / 180; }
double rad2deg(double x) { return x * 180 / pi(); }

// Checks if the SocketIO event has JSON data.
// If there is data the JSON object in string format will be returned,
// else the empty string "" will be returned.
std::string hasData(std::string s) {
  auto found_null = s.find("null");
  auto b1 = s.find_first_of("[");
  auto b2 = s.find_last_of("]");
  if (found_null != std::string::npos) {
    return "";
  }
  else if (b1 != std::string::npos && b2 != std::string::npos) {
    return s.substr(b1, b2 - b1 + 1);
  }
  return "";
}

int main()
{
  uWS::Hub h;

  // throttle pid
  PID pid_t;
  pid_t.Init(0.32, 0.03, 0.01, -1.0, 1.0, false, 70.0);


  // steering pid
  PID pid_s;
//  pid_s.Init(0.13, 0.010, 3.1, -1.0, 1.0, AUTO_TUNING);
//  // total error = 303.266
//  pid_s.Init(0.132742, 0.010995, 3.22905, -1.0, 1.0, AUTO_TUNING);
//  // total_error =
//  pid_s.Init(0.132742, 0.0104452, 3.22905, -1.0, 1.0, AUTO_TUNING);
//
//  //totl error = 491.699 @50mph
//  pid_s.Init(0.132742, 0.0115447, 3.22905, -1.0, 1.0, AUTO_TUNING);

  // // Speed set-point adaptation
//  pid_t.set_point = pid_t.max_set_point - angle * 0.35;

  // 1rst test 70mph
//  pid_s.Init(0.2, 0.00000, 3.25, -1.0, 1.0, AUTO_TUNING);

  // top speed 70mph  (exp :1.2)
//  pid_s.Init(0.132742, 0.0115447, 3.22905, -1.0, 1.0, AUTO_TUNING);

  // throttle = 1  (exp :2.0)
//  pid_s.Init(0.15, 0.0115447, 3.8, -1.0, 1.0, AUTO_TUNING);

  // throttle = 1  (exp :2.0)
//  pid_s.Init(0.105, 0.001, 3.2, -1.0, 1.0, AUTO_TUNING);


  // throttle =1 (exp :1.2) Steps = 200
//  pid_s.Init(0.105, 0.0009, 3.2, -1.0, 1.0, AUTO_TUNING);

  // final tuning (manual)
  pid_s.Init(0.095, 0.0079, 3.3, -1.0, 1.0, AUTO_TUNING);


  h.onMessage([&pid_s, &pid_t](uWS::WebSocket<uWS::SERVER> ws, char *data, size_t length, uWS::OpCode opCode) {
    // "42" at the start of the message means there's a websocket message event.
    // The 4 signifies a websocket message
    // The 2 signifies a websocket event
    static bool first_it = true;

    if (first_it == true || (pid_s.step % STEP_NUM_RUN == 0 && AUTO_TUNING)){
      // reset the simulator on first execution, at the of each run
      std::string msg("42[\"reset\", {}]");
      ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
      first_it = false;
    }

    if (length && length > 2 && data[0] == '4' && data[1] == '2')
    {
      auto s = hasData(std::string(data).substr(0, length));
      if (s != "") {
        auto j = json::parse(s);
        std::string event = j[0].get<std::string>();
        if (event == "telemetry") {
          // j[1] is the data JSON object
          double cte = std::stod(j[1]["cte"].get<std::string>());
          double speed = std::stod(j[1]["speed"].get<std::string>());
          double angle = std::stod(j[1]["steering_angle"].get<std::string>());
          double steer_value, throttle_value;

          /***********************************************************************************************************************************************************/
          /*
          * TODO: Done!
          * Calcuate steering value here, remember the steering value is
          * [-1, 1].
          * NOTE: Feel free to play around with the throttle and speed. Maybe use
          * another PID controller to control the speed!
          */
          
          // update errors and calculate steer_value at each step
          steer_value = pid_s.Update(-fabs(cte)/cte * pow(fabs(cte),1.4));

          if (DEBUG == 1){
            // DEBUG Steering
            std::cout << "Steering value: " << steer_value << std::endl;
            std::cout << "Error: " << cte <<  std::endl;
            std::cout << "Angle: " << angle <<  std::endl;
            std::cout << "P: " << pid_s.p_error << "    I: " << pid_s.i_error << "    D: " << pid_s.d_error << std::endl << std::endl;
          }

          if (RACING_MODE == 0){
            pid_t.set_point = 30.0;

            // Speed error computation
            double speed_error = pid_t.set_point - speed;
            throttle_value = pid_t.Update(speed_error);

            if (DEBUG == 1){
              // DEBUG Throttle
              std::cout << "Throttle Value: " << throttle_value << std::endl;
              std::cout << "Error: " << speed_error <<  std::endl;
              std::cout << "P: " << pid_t.p_error << "    I: " << pid_t.i_error << "    D: " << pid_t.d_error << std::endl << std::endl;
            }
          }
          else{
            throttle_value = 1.0;
            if (fabs(cte) > 0.35 && fabs(angle)> 5.5 && speed > 50){
              throttle_value = -1.0;
            }
          }

          /***********************************************************************************************************************************************************/

          json msgJson;
          msgJson["steering_angle"] = steer_value;
          msgJson["throttle"] = throttle_value;
          auto msg = "42[\"steer\"," + msgJson.dump() + "]";
//          std::cout << msg << std::endl;
          ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
        }
      } else {
        // Manual driving
        std::string msg = "42[\"manual\",{}]";
        ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
      }
    }
  });

  // We don't need this since we're not using HTTP but if it's removed the program
  // doesn't compile :-(
  h.onHttpRequest([](uWS::HttpResponse *res, uWS::HttpRequest req, char *data, size_t, size_t) {
    const std::string s = "<h1>Hello world!</h1>";
    if (req.getUrl().valueLength == 1)
    {
      res->end(s.data(), s.length());
    }
    else
    {
      // i guess this should be done more gracefully?
      res->end(nullptr, 0);
    }
  });

  h.onConnection([&h](uWS::WebSocket<uWS::SERVER> ws, uWS::HttpRequest req) {
    std::cout << "Connected!!!" << std::endl;
  });

  h.onDisconnection([&h](uWS::WebSocket<uWS::SERVER> ws, int code, char *message, size_t length) {
    ws.close();
    std::cout << "Disconnected" << std::endl;
  });

  int port = 4567;
  if (h.listen(port))
  {
    std::cout << "Listening to port " << port << std::endl;
  }
  else
  {
    std::cerr << "Failed to listen to port" << std::endl;
    return -1;
  }
  h.run();
}
