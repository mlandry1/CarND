## PID Control
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

In this project I revisit the lake race track from the Behavioral Cloning Project (P3). This time, however, a PID controller is implemented in C++ to maneuver the vehicle around the track!

The simulator will provides the cross track error (CTE) and the velocity (mph) in order to compute the appropriate steering angle.

More information is only accessible by people who are already enrolled in Term 2 of CarND. If you are enrolled, see [the project page](https://classroom.udacity.com/nanodegrees/nd013/parts/40f38239-66b6-46ec-ae68-03afd8a601c8/modules/f1820894-8322-4bb3-81aa-b26b3c6dcbaf/lessons/e8235395-22dd-4b87-88e0-d108c5e5bbf4/concepts/6a4d8d42-6a04-4aa6-b284-1697c0fd6562) for more details.

---
The Project
---

**PID Control Project**

The goals / steps of this project were the following:

1. Build a PID controller class.
2. Tune the PID hyperparameters by applying the general processing flow as described in the lessons.
3. Test the solution on the simulator!

## Rubric Points
 See the [rubric points](https://review.udacity.com/#!/rubrics/824/view) for this project.
 
### 1. Describe the effect each of the P, I, D components had in your implementation.
 
The P, or "proportional", component takes into account only the present value of the cross-track error (*cte*). The P component contributes to the control output  by `Kp * cte`.  The P component steers the car in the right direction but tends to overshoot every time thus resulting in a constant oscillation.
 
The D, or "differential", component. The D component contributes to the control output  by `Kd * d/dt * cte`. This component is thus proportional to the "speed of the error". For example, when the car is approaching the lane's center, `cte` is decreasing while still being positive.  `d/dt*cte` is on its part negative.. It is thus counteracting the action of the P component and helping with the controler's reaction damping.

The I, or "integral", component The I component contributes to the control output by `Ki * sum(cte)`. Thus it takes into account the cumulative error wich can come from several sources such as a steering bias. In the case however, the steering biais was not evident but the I component was contrbuting to eliminate the steady-state errors, in straight lines or in the curves.

While the PID controller is easy to implement, it is not so easy to tune...

### 2. Describe how the final hyperparameters were chosen.

#### Manual tuning
First, I went for a manual tuning. I used a method similar to the one described [here](https://en.wikipedia.org/wiki/PID_controller#Manual_tuning). 

##### Kp tuning
At first, I created two pid controler. The first for speed regulation, the second one for steering control. I then setuped both controller with small values of Kp at first.

##### Speed regulator final tuning
I then focused only on the speed regulator. I increased the Kp value until I reached a state of "constant oscillation". The value of the gain at this step is called the "*utlimate gain*". I then tuned Kp to half of this *ultimate gain*. The next step was to tune the Ki component of my speed regulator. I increased Ki until I reached a stable speed without any steady state error. I then added a bit of Kd to improve response time.

##### Steering regulator
Now that the speed regulator is functional, let's dive into the steering regulator's tuning. 

Starting with a small Kp value and a small speed value (~30mph) , I then increased the Kd carefuly to dampen the oscillation. I found that further relaxing of the Kp parameter helped to make room for a stronger Kd, yeilding better performance...

| Parameter | Value |
|:-------------:|:------:|
| Kp | 0.1   |
| Ki  | 0.0   |
| Kd | 3.2   |
| speed set-point | 30mph| 

[![PD_SpeedRegulated30MPH](./images/PD_SpeedRegulated30MPH.gif)](https://youtu.be/q3FgZYWJB0s)

I then increased the Ki gain just a little bit to try to compensate for any steering bias the car could have had. I found little improvement. I played arround with the other gains and I ended up with the following solution:

| Parameter | Value |
|:-------------:|:------:|
| Kp | 0.095   |
| Ki  | 0.0079 |
| Kd | 3.3       |
| speed set-point | 30mph| 

[![PID_SpeedRegulated30MPH](./images/PID_SpeedRegulated30MPH.gif)](https://www.youtube.com/watch?v=-gZUnDhlYOk)

#### Auto tuning with "Twiddle" algorithm
Then I switched to auto tuning using the Twiddle algorithm. I had a bit of trouble finding a reliable way to compare each of my "run" "total errors".

I ended up reseting the simulator after a fixed number of step. I used the following lines of code to do so:

```sh
    if (first_it == true || pid_s.step % STEP_NUM_RUN == 0){
      // reset the simulator on first execution, at the of each run
      std::string msg("42[\"reset\", {}]");
      ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
      first_it == false;
    }
```
Finaly I sadly found no significant improvement over my manually tuned gains. This was very disapointing since I spent a lot of time on the twiddle implementation in my code.

### 3. Additional stuff, I have done to improve the speed/error arround the track.

#### Throttle control (racing mode)

To increase the speed arround the track, I first went for just increasing the speed regulator' setpoint. Up until 45-50mph, the car gets by and drives arround the track. However I found that higher speeds can't be sustained into all the curves of the track. 

To reach higher speeds, I had to design a logic to slow down the car into the curves. At first I implmented something that acted directly on the Speed setpoint but it porved to slow to react. Then I stumbled onto a fellow CarND student's blog post that explained the strategy she used to control the throttle in her PID project [(Mithi's blog post)](https://medium.com/@mithi/a-review-of-udacitys-self-driving-car-engineer-nanodegree-second-term-56147f1d01ef).

Below is the code I used, wich is exactly the same as Mithi's except for values. The idea is to completely brake the car, if ever the steering angle becomes greater thant 5deg, while the speed is greater than 45mph and the cross-track error is greater than 0.4m. Otherwise it is pedal to the metal!

```sh
throttle_value = 1.0;
if (fabs(cte) > 0.35 && fabs(angle)> 5.5 && speed > 50){
  throttle_value = -1.0;
}
```
Here is the video along with the values used for the controller:

| Parameter | Value |
|:-------------:|:------:|
| Kp | 0.095   |
| Ki  | 0.0079 |
| Kd | 3.3       |
| speed set-point | RACING MODE| 

[![PID_Racing](./images/PID_Racing.gif)](https://youtu.be/L5V6FT0Frfo)

#### Non-linear error feed

In an effort to decrease the oscillation while driving in straight lines, I chose to use a power function of the cross-track error instead of using it directly to feed my steering PID. The resulting PID proved to be smoother in straight line, but harder to tune. Below is the code I used...

```sh
steer_value = pid_s.Update(-fabs(cte)/cte * pow(fabs(cte),1.2));
```

#### Anti-Wind Up

The wind-up of the integrator is commonly problematic when using a PID in the real life. That's why I also implemented the code below. This prevent the integrator to wind up by preventing incrementation of *i_error* if the controller's output magnitude is greater than *max_output* or *min_output*. The value of *i_error* can also be individualy limited.

```sh
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
```
 
## Dependencies

* cmake >= 3.5
 * All OSes: [click here for installation instructions](https://cmake.org/install/)
* make >= 4.1
  * Linux: make is installed by default on most Linux distros
  * Mac: [install Xcode command line tools to get make](https://developer.apple.com/xcode/features/)
  * Windows: [Click here for installation instructions](http://gnuwin32.sourceforge.net/packages/make.htm)
* gcc/g++ >= 5.4
  * Linux: gcc / g++ is installed by default on most Linux distros
  * Mac: same deal as make - [install Xcode command line tools]((https://developer.apple.com/xcode/features/)
  * Windows: recommend using [MinGW](http://www.mingw.org/)
* [uWebSockets](https://github.com/uWebSockets/uWebSockets)
  * Run either `./install-mac.sh` or `./install-ubuntu.sh`.
  * If you install from source, checkout to commit `e94b6e1`, i.e.
    ```
    git clone https://github.com/uWebSockets/uWebSockets 
    cd uWebSockets
    git checkout e94b6e1
    ```
    Some function signatures have changed in v0.14.x. See [this PR](https://github.com/udacity/CarND-MPC-Project/pull/3) for more details.
* Simulator. You can download these from the [project intro page](https://github.com/udacity/self-driving-car-sim/releases) in the classroom.

There's an experimental patch for windows in this [PR](https://github.com/udacity/CarND-PID-Control-Project/pull/3)

## Basic Build Instructions

1. Clone this repo.
2. Make a build directory: `mkdir build && cd build`
3. Compile: `cmake .. && make`
4. Run it: `./pid`. 
