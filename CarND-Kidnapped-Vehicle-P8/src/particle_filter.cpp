/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /* TODO: Done!
   *       Set the number of particles. Initialize all particles to first position (based on estimates of
   *       x, y, theta and their uncertainties from GPS) and all weights to 1.
   *       Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method (and others in this file).
   */

  // TODO: double check particle number
  num_particles = 500;

  default_random_engine gen;

  // Creates a normal (Gaussian) distribution for x, y and theta.
  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);

  // Particle list initialization
  for(int i=0; i<num_particles; i++){
    Particle particle_init;

    particle_init.id = i;
    particle_init.x = dist_x(gen);
    particle_init.y = dist_y(gen);
    particle_init.theta = dist_theta(gen);
    particle_init.weight = 1.0;

    particles.push_back(particle_init);
  }

  // Weight list initialization
  for(int i=0; i<num_particles; i++){
    weights.push_back(1.0);
  }

  is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
  /* TODO: Done!
   *       Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
   *       http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *       http://www.cplusplus.com/reference/random/default_random_engine/
   */

  default_random_engine gen;

  // Creates a normal (Gaussian) distribution of 0 mean..
  normal_distribution<double> dist_x(0, std_pos[0]);
  normal_distribution<double> dist_y(0, std_pos[1]);
  normal_distribution<double> dist_theta(0, std_pos[2]);;

  for(int i=0; i<num_particles; i++){
    Particle p = particles[i];

    // Compute the predicted delta in position and orientation
    double delta_x = 0.0;
    double delta_y = 0.0;
    double delta_theta = 0.0;
    double yaw = p.theta;

    //avoid division by zero
    if (fabs(yaw_rate) > 0.01) {
      delta_x = velocity/yaw_rate * ( sin(yaw + yaw_rate*delta_t) - sin(yaw) );
      delta_y = velocity/yaw_rate * ( cos(yaw) - cos(yaw + yaw_rate*delta_t) );
    }
    else {
      delta_x = velocity*delta_t*cos(yaw);
      delta_y = velocity*delta_t*sin(yaw);
    }
    // Yaw angle
    delta_theta = yaw_rate*delta_t;

    // Position/orientation update + noise
    p.x = p.x + delta_x + dist_x(gen);
    p.y = p.y + delta_y + dist_y(gen);
    p.theta = p.theta + delta_theta + dist_theta(gen);

    particles[i] = p;
  }

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
  /* TODO: Done!
   *       Find the predicted measurement that is closest to each observed measurement and assign the
   *       observed measurement to this particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
   *       implement this method and use it as a helper during the updateWeights phase.
   */
  double min_dist, act_dist;
  int min_id;

  // for each observation, find the predicted landmark within the minimum distance of the observation
  // then copy the predicted landmark id to the observation id.
  for(int i=0; i<observations.size(); i++){
    min_dist = INFINITY;
    min_id = -1;

    LandmarkObs& obs = observations[i];

    // For each predicted landmark observation
    for(int j=0; j<predicted.size(); j++){
      LandmarkObs pred = predicted[j];

      // Compute the Euclidian distance
      act_dist = dist(pred.x, pred.y, obs.x, obs.y);

      // If this predicted landmark observation is the closest up to date to the observation...
      if (min_dist > act_dist){
        min_dist = act_dist;
        min_id = j;
      }
    }
    // Copy the associated landmark index to the observation
    observations[i].id = min_id;
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   std::vector<LandmarkObs> observations, Map map_landmarks) {
  /* TODO: Done!
   *       Update the weights of each particle using a multi-variate Gaussian distribution. You can read
   *       more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
   *       according to the MAP'S coordinate system. You will need to transform between the two systems.
   *       Keep in mind that this transformation requires both rotation AND translation (but no scaling).
   *       The following is a good resource for the theory:
   *       https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *       and the following is a good resource for the actual equation to implement (look at equation
   *       3.33
   *       http://planning.cs.uiuc.edu/node99.html
   */

  // For readability
  double obs_std_x = std_landmark[0];
  double obs_std_y = std_landmark[1];

  // Precompute terms for Multi-variate Gaussian probability evaluation below
  double c1 = 1/(2*M_PI*obs_std_x*obs_std_y);
  double c2 = 1/(2*obs_std_x*obs_std_x);
  double c3 = 1/(2*obs_std_y*obs_std_y);

  int map_id, obs_id;
  double map_x, obs_x, obs_map_x;
  double map_y, obs_y, obs_map_y;

  // for each particle
  for(int i=0; i<num_particles; i++){
    Particle p = particles[i];

    // create a list of observable landmarks given each particle's position
    std::vector<LandmarkObs> observable_landmarks;

    // for each map landmark, validate if within sensor's range before adding it to
    // the observable landmark list for this particle
    for(int j=0; j<map_landmarks.landmark_list.size(); j++){
      // create a predicted landmark
      LandmarkObs pred;
      pred.id = map_landmarks.landmark_list[j].id_i;
      pred.x  = map_landmarks.landmark_list[j].x_f;
      pred.y  = map_landmarks.landmark_list[j].y_f;

      // Add only if within sensor range
      if(dist(pred.x, pred.y, p.x, p.y) <= sensor_range)
        observable_landmarks.push_back(pred);
    }

    // create a list of transformed observations
    std::vector<LandmarkObs> map_observations;

    // for each observation, transform coordinates from the vehicle frame to
    // the map frame, given current particle's postion and orientation
    for(int j=0; j<observations.size(); j++){
      LandmarkObs obs_map;

      // Observations in vehicle frame
      obs_x  = observations[j].x;
      obs_y  = observations[j].y;

      // Transform observations to the map frame given each particle
      obs_map_x = p.x + obs_x * cos(p.theta) - obs_y * sin(p.theta);
      obs_map_y = p.y + obs_x * sin(p.theta) + obs_y * cos(p.theta);

      // Build the LandmarkObs in the map frame
      obs_map.id = observations[j].id;
      obs_map.x = obs_map_x;
      obs_map.y = obs_map_y;

      // Add to the transformed list
      map_observations.push_back(obs_map);
    }

    // Nearest neighbor associations
    dataAssociation(observable_landmarks, map_observations);

    double gauss_prob;
    double delta_x, delta_y;

    p.weight = 1.0;
    // for each observation, compute the multivariate-gaussian probability
    for(int j=0; j<map_observations.size(); j++){
      LandmarkObs obs_map = map_observations[j];

      delta_x = obs_map.x - observable_landmarks[obs_map.id].x;
      delta_y = obs_map.y - observable_landmarks[obs_map.id].y;
      gauss_prob = c1 * exp(-(c2*delta_x*delta_x+c3*delta_y*delta_y));

      // multiply all the probabilities together to get the particle's weight
      p.weight = p.weight * gauss_prob;
    }
    // Update particle weight in the weight list
    weights[i] = p.weight;
  }
}

void ParticleFilter::resample() {
  /* TODO: Done!
   * Resample particles with replacement with probability proportional to their weight.
   * NOTE: You may find std::discrete_distribution helpful here.
   *       http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
  default_random_engine gen;

  // Create a discrete distribution of indexes based on the weights of the weight list
  std::discrete_distribution<int> d(weights.begin(), weights.end());
  // Create a list of new particles
  std::vector<Particle> new_particles;

  // for the number of particles, draw particles indexes with a probability equal to
  // their weight.. and build a new list.
  for(int i=0; i<num_particles; i++)
  {
    int index = d(gen);
    new_particles.push_back(particles[index]);
  }
  // replace the old particles by the new ones.
  particles = new_particles;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
  // particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates

  //Clear the previous associations
  particle.associations.clear();
  particle.sense_x.clear();
  particle.sense_y.clear();

  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;

  return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
  vector<int> v = best.associations;
  stringstream ss;
  copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}
string ParticleFilter::getSenseX(Particle best)
{
  vector<double> v = best.sense_x;
  stringstream ss;
  copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}
string ParticleFilter::getSenseY(Particle best)
{
  vector<double> v = best.sense_y;
  stringstream ss;
  copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}
