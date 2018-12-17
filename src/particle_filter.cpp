/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>
#include <vector>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {

  num_particles = 128;

  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);

  for(unsigned int i=0; i<num_particles; i++){
    Particle particle;
    particle.x = dist_x(r_engine);
    particle.y = dist_y(r_engine);
    particle.theta = dist_theta(r_engine);
    particle.weight = 1;
    particles.push_back(particle);
  }

  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {

  double term_0;
  double term_1;
  if (yaw_rate==0) {
    term_0 = 0;
    term_1 = velocity * delta_t;
  }else {
    term_0 = velocity/yaw_rate;
    term_1 = 0;
  }
  double delta_yaw = yaw_rate * delta_t;

  for(unsigned int i=0; i<num_particles; i++) {
    Particle p = particles[i];
    double new_theta = p.theta + delta_yaw;

    double sin_theta = sin(p.theta);
    double sin_new_theta = sin(new_theta);
    double cos_theta = cos(p.theta);
    double cos_new_theta = cos(new_theta);

    double new_x = p.x + term_0 * (sin_new_theta - sin_theta) + term_1 * cos_theta;
    double new_y = p.y + term_0 * (cos_theta - cos_new_theta) + term_1 * sin_theta;

    normal_distribution<double> dist_x(new_x, std_pos[0]);
    normal_distribution<double> dist_y(new_y, std_pos[1]);
    normal_distribution<double> dist_theta(new_theta, std_pos[2]);

    particles[i].x = dist_x(r_engine);
    particles[i].y = dist_y(r_engine);
    particles[i].theta = dist_theta(r_engine);
    particles[i].weight = 1;

  }


}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
  // TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
  //   observed measurement to this particular landmark.
  // NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
  //   implement this method and use it as a helper during the updateWeights phase.
  //
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
    const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {

  //Accumulate total weight for normalization
  double total_weight = 0;

  for(unsigned int i=0; i<num_particles; i++) {
    Particle p = particles[i];
    const float cosTheta = cos(p.theta);
    const float sinTheta = sin(p.theta);
    vector<LandmarkObs>::const_iterator it = observations.begin();

    vector<LandmarkObs> predicted;

    for(auto obs:observations){
      LandmarkObs map;
      map.x = p.x + (cosTheta * obs.x) - (sinTheta * obs.y);
      map.y = p.y + (sinTheta * obs.x) + (cosTheta * obs.y);
      predicted.push_back(map);

      Map::single_landmark_s nearest = find_nearest(map.x, map.y, sensor_range, map_landmarks);

      // Calc particle weight
      p.weight *= calc_observation_weight(map.x, map.y, nearest.x_f, nearest.y_f, std_landmark);
    }

    particles[i] = p;
    total_weight += p.weight;
  }

  //Normalize
  for(unsigned int i=0; i<num_particles; i++) {
    Particle particle = particles[i];
    particle.weight = particle.weight/ total_weight;
    particles[i] = particle;
  }
}

Map::single_landmark_s ParticleFilter::find_nearest(double x, double y, double sensor_range, const Map &map_landmarks){
  double min_dist=sensor_range+1;
  Map::single_landmark_s nearest;

  for(auto lm: map_landmarks.landmark_list){
    double delta_dist = dist(x, y, lm.x_f, lm.y_f);
    if(delta_dist > sensor_range){
      continue;
    }
    if(delta_dist < min_dist) {
      min_dist = delta_dist;
      nearest = lm;
    }
  }
  return nearest;

}

double ParticleFilter::calc_observation_weight(double x, double y, double min_x, double min_y, double std[]){
  double xdist_sq = (x - min_x) * (x - min_x);
  double ydist_sq = (y - min_y) * (y - min_y);
  double sigma_x = std[0];
  double sigma_y = std[1];
  double sigmax_sq = 2 * sigma_x * sigma_x;
  double sigmay_sq = 2 * sigma_y * sigma_y;
  double power = -1 * (xdist_sq/sigmax_sq + ydist_sq/sigmay_sq);
  double normalizer = 1/(2*M_PI * sigma_x * sigma_y);
  double obs_weight = normalizer * exp(power);
  return obs_weight;
}


void ParticleFilter::resample() {

  std::vector<double> probs;


  for(unsigned int i=0; i<num_particles; i++) {
    Particle p = particles[i];
    probs.push_back(p.weight);
  }

  std::discrete_distribution<> d(probs.begin(), probs.end());

  std::vector<Particle> resampled;
  for(unsigned int i=0; i<num_particles; i++) {
    Particle p = particles[d(r_engine)];
    resampled.push_back(p);
  }
  particles = resampled;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
    const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
  //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates

  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
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
