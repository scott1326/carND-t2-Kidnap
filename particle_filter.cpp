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

// DEBUG *******************************************
#include <fstream>

#include "particle_filter.h"

using namespace std;

#define NUM_PARTICLES 10

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

  // Instantiate random number gens
  default_random_engine gen;
  normal_distribution<double> dist_x(0, std[0]);
  normal_distribution<double> dist_y(0, std[1]);
  normal_distribution<double> dist_theta(0, std[2]);
  
  // Initial guess of number of particles
  num_particles = NUM_PARTICLES;

  for (int i = 0; i < num_particles; i++) {
    Particle new_particle;
    new_particle.id = i;
    new_particle.x = x + dist_x(gen);
    new_particle.y = y + dist_y(gen);
    new_particle.theta = theta + dist_theta(gen);
    new_particle.weight = 1;
    particles.push_back(new_particle);
  }

  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

  default_random_engine gen;

  for (std::vector<Particle>::iterator iter = particles.begin(); iter != particles.end(); iter++) {
 
    // update x,y,theta
    // yaw_rate != zero
    if (fabs(yaw_rate) > .001) {
      (*iter).x += (velocity / yaw_rate) * (sin((*iter).theta + (yaw_rate * delta_t)) - sin((*iter).theta));
      (*iter).y += (velocity / yaw_rate) * (cos((*iter).theta) - cos((*iter).theta + yaw_rate * delta_t));
      (*iter).theta += yaw_rate * delta_t;
    }
    else {
      // yaw_rate == zero
      (*iter).x += velocity * delta_t * cos((*iter).theta);
      (*iter).y += velocity * delta_t * sin((*iter).theta);
      (*iter).theta = (*iter).theta;
    }
    
    // declare noise generators
    normal_distribution<double> dist_x( (*iter).x, std_pos[0]);
    normal_distribution<double> dist_y( (*iter).y, std_pos[1]);
    normal_distribution<double> dist_theta( (*iter).theta, std_pos[2]);

    (*iter).x = dist_x(gen);
    (*iter).y = dist_y(gen);
    (*iter).theta = dist_theta(gen);
  }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

  // for readability
  double lm_std_x = std_landmark[0];
  double lm_std_y = std_landmark[1];
  double lm_var_x = lm_std_x * lm_std_x;
  double lm_var_y = lm_std_y * lm_std_y;

  // must clear weights in ParticleFilter as well
  weights.clear();
  for (std::vector<Particle>::iterator current_particle = particles.begin(); current_particle != particles.end(); current_particle++) {
    // set weight to 1
    (*current_particle).weight = 1;

    // DEBUG ******************************************
    (*current_particle).sense_x.clear();
    (*current_particle).sense_y.clear();
    (*current_particle).associations.clear();

    // get transformed theta
    double trans_theta = (*current_particle).theta;

    // transform each observation to map coords
    for (std::vector<LandmarkObs>::const_iterator obs_iter = observations.begin(); obs_iter != observations.end(); obs_iter++) {
      double x_obs_trans = (*current_particle).x + cos(trans_theta)*((*obs_iter).x) - sin(trans_theta)*((*obs_iter).y);
      double y_obs_trans = (*current_particle).y + sin(trans_theta)*((*obs_iter).x) + cos(trans_theta)*((*obs_iter).y);

      // for this obs, go through landmarks
      LandmarkObs nearest_landmark;
      double current_smallest_distance = sensor_range;
      std::vector<Map::single_landmark_s> const lm_list = map_landmarks.landmark_list;
      for (std::vector<Map::single_landmark_s>::const_iterator map_iter = lm_list.begin(); map_iter != lm_list.end(); map_iter++) {
        // use dist function to get closest landmark
        double lm_dist = dist(x_obs_trans, y_obs_trans, (*map_iter).x_f, (*map_iter).y_f);

        if (lm_dist < current_smallest_distance) {
          current_smallest_distance = lm_dist;
          nearest_landmark.id = (*map_iter).id_i;
          nearest_landmark.x = (*map_iter).x_f;
          nearest_landmark.y = (*map_iter).y_f;
        }
      }

      // DEBUG - can remove for final version
      (*current_particle).sense_x.push_back(x_obs_trans);
      (*current_particle).sense_y.push_back(y_obs_trans);
      (*current_particle).associations.push_back(nearest_landmark.id);

      // calc weight using multivariate-gassian
      double scale_factor = 1 / (2 * M_PI * lm_std_x * lm_std_y);
      double x_mu_2 = (x_obs_trans - nearest_landmark.x) * (x_obs_trans - nearest_landmark.x);
      double y_mu_2 = (y_obs_trans - nearest_landmark.y) * (y_obs_trans - nearest_landmark.y);
      //ofile << "XY diffs squared " << x_mu_2 << " " << y_mu_2 << "\r\n";
      //ofile << "E exp " << -(x_mu_2 / (2 * lm_var_x) + y_mu_2 / (2 * lm_var_y)) << "\r\n";
      double w = scale_factor * exp( -( x_mu_2 / (2 * lm_var_x) + y_mu_2 / (2 * lm_var_y) ) );

      // multiply weights together
      (*current_particle).weight *= w;
    }
 
    // push cumulative weight to weights vector
    // saves having to do it later in resample
    weights.push_back((*current_particle).weight);
  }

}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

  std::random_device rd;
  std::mt19937 gen(rd());
  std::discrete_distribution<> dist(weights.begin(), weights.end());

  std::vector<Particle> sampled_particles;
  for (int i = 0; i < num_particles; i++) {
    Particle new_part = particles[dist(gen)];
    sampled_particles.push_back(new_part);
  }

  particles = sampled_particles;
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
