#include <immintrin.h>
#include <math.h>
#include <stdalign.h>
#include <stdint.h>
#include <stdio.h>

typedef intptr_t intnative_t;

typedef struct {
  double position[3], velocity[3], mass;
} body;

#define SOLAR_MASS (4 * M_PI * M_PI)
#define DAYS_PER_YEAR 365.24
#define BODIES_COUNT 5

static body solar_Bodies[] = {
    {// Sun
     .mass = SOLAR_MASS},
    {// Jupiter
     {4.84143144246472090e+00, -1.16032004402742839e+00,
      -1.03622044471123109e-01},
     {1.66007664274403694e-03 * DAYS_PER_YEAR,
      7.69901118419740425e-03 * DAYS_PER_YEAR,
      -6.90460016972063023e-05 * DAYS_PER_YEAR},
     9.54791938424326609e-04 * SOLAR_MASS},
    {// Saturn
     {8.34336671824457987e+00, 4.12479856412430479e+00,
      -4.03523417114321381e-01},
     {-2.76742510726862411e-03 * DAYS_PER_YEAR,
      4.99852801234917238e-03 * DAYS_PER_YEAR,
      2.30417297573763929e-05 * DAYS_PER_YEAR},
     2.85885980666130812e-04 * SOLAR_MASS},
    {// Uranus
     {1.28943695621391310e+01, -1.51111514016986312e+01,
      -2.23307578892655734e-01},
     {2.96460137564761618e-03 * DAYS_PER_YEAR,
      2.37847173959480950e-03 * DAYS_PER_YEAR,
      -2.96589568540237556e-05 * DAYS_PER_YEAR},
     4.36624404335156298e-05 * SOLAR_MASS},
    {// Neptune
     {1.53796971148509165e+01, -2.59193146099879641e+01,
      1.79258772950371181e-01},
     {2.68067772490389322e-03 * DAYS_PER_YEAR,
      1.62824170038242295e-03 * DAYS_PER_YEAR,
      -9.51592254519715870e-05 * DAYS_PER_YEAR},
     5.15138902046611451e-05 * SOLAR_MASS}};

static void offset_Momentum(body bodies[]) {
  for (intnative_t i = 0; i < BODIES_COUNT; ++i)
    for (intnative_t m = 0; m < 3; ++m)
      bodies[0].velocity[m] -=
          bodies[i].velocity[m] * bodies[i].mass / SOLAR_MASS;
}

static void output_Energy(body bodies[]) {
  double energy = 0;
  for (intnative_t i = 0; i < BODIES_COUNT; ++i) {
    energy += 0.5 * bodies[i].mass *
              (bodies[i].velocity[0] * bodies[i].velocity[0] +
               bodies[i].velocity[1] * bodies[i].velocity[1] +
               bodies[i].velocity[2] * bodies[i].velocity[2]);

    for (intnative_t j = i + 1; j < BODIES_COUNT; ++j) {
      double position_Delta[3];

      for (intnative_t m = 0; m < 3; ++m)
        position_Delta[m] = bodies[i].position[m] - bodies[j].position[m];

      energy -= bodies[i].mass * bodies[j].mass /
                sqrt(position_Delta[0] * position_Delta[0] +
                     position_Delta[1] * position_Delta[1] +
                     position_Delta[2] * position_Delta[2]);
    }
  }

  printf("%.9f\n", energy);
}

static void advance(body bodies[]) {
#define INTERACTIONS_COUNT (BODIES_COUNT * (BODIES_COUNT - 1) / 2)
#define ROUNDED_INTERACTIONS_COUNT (INTERACTIONS_COUNT + INTERACTIONS_COUNT % 2)

  static alignas(__m128d) double position_Deltas[3][ROUNDED_INTERACTIONS_COUNT],
      magnitudes[ROUNDED_INTERACTIONS_COUNT];

  for (intnative_t i = 0, k = 0; i < BODIES_COUNT - 1; ++i)
    for (intnative_t j = i + 1; j < BODIES_COUNT; ++j, ++k)
      for (intnative_t m = 0; m < 3; ++m)
        position_Deltas[m][k] = bodies[i].position[m] - bodies[j].position[m];

  for (intnative_t i = 0; i < ROUNDED_INTERACTIONS_COUNT / 2; ++i) {
    __m128d position_Delta[3];

    for (intnative_t m = 0; m < 3; ++m)
      position_Delta[m] = ((__m128d *)position_Deltas[m])[i];

    const __m128d distance_Squared = position_Delta[0] * position_Delta[0] +
                                     position_Delta[1] * position_Delta[1] +
                                     position_Delta[2] * position_Delta[2];

    __m128d distance_Reciprocal =
        _mm_cvtps_pd(_mm_rsqrt_ps(_mm_cvtpd_ps(distance_Squared)));

    for (intnative_t j = 0; j < 2; ++j)
      distance_Reciprocal = distance_Reciprocal * 1.5 -
                            0.5 * distance_Squared * distance_Reciprocal *
                                (distance_Reciprocal * distance_Reciprocal);

    ((__m128d *)magnitudes)[i] = 0.01 / distance_Squared * distance_Reciprocal;
  }

  for (intnative_t i = 0, k = 0; i < BODIES_COUNT - 1; i++)
    for (intnative_t j = i + 1; j < BODIES_COUNT; ++j, ++k) {
      const double i_mass_magnitude = bodies[i].mass * magnitudes[k],
                   j_mass_magnitude = bodies[j].mass * magnitudes[k];

      for (intnative_t m = 0; m < 3; ++m) {
        bodies[i].velocity[m] -= position_Deltas[m][k] * j_mass_magnitude;
        bodies[j].velocity[m] += position_Deltas[m][k] * i_mass_magnitude;
      }
    }

  for (intnative_t i = 0; i < BODIES_COUNT; ++i)
    for (intnative_t m = 0; m < 3; ++m)
      bodies[i].position[m] += 0.01 * bodies[i].velocity[m];
}

int main(int argc, char *argv[]) {
  offset_Momentum(solar_Bodies);
  output_Energy(solar_Bodies);

  for (intnative_t n = atoi(argv[1]); n--; advance(solar_Bodies))
    ;

  output_Energy(solar_Bodies);
}