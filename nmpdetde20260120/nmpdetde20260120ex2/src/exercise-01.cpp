#include "NavierStokes.hpp"

// Main function.
int main(int argc, char *argv[]) {
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);

  const std::string mesh_file_name = "../mesh/mesh-pipe.msh";
  const unsigned int degree_velocity = 2;
  const unsigned int degree_pressure = 1;

  const unsigned int mu = 1;
  const double timestep = 0.025;
  const double T = 1.0;
  const double theta = 1;

  NavierStokes problem(mesh_file_name, degree_velocity, degree_pressure, mu,
                       timestep, T, theta);

  problem.run();

  return 0;
}
