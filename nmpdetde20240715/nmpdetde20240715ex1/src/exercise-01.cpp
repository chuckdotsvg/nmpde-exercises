#include "Heat.hpp"

// Main function.
int main(int argc, char *argv[]) {
  constexpr unsigned int dim = Heat::dim;
  const auto f = [](const Point<dim> &p, const double t) {
    return numbers::PI / 2. * std::sin(numbers::PI / 2. * p[0]) *
               std::cos(numbers::PI / 2. * t) +
           (numbers::PI * numbers::PI / 4. - 1) *
               std::sin(numbers::PI / 2. * p[0]) *
               std::sin(numbers::PI / 2. * t) +
           numbers::PI / 2. * std::cos(numbers::PI / 2. * p[0]) *
               std::sin(numbers::PI / 2. * t);
  };
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);

  Heat problem(2, 1., 0.5, 0.1, 1, 1, 1, 40, f);

  problem.run();

  return 0;
}
