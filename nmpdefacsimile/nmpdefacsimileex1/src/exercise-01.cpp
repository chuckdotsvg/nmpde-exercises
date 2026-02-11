#include <deal.II/base/convergence_table.h>
#include <iostream>

#include "Poisson2D.hpp"

static constexpr unsigned int dim = Poisson2D::dim;

// Exact solution.
class ExactSolution : public Function<dim> {
public:
  // Constructor.
  ExactSolution() {}

  // Evaluation.
  virtual double value(const Point<dim> &p,
                       const unsigned int /*component*/ = 0) const override {
    return std::sin(M_PI / 2. * p[0]) * p[1];
  }

  // Gradient evaluation.
  virtual Tensor<1, dim>
  gradient(const Point<dim> &p,
           const unsigned int /*component*/ = 0) const override {
    Tensor<1, dim> result;

    result[0] = M_PI / 2. * std::cos(M_PI / 2. * p[0]) * p[1];

    result[1] = std::sin(M_PI / 2. * p[0]); // derivative of p[1] is 1

    return result;
  }

  static constexpr double A = -4.0 / 15.0 * std::pow(0.5, 2.5);
};

// Main function.
int main(int /*argc*/, char * /*argv*/[]) {
  ConvergenceTable table;

  const std::vector<double> h_values = {0.1, 0.05, 0.025, 0.0125};

  constexpr unsigned int dim = Poisson2D::dim;

  // const std::string mesh_file_name = "../mesh/mesh-square-h0.100000.msh";
  const unsigned int r = 2;

  const auto mu = [](const Point<dim> & /*p*/) { return 1.0; };
  const auto sigma = mu;
  const auto f = [](const Point<dim> &p) {
    return (1. + numbers::PI * numbers::PI / 4.) *
           std::sin(numbers::PI / 2.0 * p[0]) * p[1];
  };

  const ExactSolution exact_solution;

  std::ofstream convergence_file("convergence.csv");
  convergence_file << "h,eL2,eH1" << std::endl;

  for (const auto &h : h_values) {
    const std::string mesh_file_name =
        "../mesh/mesh-square-h" + std::to_string(h) + ".msh";

    cout << "reading mesh from file: " << mesh_file_name << std::endl;

    Poisson2D problem(mesh_file_name, r, mu, sigma, f);

    problem.setup();
    problem.assemble();
    problem.solve();
    problem.output();

    const double error_L2 =
        problem.compute_error(VectorTools::L2_norm, exact_solution);
    const double error_H1 =
        problem.compute_error(VectorTools::H1_norm, exact_solution);

    table.add_value("h", h);
    table.add_value("L2", error_L2);
    table.add_value("H1", error_H1);

    convergence_file << h << "," << error_L2 << "," << error_H1 << std::endl;
  }

  table.evaluate_all_convergence_rates(ConvergenceTable::reduction_rate_log2);

  table.set_scientific("L2", true);
  table.set_scientific("H1", true);

  table.write_text(std::cout);

  // Poisson2D problem(mesh_file_name, r, mu, sigma, f);
  //
  // problem.setup();
  // problem.assemble();
  // problem.solve();
  // problem.output();

  return 0;
}
