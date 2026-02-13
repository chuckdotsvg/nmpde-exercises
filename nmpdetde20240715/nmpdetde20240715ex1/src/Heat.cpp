#include "Heat.hpp"
#include <deal.II/base/function.h>

void Heat::setup() {
  pcout << "===============================================" << std::endl;

  Triangulation<dim> mesh_serial;
  // Create the mesh.
  {
    std::cout << "Initializing the mesh" << std::endl;

    GridGenerator::subdivided_hyper_cube(mesh_serial, N_el, 0.0, 1.0, true);
    std::cout << "  Number of elements = " << mesh_serial.n_active_cells()
              << std::endl;

    // Write the mesh to file.
    //
    // Since we generate the mesh internally, we also write it to file for
    // possible inspection by the user. This would not be necessary if we read
    // the mesh from file, as we will do later on.
    std::string mesh_file_name = "mesh-" + std::to_string(N_el) + ".vtk";
    GridOut grid_out;
    std::ofstream grid_out_file(mesh_file_name);
    grid_out.write_vtk(mesh_serial, grid_out_file);
    std::cout << "  Mesh saved to " << mesh_file_name << std::endl;
  }

  pcout << "-----------------------------------------------" << std::endl;

  // Create the mesh.
  {
    // Copy the serial mesh into the parallel one.
    {
      GridTools::partition_triangulation(mpi_size, mesh);

      const auto construction_data = TriangulationDescription::Utilities::
          create_description_from_triangulation(mesh_serial, MPI_COMM_WORLD);
      mesh.create_triangulation(construction_data);
    }

    pcout << "  Number of elements = " << mesh.n_global_active_cells()
          << std::endl;
  }

  pcout << "-----------------------------------------------" << std::endl;

  // Initialize the finite element space.
  {
    pcout << "Initializing the finite element space" << std::endl;

    fe = std::make_unique<FE_SimplexP<dim>>(r);

    pcout << "  Degree                     = " << fe->degree << std::endl;
    pcout << "  DoFs per cell              = " << fe->dofs_per_cell
          << std::endl;

    quadrature = std::make_unique<QGaussSimplex<dim>>(r + 1);

    pcout << "  Quadrature points per cell = " << quadrature->size()
          << std::endl;
  }

  pcout << "-----------------------------------------------" << std::endl;

  // Initialize the DoF handler.
  {
    pcout << "Initializing the DoF handler" << std::endl;

    dof_handler.reinit(mesh);
    dof_handler.distribute_dofs(*fe);

    pcout << "  Number of DoFs = " << dof_handler.n_dofs() << std::endl;
  }

  pcout << "-----------------------------------------------" << std::endl;

  // Initialize the linear system.
  {
    pcout << "Initializing the linear system" << std::endl;

    const IndexSet locally_owned_dofs = dof_handler.locally_owned_dofs();
    const IndexSet locally_relevant_dofs =
        DoFTools::extract_locally_relevant_dofs(dof_handler);

    pcout << "  Initializing the sparsity pattern" << std::endl;
    TrilinosWrappers::SparsityPattern sparsity(locally_owned_dofs,
                                               MPI_COMM_WORLD);
    DoFTools::make_sparsity_pattern(dof_handler, sparsity);
    sparsity.compress();

    pcout << "  Initializing the system matrix" << std::endl;
    system_matrix.reinit(sparsity);

    pcout << "  Initializing vectors" << std::endl;
    system_rhs.reinit(locally_owned_dofs, MPI_COMM_WORLD);
    solution_owned.reinit(locally_owned_dofs, MPI_COMM_WORLD);
    solution.reinit(locally_owned_dofs, locally_relevant_dofs, MPI_COMM_WORLD);
  }
}

void Heat::assemble() {
  // Number of local DoFs for each element.
  const unsigned int dofs_per_cell = fe->dofs_per_cell;

  // Number of quadrature points for each element.
  const unsigned int n_q = quadrature->size();

  FEValues<dim> fe_values(*fe, *quadrature,
                          update_values | update_gradients |
                              update_quadrature_points | update_JxW_values);

  // Local matrix and vector.
  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double> cell_rhs(dofs_per_cell);

  std::vector<types::global_dof_index> dof_indices(dofs_per_cell);

  // Reset the global matrix and vector, just in case.
  system_matrix = 0.0;
  system_rhs = 0.0;

  // Evaluation of the old solution on quadrature nodes of current cell.
  std::vector<double> solution_old_values(n_q);

  // Evaluation of the gradient of the old solution on quadrature nodes of
  // current cell.
  std::vector<Tensor<1, dim>> solution_old_grads(n_q);

  for (const auto &cell : dof_handler.active_cell_iterators()) {
    if (!cell->is_locally_owned())
      continue;

    fe_values.reinit(cell);

    cell_matrix = 0.0;
    cell_rhs = 0.0;

    // Evaluate the old solution and its gradient on quadrature nodes.
    fe_values.get_function_values(solution, solution_old_values);
    fe_values.get_function_gradients(solution, solution_old_grads);

    for (unsigned int q = 0; q < n_q; ++q) {

      double f_old_loc = f(fe_values.quadrature_point(q), time - delta_t);
      double f_new_loc = f(fe_values.quadrature_point(q), time);
      for (unsigned int i = 0; i < dofs_per_cell; ++i) {
        for (unsigned int j = 0; j < dofs_per_cell; ++j) {
          // Time derivative.
          cell_matrix(i, j) += (1.0 / delta_t) *             //
                               fe_values.shape_value(i, q) * //
                               fe_values.shape_value(j, q) * //
                               fe_values.JxW(q);

          cell_matrix(i, j) += theta * eps *                //
                               fe_values.shape_grad(i, q) * //
                               fe_values.shape_grad(j, q) * //
                               fe_values.JxW(q);

          cell_matrix(i, j) +=
              theta * b * //
              fe_values.shape_grad(
                  i, q)[0] * // in practice is a one element vector, but I need
                             // the [0] to get the value out of the tensor
              fe_values.shape_value(j, q) * //
              fe_values.JxW(q);

          cell_matrix(i, j) -= theta * k *                   //
                               fe_values.shape_value(i, q) * //
                               fe_values.shape_value(j, q) * //
                               fe_values.JxW(q);
        }

        // Time derivative.
        cell_rhs(i) += (1.0 / delta_t) *             //
                       fe_values.shape_value(i, q) * //
                       solution_old_values[q] *      //
                       fe_values.JxW(q);

        // Diffusion.
        cell_rhs(i) -= (1.0 - theta) * eps *                      //
                       scalar_product(fe_values.shape_grad(i, q), //
                                      solution_old_grads[q]) *    //
                       fe_values.JxW(q);

        cell_rhs(i) -= (1.0 - theta) * b *             //
                       fe_values.shape_grad(i, q)[0] * //
                       solution_old_values[q] *        //
                       fe_values.JxW(q);

        cell_rhs(i) += (1.0 - theta) * k *           //
                       fe_values.shape_value(i, q) * //
                       solution_old_values[q] *      //
                       fe_values.JxW(q);

        cell_rhs(i) +=
            theta * f_new_loc * fe_values.shape_value(i, q) * fe_values.JxW(q);

        cell_rhs(i) += (1.0 - theta) * f_old_loc * fe_values.shape_value(i, q) *
                       fe_values.JxW(q);

        // No Forcing term.
      }
    }

    cell->get_dof_indices(dof_indices);

    system_matrix.add(dof_indices, cell_matrix);
    system_rhs.add(dof_indices, cell_rhs);
  }

  system_matrix.compress(VectorOperation::add);
  system_rhs.compress(VectorOperation::add);

  // Boundary conditions.
  //
  // So far we assembled the matrix as if there were no Dirichlet conditions.
  // Now we want to replace the rows associated to nodes on which Dirichlet
  // conditions are applied with equations like u_i = b_i. We use deal.ii
  // functions to
  {
    // We construct a map that stores, for each DoF corresponding to a Dirichlet
    // condition, the corresponding value. E.g., if the Dirichlet condition is
    // u_i = b_i, the map will contain the pair (i, b_i).
    std::map<types::global_dof_index, double> boundary_values;

    // This object represents our boundary data as a real-valued function (that
    // always evaluates to zero). Other functions may require to implement a
    // custom class derived from dealii::Function<dim>.
    Functions::ZeroFunction<dim> bc_function;

    // Then, we build a map that, for each boundary tag, stores a pointer to the
    // corresponding boundary function.
    std::map<types::boundary_id, const Function<dim> *> boundary_functions;
    boundary_functions[0] = &bc_function;
    // boundary_functions[1] = &bc_function;

    // interpolate_boundary_values fills the boundary_values map.
    VectorTools::interpolate_boundary_values(dof_handler, boundary_functions,
                                             boundary_values);

    // Finally, we modify the linear system to apply the boundary conditions.
    // This replaces the equations for the boundary DoFs with the corresponding
    // u_i = 0 equations.
    MatrixTools::apply_boundary_values(boundary_values, system_matrix, solution,
                                       system_rhs, true);
  }
}

void Heat::solve_linear_system() {
  TrilinosWrappers::PreconditionSSOR preconditioner;
  preconditioner.initialize(
      system_matrix, TrilinosWrappers::PreconditionSSOR::AdditionalData(1.0));

  ReductionControl solver_control(/* maxiter = */ 10000,
                                  /* tolerance = */ 1.0e-16,
                                  /* reduce = */ 1.0e-6);

  SolverGMRES<TrilinosWrappers::MPI::Vector> solver(solver_control);

  solver.solve(system_matrix, solution_owned, system_rhs, preconditioner);
  pcout << solver_control.last_step() << " CG iterations" << std::endl;
}

void Heat::output() const {
  DataOut<dim> data_out;

  data_out.add_data_vector(dof_handler, solution, "solution");

  // Add vector for parallel partition.
  std::vector<unsigned int> partition_int(mesh.n_active_cells());
  GridTools::get_subdomain_association(mesh, partition_int);
  const Vector<double> partitioning(partition_int.begin(), partition_int.end());
  data_out.add_data_vector(partitioning, "partitioning");

  data_out.build_patches();

  // const std::filesystem::path mesh_path(mesh_file_name);
  const std::string output_file_name = "output-";

  data_out.write_vtu_with_pvtu_record(/* folder = */ "./",
                                      /* basename = */ output_file_name,
                                      /* index = */ timestep_number,
                                      MPI_COMM_WORLD);
}

void Heat::run() {
  // Setup initial conditions.
  {
    setup();

    VectorTools::interpolate(dof_handler, Functions::ZeroFunction<dim>(),
                             solution_owned);
    solution = solution_owned;

    time = 0.0;
    timestep_number = 0;

    // Output initial condition.
    output();
  }

  pcout << "===============================================" << std::endl;

  // Time-stepping loop.
  while (time < T - 0.5 * delta_t) {
    time += delta_t;
    ++timestep_number;

    pcout << "Timestep " << std::setw(3) << timestep_number
          << ", time = " << std::setw(4) << std::fixed << std::setprecision(2)
          << time << " : ";

    assemble();
    solve_linear_system();

    // Perform parallel communication to update the ghost values of the
    // solution vector.
    solution = solution_owned;

    output();
  }
}
