#include <chrono>
#include <cstdlib>
#include <iostream>

//
[[nodiscard]] auto polynomial(const float x, const float *const poly,
                              const int degree) noexcept -> float {
  auto out = 0.0F;
  auto x_to_the_power_of = 1.0F;

  for (auto i = 0; i <= degree; ++i) {
    out += x_to_the_power_of * poly[i];
    x_to_the_power_of *= x;
  }

  return out;
}

auto polynomial_expansion(const float *const poly, const int degree,
                          const int n, float *const array) noexcept -> void {
#pragma omp parallel for schedule(dynamic, 1024)
  for (auto i = 0; i < n; ++i) {
    array[i] = polynomial(array[i], poly, degree);
  }
}

auto main(const int argc, const char *const *const argv) noexcept -> int {
  if (argc < 3) {
    std::cerr << "usage: " << argv[0] << " n degree\n";
    // return EXIT_FAILURE;
    return -1;
  }

  // TODO: atoi is an unsafe function
  const auto n = std::atoi(argv[1]);
  const auto degree = std::atoi(argv[2]);
  const auto num_iter = 1;

  auto *const array = new float[n];
  auto *const poly = new float[degree + 1];
  for (auto i = 0; i < n; ++i) {
    array[i] = 1;
  }

  for (auto i = 0; i < degree + 1; ++i) {
    poly[i] = 1;
  }

  const auto begin = std::chrono::system_clock::now();

  for (auto iter = 0; iter < num_iter; ++iter) {
    polynomial_expansion(poly, degree, n, array);
  }

  const auto end = std::chrono::system_clock::now();
  const auto total_time = std::chrono::duration<double>{(end - begin) / num_iter};

  std::cerr << array[0] << '\n';
  std::cout << n << " " << degree << " " << total_time.count() << '\n';

  // clean up
  delete[] array;
  delete[] poly;

  return EXIT_SUCCESS;
}
