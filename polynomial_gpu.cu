#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>

__global__ auto polynomial_expansion_kernel(const float *const poly,
                                            const int degree, const int n,
                                            float *const array) noexcept
    -> void {
  const auto idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx >= n) {
    return;
  }

  const auto x = array[idx];
  auto result = 0.0F;
  auto x_to_the_power_of = 1.0F;

  for (auto i = 0; i <= degree; ++i) {
    result += x_to_the_power_of * poly[i];
    x_to_the_power_of *= x;
  }

  array[idx] = result;
}

auto main(const int argc, const char *const *const argv) noexcept -> int {
  if (argc < 3) {
    std::cerr << "usage: " << argv[0] << " n degree\n";
    // return EXIT_FAILURE;
    return -1;
  }

  // atoi is an unsafe function
  const auto n = std::atoi(argv[1]);
  const auto degree = std::atoi(argv[2]);
  const auto num_iter = 1;

  // auto *const array = new float[n];
  // auto *const poly = new float[degree + 1];
  float *array;
  float *poly;
  cudaMallocManaged(&array, n * sizeof(float));
  cudaMallocManaged(&poly, (degree + 1) * sizeof(float));

  for (auto i = 0; i < n; ++i) {
    array[i] = 1;
  }

  for (auto i = 0; i < degree + 1; ++i) {
    poly[i] = 1;
  }

  const auto begin = std::chrono::system_clock::now();

  // for (auto iter = 0; iter < nbiter; ++iter) {
  //   polynomial_expansion(poly, degree, n, array);
  // }

  // launch GPU kernel
  const auto block_size = 256;
  const auto num_blocks = (n + block_size - 1) / block_size;
  polynomial_expansion_kernel<<<num_blocks, block_size>>>(poly, degree, n,
                                                          array);
  // wait for GPU to finish
  cudaDeviceSynchronize();

  const auto end = std::chrono::system_clock::now();
  const auto total_time =
      std::chrono::duration<double>{(end - begin) / num_iter};

  // check results
  auto correct = true;
  int ind;
  for (auto i = 0; i < n; ++i) {
    if (std::fabs(array[i] - (degree + 1)) > 0.01) {
      correct = false;
      ind = i;
    }
  }

  if (!correct) {
    std::cerr << "Result is incorrect. In particular array[" << ind
              << "] should be " << degree + 1 << " not " << array[ind] << '\n';
  }

  std::cerr << array[0] << '\n';
  std::cout << n << " " << degree << " " << total_time.count() << '\n';

  // clean up
  // delete[] array;
  // delete[] poly;
  cudaFree(array);
  cudaFree(poly);

  return EXIT_SUCCESS;
}
