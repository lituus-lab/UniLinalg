// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 lituus-lab
#include <stdio.h>
#include "UniLinalg.h"

int main(void) {
  ulin_init();
  printf("UniLinalg %s\n", ulin_version());

  ulin_matrix a = ulin_matrix_create(3, 3);
  double avals[9] = {1,2,1, 2,1,3, 1,1,1};
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++)
      ulin_matrix_set(a, i, j, avals[i*3+j]);
  double b[3] = {8.0, 13.0, 6.0};
  double x[3];
  ulin_matrix_lu_solve(a, b, 3, x, 3, false);
  printf("solve(a, b) = [%g, %g, %g]\n", x[0], x[1], x[2]);
  ulin_matrix_destroy(a);

  ulin_vec3 ux = {1.0, 0.0, 0.0};
  ulin_vec3 uy = {0.0, 1.0, 0.0};
  ulin_vec3 uz = ulin_vec3_cross(ux, uy);
  printf("cross(x, y) = [%g, %g, %g]\n", uz.x, uz.y, uz.z);

  ulin_vec2 v = {3.0, 4.0};
  printf("vec2(3, 4).length = %g\n", ulin_vec2_length(v));
  ulin_cleanup();
  return 0;
}
