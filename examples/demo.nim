# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 lituus-lab
import UniLinalg

echo "UniLinalg " & UniLinalgVersion

# Solve x + 2y + z = 8; 2x + y + 3z = 13; x + y + z = 6 -> (1, 2, 3).
let a = matrix[float64](3, 3, [1.0, 2.0, 1.0,
                               2.0, 1.0, 3.0,
                               1.0, 1.0, 1.0])
let x = solve(a, [8.0, 13.0, 6.0])
echo "solve(a, b) = ", x

# Vector3 cross product of the X and Y axes -> Z axis.
let cr = cross(vec3(1.0, 0.0, 0.0), vec3(0.0, 1.0, 0.0))
echo "cross(x, y) = ", cr

# Vector2 length via UniMath.sqrtNewtonGeneric.
let v = vec2(3.0, 4.0)
echo "vec2(3, 4).length = ", v.length
