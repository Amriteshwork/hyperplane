/** Least squares for small problems: solve (AᵀA + λI) x = Aᵀy by Gaussian elimination. */
export function leastSquares(A: number[][], y: number[], ridge = 1e-8): number[] {
  const p = A[0].length;
  const M = Array.from({ length: p }, (_, i) =>
    Array.from({ length: p + 1 }, (_, j) => {
      if (j === p) return A.reduce((s, row, k) => s + row[i] * y[k], 0);
      return A.reduce((s, row) => s + row[i] * row[j], 0) + (i === j ? ridge : 0);
    }),
  );
  for (let col = 0; col < p; col++) {
    let pivot = col;
    for (let r = col + 1; r < p; r++) if (Math.abs(M[r][col]) > Math.abs(M[pivot][col])) pivot = r;
    [M[col], M[pivot]] = [M[pivot], M[col]];
    for (let r = 0; r < p; r++) {
      if (r === col) continue;
      const factor = M[r][col] / M[col][col];
      for (let c = col; c <= p; c++) M[r][c] -= factor * M[col][c];
    }
  }
  return M.map((row, i) => row[p] / row[i]);
}
