/**
 * `numeric__age` → `age`, and a product term `numeric__age numeric__bmi` → `age × bmi`: drop each
 * preprocessing prefix. Only a space before another `step__` prefix separates factors, since column
 * names may contain spaces.
 */
export function featureLabel(name: string): string {
  return name
    .split(/ (?=[a-z]+__)/)
    .map((factor) => {
      const split = factor.indexOf('__');
      return split === -1 ? factor : factor.slice(split + 2);
    })
    .join(' × ');
}
