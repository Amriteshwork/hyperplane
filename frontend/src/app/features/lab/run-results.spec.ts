import { featureLabel } from './run-results';

describe('featureLabel', () => {
  it('drops the preprocessing prefix', () => {
    expect(featureLabel('numeric__bmi')).toBe('bmi');
    expect(featureLabel('categorical__island_Dream')).toBe('island_Dream');
  });

  it('keeps powers and joins the factors of a product term', () => {
    expect(featureLabel('numeric__age^2')).toBe('age^2');
    expect(featureLabel('numeric__age numeric__bmi')).toBe('age × bmi');
    expect(featureLabel('numeric__a^2 categorical__sex_M')).toBe('a^2 × sex_M');
  });

  it('keeps spaces inside a column name', () => {
    expect(featureLabel('numeric__body mass')).toBe('body mass');
    expect(featureLabel('numeric__body mass numeric__flipper length')).toBe(
      'body mass × flipper length',
    );
  });

  it('leaves names without a prefix unchanged', () => {
    expect(featureLabel('x0 x1')).toBe('x0 x1');
  });
});
