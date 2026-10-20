import { schemaDefaults, schemaFields } from './json-schema';
import { validateField } from './validation';

describe('schemaFields', () => {
  it('unwraps Optional, inlines $ref, and humanises missing titles', () => {
    const [field] = schemaFields({
      properties: { learning_rate: { anyOf: [{ $ref: '#/$defs/Rate' }, { type: 'null' }] } },
      $defs: { Rate: { type: 'number', minimum: 0 } },
    });
    expect(field).toMatchObject({
      key: 'learning_rate',
      title: 'Learning rate',
      type: 'number',
      minimum: 0,
      nullable: true,
      level: 'beginner',
    });
  });

  it('reads a one-value Literal (`const`) as an enum, so it renders as a choice', () => {
    const [field] = schemaFields({
      properties: {
        class_weight: { anyOf: [{ const: 'balanced', type: 'string' }, { type: 'null' }] },
      },
    });
    expect(field).toMatchObject({ enum: ['balanced'], nullable: true, type: 'string' });
  });

  it('collects declared defaults only', () => {
    expect(
      schemaDefaults({ properties: { a: { type: 'integer', default: 3 }, b: { type: 'string' } } }),
    ).toEqual({ a: 3 });
  });
});

describe('validateField', () => {
  const [count] = schemaFields({
    properties: { count: { type: 'integer', minimum: 1, maximum: 10 } },
    required: ['count'],
  });

  it.each([
    [undefined, 'This field is required.'],
    [Number.NaN, 'Enter a number.'],
    [1.5, 'Enter a whole number.'],
    [0, 'Must be at least 1.'],
    [11, 'Must be at most 10.'],
    [5, null],
  ])('%s → %s', (value, message) => {
    expect(validateField(count, value)).toBe(message);
  });
});
