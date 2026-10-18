import { TestBed } from '@angular/core/testing';

import { JsonSchema, SchemaValues } from './json-schema';
import { JsonSchemaForm } from './json-schema-form';

/** Shaped like Pydantic v2 output for a hyperparameter model. */
const schema: JsonSchema = {
  title: 'Params',
  type: 'object',
  properties: {
    fit_intercept: {
      title: 'Fit Intercept',
      type: 'boolean',
      default: true,
      description: 'Learn a bias term.',
      level: 'beginner',
    },
    alpha: {
      title: 'Alpha',
      type: 'number',
      default: 1,
      exclusiveMinimum: 0,
      description: 'Regularisation strength.',
      level: 'beginner',
    },
    solver: { $ref: '#/$defs/Solver', default: 'auto', level: 'advanced' },
    max_iter: {
      title: 'Max Iter',
      anyOf: [{ type: 'integer', minimum: 1 }, { type: 'null' }],
      default: null,
      level: 'advanced',
    },
  },
  $defs: { Solver: { title: 'Solver', type: 'string', enum: ['auto', 'svd', 'lsqr'] } },
};

async function setup(value: SchemaValues = {}) {
  const fixture = TestBed.createComponent(JsonSchemaForm);
  fixture.componentRef.setInput('schema', schema);
  fixture.componentRef.setInput('value', value);
  const emitted: SchemaValues[] = [];
  const validity: boolean[] = [];
  fixture.componentInstance.value.subscribe((v) => emitted.push(v));
  fixture.componentInstance.validChange.subscribe((v) => validity.push(v));
  await fixture.whenStable();
  return { fixture, el: fixture.nativeElement as HTMLElement, emitted, validity };
}

function field(el: HTMLElement, key: string): HTMLElement {
  const node = el.querySelector<HTMLElement>(`[data-field="${key}"]`);
  if (!node) throw new Error(`field ${key} not rendered`);
  return node;
}

describe('JsonSchemaForm', () => {
  it('renders beginner fields first and tucks advanced ones into a collapsed section', async () => {
    const { el } = await setup();
    const trigger = el.querySelector<HTMLButtonElement>('[data-slot="collapsible-trigger"]');
    const advanced = el.querySelector('[data-slot="collapsible-content"]');

    const order = [...el.querySelectorAll('[data-field]')].map((n) => n.getAttribute('data-field'));
    expect(order.slice(0, 2)).toEqual(['fit_intercept', 'alpha']);
    expect(trigger?.textContent).toContain('Advanced (2)');
    expect(trigger?.getAttribute('aria-expanded')).toBe('false');
    expect(advanced?.contains(field(el, 'solver'))).toBe(true);
  });

  it('labels each control and links its description as help text', async () => {
    const { el } = await setup();
    const alpha = field(el, 'alpha');
    const input = alpha.querySelector('input') as HTMLInputElement;
    const label = alpha.querySelector('label') as HTMLLabelElement;

    expect(label.textContent).toContain('Alpha');
    expect(label.htmlFor).toBe(input.id);
    expect(input.value).toBe('1');
    const help = alpha.querySelector(`#${input.getAttribute('aria-describedby')}`);
    expect(help?.textContent).toContain('Regularisation strength.');
  });

  it('resolves $ref enums to a select and Optional[int] to a nullable number input', async () => {
    const { el } = await setup();
    const select = field(el, 'solver').querySelector('select') as HTMLSelectElement;
    const maxIter = field(el, 'max_iter').querySelector('input') as HTMLInputElement;

    expect([...select.options].map((o) => o.text)).toEqual(['auto', 'svd', 'lsqr']);
    expect(select.selectedOptions[0].text).toBe('auto');
    expect(maxIter.type).toBe('number');
    expect(maxIter.placeholder).toBe('None');
  });

  it('emits the full value object, defaults included, when a field changes', async () => {
    const { fixture, el, emitted } = await setup();
    const select = field(el, 'solver').querySelector('select') as HTMLSelectElement;
    select.value = '1';
    select.dispatchEvent(new Event('change'));
    await fixture.whenStable();

    expect(emitted.at(-1)).toEqual({
      fit_intercept: true,
      alpha: 1,
      solver: 'svd',
      max_iter: null,
    });
  });

  it('shows a validation message and reports invalid when a bound is broken', async () => {
    const { fixture, el, validity } = await setup();
    const input = field(el, 'alpha').querySelector('input') as HTMLInputElement;
    input.value = '0';
    input.dispatchEvent(new Event('input'));
    await fixture.whenStable();

    const error = field(el, 'alpha').querySelector('[role="alert"]');
    expect(error?.textContent).toContain('Must be greater than 0.');
    expect(input.getAttribute('aria-invalid')).toBe('true');
    expect(input.getAttribute('aria-describedby')).toContain(error?.id);
    expect(validity).toEqual([true, false]);
  });

  it('rejects a non-integer for an integer field but accepts clearing an optional one', async () => {
    const { fixture, el } = await setup({ max_iter: 2.5 });
    expect(field(el, 'max_iter').textContent).toContain('Enter a whole number.');

    const input = field(el, 'max_iter').querySelector('input') as HTMLInputElement;
    input.value = '';
    input.dispatchEvent(new Event('input'));
    await fixture.whenStable();
    expect(field(el, 'max_iter').querySelector('[role="alert"]')).toBeNull();
  });

  it('toggles a boolean through its switch', async () => {
    const { fixture, el, emitted } = await setup();
    const toggle = field(el, 'fit_intercept').querySelector<HTMLElement>('[role="switch"]');
    toggle?.click();
    await fixture.whenStable();
    expect(emitted.at(-1)?.['fit_intercept']).toBe(false);
  });
});
