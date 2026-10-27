import { TestBed } from '@angular/core/testing';

import { NetworkSummary } from '../../core/api/models/network-summary';
import { NetworkDiagram } from './network-diagram';

const network: NetworkSummary = {
  layers: [
    { kind: 'input', units: 2, activation: null, shown: ['numeric__bmi', 'numeric__age'] },
    { kind: 'hidden', units: 100, activation: 'relu', shown: ['neuron 1', 'neuron 2'] },
    { kind: 'output', units: 1, activation: 'identity', shown: ['prediction'] },
  ],
  parameters: 401,
  weights: [
    [
      [0.5, -1],
      [0.25, 0],
    ],
    [[2], [-0.5]],
  ],
};

function render(): HTMLElement {
  const fixture = TestBed.createComponent(NetworkDiagram);
  fixture.componentRef.setInput('network', network);
  fixture.detectChanges();
  return fixture.nativeElement as HTMLElement;
}

describe('NetworkDiagram', () => {
  it('draws every shown unit and one line per weight', () => {
    const element = render();
    expect(element.querySelectorAll('circle')).toHaveLength(5);
    expect(element.querySelectorAll('line')).toHaveLength(6);
  });

  it('colours lines by sign and scales them within each pair of layers', () => {
    const lines = [...render().querySelectorAll('line')];
    expect(lines[1].getAttribute('class')).toContain('stroke-fig-negative');
    expect(lines[0].getAttribute('class')).toContain('stroke-fig-1');
    // -1 is the largest input → hidden weight; 2 the largest hidden → output one.
    expect(lines[1].getAttribute('stroke-width')).toBe('3');
    expect(lines[4].getAttribute('stroke-width')).toBe('3');
    expect(lines[0].querySelector('title')?.textContent).toBe('bmi → neuron 1: +0.500');
  });

  it('labels inputs without their prefix and counts the units not drawn', () => {
    const element = render();
    const text = element.textContent ?? '';
    expect(text).toContain('bmi');
    expect(text).not.toContain('numeric__');
    expect(text).toContain('+98 more');
    expect(text).toContain('100 · relu');
    expect(element.querySelector('svg')?.getAttribute('aria-label')).toContain(
      '401 weights and biases',
    );
  });
});
