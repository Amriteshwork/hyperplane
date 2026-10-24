import { Type } from '@angular/core';
import { ComponentFixture, TestBed } from '@angular/core/testing';

import { BackpropStepsFigure } from './backprop-steps-figure';
import { BoundaryPlaygroundFigure } from './boundary-playground-figure';
import { EarlyStoppingFigure } from './early-stopping-figure';
import { DEFAULT_FIGURES } from './figure-registry';
import { NetworkShapeFigure } from './network-shape-figure';
import { ReluSumFigure } from './relu-sum-figure';
import { XorNetworkFigure } from './xor-network-figure';

async function mount<T>(type: Type<T>): Promise<[ComponentFixture<T>, HTMLElement]> {
  const fixture = TestBed.createComponent(type);
  await fixture.whenStable();
  return [fixture, fixture.nativeElement];
}

function button(el: HTMLElement, label: string): HTMLButtonElement {
  const found = [...el.querySelectorAll('button')].find((b) => b.textContent?.includes(label));
  if (!found) throw new Error(`no button "${label}"`);
  return found;
}

function slide(el: HTMLElement, label: string, value: number): void {
  const input = [...el.querySelectorAll('label')]
    .find((l) => l.textContent?.includes(label))
    ?.querySelector('input');
  if (!input) throw new Error(`no slider "${label}"`);
  input.value = String(value);
  input.dispatchEvent(new Event('input'));
}

/** Animation frames run only when the test says so. */
function manualFrames(): () => void {
  // Angular's zoneless scheduler also uses frames, so cancel by id, not wholesale.
  let queue = new Map<number, FrameRequestCallback>();
  let id = 0;
  vi.spyOn(globalThis, 'requestAnimationFrame').mockImplementation((cb) => {
    queue.set(++id, cb);
    return id;
  });
  vi.spyOn(globalThis, 'cancelAnimationFrame').mockImplementation((handle) => queue.delete(handle));
  return () => {
    const now = queue;
    queue = new Map();
    now.forEach((cb) => cb(0));
  };
}

describe('figures', () => {
  afterEach(() => vi.restoreAllMocks());

  it('registers a loader for every figure that resolves to a component', async () => {
    for (const load of Object.values(DEFAULT_FIGURES))
      expect(typeof (await load())).toBe('function');
  });

  it('XOR network outputs 1 only for mixed inputs', async () => {
    const [fixture, el] = await mount(XorNetworkFigure);
    const output = () => el.querySelector('svg')?.getAttribute('aria-label') ?? '';
    for (const [corner, want] of [
      ['(0, 1)', '1.00'],
      ['(1, 1)', '0.00'],
      ['(0, 0)', '0.00'],
    ]) {
      button(el, corner).click();
      await fixture.whenStable();
      expect(output()).toContain(`XOR = ${want}`);
    }
  });

  it('network shape counts weights and biases per layer', async () => {
    const [fixture, el] = await mount(NetworkShapeFigure);
    slide(el, 'Input features', 10);
    slide(el, 'Neurons per hidden layer', 100);
    await fixture.whenStable();
    expect(el.textContent).toMatch(/Total\s*1,201 weights and biases/);
  });

  it('more ReLU neurons fit the curve better', async () => {
    const [fixture, el] = await mount(ReluSumFigure);
    const error = () =>
      Number(
        /error (\d+\.\d+)/.exec(el.querySelector('svg')?.getAttribute('aria-label') ?? '')?.[1],
      );
    slide(el, 'Hidden ReLU neurons', 1);
    await fixture.whenStable();
    const one = error();
    slide(el, 'Hidden ReLU neurons', 12);
    await fixture.whenStable();
    expect(error()).toBeLessThan(one / 10);
  });

  it('backprop stepper walks to an update that lowers the loss', async () => {
    const [fixture, el] = await mount(BackpropStepsFigure);
    const next = button(el, 'Next step');
    while (!next.disabled) {
      next.click();
      await fixture.whenStable();
    }
    const text = el.querySelector('[aria-live]')?.textContent ?? '';
    const [before, after] =
      /drops from (\d+\.\d+) to (\d+\.\d+)/.exec(text)?.slice(1).map(Number) ?? [];
    expect(after).toBeLessThan(before);
    button(el, 'Start over').click();
    await fixture.whenStable();
    expect(el.textContent).toContain('Step 1 of 11');
  });

  it('boundary playground trains when asked and resets on a new dataset', async () => {
    const frame = manualFrames();
    const [fixture, el] = await mount(BoundaryPlaygroundFigure);
    const epoch = () => el.querySelector('dl dd')?.textContent?.trim();
    button(el, 'Train').click();
    for (let i = 0; i < 4; i++) frame();
    await fixture.whenStable();
    expect(epoch()).toBe('20');
    expect(button(el, 'Pause')).toBeTruthy();
    button(el, 'XOR').click();
    await fixture.whenStable();
    expect(epoch()).toBe('0');
  });

  it('early stopping trains live and can jump to the best epoch', async () => {
    const frame = manualFrames();
    const [fixture, el] = await mount(EarlyStoppingFigure);
    for (let i = 0; i < 100; i++) frame();
    await fixture.whenStable();
    expect(el.textContent).not.toContain('Training…');
    const best = /lowest \(([\d.]+)\) at epoch (\d+)/.exec(
      el.querySelectorAll('svg')[1].getAttribute('aria-label') ?? '',
    );
    expect(Number(best?.[2])).toBeLessThan(1000); // validation loss turned back up: it overfits
    button(el, 'Jump to the best epoch').click();
    await fixture.whenStable();
    expect(el.textContent).toMatch(
      new RegExp(`fit at epoch ${Math.round(Number(best?.[2]) / 5) * 5}\\b`),
    );
  });
});
