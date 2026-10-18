import { TestBed } from '@angular/core/testing';
import { provideRouter } from '@angular/router';

import { provideMarkdownView } from '../markdown-view/provide-markdown-view';
import { CodeAnnotation, CodeExplainer } from './code-explainer';

const code = [
  'import pandas as pd',
  'from sklearn.linear_model import LinearRegression',
  '',
  'df = pd.read_csv("data.csv")',
  'X, y = df.drop(columns="target"), df["target"]',
  'model = LinearRegression().fit(X, y)',
].join('\n');

const annotations: CodeAnnotation[] = [
  {
    id: 'fit',
    startLine: 6,
    endLine: 6,
    short: 'Fit the model.',
    deep: 'Solves the **normal equation**.',
    concepts: [{ id: 'normal-equation', title: 'Normal equation' }],
  },
  {
    id: 'load',
    startLine: 4,
    endLine: 5,
    short: 'Load the data and split off the target.',
    deep: 'Reads the CSV into a DataFrame.',
    concepts: [],
  },
];

async function setup() {
  TestBed.configureTestingModule({ providers: [provideRouter([]), provideMarkdownView()] });
  const fixture = TestBed.createComponent(CodeExplainer);
  fixture.componentRef.setInput('code', code);
  fixture.componentRef.setInput('annotations', annotations);
  fixture.componentRef.setInput('filename', 'train.py');
  await fixture.whenStable();
  const el: HTMLElement = fixture.nativeElement;
  const block = (id: string) => el.querySelector<HTMLButtonElement>(`[data-anchor="${id}"]`);
  const panel = () => el.querySelector('aside') as HTMLElement;
  const click = async (target: HTMLElement | null) => {
    target?.click();
    await fixture.whenStable();
  };
  return { fixture, el, block, panel, click };
}

describe('CodeExplainer', () => {
  it('renders every line and turns each annotated span into one selectable block', async () => {
    const { el, block } = await setup();
    expect(el.querySelectorAll('.line').length).toBe(6);
    expect(block('load')?.querySelectorAll('.line').length).toBe(2);
    expect(block('load')?.textContent).toContain('pd.read_csv');
    expect(block('fit')?.getAttribute('aria-pressed')).toBe('false');
    expect(el.textContent).toContain('train.py');
  });

  it('shows the short explanation and concept chips for the clicked block', async () => {
    const { block, panel, click } = await setup();
    expect(panel().textContent).toContain('Select a highlighted block');

    await click(block('fit'));
    expect(block('fit')?.getAttribute('aria-pressed')).toBe('true');
    expect(panel().textContent).toContain('Line 6');
    expect(panel().textContent).toContain('Fit the model.');
    const chip = panel().querySelector<HTMLAnchorElement>('a[href="/concepts/normal-equation"]');
    expect(chip?.textContent?.trim()).toBe('Normal equation');
  });

  it('switches to the deep explanation and keeps that depth across blocks', async () => {
    const { block, panel, click } = await setup();
    await click(block('load'));
    const deep = [...panel().querySelectorAll('button')].find(
      (b) => b.textContent?.trim() === 'deep',
    );
    await click(deep ?? null);

    expect(deep?.getAttribute('aria-pressed')).toBe('true');
    expect(panel().textContent).toContain('Reads the CSV');
    await click(block('fit'));
    expect(panel().querySelector('strong')?.textContent).toBe('normal equation');
  });

  it('walks annotations in line order with previous/next', async () => {
    const { panel, click, block } = await setup();
    const button = (label: string) =>
      [...panel().querySelectorAll('button')].find((b) => b.textContent?.includes(label)) ?? null;

    await click(button('Start from the top'));
    expect(block('load')?.getAttribute('aria-pressed')).toBe('true');
    expect(button('Previous')?.disabled).toBe(true);

    await click(button('Next'));
    expect(block('fit')?.getAttribute('aria-pressed')).toBe('true');
    expect(button('Next')?.disabled).toBe(true);
  });

  it('copies the exact code to the clipboard', async () => {
    const writeText = vi.fn().mockResolvedValue(undefined);
    Object.defineProperty(navigator, 'clipboard', { value: { writeText }, configurable: true });
    const { el, fixture } = await setup();
    const copy = [...el.querySelectorAll('button')].find((b) => b.textContent?.includes('Copy'));
    copy?.click();
    await fixture.whenStable();

    expect(writeText).toHaveBeenCalledWith(code);
    expect(copy?.textContent).toContain('Copied');
  });
});
