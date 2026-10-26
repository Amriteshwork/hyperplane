import { TestBed } from '@angular/core/testing';
import { provideRouter, withComponentInputBinding } from '@angular/router';
import { RouterTestingHarness } from '@angular/router/testing';

import { TrackOut } from '../../core/api/models/track-out';
import { provideApiTesting, respond, settle } from '../../testing/api-testing';
import { TRACK_ROUTES } from './track.routes';

const concept = (id: string, title: string) => ({
  id,
  title,
  summary: `${title}.`,
  algorithms: [],
});

const track: TrackOut = {
  id: 'transformers',
  title: 'Transformers, from scratch',
  summary: 'NumPy to GPT.',
  body: 'Every idea, **in order**.',
  goal: { id: 'transformer', title: 'Transformer' },
  chapters: [
    {
      title: 'The maths you need',
      summary: 'Arrays first.',
      concepts: [concept('numpy-basics', 'NumPy basics'), concept('softmax', 'Softmax')],
    },
    {
      title: 'The transformer',
      summary: 'Snap it together.',
      concepts: [concept('transformer', 'Transformer')],
    },
  ],
  resources: [],
};

async function open(status = 200) {
  TestBed.configureTestingModule({
    providers: [
      provideRouter(
        [{ path: 'tracks/:trackId', children: TRACK_ROUTES }],
        withComponentInputBinding(),
      ),
      provideApiTesting(),
    ],
  });
  const harness = await RouterTestingHarness.create();
  const navigation = harness.navigateByUrl('/tracks/transformers');
  await respond('/api/v1/tracks/transformers', status === 200 ? track : { title: 'nope' }, {
    status,
    statusText: status === 200 ? 'OK' : 'Not Found',
  });
  await navigation;
  await settle();
  return harness.routeNativeElement as HTMLElement;
}

describe('TrackPage', () => {
  it('shows the intro, a start button and every chapter, numbered across the track', async () => {
    const el = await open();
    const text = el.textContent?.replace(/\s+/g, ' ');

    expect(el.querySelector('h1')?.textContent).toContain('Transformers, from scratch');
    expect(text).toContain('2 chapters · 3 concepts · ends at Transformer');
    expect(el.querySelector('strong')?.textContent).toBe('in order');
    const start = [...el.querySelectorAll('a')].find((a) => a.textContent?.includes('Start'));
    expect(start?.getAttribute('href')).toBe('/concepts/numpy-basics');

    const chapters = el.querySelectorAll('section[aria-labelledby="route"] > ol > li');
    expect([...chapters].map((c) => c.querySelector('h3')?.textContent)).toEqual([
      'The maths you need',
      'The transformer',
    ]);
    const steps = [...el.querySelectorAll('section[aria-labelledby="route"] li li a')];
    expect(steps.map((a) => a.getAttribute('href'))).toEqual([
      '/concepts/numpy-basics',
      '/concepts/softmax',
      '/concepts/transformer',
    ]);
    expect(steps[2].textContent?.replace(/\s+/g, ' ').trim()).toMatch(/^3 Transformer/);
  });

  it('says so when the track does not exist', async () => {
    const el = await open(404);
    expect(el.textContent).toContain('Track not found');
  });
});
