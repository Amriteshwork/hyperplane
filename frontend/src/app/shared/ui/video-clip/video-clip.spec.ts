import { TestBed } from '@angular/core/testing';

import { VideoClip, VideoClipData } from './video-clip';

async function render(video: VideoClipData) {
  const fixture = TestBed.createComponent(VideoClip);
  fixture.componentRef.setInput('video', video);
  await fixture.whenStable();
  return { fixture, el: fixture.nativeElement as HTMLElement };
}

describe('VideoClip', () => {
  it('loads nothing from YouTube until the reader presses play', async () => {
    const { fixture, el } = await render({ title: 'Soft lookup', youtubeId: 'dQw4w9WgXcQ' });
    expect(el.querySelector('iframe')).toBeNull();

    el.querySelector('button')?.click();
    await fixture.whenStable();

    const frame = el.querySelector('iframe');
    expect(frame?.getAttribute('src')).toBe(
      'https://www.youtube-nocookie.com/embed/dQw4w9WgXcQ?autoplay=1&rel=0',
    );
    expect(frame?.getAttribute('title')).toBe('Soft lookup');
  });

  it.each([null, 'x"><script>'])('shows "coming soon" without a valid id (%s)', async (id) => {
    const { el } = await render({ title: 'Clocks', youtubeId: id });
    expect(el.querySelector('button')).toBeNull();
    expect(el.textContent).toContain('Animation coming soon');
  });
});
