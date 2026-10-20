import { InjectionToken } from '@angular/core';

/** Opens a server-sent events stream; tests provide a fake. */
export type EventSourceFactory = (url: string) => EventSource;

export const EVENT_SOURCE_FACTORY = new InjectionToken<EventSourceFactory>('EVENT_SOURCE_FACTORY', {
  providedIn: 'root',
  factory: () => (url) => new EventSource(url, { withCredentials: true }),
});
