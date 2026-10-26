import { Routes } from '@angular/router';

import { provideMarkdownView } from '../../shared/ui/markdown-view/provide-markdown-view';
import { TrackPage } from './track-page';
import { TrackStore } from './track.store';

export const TRACK_ROUTES: Routes = [
  { path: '', component: TrackPage, providers: [TrackStore, provideMarkdownView()] },
];
