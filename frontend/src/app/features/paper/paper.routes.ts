import { Routes } from '@angular/router';

import { provideMarkdownView } from '../../shared/ui/markdown-view/provide-markdown-view';
import { PaperPage } from './paper-page';
import { PaperStore } from './paper.store';

export const PAPER_ROUTES: Routes = [
  { path: '', component: PaperPage, providers: [PaperStore, provideMarkdownView()] },
];
