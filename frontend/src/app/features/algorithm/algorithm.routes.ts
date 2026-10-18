import { Routes } from '@angular/router';

import { provideMarkdownView } from '../../shared/ui/markdown-view/provide-markdown-view';
import { AlgorithmShell } from './algorithm-shell';
import { AlgorithmStore } from './algorithm.store';
import { CodeTab } from './code-tab';
import { ConceptsTab } from './concepts-tab';
import { OverviewTab } from './overview-tab';
import { ResourcesTab } from './resources-tab';

export const ALGORITHM_ROUTES: Routes = [
  {
    path: '',
    component: AlgorithmShell,
    providers: [AlgorithmStore, provideMarkdownView()],
    children: [
      { path: '', pathMatch: 'full', redirectTo: 'overview' },
      { path: 'overview', component: OverviewTab },
      { path: 'concepts', component: ConceptsTab },
      { path: 'resources', component: ResourcesTab },
      { path: 'code', pathMatch: 'full', redirectTo: 'code/train' },
      { path: 'code/:variant', component: CodeTab },
    ],
  },
];
