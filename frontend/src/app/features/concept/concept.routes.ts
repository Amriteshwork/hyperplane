import { Routes } from '@angular/router';

import { provideMarkdownView } from '../../shared/ui/markdown-view/provide-markdown-view';
import { ConceptPage } from './concept-page';
import { ConceptStore } from './concept.store';

export const CONCEPT_ROUTES: Routes = [
  { path: '', component: ConceptPage, providers: [ConceptStore, provideMarkdownView()] },
];
