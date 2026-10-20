import { Routes } from '@angular/router';

/** One lazy chunk per feature (ARCHITECTURE §9). */
export const routes: Routes = [
  {
    path: '',
    pathMatch: 'full',
    title: 'Hyperplane',
    loadChildren: () => import('./features/home/home.routes').then((m) => m.HOME_ROUTES),
  },
  {
    path: 'learn',
    title: 'Learn · Hyperplane',
    loadChildren: () => import('./features/catalog/catalog.routes').then((m) => m.CATALOG_ROUTES),
  },
  {
    path: 'learn/algorithms/:algorithmId',
    title: 'Algorithm · Hyperplane',
    loadChildren: () =>
      import('./features/algorithm/algorithm.routes').then((m) => m.ALGORITHM_ROUTES),
  },
  {
    path: 'concepts',
    pathMatch: 'full',
    title: 'Concepts · Hyperplane',
    loadChildren: () =>
      import('./features/concept-index/concept-index.routes').then((m) => m.CONCEPT_INDEX_ROUTES),
  },
  {
    path: 'concepts/:conceptId',
    title: 'Concept · Hyperplane',
    loadChildren: () => import('./features/concept/concept.routes').then((m) => m.CONCEPT_ROUTES),
  },
  {
    path: 'lab',
    title: 'Lab · Hyperplane',
    loadChildren: () => import('./features/lab/lab.routes').then((m) => m.LAB_ROUTES),
  },
  {
    path: 'runs',
    title: 'Runs · Hyperplane',
    loadChildren: () => import('./features/runs/runs.routes').then((m) => m.RUNS_ROUTES),
  },
  {
    path: '**',
    title: 'Not found · Hyperplane',
    loadComponent: () => import('./core/layout/not-found-page').then((m) => m.NotFoundPage),
  },
];
