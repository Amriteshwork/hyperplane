import { Routes } from '@angular/router';

import { LabPage } from './lab-page';
import { LabStore } from './lab.store';

export const LAB_ROUTES: Routes = [{ path: '', component: LabPage, providers: [LabStore] }];
