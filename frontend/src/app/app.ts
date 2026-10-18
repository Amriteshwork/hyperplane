import { ChangeDetectionStrategy, Component } from '@angular/core';
import { RouterOutlet } from '@angular/router';

import { Footer } from './core/layout/footer';
import { Header } from './core/layout/header';

/** Layout shell: header, routed page, footer. */
@Component({
  selector: 'app-root',
  imports: [RouterOutlet, Header, Footer],
  changeDetection: ChangeDetectionStrategy.OnPush,
  templateUrl: './app.html',
  host: { class: 'flex min-h-dvh flex-col' },
})
export class App {}
