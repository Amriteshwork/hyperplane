import { ChangeDetectionStrategy, Component, inject } from '@angular/core';
import { RouterLink, RouterLinkActive } from '@angular/router';
import { NgIcon, provideIcons } from '@ng-icons/core';
import { lucideMenu, lucideMoon, lucideSun } from '@ng-icons/lucide';
import { HlmButton } from '@spartan-ng/helm/button';
import { HlmSheetImports } from '@spartan-ng/helm/sheet';

import { ThemeService } from '../theme/theme.service';
import { NAV_ITEMS } from './nav-items';

@Component({
  selector: 'app-header',
  imports: [RouterLink, RouterLinkActive, NgIcon, HlmButton, HlmSheetImports],
  providers: [provideIcons({ lucideMenu, lucideMoon, lucideSun })],
  changeDetection: ChangeDetectionStrategy.OnPush,
  templateUrl: './header.html',
})
export class Header {
  protected readonly theme = inject(ThemeService);
  protected readonly navItems = NAV_ITEMS;
}
