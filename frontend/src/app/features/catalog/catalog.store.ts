import { Injectable, computed, inject, resource } from '@angular/core';

import { Api } from '../../core/api/api';
import { listFamilies } from '../../core/api/fn/learn/list-families';
import { errorMessage } from '../../core/http/http-errors';

/** Families with their algorithms, available and upcoming. Provided by the catalog route. */
@Injectable()
export class CatalogStore {
  private readonly api = inject(Api);

  private readonly page = resource({ loader: () => this.api.invoke(listFamilies, { limit: 100 }) });

  readonly families = computed(() => (this.page.hasValue() ? this.page.value().items : []));
  readonly loading = this.page.isLoading;
  readonly error = computed(() => (this.page.error() ? errorMessage(this.page.error()) : null));

  reload(): void {
    this.page.reload();
  }
}
