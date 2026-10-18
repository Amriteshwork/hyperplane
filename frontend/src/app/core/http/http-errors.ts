import { HttpErrorResponse } from '@angular/common/http';

export function isNotFound(error: unknown): boolean {
  return error instanceof HttpErrorResponse && error.status === 404;
}

/** The `detail` of an RFC 9457 problem response, or a generic fallback. */
export function errorMessage(error: unknown): string {
  if (error instanceof HttpErrorResponse) {
    const body: unknown = error.error;
    if (typeof body === 'object' && body !== null && 'detail' in body) {
      const detail = (body as { detail: unknown }).detail;
      if (typeof detail === 'string' && detail) return detail;
    }
    if (error.status === 0) return 'Can’t reach the server. Check your connection and try again.';
  }
  return 'Something went wrong. Please try again.';
}
