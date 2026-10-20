/** Hands a blob to the browser as a file download. */
export function saveFile(document: Document, blob: Blob, filename: string): void {
  const view = document.defaultView;
  if (!view) return;
  const url = view.URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  link.click();
  view.URL.revokeObjectURL(url);
}
