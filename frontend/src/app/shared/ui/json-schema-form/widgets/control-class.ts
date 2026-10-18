/**
 * Spartan's input look, applied to plain controls: `hlmInput` owns `aria-describedby`
 * through its field directive, which would drop our help/error links.
 */
export const CONTROL_CLASS =
  'h-9 w-full min-w-0 rounded-md border border-input bg-transparent px-2.5 py-1 text-base shadow-xs outline-none transition-[color,box-shadow] placeholder:text-muted-foreground focus-visible:border-ring focus-visible:ring-3 focus-visible:ring-ring/50 aria-invalid:border-destructive aria-invalid:ring-3 aria-invalid:ring-destructive/20 md:text-sm dark:bg-input/30';
