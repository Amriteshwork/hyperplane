/** The number a range or number input now holds, from its `input` event. */
export function sliderValue(event: Event): number {
  return Number((event.target as HTMLInputElement).value);
}
