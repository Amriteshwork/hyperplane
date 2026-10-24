import { DecimalPipe } from '@angular/common';
import { ChangeDetectionStrategy, Component, computed, signal } from '@angular/core';

import { FigureFrame } from './figure-frame';

// x → z = w₁x + b₁ → h = tanh z → ŷ = w₂h + b₂ → L = (ŷ − y)²
const START = { x: 1.5, y: 0.5, w1: 0.8, b1: -0.2, w2: 1.3, b2: 0.1 };
const LEARNING_RATE = 0.1;

type Highlight = 'x' | 'z' | 'h' | 'yhat' | 'loss' | 'w1' | 'w2';

interface Step {
  title: string;
  /** Plain-words explanation, then the arithmetic. */
  text: string;
  maths: string;
  highlight: Highlight[];
  /** Show the backward-pass values (orange) from this step on. */
  backward: boolean;
}

/** Walks one training example through a forward pass, then backpropagates, edge by edge. */
@Component({
  selector: 'app-backprop-steps-figure',
  imports: [DecimalPipe, FigureFrame],
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <app-figure-frame
      heading="Backpropagation, one step at a time"
      hint="Step forward through the network, then backwards: every number is computed live."
    >
      <svg
        viewBox="0 0 360 150"
        class="w-full"
        role="img"
        [attr.aria-label]="step().title + ': ' + step().text"
      >
        @for (edge of edges; track edge.id) {
          <line
            [attr.x1]="edge.x1"
            [attr.x2]="edge.x2"
            y1="60"
            y2="60"
            [class]="lit(edge.id) ? 'stroke-fig-2' : 'stroke-border'"
            [attr.stroke-width]="lit(edge.id) ? 3 : 1.5"
          />
          <text
            [attr.x]="(edge.x1 + edge.x2) / 2"
            y="50"
            text-anchor="middle"
            class="fill-foreground text-[10px]"
          >
            {{ edge.label }}
          </text>
          @if (step().backward && edge.grad !== null) {
            <text
              [attr.x]="(edge.x1 + edge.x2) / 2"
              y="80"
              text-anchor="middle"
              class="fill-fig-2 text-[9px] tabular-nums"
            >
              ∂L/∂{{ edge.label.split(' ')[0] }} = {{ edge.grad() | number: '1.3-3' }}
            </text>
          }
        }
        @for (node of nodes; track node.id) {
          <circle
            [attr.cx]="node.x"
            cy="60"
            r="19"
            [class]="lit(node.id) ? 'fill-fig-1 stroke-fig-1' : 'fill-card stroke-border'"
            [attr.fill-opacity]="lit(node.id) ? 0.25 : 1"
            stroke-width="1.5"
          />
          <text
            [attr.x]="node.x"
            y="64"
            text-anchor="middle"
            class="fill-foreground text-[11px] font-semibold"
          >
            {{ node.label }}
          </text>
          @if (index() >= node.shownFrom) {
            <text
              [attr.x]="node.x"
              y="100"
              text-anchor="middle"
              class="fill-fig-1 text-[10px] tabular-nums"
            >
              {{ node.value() | number: '1.3-3' }}
            </text>
          }
          @if (step().backward && node.grad !== null) {
            <text
              [attr.x]="node.x"
              y="116"
              text-anchor="middle"
              class="fill-fig-2 text-[9px] tabular-nums"
            >
              δ {{ node.grad() | number: '1.3-3' }}
            </text>
          }
        }
        <text x="10" y="140" class="fill-fig-1 text-[9px]">blue: forward values</text>
        <text x="350" y="140" text-anchor="end" class="fill-fig-2 text-[9px]">
          orange: gradients (backward)
        </text>
      </svg>

      <div class="mt-2 rounded-md bg-muted/50 p-3" aria-live="polite">
        <p class="text-xs text-muted-foreground">Step {{ index() + 1 }} of {{ steps().length }}</p>
        <p class="mt-1 text-sm font-semibold">{{ step().title }}</p>
        <p class="mt-1 text-sm">{{ step().text }}</p>
        <p class="mt-2 font-mono text-xs leading-5">{{ step().maths }}</p>
      </div>
      <div class="mt-3 flex flex-wrap gap-2">
        <button
          type="button"
          class="rounded-md border px-3 py-1.5 text-xs hover:bg-muted disabled:opacity-50"
          [disabled]="index() === 0"
          (click)="index.set(index() - 1)"
        >
          ← Back
        </button>
        <button
          type="button"
          class="rounded-md border bg-primary px-3 py-1.5 text-xs text-primary-foreground hover:opacity-90 disabled:opacity-50"
          [disabled]="index() === steps().length - 1"
          (click)="index.set(index() + 1)"
        >
          Next step →
        </button>
        <button
          type="button"
          class="rounded-md px-3 py-1.5 text-xs text-muted-foreground hover:bg-muted"
          (click)="index.set(0)"
        >
          Start over
        </button>
      </div>
      <p caption>
        The forward pass keeps every intermediate value. The backward pass reuses them: each
        gradient is the gradient one step closer to the loss, times one local slope. That is the
        chain rule, applied once per edge, which is why backpropagation costs about as much as one
        more forward pass, however many weights there are.
      </p>
    </app-figure-frame>
  `,
})
export class BackpropStepsFigure {
  protected readonly index = signal(0);

  // Forward pass.
  private readonly z = START.w1 * START.x + START.b1;
  private readonly h = Math.tanh(this.z);
  private readonly yhat = START.w2 * this.h + START.b2;
  private readonly loss = (this.yhat - START.y) ** 2;
  // Backward pass: each δ is ∂L/∂(that value).
  private readonly dYhat = 2 * (this.yhat - START.y);
  private readonly dW2 = this.dYhat * this.h;
  private readonly dH = this.dYhat * START.w2;
  private readonly dZ = this.dH * (1 - this.h ** 2);
  private readonly dW1 = this.dZ * START.x;
  // One gradient-descent step on every weight and bias, then the loss again.
  private readonly next = {
    w1: START.w1 - LEARNING_RATE * this.dW1,
    b1: START.b1 - LEARNING_RATE * this.dZ,
    w2: START.w2 - LEARNING_RATE * this.dW2,
    b2: START.b2 - LEARNING_RATE * this.dYhat,
  };
  private readonly nextLoss =
    (this.next.w2 * Math.tanh(this.next.w1 * START.x + this.next.b1) + this.next.b2 - START.y) ** 2;

  protected readonly nodes = [
    { id: 'x', label: 'x', x: 24, value: () => START.x, grad: null, shownFrom: 0 },
    { id: 'z', label: 'z', x: 108, value: () => this.z, grad: () => this.dZ, shownFrom: 1 },
    { id: 'h', label: 'h', x: 180, value: () => this.h, grad: () => this.dH, shownFrom: 2 },
    {
      id: 'yhat',
      label: 'ŷ',
      x: 260,
      value: () => this.yhat,
      grad: () => this.dYhat,
      shownFrom: 3,
    },
    { id: 'loss', label: 'L', x: 336, value: () => this.loss, grad: null, shownFrom: 4 },
  ];

  protected readonly edges = [
    { id: 'w1', label: 'w₁ = 0.8', x1: 43, x2: 89, grad: () => this.dW1 },
    { id: 'tanh', label: 'tanh', x1: 127, x2: 161, grad: null },
    { id: 'w2', label: 'w₂ = 1.3', x1: 199, x2: 241, grad: () => this.dW2 },
    { id: 'sq', label: '(ŷ − y)²', x1: 279, x2: 317, grad: null },
  ];

  protected readonly steps = computed<Step[]>(() => {
    const f = (v: number) => v.toFixed(3);
    return [
      {
        title: 'One training example',
        text: `The input is x = ${START.x} and the right answer is y = ${START.y}. The network has one hidden neuron with a tanh bend.`,
        maths: `z = w₁·x + b₁   h = tanh(z)   ŷ = w₂·h + b₂   L = (ŷ − y)²`,
        highlight: ['x'],
        backward: false,
      },
      {
        title: 'Forward: the weighted sum',
        text: 'The hidden neuron multiplies the input by its weight and adds its bias.',
        maths: `z = ${START.w1} × ${START.x} + (${START.b1}) = ${f(this.z)}`,
        highlight: ['w1', 'z'],
        backward: false,
      },
      {
        title: 'Forward: the bend',
        text: 'The activation squashes z. Keep h: the backward pass needs it.',
        maths: `h = tanh(${f(this.z)}) = ${f(this.h)}`,
        highlight: ['z', 'h'],
        backward: false,
      },
      {
        title: 'Forward: the prediction',
        text: 'The output neuron is linear: a weighted sum of the hidden neuron.',
        maths: `ŷ = ${START.w2} × ${f(this.h)} + ${START.b2} = ${f(this.yhat)}`,
        highlight: ['w2', 'yhat'],
        backward: false,
      },
      {
        title: 'The loss',
        text: `The prediction is ${f(this.yhat - START.y)} too high. The squared error measures how bad that is.`,
        maths: `L = (${f(this.yhat)} − ${START.y})² = ${f(this.loss)}`,
        highlight: ['yhat', 'loss'],
        backward: false,
      },
      {
        title: 'Backward: start at the output',
        text: 'How much does L change if ŷ changes a little? That is the first gradient.',
        maths: `∂L/∂ŷ = 2(ŷ − y) = ${f(this.dYhat)}`,
        highlight: ['loss', 'yhat'],
        backward: true,
      },
      {
        title: 'Backward: the output weight',
        text: 'ŷ = w₂·h, so nudging w₂ moves ŷ by h per unit. Multiply by the gradient so far.',
        maths: `∂L/∂w₂ = ∂L/∂ŷ × h = ${f(this.dYhat)} × ${f(this.h)} = ${f(this.dW2)}`,
        highlight: ['w2', 'yhat'],
        backward: true,
      },
      {
        title: 'Backward: through the weight to h',
        text: 'The error flows back along the same edge it came forward on, scaled by w₂.',
        maths: `∂L/∂h = ∂L/∂ŷ × w₂ = ${f(this.dYhat)} × ${START.w2} = ${f(this.dH)}`,
        highlight: ['h', 'w2'],
        backward: true,
      },
      {
        title: 'Backward: through the bend',
        text: 'Multiply by the activation’s slope at this point: tanh′(z) = 1 − h².',
        maths: `∂L/∂z = ∂L/∂h × (1 − h²) = ${f(this.dH)} × ${f(1 - this.h ** 2)} = ${f(this.dZ)}`,
        highlight: ['h', 'z'],
        backward: true,
      },
      {
        title: 'Backward: the first weight',
        text: 'z = w₁·x, so the gradient of w₁ is the error at z times its input.',
        maths: `∂L/∂w₁ = ∂L/∂z × x = ${f(this.dZ)} × ${START.x} = ${f(this.dW1)}`,
        highlight: ['w1', 'z'],
        backward: true,
      },
      {
        title: 'Update every weight',
        text: `Step each weight and bias against its gradient (learning rate ${LEARNING_RATE}), and the loss drops from ${f(this.loss)} to ${f(this.nextLoss)}. Training repeats this for every batch, thousands of times.`,
        maths:
          `w₁: ${START.w1} → ${f(this.next.w1)}   w₂: ${START.w2} → ${f(this.next.w2)}   ` +
          `b₁: ${START.b1} → ${f(this.next.b1)}   b₂: ${START.b2} → ${f(this.next.b2)}`,
        highlight: ['w1', 'w2', 'loss'],
        backward: true,
      },
    ];
  });

  protected readonly step = computed(() => this.steps()[this.index()]);

  protected lit(id: string): boolean {
    return (this.step().highlight as string[]).includes(id);
  }
}
