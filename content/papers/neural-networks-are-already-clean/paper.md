---
title: Neural Networks Are Already Clean, explained
summary: A 2026 essay arguing that a network's own neurons are readable building blocks, and that the famous "messy" neurons were misread. Section by section, in plain words, with the key ideas animated.
order: 2
source: goodfire-already-clean
concepts: [vectors, gradient-descent, neurons, activation-functions, backpropagation, self-attention, residual-connections, layer-normalization, transformer]
track: transformers
videos:
  v-not-cat-car:
    title: "Cats and cars, or one V?"
    youtube: uBSXP3hEYlU
  recursive-generalization:
    title: A neuron that grows like a branch
    youtube: UuW-7TKuIp8
  generalize-or-pack:
    title: "Squeezed: coarser neurons, not packed ones"
    youtube: nhpoWmSy3ho
  swiglu-sign-regimes:
    title: "Two ways to fire: SwiGLU's double negative"
    youtube: Iepq5dhMqXA
---

> **Where this comes from.** This walkthrough is derived from the essay
> [**Neural Networks Are Already Clean**](https://already-clean.goodfire.com/) by **Nick
> Cammarata** (Goodfire, 2026), with contributions from Curt Tigges and Tarun Gogineni. Every
> finding, number and example below is theirs; this page only reorders, simplifies and explains
> it for readers new to interpretability. Quotes are taken from the essay's
> [text version](https://already-clean.goodfire.com/essay.md). **All figures and animations are
> captured from the original essay**, with credit under each; click one to see it full size, or
> follow its caption link to the interactive version (most let you hover, click and replay).

**Mechanistic interpretability** is the attempt to understand a trained neural network the way
a biologist understands a cell: open it up, find the parts, and work out what each part does
and how they connect. The obvious parts to study are the network's own **neurons**: each one
takes a weighted sum of its inputs and bends it with an activation function.

For about five years the field believed that plan was broken. Many neurons seemed
**polysemantic**: one neuron firing for cat faces *and* car fronts, or for citations, HTTP
requests *and* Korean text. The leading explanation, **superposition**, said that a network
short on neurons packs unrelated things into the same neuron because they rarely collide. If
that's true, a neuron's mixture tells you nothing, and you must first "untangle" the network
with a second model (a sparse autoencoder) before studying anything.

This essay, written by one of the people who first proposed superposition, argues the
opposite: the famous messy neurons were **misread**, by weak tools and by looking for the
wrong kind of thing. Neurons mostly group **related** things, and what remains is a branching
pattern the author calls **recursive generalization**. So we can study networks directly.
**Neural networks are already clean.**

![A city of connected neighbourhoods beside a knot woven from the same buildings](/paper-media/neural-networks-are-already-clean/city-knot.webp)

*The essay's opening image: a network as a city, whose districts make sense, versus a knot that has to be untangled first. Figure: [Cammarata, Neural Networks Are Already Clean](https://already-clean.goodfire.com/#city-knot) (Goodfire, 2026).*

**How to read this page.** Each section follows one part of the essay, in order. Each starts
with **what the essay says** (a short quote), explains it **in plain words**, sometimes adds a
small calculation or an animation, and says **why it matters**. Links at the end of each
section go to this site's concept pages, which build the ideas from scratch. The essay is an
informal *working hypothesis*, not a peer-reviewed paper; the author puts their own confidence
at about 85%.

## 1 · The short version: two explanations, one bet

> **The essay says:** *"What looked like arbitrary packing was mostly a motif I'll call
> recursive generalization, which is much less concerning. This means we can study networks
> directly, instead of first untangling them."*

**In plain words.** There are two ways to do interpretability:

- **Study the network itself.** Look at its neurons and circuits, watch data flow through
  them, form guesses, and test them. Ordinary science.
- **Untangle first.** Assume the neurons are scrambled, train a second model (a *decomposition*
  such as a sparse autoencoder) to rebuild the network as a long list of cleaner "features",
  and study that list instead.

The field started with the first and switched to the second because neurons looked like
tangled knots. The essay's main claims:

1. **The evidence was misread.** The canonical "polysemantic" neurons in image models turn out
   to have one sensible job once you look properly.
2. **The worst layer was a fixable learning failure.** The last layer of InceptionV1, the 2014
   image model the field first studied, looked almost entirely uninterpretable. The same-sized
   network built with modern parts (residual connections, LayerNorm) has a last layer where
   **99–100%** of neurons are easily readable.
3. **Language models are far cleaner than measured.** In 2023, Anthropic's
   one-layer model scored a median **0 out of 14** on neuron interpretability, and OpenAI's
   automated project found about **1.7%** of GPT-2's neurons interpretable. With better tools,
   preliminary audits of **Qwen3-32B** find about **90%** of neuron "poles" with one main job,
   **~7%** with two related jobs, and only **2–4%** not yet understood.
4. **What's left is recursive generalization,** not superposition, and that's good news.

The essay ends with a four-step argument, and it's worth having in mind from the start:

1. Networks use **one** learning principle (generalization), not two, so nothing blocks
   ordinary science on them.
2. With modern models and tools, that science works much better than most people expect.
3. Understanding **compounds**: like a crossword, every answer constrains the others, so it
   gets easier as the map fills in.
4. *(Highly speculative)* This might be most of what interpretability needs.

**Why it matters.** If neurons are knots, interpretability needs a clever untangling step that
may never be exact. If they're clean, the bottleneck is just doing a lot of careful science,
which automated research agents can now help with at scale.

**Go deeper:** [Neurons and layers](/concepts/neurons) ·
[Activation functions](/concepts/activation-functions)

## 2 · Cars and animals: the neuron that was a V

> **The essay says:** *"I spent a few seconds wondering what one thing this might be. Almost
> instantly it was obvious: it's not a cat and car neuron, it's a V-shape neuron."*

**In plain words.** The most famous polysemantic neuron is **4e:55** in InceptionV1 (layer 4e,
neuron 55). Its strongest images are cat faces and the fronts of cars. Papers from 2017 onward,
including ones the author co-wrote, called it a genuine "cat OR car" mixture.

Asking "what single thing do these share?" gives the answer: a **V shape**. A cat's face
narrows to a V at the chin; a car's front narrows to a V at the grille. And the tests agree:

- Its **circuit connections** include V-shaped flames, V-shaped water jets from a fireboat, and
  V-shaped fox faces. Why would a cat-and-car mixture connect strongly to a flame?
- Its **negative pole**, the inputs that push it most negative, is an **O**: round where the V
  is pointed.
- New predictions hold: other pointed things like bullet-train noses and pool-table racks
  drive it too.

![Neuron 4e:55 with its V-shaped inputs and readers, and photos sorted into pointed and round](/paper-media/neural-networks-are-already-clean/v-figure.webp)

*Neuron 4e:55. Top: its strongest inputs and readers (fox snouts, wolf faces, flames, coyote faces) and renders of its pointed and round sides. Bottom: pointed things drive it and round ones push it negative, whatever the label. Figure: [Cammarata, Neural Networks Are Already Clean](https://already-clean.goodfire.com/#v-figure) (Goodfire, 2026).*

::video[v-not-cat-car]

**Negative poles.** A neuron computes a weighted sum of its inputs, $z = \sum_i w_i x_i + b$,
before its activation function. Each weight is a **vote**: a positive weight votes for the
neuron's positive side, a negative weight for its negative side. Even in a ReLU network, where
everything below zero is cut off, the negative votes are shaped by learning, because each
weight gets a gradient whenever the neuron is on. So *what pushes a neuron down* is as
informative as what pushes it up. It tells you what distinction the neuron draws: V versus O.

The second canonical example was a circuit from the 2020 essay *Zoom In*, where a **car**
neuron seemed to spread its signal into a set of unrelated **dog** neurons, as if storing the
car in a "high-dimensional hiding place" to read back later. That was the original picture of
superposition. It also falls apart:

- **Cars and dogs were grouped by shared pose**: the way they face the camera. InceptionV1 has
  over a hundred such **generalized pose neurons**, each a clean detector of one pose shared
  across cars, dogs and other objects. Nothing downstream reads the car back out of the dogs;
  later layers keep splitting it into car specialists and finer poses.
- **The pictures were misleading.** The icons of what each neuron "wants" come from **feature
  visualization**: start from noise and run gradient ascent on the *input image* to make the
  neuron fire as much as possible. Late in InceptionV1, that optimization takes the easiest
  route, which often means dogs (ImageNet has a lot of dog photos and dog neurons). Nudged even
  slightly toward the routes real images take, two of the "dog" neurons turn out to be car
  neurons, and two share a pose between cars and dogs.

![What we thought: a car hidden among dog neurons. What's going on: car neurons and shared car-and-dog pose neurons](/paper-media/neural-networks-are-already-clean/car-circuit.webp)

*Left, the 2020 reading: the car stored among dog neurons. Right, visualized through the car: two are car neurons and two share a pose between cars and dogs. Figure: [Cammarata, Neural Networks Are Already Clean](https://already-clean.goodfire.com/#car-circuit) (Goodfire, 2026).*

![The same pose neuron in DINOv2 and three ConvNeXt sizes, grouping dogs and vehicles that face the same way](/paper-media/neural-networks-are-already-clean/pose-everywhere.webp)

*The same generalized pose neuron in DINOv2 and in large and medium ConvNeXts: dogs and vehicles grouped by how they face the camera. The smallest model only has a rough version. Figure: [Cammarata, Neural Networks Are Already Clean](https://already-clean.goodfire.com/#pose-everywhere) (Goodfire, 2026).*

The essay draws a general lesson: people (and AI interpreters) describe neurons as **lists of
nouns** ("cats and cars") when they're often a **shared pattern** ("a V"). A neuron is better
thought of as a **verb** than a noun: its job is to draw a distinction downstream neurons can
use. A useful trick: *if you couldn't use groups of nouns at all, how would you describe this
neuron?*

**Why it matters.** The two examples that convinced the field neurons could be arbitrary
mixtures were the cat-and-car neuron and the car-into-dogs circuit. Both turn out to be
ordinary generalization. To the author's knowledge, no one has published a second circuit with
the "superposition" shape.

**Go deeper:** [Neurons and layers](/concepts/neurons) ·
[Gradient descent](/concepts/gradient-descent) (feature visualization is gradient ascent on
the image instead of the weights)

## 3 · A history of microscope errors

> **The essay says:** *"The story of interpretability, as I see it, has largely been one of
> finding out that what was under the microscope was better than we thought. The microscope
> just wasn't tuned right."*

**In plain words.** Two things can make a neuron look messy: a bad **microscope** (the tools we
look with) or a badly trained **specimen** (the network itself). Both happened.

**The microscope.** Early feature visualizations were blurry and psychedelic. The fix was
*transformation robustness*: jitter, rotate and rescale the image during optimization so it
can't exploit pixel-level tricks. The same neurons came out crisp. Deeper in the network it
broke again, through the reward hacking from section 2 and slightly wrong regularization. A
few small tweaks, and much deeper layers became visible. Same neurons, better microscope.

![Two InceptionV1 neurons rendered with 2009-style, 2017 and 2026 feature visualization](/paper-media/neural-networks-are-already-clean/microscope-timeline.webp)

*The same two InceptionV1 neurons with 2009-style, 2017 and 2026 feature visualization, beside their strongest photos. Small regularization tweaks bring out first the middle neuron, then the late one. Figure: [Cammarata, Neural Networks Are Already Clean](https://already-clean.goodfire.com/#microscope-timeline) (Goodfire, 2026).*

**The specimen.** InceptionV1's last layer really was hard to read. One researcher estimated
"essentially no interpretable neurons"; the author remembers about 5%. The standard
explanation: there are too many high-level concepts for the neurons available, so the network
packs unrelated ones together. But here's what happens when you keep the **same parameter
count and the same data** and change only the architecture:

![A late-layer InceptionV1 neuron: psychedelic optimized views and photos of striped fabric](/paper-media/neural-networks-are-already-clean/inception-last-layer.webp)

*A late-layer InceptionV1 neuron, the kind that convinced people the last layer was hopeless. Figure: [Cammarata, Neural Networks Are Already Clean](https://already-clean.goodfire.com/#inception-last-layer) (Goodfire, 2026).*

| Last layer of a ~7M-parameter ImageNet model | Easily interpretable neurons |
| --- | --- |
| InceptionV1 (2014), tools of that era | ~0–5% |
| Same model, modern tools (fixed feature visualization, attribution views) | ~50% |
| Same architecture, retrained with modern regularization | ~70% |
| ConvNeXt of the same size (residual connections, LayerNorm, …) | ~99% |

The improvements weren't designed for interpretability at all. Residual connections were
invented to help *deep networks train*; they also seem to make deep layers conceptually clean.

![Last-layer neuron of a ConvNeXt with its 20 strongest images, all ladybugs](/paper-media/neural-networks-are-already-clean/ladybug-figure.webp)

*The ConvNeXt's last layer: each neuron's 20 strongest responses out of 50,000 images, in order. This one is ladybugs, nearly all the way down. Figure: [Cammarata, Neural Networks Are Already Clean](https://already-clean.goodfire.com/#ladybug-figure) (Goodfire, 2026).*

**Why it matters.** The last layer of InceptionV1 was a key piece of evidence that networks are
fundamentally scrambled. It turned out to be a network that hadn't learned well, seen through
tools that hadn't been tuned. Neither problem needs a new theory of learning.

**Go deeper:** [Residual connections](/concepts/residual-connections) ·
[Layer normalization](/concepts/layer-normalization)

## 4 · Recursive generalization: neurons grow like branches

> **The essay says:** *"As a neuron grows, it sprouts offshoots the way a tree branch sprouts
> sub-branches. For instance, one neuron that detected 'iy' grew to detect 'diy,' and then 'do
> it yourself,' which doesn't contain 'iy' at all."*

**In plain words.** If the old InceptionV1 wasn't packing things arbitrarily, what *was* it
doing? The author retrained it, saving checkpoints, and **watched neurons form**. One started
on yellow flowers (rapeseed fields), in photos that sometimes had a dog. Then it grew along each
facet of those photos: toward other nature, toward green objects like balls, toward dogs
without flowers, then broccoli, until the fields were nearly gone. Every step was a small,
sensible move from the last. Only the end result looked like a random mixture.

The same thing, trained in a ConvNeXt, reached for balls and fruit early on, then *let them
go* and stayed on the fields.

![Two neurons over training: the InceptionV1 one spreads from fields into balls, fruit, dogs and broccoli; the ConvNeXt one stays on fields](/paper-media/neural-networks-are-already-clean/neuron-overflow.webp)

*Two neurons trained on the same data, both starting on rapeseed fields. InceptionV1's keeps spreading (balls, fruit, dogs, broccoli) until the fields are nearly gone; ConvNeXt's lets the offshoots go. Figure: [Cammarata, Neural Networks Are Already Clean](https://already-clean.goodfire.com/#neuron-overflow) (Goodfire, 2026).*

This is **recursive generalization**: a neuron generalizing along more than one axis, often
first one, then sprouting outward in ways that are easy to follow locally, but that stop it
being "monosemantic" (one meaning) in the end.

![Branching diagram: iy grows to Miyazaki and diy, and diy grows to do it yourself](/paper-media/neural-networks-are-already-clean/recursive-generalization.webp)

*The essay's own sketch: from “iy” to “diy” to “do it yourself”, which has no “iy” at all. Conceptual illustration. Figure: [Cammarata, Neural Networks Are Already Clean](https://already-clean.goodfire.com/#recursive-generalization) (Goodfire, 2026).*

::video[recursive-generalization]

Examples from language models:

- An **"iy"** neuron fires in "Miyazaki", then on "diy", then on "do it yourself", which has no
  "iy" left at all.
- A **planetary geology** neuron also fires on code for **Cesium**, a JavaScript library for
  mapping planets and globes.
- An **"enemy"** neuron fires for a Russian poem about enemies.

![A planetary-geology neuron's top firings grouped by the circuit routes that feed them](/paper-media/neural-networks-are-already-clean/route-split.webp)

*A planetary-geology neuron's strongest firings mix lunar crust with Cesium and IDL code. Grouped by the circuits that feed them, they split into three facets of one core (planetary surfaces) and one offshoot (the code), which arrives by its own route. Figure: [Cammarata, Neural Networks Are Already Clean](https://already-clean.goodfire.com/#route-split) (Goodfire, 2026).*

The idea first appeared in the 2021 paper *Multimodal Neurons* under the name "conjoined
features", but no one, including the author, suggested it could be the general explanation.

**Why it matters.** A branch is **easy to explain**: everything on it traces back to one
origin, and it grows out in a particular order over training. That's a narrow, testable
prediction. And the essay finds the offshoots are both rarer and less important than a neuron's
main job: they look like whatever was nearby and easy to grab, not hidden "sleeper" features.

**Go deeper:** [Gradient descent](/concepts/gradient-descent) ·
[Backpropagation](/concepts/backpropagation)

## 5 · Superposition: the opposite prediction

> **The essay says:** *"With generalization, things are put together because of a shared
> pattern that it's useful to have in one place, so it can be built on. With superposition,
> unrelated things are placed together because they don't interfere. These have opposite
> predictions."*

**In plain words.** Both explanations say a neuron can respond to several things. They
disagree about **which** things:

| | Generalization | Superposition |
| --- | --- | --- |
| Why things share a neuron | They share a useful pattern | They rarely occur together, so they don't collide |
| What ends up together | **Similar** things (local) | **Unrelated** things (non-local) |
| Example | A floppy-ear neuron: many ears at many angles | A floppy-ear + Latin-poetry neuron |
| Squeeze the layer smaller | Neurons get **coarser** | Neurons get **packed** with mixtures |
| What a grouping teaches you | How the network thinks | Almost nothing |

The last two rows are the testable part, and section 7 runs that test.

![Arbitrary packing puts unrelated shapes in each neuron; generalization puts similar ones together](/paper-media/neural-networks-are-already-clean/packing-vs-growing.webp)

*The same 32 things in both panels; similar colours and shapes are similar things. With fewer neurons, generalizing neurons get coarser, never fractured. (Interactive in the original: drag the neuron count.) Figure: [Cammarata, Neural Networks Are Already Clean](https://already-clean.goodfire.com/#packing-vs-growing) (Goodfire, 2026).*

**A tangle in the word itself.** "Superposition" gets used in two different senses:

1. **"More features than dimensions."** A 100-dimensional space can hold far more than 100
   meaningful directions. But even a perfectly clean network satisfies this, because neurons
   **compose**: wheel, window and body neurons combine into a car neuron in the next layer, so
   "car" lives both in that combination and in the car neuron. The essay doesn't think this
   sense is bad news.
2. **A learning principle**: neurons learning *unrelated* things together because they rarely
   interfere. This is the sense the essay argues against, because almost all of the good or bad
   news for interpretability hangs on it.

When the essay says "superposition", it means sense 2.

**Why it matters.** Untangling methods are only *necessary* if sense 2 is common. If neurons
group things by generalization, the grouping itself is a clue, and splitting it apart throws
the clue away.

**Go deeper:** [Vectors](/concepts/vectors) (directions in a space) ·
[Neurons and layers](/concepts/neurons)

## 6 · Superposition of the gaps

> **The essay says:** *"I now think I was making superposition the null hypothesis when it
> should have been the last hypothesis, since it's the most radical and the least testable. It
> just looks like things that don't make sense."*

**In plain words.** Calling a neuron "polysemantic" is often a claim about the **observer**,
not the neuron: it says nobody has a good explanation yet. The essay lists five possible
causes, roughly in the order you should suspect them:

1. **Bad tools**: the microscope isn't good enough.
2. **Bad explanations**: built from those tools.
3. **Bad learning**: the network learned poorly (architecture, data, …).
4. **Recursive generalization**: the neuron branched in a few related directions.
5. **Superposition**: the network packed unrelated things together.

The first four can each be **tested**: a better tool either resolves the neuron or it doesn't;
retraining with a better architecture shows whether learning was the problem; branches can be
watched as they grow. Superposition's only symptom is "this doesn't make sense", which is also
what every other cause looks like before you find it.

The essay gives several cases where the "mess" dissolved:

- A neuron in a frontier model fired on **"pr"**, but also on **Base64 URL strings** and some
  **Bengali**. Decode the URLs: "pr" words. Transliterate the Bengali: "pr".
- A "polysemantic **comma** neuron" in a Qwen model turned out to track **Colorado**: phone
  numbers, slang, links to other Colorado neurons. Turning it up made the model want to go
  hiking. The commas were where it kept a running summary of the context. Attention tends to
  collect information on punctuation, a pattern the essay calls the **summary motif**.
- Neurons that looked like "coffee and universities" made sense once examples included
  **long context**: the reason was paragraphs further up.

![The pr neuron firing in English, Base64 and Bengali](/paper-media/neural-networks-are-already-clean/pr-neuron.webp)

*The “pr” neuron: decoded or transliterated, the Base64 and Bengali “noise” turns out to be “pr”. Figure: [Cammarata, Neural Networks Are Already Clean](https://already-clean.goodfire.com/#pr-neuron) (Goodfire, 2026).*

![Colorado neuron: commas stay dark until Colorado appears, then brighten](/paper-media/neural-networks-are-already-clean/colorado-neuron.webp)

*The “comma” neuron: commas stay dark until Colorado shows up, then brighten as it builds. Steering it turns a weekend plan into a hike. Figure: [Cammarata, Neural Networks Are Already Clean](https://already-clean.goodfire.com/#colorado-neuron) (Goodfire, 2026).*

![A paragraph about coffee and a university, whose real topic is several paragraphs earlier](/paper-media/neural-networks-are-already-clean/context-neuron.webp)

*The last paragraph alone suggests a neuron for coffee and universities; the reason is further up. Figure: [Cammarata, Neural Networks Are Already Clean](https://already-clean.goodfire.com/#context-neuron) (Goodfire, 2026).*

The author was a "staunch superposition believer" and began this work expecting to hit
intractable mixtures. They kept not arriving. Each time the strong activations came back clean,
the suspicion moved: to the weak activations, then to combinations of neurons, then to neurons
not yet checked. Anthropic's 2022 *Softmax Linear Units* paper had already noted that
superposition "is often treated as the default hypothesis simply because it seems intuitively
more compelling than other explanations, but there is little evidence."

**Why it matters.** The risk runs the other way too: assuming every neuron *must* have a tidy
story. The essay's defence is that real explanations are narrow and easy to break ("it's a
Colorado neuron": steer it and see), and that it's fine to say "this one might do two unrelated
things, we're not sure."

**Go deeper:** [Self-attention](/concepts/self-attention) (how information gets moved onto
commas)

## 7 · Image models: what neurons do under pressure

> **The essay says:** *"If it switched to superposition, we should see overlapping mixtures on
> top of each other... But that's not what we see. The neurons stay interpretable and
> generalized, but coarser."*

**In plain words.** Small image models are where the field's worry started, and they're the
easiest to see clearly, so the author ran the tests there. Three ConvNeXt models spanning about
**300× in size**: 92k, 7.06M (the same as InceptionV1) and 28.6M parameters.

**Neurons as gates, not nouns.** The essay suggests thinking of a neuron less as "the car
neuron" and more as **control flow**: what it lets through and what it keeps out. Early on, cars
and dogs share a pose neuron. Further along, specific car detectors (taxi, ambulance) don't want
dogs, so a neuron in front of them fires **positively for cars and negatively for dogs**. Its
negative side is a gate that sends the dogs elsewhere. Last-layer neurons in the large model
each respond to one thing and *negatively to a close cousin*: taxis vs ambulances, broccoli vs
cauliflower, spoonbills vs flamingos. The smallest model stops earlier, with one neuron for cars,
trains and carts against boats and planes.

![Cars and dogs share a pose neuron; a gate sends cars to car detectors and dogs elsewhere](/paper-media/neural-networks-are-already-clean/gate-neuron.webp)

*Cars and dogs share a pose early on; a gate in front of the car detectors lets cars through and turns dogs away with its negative side. Figure: [Cammarata, Neural Networks Are Already Clean](https://already-clean.goodfire.com/#gate-neuron) (Goodfire, 2026).*

![One sports car through five ConvNeXt blocks: what it is grouped with and separated from at each](/paper-media/neural-networks-are-already-clean/car-path-distinctions.webp)

*One sports car through the last five blocks of the large ConvNeXt. Above: what each neuron groups it with (tables and cabinets, then convertibles). Below: what it separates it from (snakes and dogs, then station wagons). Figure: [Cammarata, Neural Networks Are Already Clean](https://already-clean.goodfire.com/#car-path-distinctions) (Goodfire, 2026).*

![Last-layer neurons with a positive and a negative pole: taxis vs ambulances, broccoli vs cauliflower, spoonbills vs flamingos](/paper-media/neural-networks-are-already-clean/negative-poles.webp)

*Last-layer neurons in the large ConvNeXt: each responds to one thing and negatively to a close cousin. Figure: [Cammarata, Neural Networks Are Already Clean](https://already-clean.goodfire.com/#negative-poles) (Goodfire, 2026).*

![The small model's neuron: ground vehicles positive, boats and planes negative](/paper-media/neural-networks-are-already-clean/tiny-vehicles-positive.webp)

*The smallest model stops earlier: ground vehicles on the positive side, boats and planes (and an oxcart on water) on the negative side. Figure: [Cammarata, Neural Networks Are Already Clean](https://already-clean.goodfire.com/#tiny-vehicles-positive) (Goodfire, 2026).*

**The squeeze test.** Layer 3b of InceptionV1 holds the well-understood **curve detectors**,
and the model has no residual connection to route around it. Retrain the whole model with that
layer cut from **480 neurons to 10** (48× fewer). If superposition kicked in, each of the 10
should be a pile of overlapping unrelated detectors. Instead, they're **coarser versions of the
same ideas**: one "curves and loops" neuron instead of ten curve detectors, two circle neurons
instead of several. The same happens when the last layer of the medium ConvNeXt is cut from
**1,024 to 13** neurons.

![The ten neurons of the squeezed layer: dark circles, yellow, vertical edges, curves and loops, high frequency, red, slanted edges, eyes and faces, circles, colourful marks](/paper-media/neural-networks-are-already-clean/crushed-3b-figure.webp)

*InceptionV1 retrained with layer 3b cut from 480 neurons to 10, with no way around it. Each row is one neuron and its five strongest crops: coarser, still readable. 58% top-1 accuracy. Figure: [Cammarata, Neural Networks Are Already Clean](https://already-clean.goodfire.com/#crushed-3b-figure) (Goodfire, 2026).*

![The medium ConvNeXt's last layer cut to 13 neurons, e.g. beetles vs llamas](/paper-media/neural-networks-are-already-clean/crushed-last-layer.webp)

*The medium ConvNeXt with its final layer cut from 1,024 to 13 neurons: each neuron is still a readable contrast, just broader. Figure: [Cammarata, Neural Networks Are Already Clean](https://already-clean.goodfire.com/#crushed-last-layer) (Goodfire, 2026).*

::video[generalize-or-pack]

**Could the curves be hiding in combinations of the 10?** Train a sparse autoencoder on the
squeezed layer, which is designed to pull out exactly such hidden directions. It finds broad,
bunched-together features, not ten clean curves. And on full-size models, the essay finds that
**88%** of an SAE's activity on the last layer is near-copies of single neurons; removing every
SAE direction that matches no neuron **leaves accuracy unchanged**. When the model finally
understood baseball players, it was on a neuron.

![Ten original curve detectors, the single curves-and-loops neuron, and the closest SAE features: broad and bunched](/paper-media/neural-networks-are-already-clean/c3b-squeeze.webp)

*Left, the ten original curve detectors. Middle, the squeezed layer, where they became one “curves & loops” neuron. Right, an SAE trained on the squeezed layer: the feature closest to each curve detector is broad and bunched together, not ten curves. Figure: [Cammarata, Neural Networks Are Already Clean](https://already-clean.goodfire.com/#c3b-squeeze) (Goodfire, 2026).*

![An SAE feature, animals at the water's surface, built from four existing neurons](/paper-media/neural-networks-are-already-clean/convnext-sae-residual.webp)

*What the SAE added on the last layer: loose groupings built from existing neurons, like “animals at the water's surface”. Figure: [Cammarata, Neural Networks Are Already Clean](https://already-clean.goodfire.com/#convnext-sae-residual) (Goodfire, 2026).*

![Five blocks of circuits feeding a monastery neuron: stone walls, roofs and arches, bell towers, castles](/paper-media/neural-networks-are-already-clean/ladybug-circuit.webp)

*Late layers are organized too: the circuit into a monastery neuron, built from stone walls, roofs and arches, bell towers and castles. Figure: [Cammarata, Neural Networks Are Already Clean](https://already-clean.goodfire.com/#ladybug-circuit) (Goodfire, 2026).*

**Rare things and memorization.** Maybe rare concepts get a secret hiding place? The author
taught the 7.06M model a nonsense fact: *this photo of an umbrella is a bee*. In nine training
steps, its top guess went umbrella → maypole → bee. No hidden umbrella-bee feature formed.
The umbrella neurons faded and **existing bee neurons came on**, almost all in the last three
blocks. Memorization took an off-ramp between two routes the network already had. Even a model
trained to map random noise to random labels partly leaned on real textures it knew.

![Before and after teaching the model an umbrella is a bee: umbrella neurons fade and bee neurons switch on](/paper-media/neural-networks-are-already-clean/umbrella-bee.webp)

*An umbrella switches to the bee route. Bars show each neuron's response to this photo before and after the model learned “bee”. Figure: [Cammarata, Neural Networks Are Already Clean](https://already-clean.goodfire.com/#umbrella-bee) (Goodfire, 2026).*

![A stray hidden feature versus an off-ramp between two existing routes](/paper-media/neural-networks-are-already-clean/memorization-ramp.webp)

*One way to memorize a nonsense label would be a feature connected to nothing (left). What happens is an off-ramp between two routes the network already has (right). Figure: [Cammarata, Neural Networks Are Already Clean](https://already-clean.goodfire.com/#memorization-ramp) (Goodfire, 2026).*

![Unusual images read with ordinary neurons: a giant banana as a vault, a hummingbird as ice cream, a wheelbarrow as a lawn mower](/paper-media/neural-networks-are-already-clean/rare-image-examples.webp)

*The images furthest from everything else are still read with ordinary neurons: a giant banana as vaulted ceilings, a hummingbird as ice cream, a fallen wheelbarrow as a lawn mower. Figure: [Cammarata, Neural Networks Are Already Clean](https://already-clean.goodfire.com/#rare-image-examples) (Goodfire, 2026).*

**Why it matters.** This is the direct test the table in section 5 promised. Squeezed hard, the
network didn't change learning principles; its neurons just got broader. Broad neurons are easy
to break down once you know what to look for, and look like arbitrary mixtures when you don't.

**Go deeper:** [Neurons and layers](/concepts/neurons) ·
[Residual connections](/concepts/residual-connections)

## 8 · Do weak activations matter?

> **The essay says:** *"If a weak activation looks unrelated, don't be too worried unless
> you've looked at it carefully. Of the probably hundreds I've followed by hand, almost all
> turned out to be related."*

**In plain words.** Suppose a neuron's *strong* firings are all clean. Could unrelated things be
hidden in its *weak* firings? The essay splits this into three questions.

**Are they generalization or superposition?** Generalization, the essay argues. Weak activations
are vastly more common: in **100 million tokens** through one Qwen3-32B neuron, almost all sit
near zero and only about **160** reach the strong half. A packed feature hiding there would
collide with orders of magnitude more things. And you can **remove almost all weak activations
across the whole network** for very little cost. What you lose are fine distinctions between
answers the network was already unsure about: near-ties tipping the other way. The places it
costs most are easy to explain (early vision layers that work like brightness dials, a final
language layer that suppresses single tokens).

![100 million activations of one neuron: a dense crowd near zero thinning to a few strong ones](/paper-media/neural-networks-are-already-clean/weak-crowd-strip.webp)

*100 million tokens through one Qwen3-32B neuron. Almost all sit near zero; about 160 reach the strong half. Figure: [Cammarata, Neural Networks Are Already Clean](https://already-clean.goodfire.com/#weak-crowd-strip) (Goodfire, 2026).*

![Removing the weakest 75% of activations changes 1 in 10 answers, all near ties; removing the strongest 7% changes 96 in 100](/paper-media/neural-networks-are-already-clean/weak-close-calls.webp)

*Remove the weakest 75% of activations and 1 answer in 10 changes, all near-ties (husky ↔ Eskimo dog). Remove the strongest 7%, the same total activation, and 96 in 100 change. Figure: [Cammarata, Neural Networks Are Already Clean](https://already-clean.goodfire.com/#weak-close-calls) (Goodfire, 2026).*

Going down from a neuron's maximum, meaning **radiates outward**. CLIP's Trump neuron: photos
of Trump, then art and people related to him, then Fox News studios. Something that looks
unrelated far down the range usually has a traceable route, and what fired weakly here may
have fired strongly one layer back. (A Zen koan neuron fires weakly on recursion in maths:
many koans turn on self-reference.)

![Every strong firing of five neurons (Titanic, Baroque, Christmas, Rainbow, Colorado) drawn in rings by strength and coloured by topic](/paper-media/neural-networks-are-already-clean/five-circles.webp)

*Every strong firing of five Qwen3-32B neurons. The centre is each neuron's maximum, rings are fractions of it, colour is what a firing is about. Meaning radiates outward; grey is hard to place. Figure: [Cammarata, Neural Networks Are Already Clean](https://already-clean.goodfire.com/#five-circles) (Goodfire, 2026).*

**Percentiles or raw activations?** Percentiles sound reassuring but mislead. Using the essay's
rule of thumb that 1 token in 100 is about a quarter of a page (so a page ≈ 400 tokens):

$$
\text{99th percentile} = \tfrac{1}{100} \text{ tokens} \approx 4 \text{ per page}, \qquad
\text{99.9th} = \tfrac{1}{1000} \approx \text{once every } 2.5 \text{ pages}
$$

and the 99.99th is about once every 25 pages. A neuron for a niche feeling shouldn't come up four
times a page, so "top 1%" can already be deep in the weak tail. Each neuron's own raw range is
more telling: some keep their meaning far down, others (like a Christmas neuron) drop off a cliff
at about 5% of their maximum.

**Do you need them to label a neuron?** Usually not. Neurons are **analogizers**: a Titanic
neuron fires a little on any boat voyage because the voyage is a bit Titanic-ish. The label from
the top activations usually holds up.

**Why it matters.** Weak activations were the last refuge of "it's packed somewhere we haven't
looked". The essay's measurements suggest they're mostly the same meaning, fading.

**Go deeper:** [Activation functions](/concepts/activation-functions)

## 9 · Language models: Santa on the negative pole

> **The essay says:** *"What we see is about 90% of neurons having basically one job, with
> recursive generalization offshoots that are usually related... I've never seen a neuron with
> something like ten unrelated things it does."*

**In plain words.** This part of the essay, done with Curt Tigges, takes the same approach to
large language models, mainly **Qwen3-32B**, using the network's own neurons rather than a
replacement model.

**Circuits in neurons, not features.** Anthropic's *Biology of a Large Language Model* traced
a medical-diagnosis circuit in Claude 3.5 Haiku using *transcoder features*, a trained
replacement for the model's layers. Their largest one predicts the same next word as the real
model only about half the time, so their graphs need "error nodes". The same circuit can be
traced in Qwen's own neurons with no reconstruction error: the diagnosis hub is **four
neurons**, and removing them makes the model ask about itching instead, the telltale sign of a
*different* pregnancy liver condition, its next guess. The essay also shows what it believes is
the first published attribution graph of a **multi-trillion-parameter model** (Kimi K3), reasoning
from a found wallet to "return it".

![The same preeclampsia diagnosis circuit in Claude transcoder features and in Qwen neurons](/paper-media/neural-networks-are-already-clean/preeclampsia-circuit.webp)

*The same diagnosis circuit in Anthropic's transcoder features (left) and in Qwen3-32B's own neurons (right). Figure: [Cammarata, Neural Networks Are Already Clean](https://already-clean.goodfire.com/#preeclampsia-circuit) (Goodfire, 2026).*

![A Kimi K3 attribution graph from wallet, cash and ID through lost property and honesty to 'return'](/paper-media/neural-networks-are-already-clean/wallet-compact.webp)

*A found wallet in a trillion-parameter model: wallet, cash and the owner's ID feed lost property and honesty, which push “return”. Figure: [Cammarata, Neural Networks Are Already Clean](https://already-clean.goodfire.com/#wallet-compact) (Goodfire, 2026).*

**Santa lives on the negative pole.** In one graph, labels from standard tools read "capital
cities" because they ignore each neuron's negative side. Remove the **eight** neurons whose
*negative* side is Santa lore and the model's confidence in "Santa" falls from near-certain to
**6%**. Remove the **twenty** neurons actually labelled Christmas or Santa and it only falls to
**76%**. The strong positives *and* negatives matter; the weak activations mostly don't.

![Spelling-fragment neurons feeding one Miyazaki neuron, which feeds associations like Spirited Away](/paper-media/neural-networks-are-already-clean/miyazaki-circuit.webp)

*From spelling fragments to “Miyazaki”: general spelling pieces feed one Miyazaki neuron, which fires on the name in every script and feeds later associations like “Spirited Away”. Figure: [Cammarata, Neural Networks Are Already Clean](https://already-clean.goodfire.com/#miyazaki-circuit) (Goodfire, 2026).*

**SwiGLU's double negative.** Modern language models use a **SwiGLU** layer, where each neuron
multiplies two numbers instead of applying one bend:

$$
h = \operatorname{SiLU}(g) \cdot v, \qquad g = \mathbf{w}_g \cdot \mathbf{x}, \quad
v = \mathbf{w}_v \cdot \mathbf{x}, \qquad \operatorname{SiLU}(g) = g \, \sigma(g)
$$

Here $g$ is the **gate** and $v$ the **value**. $\operatorname{SiLU}$ is mostly zero for negative
$g$, but not quite: it dips to about $-0.28$. So when **both** are negative, the product is
**positive**, and the neuron fires for a second, quite different reason. Take $g = -1.5$ and
$v = -4$: $\operatorname{SiLU}(-1.5) \approx -0.27$, so $h \approx +1.1$.

::video[swiglu-sign-regimes]

This "echo" usually looks like the neuron's negative pole showing up on its positive side. It's
usually small, but for about **4%** of neurons it fills the whole top of the list: in one
"enemy" neuron, **131 of 133** mid-range responses are − × −, while all **93** of its strongest
are + × +. Showing the sign combinations separately clears the noise without replacing the
neuron. A tool that mixed neurons together would have hidden it.

![A neuron whose top list is all double-negative leak, and its real job once separated](/paper-media/neural-networks-are-already-clean/double-negative-example.webp)

*A neuron whose whole top list is SwiGLU leak (Colossians, both gate and value negative). Separate the leak and its real job appears: the token that completes “Russia” in any script. Figure: [Cammarata, Neural Networks Are Already Clean](https://already-clean.goodfire.com/#double-negative-example) (Goodfire, 2026).*

![Share of double-negative responses by activation band: the noisy middle is almost all minus times minus](/paper-media/neural-networks-are-already-clean/swiglu-audit.webp)

*The double-negative regime fills the noisy middle: 131/133 mid-range responses are − × −, while the strongest tail is + × +. Figure: [Cammarata, Neural Networks Are Already Clean](https://already-clean.goodfire.com/#swiglu-audit) (Goodfire, 2026).*

**Other findings in this section:**

- **Experts are specialized.** In mixture-of-experts models, experts get narrower with depth,
  and one is devoted entirely to bosses in RPG board games. The essay calls experts an
  *anti-superposition* architecture: a small linear router couldn't recognise ten thousand
  unrelated things to route them to one packed expert.
- **Explanations that can lose.** In a *blind maximum* test, one agent writes a short
  explanation of a neuron; a second agent sees **only** that explanation and writes a sentence
  meant to fire it maximally. The sentence often fires it harder than any example in **10
  billion** real tokens.
- **Old models revisited.** GPT-2 XL, where under 2% of neurons came out clean in 2023: about
  **70%** look like roughly one thing with today's approach. The *Towards Monosemanticity*
  neuron for "academic citations, English dialogue, HTTP requests, and Korean text" may be a
  **"don't say a dot"** neuron: most of its firings sit on or near a dot, and eight of its ten
  strongest negative output connections are tokens starting with one.

![All 10,240 Qwen3.6 experts, one row per layer, sorted from narrow to broad](/paper-media/neural-networks-are-already-clean/expert-organization.webp)

*All 10,240 experts in Qwen3.6, one row per layer, sorted from the narrowest experts to the broadest. Figure: [Cammarata, Neural Networks Are Already Clean](https://already-clean.goodfire.com/#expert-organization) (Goodfire, 2026).*

![Agent 1 explains a neuron; agent 2 writes a sentence from the explanation alone that beats the best corpus example](/paper-media/neural-networks-are-already-clean/blind-maximum.webp)

*Blind maximum: agent 2 sees only the explanation, and its sentence fires the neuron harder than anything found in the corpus. Figure: [Cammarata, Neural Networks Are Already Clean](https://already-clean.goodfire.com/#blind-maximum) (Goodfire, 2026).*

![The Towards Monosemanticity neuron and a Qwen3.6 neuron doing the same don't-say-dot job](/paper-media/neural-networks-are-already-clean/period-neuron-figure.webp)

*Don't say dot: the Towards Monosemanticity neuron in its own viewer (left) and a Qwen3.6 neuron doing the same job (right). Remove either and the next period gets likelier. Figure: [Cammarata, Neural Networks Are Already Clean](https://already-clean.goodfire.com/#period-neuron-figure) (Goodfire, 2026).*

**Why it matters.** Small, simplified models were chosen as easy starting points, but their
simplifications made neurons learn worse, and smaller models have broader neurons. Big modern
models have narrower neurons, so patterns are easier to spot. The essay's phrase: by adopting
the architectures and scale the field wanted anyway, we may have been "slouching towards
monosemanticity" without knowing it.

**Go deeper:** [Activation functions](/concepts/activation-functions) ·
[Transformer](/concepts/transformer)

## 10 · The residual stream and attention

> **The essay says:** *"Find what the architecture is doing and work with it, rather than
> fighting the network."*

**In plain words.** Neurons aren't the only place things could be packed.

**The residual stream** is the running vector each layer reads from and adds to. The worry is a
continuous kind of packing: many meanings woven through the same dimensions, tolerating some
interference. In practice, the essay hasn't hit it as a bottleneck: what neurons write to and
read from the stream is readable, and attribution graphs come out clean.

In vision models you can check by eye. For years every map of a ConvNeXt's residual stream came
out as one big arc. The reason: the network put almost all its activation mass into **one
dimension nothing reads**, and used it as a knob. Since **LayerNorm** divides each vector by its
overall size, growing that one coordinate shrinks every other coordinate after normalization.
Remove it, and the maps made sense. Squeezing the stream to a quarter and an eighth of its width
(down to 46 dimensions) cost accuracy (72% → 60% → 39%) but the maps just got a bit messier;
a dark-bird-on-water neuron leaked onto alligators and a swimming retriever, not onto clocks.

![The ConvNeXt residual stream drawn as one map, with image locations traced from first layer to last](/paper-media/neural-networks-are-already-clean/rs-jt.webp)

*The residual stream at every layer of the ConvNeXt, drawn as one map. Each line follows one image location from the first layer to the last. Animal-face locations start apart, sorted by fur colour, and gather; red patches start together and become different things. Figure: [Cammarata, Neural Networks Are Already Clean](https://already-clean.goodfire.com/#rs-jt) (Goodfire, 2026).*

"More features than dimensions" in the stream isn't packing on its own: a vector space can hold
many more meaningful directions than it has axes. It only matters if you want one big list of
features, and the essay doesn't think that's the right goal.

**Attention** is the part the essay understands least. But it can be broken into questions:
*where* is it looking (nearly free to see), *what* is it looking for and *what* does it do when
it finds it (both answerable in terms of the neurons it reads and writes). In the diagnosis
example, three heads read a headache neuron, a migraine neuron and a brain neuron at the comma
after "headache", and write mostly onto the preeclampsia neuron. Ablating those heads drops
preeclampsia by a quarter. Neurons and heads explain each other.

![Attention moving symptoms onto the summary token: headache to preeclampsia, music to musical content](/paper-media/neural-networks-are-already-clean/attn-reads.webp)

*The chips under tokens are neurons that fire there. You can follow much of what attention does by watching which neurons it affects. Figure: [Cammarata, Neural Networks Are Already Clean](https://already-clean.goodfire.com/#attn-reads) (Goodfire, 2026).*

![Three cases where a neuron and an attention head only make sense together](/paper-media/neural-networks-are-already-clean/attn-mutual.webp)

*Understanding attention and neurons through each other: three cases in Qwen3-32B where a neuron and a head only make sense together. Figure: [Cammarata, Neural Networks Are Already Clean](https://already-clean.goodfire.com/#attn-mutual) (Goodfire, 2026).*

**Why it matters.** The same lesson as the microscope: what looked like mess was an
architectural quirk that, once understood, was easy to work with.

**Go deeper:** [Residual connections](/concepts/residual-connections) ·
[Layer normalization](/concepts/layer-normalization) ·
[Self-attention](/concepts/self-attention)

## 11 · Zooming out: a city, not a knot

> **The essay says:** *"If you assume a network is clean and you're wrong, a messy neuron is a
> lead... If you assume it's packed, not worth studying directly, and you're wrong, a messy
> neuron is the end, because you stop studying it."*

**In plain words.** By "clean", the essay doesn't mean every neuron matches a human word. It
means the network's parts are **built to be read by the network's own downstream parts**. A
network is "almost a fractal of explanations": coherent groupings at the large scale, and
recursive generalization at the small one. The goal for interpretability becomes: **become
another reader.**

![Animated zoom from three ConvNeXt networks into the medium one: the dog district, a circuit, one neuron, its weights](/paper-media/neural-networks-are-already-clean/recursive-explainability.webp)

*Nothing here has been changed or recombined: zooming from three ConvNeXts into the medium one, then its dog district, a circuit, one neuron and its weights. Every district, circuit and neuron is the network's own. Figure: [Cammarata, Neural Networks Are Already Clean](https://already-clean.goodfire.com/#recursive-explainability) (Goodfire, 2026).*

**Two styles of interpretability.**

- **Enumerative**: produce a big list of every feature, then audit it. Polysemanticity is a
  serious threat here, because every mixed item multiplies the work.
- **Exploratory**: start from questions, zoom in and out across neurons, attention and circuits
  like Google Maps, and ignore the parts you don't need. This style tolerates neurons with a
  couple of meanings, and doesn't need "feature" to be defined at all.

The essay bets on exploratory, and on **motifs** (recurring patterns like the summary motif,
gates on negative poles, double negatives) over perfect monosemanticity.

**A city, not a knot.** A bigger system isn't necessarily harder to navigate: in a small town,
"the financial district" barely exists, but in New York you can find it on a map in a minute.
In the ConvNeXts, agents found dog-ear neurons *faster* in the larger models, where several ear
neurons feed one hanging-ears neuron. The danger is imposing our own map: arriving in Tokyo and
rebuilding "Brooklyn" from seventy disconnected pieces makes Tokyo look messier than it is. An
SAE trained on the V neuron's layer splits the V into a car feature and a cat feature, further
from what the neuron does. Probes and dictionaries still have a place, but at the end, not the
start.

![The dog-ear neurons in small, medium and large ConvNeXts: coarse in the small one, a branching family in the large](/paper-media/neural-networks-are-already-clean/source-figure-17.webp)

*How the ear family changes with scale across three networks about 300× apart in size. Bigger is easier to navigate: in the large model, pale, wavy and head-side ear neurons all feed one hanging-ears neuron. Figure: [Cammarata, Neural Networks Are Already Clean](https://already-clean.goodfire.com/#source-figure-17) (Goodfire, 2026).*

![Map of every neuron in the medium ConvNeXt drawn as a city of islands: stages, districts like dog faces, birds, bugs and lizards](/paper-media/neural-networks-are-already-clean/convnext-map-still.webp)

*Tethered abstraction: every neuron in the medium ConvNeXt drawn as a city, with districts (birds, bugs and spiders, dog faces) you can zoom into down to single neurons. Figure: [Cammarata, Neural Networks Are Already Clean](https://already-clean.goodfire.com/#convnext-map-still) (Goodfire, 2026).*

**The asymmetry.** Assume clean and be wrong: each messy neuron is a lead, and investigation
sharpens it. Assume packed and be wrong: each messy neuron is a dead end, and you stop looking.
The author made the second assumption for years and spent "most of a decade figuring out how to
look at a model rather than looking at it."

**What a solution might look like.** A laboratory of automated scientists that label neurons,
find and test motifs, and follow data through circuits, at a scale no human team could reach.
Each step is simple, and the pieces already exist. It echoes Rich Sutton's *bitter lesson*:
the simple method at enormous scale beats the clever one. Maybe interpretability has a bitter
lesson too: **scaling ordinary investigation beats engineering a clever decomposition.**

![Animated map of a network with agents' explanations spreading across its districts](/paper-media/neural-networks-are-already-clean/many-agents.webp)

*Guided by a mechanistic constitution, agents propose and test explanations at many scales, each grounded in individual neurons, and keep, drop or replace them. Figure: [Cammarata, Neural Networks Are Already Clean](https://already-clean.goodfire.com/#many-agents) (Goodfire, 2026).*

**Why it matters, and the caveats.** The author is open that this is a working hypothesis, about
85% confident, and that the headline numbers depend on thresholds and are preliminary. A
technical companion paper on language-model neurons is promised. If the hypothesis holds, the
textbook picture is basically right: **neurons generalize, work together, and build
abstractions over layers.** In the essay's words: *"If the network isn't a
knot, untangling it can make it look like one."*

**Go deeper:** [Transformers, from scratch](/tracks/transformers) builds every piece this essay
inspects, from a single neuron to a working GPT. Then read the
[original essay](https://already-clean.goodfire.com/) with its interactive figures.
