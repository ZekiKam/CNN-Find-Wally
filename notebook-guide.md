# Teacher's guide: Find Wally with PyTorch

This guide goes with `find-wally-pytorch.ipynb`. It follows the notebook step by step. For each step it covers what the code does line by line, the theory behind it, why I made each design choice, answers to every **Think** question, and what tends to go wrong.

---

## Contents

0. [Setting up](#step-0-setting-up)
1. [Making our own training data](#step-1-making-our-own-training-data)
2. [Feeding images to PyTorch](#step-2-feeding-images-to-pytorch)
3. [Building the CNN from scratch](#step-3-building-the-cnn-from-scratch)
4. [The loss function and the optimiser](#step-4-the-loss-function-and-the-optimiser)
5. [Training the model](#step-5-training-the-model)
6. [How good are the boxes? IoU](#step-6-how-good-are-the-boxes-iou)
7. [Testing on unseen images](#step-7-testing-on-unseen-images)
8. [Results when I tested it](#results-when-i-tested-it)
9. [Decisions I made where your plan left a choice](#decisions-i-made-where-your-plan-left-a-choice)

---

## The big picture

The whole notebook is one idea: **a CNN can learn to output four numbers, the corners of a box around Wally, if we show it enough pictures where we already know the answer.**

```
backgrounds + Waldos ──paste──▶ 5000 images + boxes ──split──▶ 4000 train / 1000 test
                                                                     │
                     Dataset ──▶ DataLoader (batches of 32) ─────────┘
                                         │
   image (3x256x256) ──▶ CNN ──▶ 16x16 heatmap + size ──▶ 4 numbers (x1, y1, x2, y2)
                                         │
                      MSE loss vs true box ──▶ backward ──▶ Adam step
                                         │
                       after training: IoU on the 1000 test images
```

There is **no classification head** in this version, because the dataset only has Waldo (no Wilma). The model only answers "where?", never "what?" or "is he there?". That matters for one of the Step 7 questions.

---

## Step 0: Setting up

### Imports

| Import | Why we need it |
|---|---|
| `os`, `csv` | Listing image files, making folders, reading and writing `labels.csv` |
| `random` | Picking backgrounds, Waldos, angles, positions |
| `time` | Timing data generation and training |
| `numpy` | The colour shift in augmentation, converting images to arrays |
| `matplotlib.pyplot`, `matplotlib.patches` | Showing images, drawing boxes, learning curves |
| `PIL.Image`, `PIL.ImageEnhance` | Opening, rotating, pasting images, brightness change |
| `torch`, `torch.nn` | Tensors, layers, loss, optimiser |
| `torch.utils.data` | `Dataset`, `DataLoader`, `random_split` |
| `torchvision.ops.box_iou` | The IoU metric |

No pandas, Keras or scikit-learn: everything is in your tech stack.

### Settings cell

All the numbers that might change are in one place:

- `BACKGROUND_DIR`, `WALDO_DIR`: your Kaggle input paths, exactly as you gave them.
- `OUTPUT_DIR = '/kaggle/working/generated images'`: Kaggle's input folders are read-only, so generated files must go in `/kaggle/working`. Inside it are `train images` and `test images`, the folder names from your plan.
- `IMG_SIZE = 256`: all backgrounds are 256x256. Step 1.1 still resizes every background to `IMG_SIZE`, so one odd-sized file can't break the model. The model's flatten size is worked out from `IMG_SIZE` too (see Step 3).
- `NUM_IMAGES`, `NUM_TRAIN`, `NUM_TEST`: 5000, 4000, 1000, as in your plan.
- `TRAIN_IMAGES_PER_EPOCH = 1000`, `NUM_EPOCHS = 10`: "10 epochs of 1000 images each, each epoch uses the same training data". See [Decisions](#decisions-i-made-where-your-plan-left-a-choice).
- `BATCH_SIZE = 32`, `LEARNING_RATE = 1e-3`: see Steps 2 and 4.

### Seeds and device

```python
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
```

Three seeds, because three separate random number generators are in use: Python's `random` (positions, choices, angles), NumPy's (colour shift), and PyTorch's (starting weights, shuffling, `random_split`).

`device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')` picks the GPU if Kaggle has one switched on. Every tensor and the model must be on the same device, which is why you'll see `.to(device)` throughout.

**Think: if you change the seed, should the results be exactly the same? Roughly the same?**
Not exactly: different positions, different starting weights, a different split. But they should be roughly the same (similar IoU). If a different seed gives a wildly different result, training is unstable, and that's worth knowing. Also note that on a GPU, results can differ very slightly even with the same seed, because some GPU operations aren't perfectly deterministic.

---

## Step 1: Making our own training data

### Theory: synthetic data

To learn "where is Wally", the model needs examples **with answers** (labels). Hand-labelling thousands of real images means drawing thousands of boxes. Synthetic data flips this: we **create** each image by pasting Wally at a position we choose, so we know the box exactly. It's free, perfect and unlimited.

The catch is the **domain gap**: the model learns what *our* pasted images look like. A real Where's Wally page has Wally drawn into the scene (overlapping people, partly hidden, different scale). A model trained only on pasted Wallys can fail on real pages. Variety in the synthetic data (many backgrounds, augmented Wallys) narrows that gap but doesn't close it.

### 1.1 Loading the backgrounds

```python
background_paths = sorted(os.path.join(BACKGROUND_DIR, f) for f in os.listdir(BACKGROUND_DIR)
                          if f.lower().endswith(('.png', '.jpg', '.jpeg')))
backgrounds = [Image.open(p).convert('RGB') for p in background_paths]
```

- `sorted` makes the order the same on every machine. `os.listdir` order isn't guaranteed, and with a seed we want the same results everywhere.
- The `endswith` filter skips stray non-image files.
- `.convert('RGB')` makes every background 3-channel, even if one was saved as greyscale or with an alpha channel.
- `.resize((IMG_SIZE, IMG_SIZE))` guarantees every background is exactly the size the model expects.
- We load all 286 into memory **once**. At 256x256x3 bytes each that's about 56 MB, which is fine. In the earlier TensorFlow notebook, re-opening files for every sample was one of the main reasons training took an hour.

**Think: why lots of different backgrounds rather than one?**
With one background, the model can cheat: "find whatever is different from the usual picture". It never has to learn what Wally looks like. With 286 backgrounds it has to learn Wally's features (red and white stripes, bobble hat, glasses). One background also means it would almost certainly fail on any new background.

### 1.2 Loading the Waldos

`.convert('RGBA')` guarantees 4 channels. The **alpha channel** is the transparency of each pixel: 0 is fully see-through, 255 is solid. We print each Waldo's size so students can see they differ (about 25x50 for the standing ones and 50x50 for the square one). Different sizes mean the box size changes from image to image, which is why the model predicts all four corners rather than just a position.

### 1.3 Augmentation

```python
waldo = waldo.rotate(angle, expand = True, resample = Image.BICUBIC)
```

- `angle` is between -10° and +10°.
- `expand=True` enlarges the canvas so the rotated Wally isn't clipped. The new corners are transparent.
- `BICUBIC` gives a smoother rotated image than the default nearest-neighbour.

```python
rgb = waldo.convert('RGB'); alpha = waldo.getchannel('A')
rgb = ImageEnhance.Brightness(rgb).enhance(random.uniform(0.8, 1.2))
```

We split off the alpha channel first so that brightness and colour changes don't touch the transparency. `enhance(0.8)` gives 80% brightness and `enhance(1.2)` gives 120%.

```python
pixels = np.array(rgb).astype(np.float32)
shift = np.random.uniform(0.9, 1.1, size = 3)
pixels = np.clip(pixels * shift, 0, 255).astype(np.uint8)
```

This is the NumPy part. `pixels` has shape `(height, width, 3)`. `shift` has shape `(3,)`, so **broadcasting** multiplies every pixel's R by `shift[0]`, G by `shift[1]` and B by `shift[2]`. We convert to float first because `uint8` maths wraps around (250 x 1.1 would overflow), then clip back to 0–255.

`putalpha(alpha)` reattaches the transparency.

**Theory: why augment?** Data augmentation makes the training set more varied without new data. It teaches the model that these changes don't matter, so it learns the *idea* of Wally rather than 5 exact pixel patterns. That's the defence against overfitting to 5 sprites.

**Think: why only a small rotation? What would a 90° rotation do to the box?**
Real Wallys are upright, so big rotations would teach the model things it'll never need. Also, the box is axis-aligned: a rotated tall-thin Wally needs a bigger, squarer box that includes lots of background, which makes the target less precise. At 90° the box would be wide and short, a completely different shape from anything at test time.

**Think: a bad augmentation?**
Changing his colours a lot, for example turning red stripes green. Red and white stripes are Wally's most distinctive feature. Strong hue changes would force the model to ignore colour, throwing away the best clue. Flipping him upside down is another bad one. A horizontal mirror flip would be fine.

### 1.4 Pasting Wally and the bounding box

```python
background = random.choice(backgrounds).copy()
```

`.copy()` matters: `paste` modifies the image in place, so without it we'd draw Wally permanently onto the stored background, and every later image would contain old Wallys too. That's a good bug to mention.

```python
x = random.randint(0, IMG_SIZE - w); y = random.randint(0, IMG_SIZE - h)
background.paste(waldo, (x, y), mask = waldo)
```

`(x, y)` is the top-left corner of the pasted picture. `mask=waldo` uses Waldo's alpha channel to decide which pixels get copied, so only solid pixels replace the background.

```python
left, top, right, bottom = waldo.getchannel('A').getbbox()
box = (x + left, y + top, x + right, y + bottom)
```

`getbbox()` returns the smallest rectangle around the non-zero (non-transparent) pixels **within Waldo's own picture**. After rotation there are transparent corners, so the tight box is smaller than the picture. Adding `x` and `y` moves it to where we pasted him.

**Box format `(x1, y1, x2, y2)`:** the top-left and bottom-right corners, in pixels, with **y measured down from the top**. This is what `torchvision.ops.box_iou` expects, so we use it everywhere and avoid conversions.

Note that `getbbox` gives `right` and `bottom` as *exclusive* edges (one past the last pixel). That's the convention PIL and torchvision both use, so widths are simply `x2 - x1`.

### 1.5 Drawing boxes

`patches.Rectangle((x, y), width, height, ...)` wants a corner plus width and height, not two corners, hence `x2 - x1` and `y2 - y1`. `facecolor='none'` makes it an outline. With `imshow`, matplotlib already puts y = 0 at the top, so the box lines up with the image without any flipping.

**Think: why does x only go up to `IMG_SIZE - w`?**
`x` is Wally's **left** edge. If `x` could be 256, he'd start at the right edge and be pasted off the image. The largest legal `x` puts his right edge exactly at the image edge: `256 - w`.

### 1.6 Generating 5000 images and splitting 80/20

```python
train_ids, test_ids = random_split(range(NUM_IMAGES), [NUM_TRAIN, NUM_TEST],
                                   generator = torch.Generator().manual_seed(SEED))
```

`random_split` shuffles the indices 0–4999 and deals out 4000 and 1000. It's normally used on a `Dataset`, but it works on anything with a length, including `range`. Its own seeded `generator` makes the split repeatable even if other random calls change. We make `test_ids` a `set` because `i in test_ids` on a set is instant, while on a list it would scan 1000 items, 5000 times.

Then we generate each image and save it to the train or test folder. We write one `labels.csv` per folder with columns `filename, x1, y1, x2, y2`, the same layout as most real detection datasets.

**Think: why must test images never be used for training?**
The test set estimates how the model does on images it has **never seen**. If it trained on them, it could memorise their answers and the score would be falsely high. That's like giving students the exam answers to revise from.

**Think: can a background appear in both sets? Does it matter?**
Yes. Every image picks from the same 286 backgrounds, so nearly every background appears in both sets (5000 images ÷ 286 ≈ 17 uses each). It's a fair test of **new positions and new augmented Wallys**, but **not** of new backgrounds. If you want to measure "performance on unseen backgrounds" (from your original plan), split the **backgrounds** first, say 230 for training and 56 for testing, then generate each set from its own backgrounds. That's a nice extension for strong students. Expect the IoU to drop a little.

---

## Step 2: Feeding images to PyTorch

### Theory: Dataset and DataLoader

- A **`Dataset`** is anything with `__len__` (how many examples) and `__getitem__(i)` (give me example i as tensors). PyTorch doesn't care how you store the data.
- A **`DataLoader`** wraps a `Dataset` and handles batching, shuffling and (optionally) loading in parallel worker processes.

Separating them means the same training loop works for any data.

### 2.1 `WaldoDataset`

`__init__` reads `labels.csv` once into a list of `(filename, [x1, y1, x2, y2])`. Images are **not** loaded here, only when asked for. That's the usual pattern for big datasets that don't fit in memory.

`__getitem__`:

```python
image = torch.tensor(np.array(image), dtype = torch.float32).permute(2, 0, 1) / 255
```

- `np.array(image)` gives shape `(256, 256, 3)`: height, width, channels (**channels last**, how PIL, NumPy and matplotlib store images).
- `.permute(2, 0, 1)` reorders to `(3, 256, 256)`: **channels first**, which is what `nn.Conv2d` expects.
- `/ 255` scales pixels to 0–1. Neural networks train best when inputs are small and of similar size. Raw 0–255 values make the first layer's outputs huge, so training is unstable.

```python
box = torch.tensor(box, dtype = torch.float32) / IMG_SIZE
```

Box in pixels → fraction of the image (0–1). **This is the single most important fix from the TensorFlow version**, where box targets in raw pixels made the loss start around 20,000. A freshly created network outputs numbers near 0. With pixel targets, its first guesses are off by about 128 on average, the squared errors are around 15,000, and the gradients are enormous, so the first updates wreck the weights. With 0–1 targets the starting loss is around 0.3 and learning is smooth.

### 2.2 The same 1000 images every epoch

```python
train_subset, _ = random_split(train_dataset, [1000, 3000], generator = ...)
```

We pick 1000 of the 4000 training images once, with a fixed seed. Every epoch then reuses exactly those 1000. The other 3000 sit unused unless a student sets `TRAIN_IMAGES_PER_EPOCH = 4000`.

### 2.3 DataLoader

`shuffle=True` on training reorders the 1000 images every epoch, so batches are mixed differently each time. Without it, the model would see the same batches in the same order, and the gradient steps would be correlated. No shuffle for testing, since order doesn't affect the score, and a fixed order means index `i` in our predictions matches `test_dataset[i]`. **Step 7 relies on that**, so don't let students switch it on.

The printed shapes, `(32, 3, 256, 256)` and `(32, 4)`, are worth pausing on. The first dimension is always the batch.

**Think: 1000 / 32 = 31.25, so why 32 batches?**
31 full batches of 32 (992 images), then one last batch with the remaining 8. `DataLoader` keeps the short final batch by default (`drop_last=False`).

**Think: how many weight updates in 10 epochs?**
One per batch: 32 x 10 = **320 updates**.

### Why batches of 32

- Batch of 1: very noisy gradient (based on one image), and slow because the GPU does one image at a time.
- Whole dataset: smooth gradient but only one update per epoch, and it might not fit in memory.
- 32 is a common middle ground: enough images for a decent gradient estimate, and many updates per epoch.

---

## Step 3: Building the CNN from scratch

### Theory recap: why a CNN and not a plain network?

A plain fully connected layer from a 256x256x3 image to just 64 neurons needs 196,608 x 64 = **12.6 million** weights, and it learns "Wally at position (40, 80)" separately from "Wally at (41, 80)". A convolution layer instead slides the same small filter across the whole image:

- **Weight sharing:** a 3x3 filter has 27 weights (x 3 input channels), no matter how big the image is.
- **Translation equivariance:** a stripe detector that works in one place works everywhere, because it's the same filter.
- **Local patterns first:** early layers see small patches (edges, colours). After each pooling, later layers effectively see larger areas (stripes, a face, a whole Wally).

### The layers

```python
nn.Conv2d(3, 16, kernel_size = 3, padding = 1)
```

16 filters, each 3x3 across all 3 input channels. Output: 16 **feature maps**, one per filter, each showing "how strongly my pattern appears at each position". `padding=1` adds a 1-pixel border of zeros so a 3x3 filter's output is the same width and height as its input (256 → 256).

`nn.ReLU()`: `max(0, x)`. This **non-linearity** is essential: two linear layers in a row are equivalent to one linear layer, so without ReLU depth would add nothing.

`nn.MaxPool2d(2)`: for each 2x2 patch, keep the maximum. This halves width and height (256 → 128 → 64 → 32 → 16; with an odd size it rounds down). It:
- cuts the computation for later layers by 4x,
- makes each later neuron "see" a bigger area of the original image (its **receptive field**),
- gives a little tolerance to small shifts.

### The size table

| After | Shape | Parameters |
|---|---|---|
| input | 3 x 256 x 256 | |
| Conv 3→16, pool | 16 x 128 x 128 | 3·3·3·16 + 16 = **448** |
| Conv 16→32, pool | 32 x 64 x 64 | 3·3·16·32 + 32 = **4,640** |
| Conv 32→64, pool | 64 x 32 x 32 | 3·3·32·64 + 64 = **18,496** |
| Conv 64→64, pool | 64 x 16 x 16 | 3·3·64·64 + 64 = **36,928** |
| Score: 1x1 conv 64→1 | 1 x 16 x 16 | 64 + 1 = **65** |
| Size: Linear 64→2 | 2 | 64·2 + 2 = **130** |
| **Total** | | **60,707** |

### `FEATURE_SIZE = IMG_SIZE // 16`

Four MaxPools each halve the size, so the final grid is the image size divided by 2⁴ = 16 (rounded down). The heatmap is `FEATURE_SIZE x FEATURE_SIZE`, and `square_centres` needs to know how many squares there are. Working it out from `IMG_SIZE` means changing the image size can't silently break the model.

That exact bug happened in the first draft of this notebook, which assumed 250x250 images and hard-coded a 15x15 grid. The error, `mat1 and mat2 shapes cannot be multiplied (32x16384 and 14400x64)`, is a good one to show students: 32 is the batch size, 16,384 (64 x 16 x 16) is what came out of the conv layers, and 14,400 (64 x 15 x 15) is what the Linear layer expected.

### Why not just Flatten → Linear?

Basically: 
The first model had to learn Wally separately for every position, so it gave up and guessed the centre. The second asks every patch the same ‘is this Wally?’ question and takes the hottest spot, so it only has to learn what Wally looks like once.

The first approach (flatten and regress):
After the conv layers, we had a 16×16 grid of “what’s here” descriptions. Flatten tipped all of them into one long list, and a single layer had to read that list and output four coordinates. That layer had a separate set of weights for every square. So it had to learn “Wally in the top-left means these numbers” and “Wally in the bottom-right means those numbers” as unrelated facts, one square at a time, and 1,000 pictures weren’t nearly enough for that. It found a shortcut instead. Wally could be anywhere, so the answer that’s least wrong on average is the middle of the picture, and it guessed the middle every time. The loss stopped improving and IoU was 0.03.

The second approach (heatmap):
Instead of one big question, we ask every square the same small question: “how Wally-like are you?” One shared rule scores all 256 squares, the same way a single filter slides over the whole image. If it learns to recognise Wally in one corner, it recognises him in every corner for free. The scores become a heatmap that adds up to 1. The centre is then just the average square position, weighted by those scores, so it lands where the heat is. In short, the model only has to learn what Wally looks like, and the heatmap works out where he is.

### The heatmap head, step by step

```python
self.score = nn.Conv2d(64, 1, kernel_size = 1)
```

A **1x1 convolution** is a tiny Linear layer (64 inputs → 1 output) applied **separately to every square** of the grid, with the same weights everywhere. It gives each square a score: *how Wally-like are this square's 64 features?* It has 65 weights, and every one of the 1000 training images teaches it, whichever square Wally is in. That's the sample efficiency the flatten head lacked.

```python
scores = self.score(features).flatten(start_dim = 1)     # (B, 256)
probs = scores.softmax(dim = 1)
```

`flatten(start_dim=1)` turns each image's 1x16x16 scores into 256 numbers, keeping the batch dimension. **Softmax** turns them into probabilities: `exp(score) / sum of exp(all scores)`. They are all positive and add up to 1, so they're the model's belief about which square Wally is in. `.view(-1, 16, 16)` puts them back into a grid: the **heatmap**.

```python
self.register_buffer('square_centres', (torch.arange(FEATURE_SIZE) + 0.5) / FEATURE_SIZE)
```

The centre of each square as a fraction of the image: square 0 covers 0–1/16, so its centre is 0.5/16 ≈ 0.03, and square 15's centre is 15.5/16 ≈ 0.97. `register_buffer` stores this tensor inside the model so `.to(device)` moves it to the GPU too, but it's **not a weight**. The optimiser never changes it.

```python
centre_x = (heatmap.sum(dim = 1) * self.square_centres).sum(dim = 1)
centre_y = (heatmap.sum(dim = 2) * self.square_centres).sum(dim = 1)
```

The heatmap has shape `(B, rows, columns)`. Summing over `dim=1` (the rows) leaves the total probability of each **column**: "how likely is Wally to be in column j?". Then the **expected value** is Σ P(column j) x centre of column j. That's the probability-weighted average x position. Summing over `dim=2` (the columns) gives the same for rows, so y.

This trick is called **soft-argmax**. A hard argmax ("pick the brightest square") has no gradient: nudging a score slightly doesn't change which square is biggest, so backprop gets nothing. The weighted average changes **smoothly** as scores change, so the loss can flow back through it into the score layer and the conv layers. It also gives positions **between** square centres when the model spreads probability over neighbours, so it's more precise than 16-pixel steps.

```python
width, height = self.size(features.amax(dim = (2, 3))).unbind(dim = 1)
```

`amax(dim=(2, 3))` takes, for each of the 64 feature maps, its biggest value anywhere in the grid. That's **global max pooling**, which gives 64 numbers per image that say *what* was found but not *where*. A Linear layer turns them into width and height. Position doesn't matter for size (a square Waldo is square wherever he stands), so throwing position away here is fine, and it's the same "what vs where" idea from the TensorFlow notebook. `unbind(dim=1)` splits the `(B, 2)` result into two `(B,)` tensors.

```python
return torch.stack([centre_x - width / 2, centre_y - height / 2,
                    centre_x + width / 2, centre_y + height / 2], dim = 1)
```

This converts centre and size to corners, `(x1, y1, x2, y2)`, shape `(B, 4)`, all as fractions of the image. The output format is identical to the flatten version, so the loss, training loop, IoU and plotting don't change at all.

### `heatmap()` and `forward()`

`forward` defines what happens to a batch. You never call it directly: `model(images)` calls it for you, plus PyTorch's bookkeeping. I split the first half into a separate `heatmap()` method so Step 3's shape check and Step 7's heatmap plot can look inside the model.

### Link to YOLO

This head is a one-object version of the idea behind YOLO, from the single-obj-detection workshop: **cut the image into a grid, and let every grid cell make a prediction with shared weights.** YOLO predicts a confidence and a box for each cell. Ours predicts a score for each cell and turns the scores into one box. If students have seen YOLO, this is the bridge.

### Shape check

Pushing a fake batch of zeros through shows features `(2, 64, 16, 16)`, heatmap `(2, 16, 16)` adding up to 1, and output `(2, 4)`. If someone changes the architecture and the sizes don't match, it fails here, before any training time is wasted.

**Think: before training, where will the predicted centre be?**
The scores start out almost equal, so softmax gives each square about 1/256 and the weighted average lands at the **middle** of the image (0.5, 0.5). That's sensible: if you have no idea where Wally is, the middle minimises your average squared error. It's also what the broken flatten model got stuck predicting. During training you can see the moment the model "gets it": the loss drops suddenly (usually somewhere between epochs 3 and 7) as the heatmap goes from flat to a sharp spot.

**Think: about 60,000 weights, and 448 parameters looking at 65,536 pixels?**
Weight sharing. The same 16 filters (each 3x3x3) are reused at every one of the 65,536 positions. A fully connected layer would need separate weights for every pixel and neuron: 196,608 inputs x 16 outputs = 3.1 million weights, and it would have to learn every pattern separately at every position. The heatmap head extends the same idea: 65 weights, shared across all 256 squares.

**Think: why softmax instead of picking the highest score?**
"Pick the biggest" (argmax) isn't differentiable. A tiny change to a score almost never changes which square wins, so the gradient is zero almost everywhere and the network can't learn from it. Softmax plus a weighted average is smooth, so every score gets a gradient saying "be higher if you're near Wally, lower if not".

**Think: why is `amax` fine for size but not position?**
Max pooling over the whole grid keeps *how strongly* each feature fired but forgets *where*. Size doesn't depend on where Wally is, so nothing useful is lost. Position is exactly the thing that's lost, so it has to come from the heatmap, which keeps the grid.

**Try it: remove one MaxPool.**
The grid becomes 32x32 (set `FEATURE_SIZE = IMG_SIZE // 8`). The number of weights barely changes, because the head is shared across squares, but the last conv layer does 4x the work, so it's slower. The heatmap is finer (8-pixel squares), but each square sees a smaller area. Whether the boxes improve is a good experiment.

**Try it: put the Flatten head back.**
This reproduces the failure above. Watch the loss stall at about 0.058 and every red box sit in the middle. It's a memorable way to show why weight sharing matters.

---

## Step 4: The loss function and the optimiser

### MSE

`nn.MSELoss()` averages `(prediction − target)²` over every number in the batch: 32 images x 4 coordinates = 128 squared errors per batch.

- Squaring makes all errors positive, so they don't cancel out.
- It punishes big errors far more than small ones: being 0.2 off costs 4x as much as 0.1.
- It's smooth, so its gradient is simple: `2 × (prediction − target)`.

Because there's only one output (the box), there's only one loss. There's no combined loss to balance. If you add a "is Wally there?" head later, you'd add a binary cross-entropy term and need to weight the two, which is a nice link back to the theory.

### Adam

Gradient descent updates each weight by `−learning_rate × gradient`. Adam improves on that in two ways:

- **Momentum:** it keeps a running average of recent gradients, so it keeps moving in a consistent direction and smooths out noise from individual batches.
- **Per-weight step sizes:** it also tracks how large each weight's gradients usually are, and divides by that. Weights with consistently big gradients take smaller steps, and rarely-updated ones take bigger steps.

The result is that Adam usually works well with little tuning.

### Learning rate 0.001

This is Adam's usual default. With the heatmap head, I compared 0.0005 and 0.001. Both learned, but at 0.0005 the "click" (the sudden drop in loss) came late, around epoch 7–9, which is too close to the end of 10 epochs for comfort. At 0.001 it came around epoch 4–6, and all 3 seeds I tried ended with 97–98% of test boxes above IoU 0.5.

**Think: if the model is off by 0.1 on every coordinate, what's the MSE? How many pixels?**
0.1² = **0.01**. In pixels, 0.1 x 256 ≈ **26 pixels**, which is a whole Wally-width.

**Think: why might a learning rate that's too large get stuck?**
Big steps overshoot the minimum and bounce around. Worse, one big step can push many ReLU neurons into always outputting 0 ("dead ReLUs"). Their gradient is then 0 too, so they can never recover. If enough die, the heatmap stays flat, the network outputs the middle box for every image, and the loss stops improving.

---

## Step 5: Training the model

### The five lines that are deep learning

```python
predictions = model(images)                  # 1. forward pass
loss = loss_function(predictions, boxes)     # 2. how wrong are we?
optimizer.zero_grad()                        # 3. clear old gradients
loss.backward()                              # 4. backpropagation
optimizer.step()                             # 5. update the weights
```

1. **Forward pass:** PyTorch runs the batch through every layer and quietly records each operation in a **computation graph**.
2. **Loss:** a single number (a 0-dimensional tensor) at the end of that graph.
3. **`zero_grad()`:** every weight has a `.grad` field. PyTorch **adds** each new gradient to it rather than replacing it. That's useful for advanced tricks, but normally we must reset it to zero before each batch.
4. **`backward()`:** walks the graph backwards using the chain rule, filling in `.grad` for every weight. That's "how much would the loss change if I nudged this weight?".
5. **`step()`:** Adam reads every `.grad` and updates every weight.

Steps 3 and 4 can go in either order relative to step 1. What matters is that `zero_grad` happens before `backward` on each batch.

### Other details in the loop

- `images.to(device)`, `boxes.to(device)`: the data must be on the same device as the model (GPU or CPU).
- `running_loss += loss.item() * len(images)`: `.item()` turns the 1-element tensor into a plain Python number (and detaches it from the graph, so we don't keep every graph in memory). We multiply by the batch size because the last batch is smaller. Dividing the total by 1000 then gives the true average per image.
- `model.train()` / `model.eval()`: switch layers like Dropout and BatchNorm between modes. Our model has neither, so it changes nothing here, but forgetting it in a model that has them is a classic bug. That's why it's a good habit.
- `torch.no_grad()` in `test_loss`: we're not going to call `backward`, so there's no need to record the graph. It's faster and uses less memory.

### Why check the test loss every epoch

It lets students **see** generalisation. Training loss only tells you the model fits the training images. Test loss tells you it works on new ones. Strictly, a set used to make decisions during training should be called a **validation** set, and the test set kept for the very end. The single-obj-detection notebook explains that. Here we only *watch* the test loss and don't make any decisions with it, so it doesn't leak.

### Learning curves

Expect a distinctive shape: **flat for the first 2–5 epochs at about 0.058, then a sudden cliff**, then a slow flattening near zero. The flat part is the model predicting the middle of the image while the heatmap is still blurry. The cliff is the moment the score layer locks onto Wally's features. If a student's run is still flat at epoch 10, they've hit an unlucky start: rerun the model cell and training (a new model gets new random weights). In my run, train and test loss stayed close, so with this model 1000 images didn't overfit.

**Think: what if we delete `optimizer.zero_grad()`?**
Gradients pile up across batches: after 10 batches, `.grad` is the sum of 10 batches' gradients. The steps grow larger and larger and point in a stale direction, so the loss jumps around, often explodes or gets stuck. It's a great live demo.

**Think: gap between training and test loss?**
Test loss much higher than train means **overfitting**: the model is memorising the 1000 training images (including their exact backgrounds and positions) rather than learning Wally. Fixes: more data (`TRAIN_IMAGES_PER_EPOCH = 4000`), more augmentation, or a smaller model. If both are similar, it's generalising well. If both are high, it's **underfitting**: train longer or use a bigger model.

**Think: loss 0.0002, is that good?**
√0.0002 ≈ 0.014, and 0.014 x 256 ≈ **3.6 pixels** typical error per coordinate. Wally is about 25 pixels wide, so that's very good. Compare with the stuck value of about 0.058 in epochs 1–2: √0.058 x 256 ≈ 62 pixels, which is "guessing the middle". This shows students why a loss value on its own is hard to interpret, which motivates IoU in Step 6.

---

## Step 6: How good are the boxes? IoU

### Theory

$$\text{IoU} = \frac{|A \cap B|}{|A \cup B|} = \frac{\text{overlap}}{\text{area}_A + \text{area}_B - \text{overlap}}$$

- It doesn't depend on scale: being 10 pixels off on a huge box is fine, but on a tiny box it's terrible, and IoU reflects that while MSE doesn't.
- It punishes both wrong position and wrong size.
- **IoU > 0.5** is the classic threshold for a correct detection (from the PASCAL VOC benchmark). Stricter benchmarks average over thresholds from 0.5 to 0.95.

### `box_iou` and the diagonal

`box_iou(A, B)` with `A` of shape `(N, 4)` and `B` of shape `(M, 4)` returns an **N x M** table: the IoU of every A box with every B box. That's designed for detectors with many boxes. We have exactly one prediction per image and want it compared only with its own true box: entries `(0,0), (1,1), …`, the **diagonal**. For 1000 test images it computes a 1000 x 1000 table (a million numbers) and keeps 1000. That's wasteful but quick, and simpler to explain than the alternative.

We multiply by `IMG_SIZE` to go back to pixels. IoU is the same either way (it's a ratio), but pixel boxes are what we draw in Step 7.

One subtle point: if the model ever predicted `x2 < x1` (a "negative-width" box), `box_iou` would give a meaningless result. With a well-trained model this doesn't happen, but early in training it can, which is one reason IoU is measured after training.

**Think: the hand example?**
Box A is 0–10 x 0–10 (area 100). Box B is 5–15 x 0–10 (area 100). Overlap is 5–10 x 0–10 = **50**. Union is 100 + 100 − 50 = **150**. IoU = 50/150 = **0.33**. So shifting a box by **half** its width already drops IoU to a third, which is below the 0.5 threshold.

**Think: Wally is about 25x50. How far sideways before IoU < 0.5?**
Shift by d pixels: overlap = (25 − d) x 50, union = 2 x 1250 − overlap. Setting IoU = 0.5 gives (25 − d) x 50 = 0.5 x (2500 − (25 − d) x 50), so 1.5 x (25 − d) x 50 = 1250, so 25 − d ≈ 16.7, so **d ≈ 8 pixels**. Being off by only 8 pixels sideways (a third of Wally's width) already fails the 0.5 test. That shows students how demanding IoU is for small objects.

### Histogram

It shows the spread: most images high, plus a tail of failures. The mean IoU alone hides that tail.

---

## Step 7: Testing on unseen images

### Single examples

For 8 random test images, we draw the green (true) and red (predicted) boxes, and give the IoU and the **centre error**: the straight-line distance in pixels between the two box centres, √((tx − px)² + (ty − py)²). That's the "error for each test" from your plan, in units students can picture.

`image.permute(1, 2, 0)` converts the tensor back to channels-last for matplotlib, the reverse of Step 2.

`true_boxes[i]` and `pred_boxes[i]` line up with `test_dataset[i]` **only because the test loader doesn't shuffle**.

### Where is the model looking?

This cell draws the model's 16x16 heatmap over 4 test images.

- `image.unsqueeze(0)` adds a batch dimension, turning `(3, 256, 256)` into `(1, 3, 256, 256)`, because the model always expects a batch.
- `extent=(0, IMG_SIZE, IMG_SIZE, 0)` stretches the 16x16 heatmap to cover the 256x256 picture. The last two numbers are bottom then top, which keeps y pointing down like the image.
- `alpha=(heat / heat.max()).numpy()` gives every square its own transparency: the most likely square is solid, and unlikely squares are invisible. That way the picture isn't darkened. (An alpha array needs matplotlib 3.3 or newer, which Kaggle has.)
- `cmap='autumn'` colours squares from red (less likely) to yellow (most likely).
- The title shows the probability of the single most likely square.

This is the most "explainable" plot in the notebook: students can literally see what the network thinks.

**Think: one bright square or spread out?**
A single bright square means the model is confident. A spread over 2–4 neighbouring squares is normal when Wally straddles a square boundary, and the weighted average then puts the centre between them, which is correct. Bright spots in **separate** places mean the model is torn between two candidates, for example Wally and a stripy distractor. The centre then lands between them, on neither, which usually gives a bad box. You'll often see this in the hardest images below.

### Hardest images

`ious.argsort()[:4]` gives the indices of the 4 lowest IoUs. Looking at failures is how you learn what to fix.

**Think: why did the model get these wrong?**
Usually one of these:
- Wally is on a busy red-and-white background (stripes everywhere), so he's camouflaged.
- He's near the image edge, where padding zeros change what the filters see and there were fewer training examples.
- It's the square 50x50 Waldo, which is rarer (1 in 5) and a different shape, so the box size is wrong even when the position is right.
- A dark or colour-shifted augmentation made him blend in.

**Think: no Wally? Two Wallys?**
The model **always** outputs exactly one box, because softmax always adds up to 1. With no Wally, it'll confidently draw a box somewhere, often on something red and stripy. It has no way to say "not here", because we never gave it a "confidence" output or images without Wally. With two Wallys, it'll typically pick one, or put a box between them (the average of two answers is what MSE rewards when it's unsure). Real detectors like YOLO solve both problems by predicting many boxes, each with a confidence score. That's a good bridge to the single-obj-detection workshop.

**Think: would it work on a real Where's Wally page?**
Probably badly, for three reasons:
- **Scale:** a real page is huge (thousands of pixels) and Wally is tiny. Squash it to 256x256 and Wally becomes a few pixels.
- **Domain gap:** real Wally is drawn into the scene, partly hidden behind people, at different sizes, in the same art style as everything around him. Pasted Wally has crisp edges and is never hidden.
- **Distractors:** real pages are full of red-and-white things deliberately placed to fool you.

To make it work you'd slide the model over 256x256 tiles of the page, and train with real examples or much harder synthetic ones (occlusion, scale changes, decoys).

---

## Results when I tested it

I ran the whole notebook end to end on a 4-core CPU with no GPU. I used stand-in data built to match yours: 286 backgrounds of 256x256 cut from a Where's Wally beach scene, and 5 Waldo sprites of 24–27 x 48–52 plus one 50x50. Your real images will differ, so expect slightly different numbers.

| | Result |
|---|---|
| Model size | 60,707 weights and biases |
| Generating 5000 images | 97 s (CPU) |
| Training, 10 epochs x 1000 images | 5.2 min on CPU, about 31 s per epoch including the test loss |
| Mean IoU on 1000 test images | **0.72** |
| Correct detections (IoU > 0.5) | **98.7%** |
| Final train / test loss | 0.0002 / 0.0002 |

The loss was stuck at about 0.058 for epochs 1–2, dropped in epoch 3, and was near its final value by epoch 5. With the same settings and 3 different seeds (in a quicker test script), the results were IoU 0.70–0.73 and 97–98% correct.

Even the 4 worst test images had IoU 0.41–0.44. Those boxes were a few pixels off, not in the wrong place.

**On a Kaggle GPU** I'd expect training to take well under a minute. Most of the time would then go on reading PNG files, not on the maths. I haven't been able to measure it.

**For comparison, the Flatten → Linear head** on the same data: the loss was stuck at about 0.058 for all 10 epochs, the mean IoU was 0.03 to 0.04, and under 1% of detections were correct. Zeki saw the same on Kaggle (IoU 0.03, 0.8%).

### How many images to generate

Only 1000 training images are used, so generating 5000 wastes about a minute. **1250 in total (1000 train, 250 test)** keeps the 80/20 split. 250 test images measure IoU to within about ±3%. Set `NUM_IMAGES = 1250`, `NUM_TRAIN = 1000`, `NUM_TEST = 250`.

---

## Decisions I made where your plan left a choice

- **Epochs of 1000 images.** Your plan says 4000 training images and "10 epochs of 1000 images each, each epoch uses the same training data". I read that as: pick 1000 of the 4000 once, and reuse them every epoch. The setting `TRAIN_IMAGES_PER_EPOCH` makes it a one-line change to use all 4000, and that's listed as a "Where next" experiment.
- **No classification head.** Your dataset only has Waldo, so there's nothing to classify. Adding a "Wally or no Wally?" head is listed as an extension.
- **Box format `(x1, y1, x2, y2)`, scaled to 0–1.** This matches `box_iou`, so there are no conversions. The model predicts corners directly.
- **Box from the alpha channel after rotation.** That gives a tight box around the visible Wally rather than around the rotated picture's transparent corners.
- **A heatmap head instead of Flatten → Linear.** The flatten head couldn't learn on 286 varied backgrounds (see Step 3). The heatmap head learned in every run I tried.
- **No BatchNorm, no Dropout, learning rate 0.001.** BatchNorm made the heatmap model worse in my tests (IoU 0.36 instead of about 0.7). These choices also keep the model simple to explain.
- **The notebook is complete (solved)**, like single-obj-detection.ipynb, with **Think** questions and **Try it** experiments. If you'd rather have blanks for students to fill in, the natural places are the five training-loop lines, the `permute`/`/255` line in the Dataset, and the `nn.Linear(64 * FEATURE_SIZE * FEATURE_SIZE, 64)` size.
