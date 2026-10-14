# Teacher's guide: Can you find Wally?

This guide goes with `sem1-workshop-5-todo.ipynb` (the student version) and `sem1-workshop-5-answers.ipynb` (every blank filled in). It follows the notebook step by step. For each step it covers what the code does, the theory behind it, the answer to every `# TODO`, the answer to every **Think** question, and what tends to go wrong.

---

## Contents

- [The big picture](#the-big-picture)
- [Running the session](#running-the-session)
- [Step 0: Setting up](#step-0-setting-up)
- [Step 1: Generating our own training data](#step-1-generating-our-own-training-data)
- [Step 2: Feeding images to PyTorch](#step-2-feeding-images-to-pytorch)
- [Step 3: Building our CNN from scratch](#step-3-building-our-cnn-from-scratch)
- [Step 4: The loss function and the optimiser](#step-4-the-loss-function-and-the-optimiser)
- [Step 5: Training the model](#step-5-training-the-model)
- [Step 6: How good are the boxes? IoU](#step-6-how-good-are-the-boxes-iou)
- [Step 7: Testing on unseen images](#step-7-testing-on-unseen-images)
- [What's next answers](#whats-next-answers)
- [Results from the Kaggle run](#results-from-the-kaggle-run)
- [All TODO answers on one page](#all-todo-answers-on-one-page)
- [Errors students will hit](#errors-students-will-hit)
- [Small fixes for the notebook text](#small-fixes-for-the-notebook-text)

---

## The big picture

The whole notebook is one idea: **a CNN can learn to output four numbers, the corners of a box around Wally, if we show it enough pictures where we already know the answer.**

```
286 backgrounds + 5 Waldos ──paste──▶ 5000 images + boxes ──split──▶ 4000 train / 1000 test
                                                                          │
                       WaldoDataset ──▶ DataLoader (batches of 32) ◀──────┘
                                              │
   image (3x256x256) ──▶ 4 conv blocks ──▶ 16x16 grid ──▶ heatmap + size ──▶ (x1, y1, x2, y2)
                                              │
                          MSE loss vs true box ──▶ backward ──▶ Adam step   (x 320)
                                              │
                           after training: IoU on the 1000 test images
```

There is **no classification** in this notebook. The model only answers "where is Wally?", never "is Wally here?". It always assumes there is exactly one Wally and always draws exactly one box. This comes up again in the Step 7 Think questions.

---

## Running the session

The notebook is meant for 45 minutes after the 30-minute talk. These are rough timings for a first-timer:

| Part | Time | Notes |
|---|---|---|
| Step 0 | 2 min | Make sure everyone has the **GPU** switched on before they start |
| Step 1 | 12 min | Most of the TODOs are here. Generating the images takes about 70 s |
| Step 2 | 7 min | Shapes and batches. Worth stopping everyone at the printed shapes |
| Step 3 | 8 min | Mostly reading. Two short TODOs |
| Steps 4 and 5 | 6 min | Training takes about 1.6 min on a GPU, a good time for the Think questions |
| Step 6 | 4 min | IoU by hand first, then the real numbers |
| Step 7 | 6 min | The fun part: pictures, heatmaps, worst cases |

If a student falls behind, the quickest catch-up is to copy the missing cells from `sem1-workshop-5-answers.ipynb`.

---

## Step 0: Setting up

### Imports

| Import | Why we need it |
|---|---|
| `os`, `csv` | Listing files, making folders, reading and writing `labels.csv` |
| `random`, `time` | Random choices and positions, timing |
| `numpy` | The colour shift in augmentation, turning images into arrays |
| `matplotlib.pyplot`, `matplotlib.patches` | Showing images, drawing boxes and learning curves |
| `PIL.Image`, `PIL.ImageEnhance` | Opening, rotating and pasting images, changing brightness |
| `torch`, `torch.nn` | Tensors, layers, loss, optimiser |
| `torch.utils.data` | `Dataset`, `DataLoader`, `random_split` |
| `torchvision.ops.box_iou` | The IoU score |

`shutil` is imported later, in 1.6, where it's used.

### 0.1 Settings

Every number a student might want to change is in one cell:

- `background_directory`, `waldo_directory`: the Kaggle input folders. Kaggle's input folders are read-only, so...
- `output_directory = '/kaggle/working/generated images'`: ...generated images go in `/kaggle/working`, the one folder Kaggle lets us write to.
- `image_size = 256`: all 286 backgrounds are 256x256. The model works out its grid size from this.
- `images_num`, `train_num`, `test_num`: 5000, 4000, 1000 (an 80/20 split).
- `images_per_epoch = 1000`, `epochs_num = 10`: each epoch reuses the same 1000 training images (see 2.2).
- `model_batch_size = 32`, `learning_rate = 1e-3`: see Steps 2 and 4.

### 0.2 Seed and device

```python
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
```

Three seeds, because three separate random number generators are in use:

- Python's `random`: background choice, Wally choice, position, angle, brightness and colour shift.
- NumPy's: not actually used for random numbers in the final code (the colour shift uses `random.uniform`), but it's a good habit to seed it.
- PyTorch's: starting weights and DataLoader shuffling.

`device` is the GPU (`cuda`) if Kaggle has one switched on, otherwise the CPU. The model and every batch must be on the same device, which is why you'll see `.to(device)` in the training loop. If a student sees `Using device: cpu`, they forgot to switch on the GPU: training will take 10+ minutes instead of under 2.

---

## Step 1: Generating our own training data

### Theory: synthetic data

To learn "where is Wally?", the model needs examples **with answers** (labels). Hand-labelling means drawing thousands of boxes. Synthetic data flips this round: we **make** each image by pasting Wally where we choose, so we know the box exactly. It's free, perfectly accurate and unlimited.

The catch is the **domain gap**: the model learns what *our* pasted images look like. A real Where's Wally page has Wally drawn into the scene, partly hidden, at different sizes. Variety in the synthetic data (286 backgrounds, augmented Wallys) narrows that gap but doesn't close it. This comes back in the last Think question.

### 1.1 Loading the backgrounds

The first loop builds `background_paths`: every file in the folder joined onto the folder path, then `sorted`. `os.listdir` doesn't promise any order, so sorting makes everyone's list identical, which the seed needs.

The second loop opens each file, `.convert('RGB')` makes sure it has exactly 3 colour channels, and `.resize((image_size, image_size))` guarantees 256x256. All 286 are kept in memory **once**, about 56 MB. Re-opening files for every one of 5000 images would be much slower.

**TODO answer:** `backgrounds.append(image)`

**Think: why lots of different backgrounds rather than one?**
With one background, the model can cheat: it just has to find "whatever is different from the usual picture". It never needs to learn what Wally looks like, and it would fail on any new background. With 286 backgrounds it has to learn Wally's own features: red and white stripes, bobble hat, glasses.

### 1.2 Loading the Waldos

Same loop as the backgrounds, but `.convert('RGBA')` keeps the 4th channel, **alpha**: how see-through each pixel is (0 = fully transparent, 255 = solid). That's what lets us paste Wally without the white rectangle around him.

The printout shows the five Waldos are different sizes: 25x50, 22x50, 51x51, 42x51 and 50x50. Different sizes mean the box size changes from picture to picture, which is why the model has to predict width and height as well as position.

**TODO answer:** `.convert('RGBA')`

### 1.3 Augmentation: tweaking Wally each time

`augment_waldo` makes three changes:

1. **Rotation:** `wally.rotate(angle, expand=True)` with `angle` between -10° and +10°. `expand=True` makes the canvas bigger so his hat and feet aren't cut off. The new corners are transparent.
2. **Brightness:** the alpha channel is split off first, so brightness and colour changes don't touch the transparency. `ImageEnhance.Brightness(colours).enhance(brightness)` with `brightness` between 0.8 (80%, darker) and 1.2 (120%, brighter).
3. **Colour shift:** `np.array(colours)` gives a `(height, width, 3)` array. Each of the 3 colour channels is multiplied by its own random number between 0.9 and 1.1. We convert to `float` first because image numbers (`uint8`) wrap around: 250 x 1.1 would become 19, not 275. `np.clip` keeps everything between 0 and 255, and `putalpha(alpha)` puts the transparency back.

**Theory: why augment?** Augmentation makes the training set more varied without collecting new data. It teaches the model that these changes don't matter, so it learns the *idea* of Wally rather than 5 exact pixel patterns. It's a defence against overfitting.

**TODO answers:** `wally.rotate(angle, expand=True)` and `np.array(colours)`

**Think: why only a small rotation? What would a 90° rotation do to the box?**
Real Wallys are upright, so big rotations teach the model something it will never need. Also, our boxes are always upright rectangles. A tilted Wally needs a bigger box with background in the corners, which makes the target less precise. At 90°, a tall, thin Wally becomes short and wide: a completely different box shape from anything at test time.

**Think: an augmentation that would be a bad idea?**
Turning his red stripes green. Red and white stripes are Wally's most useful clue, so big colour changes force the model to ignore colour and throw that clue away. Flipping him upside down is another bad one. A left-right mirror flip would be fine.

### 1.4 Pasting Wally onto a background

`make_sample` builds one training picture:

1. `random.choice(backgrounds).copy()`: `.copy()` is important. `paste` changes the image it's called on, so without the copy we'd paste Wally permanently onto the stored background, and later pictures would contain old Wallys too. This makes a good "spot the bug" question.
2. `augment_waldo(random.choice(waldos))`, then read his width and height.
3. `x` and `y` are the top-left corner where his picture goes, chosen so he fits completely inside the image.
4. `background.paste(wally, (x, y), mask = wally)`: the mask uses his alpha channel, so only his solid pixels replace the background.
5. `wally.getchannel('A').getbbox()` finds the smallest box around his non-transparent pixels, **measured inside his own picture**. After rotation there are transparent corners, so this box is tighter than the picture. Adding `x` and `y` moves it to where we pasted him.

**Box format `(x1, y1, x2, y2)`:** top-left and bottom-right corners, in pixels, with **y counting down from the top**. That's what `box_iou` expects, so we use it everywhere.

**TODO answers:** `random.choice(backgrounds)`, `random.choice(waldos)`, `image_size - wally_width`, `image_size - wally_height`

**Think: why does `x` only go up to `image_size - wally_width`?**
`x` is Wally's **left** edge. If `x` could be 256, he'd start at the right edge and be pasted off the picture. The largest safe `x` puts his right edge exactly on the image edge: `256 - wally_width`.

### 1.5 Drawing boxes

`patches.Rectangle` wants a corner plus a width and height, not two corners, so `draw_box` works out `width = x2 - x1` and `height = y2 - y1`. `facecolor='none'` gives an outline only, and `plt.gca()` ("get current axes") adds it to whichever plot is active. Because `imshow` already puts y = 0 at the top, the box lines up with the picture without any flipping.

**TODO answers:** `width = x2 - x1`, `height = y2 - y1`

### 1.6 Making 5000 images and splitting 80/20

`shutil.rmtree` deletes the old output folder so images from a previous run don't get mixed in, and `os.makedirs` makes fresh `train images` and `test images` folders.

```python
train_ids, test_ids = random_split(range(images_num), [train_num, test_num], generator = torch.Generator().manual_seed(seed))
```

`random_split` shuffles the numbers 0 to 4999 and deals them into a pile of 4000 and a pile of 1000. It's normally used on a `Dataset`, but it works on anything with a length, including `range`. Its own seeded `generator` makes the split the same every time. `test_ids` becomes a `set` because `i in test_ids` is instant for a set but has to scan 1000 items in a list, 5000 times over.

The next cell generates each image and saves it into the train or test folder, recording `[file_name, x1, y1, x2, y2]`. This takes about **70 seconds**. Then `save_labels` writes one `labels.csv` per folder: the correct answers.

**TODO answers:** `[train_num, test_num]`; in the `else` branch `train_directory` and `train_labels`; `save_labels(train_directory, train_labels)`

**Think: why must the test images never be used for training?**
The test set tells us how the model does on pictures it has **never seen**. If it trained on them, it could memorise their answers and the score would look better than it really is, like revising from the exam paper.

**Extra question if a student asks: can the same background be in both sets?**
Yes. 5000 images from 286 backgrounds means each one is used about 17 times, so nearly every background appears in both sets. The test checks **new positions and new augmented Wallys**, not new backgrounds. To test unseen backgrounds you'd split the backgrounds first (say 230 for training and 56 for testing). That's a good extension for strong students, and the IoU would drop a little.

---

## Step 2: Feeding images to PyTorch

### Theory: Dataset and DataLoader

- A **`Dataset`** answers two questions: how many examples are there (`__len__`), and give me example number `i` (`__getitem__`). PyTorch doesn't care how the data is stored.
- A **`DataLoader`** wraps a Dataset and does the batching and shuffling.

Splitting the jobs this way means the same training loop works for any dataset.

### 2.1 `WaldoDataset`

`__init__` reads `labels.csv` once into `self.rows`, a list of `(file_name, box)`. `next(reader_object)` skips the header row, and `float(...)` is needed because a CSV stores everything as text. The images themselves are **not** loaded here, only when they're asked for. That's the usual pattern for datasets too big to fit in memory.

`__getitem__(i)`:

- `np.array(image)` gives shape `(256, 256, 3)`: height, width, channels. That's **channels last**, how PIL, NumPy and matplotlib store pictures.
- `.permute(2, 0, 1)` reorders it to `(3, 256, 256)`: **channels first**, which is what `nn.Conv2d` expects.
- `/ 255` scales pixels to 0–1.
- `/ image_size` scales the box to 0–1.

**Why scale the box?** A brand-new network outputs numbers near 0. If the targets were in pixels, its first guesses would be off by about 100, the squared errors would be about 10,000, and the first updates would wreck the weights. With 0–1 targets, learning is smooth. This was the biggest fix compared with the original TensorFlow version, where the loss started around 20,000.

**TODO answers:** `/ 255` and `/ image_size`

The next cell checks one example: shape `(3, 256, 256)`, pixels 0.0 to 1.0, and a box like `tensor([0.7578, 0.5273, 0.9297, 0.7266])`.

### 2.2 The same 1000 images every epoch

`random_split(train_dataset, [images_per_epoch, leftovers], ...)` picks 1000 of the 4000 training images once, with a fixed seed, and every epoch reuses those. `_` is Python's way of saying "I don't need this value", here the 3000 leftovers. Setting `images_per_epoch = 4000` uses them all (one of the What's next ideas).

**TODO answer:** `leftovers = len(train_dataset) - images_per_epoch`

### 2.3 DataLoader

`shuffle=True` for training mixes the 1000 images into different batches every epoch, so the model can't learn anything from the order. `shuffle=False` for testing, because order doesn't change the score. Step 7 also **relies** on it: prediction number `i` only lines up with `test_dataset[i]` because the test set isn't shuffled.

The printed shapes, `(32, 3, 256, 256)` and `(32, 4)`, are worth pausing on: the first number is always the batch.

**TODO answers:** `shuffle = True` (train), `shuffle = False` (test)

**Think: how many updates in 10 epochs?**
1000 / 32 = 31.25, so each epoch is 31 full batches plus one last batch of 8: **32 batches**. One update per batch, so 32 x 10 = **320 updates**.

**Why batches of 32?**
- Batch of 1: very noisy updates (based on one picture) and slow.
- All 1000 at once: smooth, but only one update per epoch, and it might not fit in memory.
- 32 is the usual middle ground: a decent average and plenty of updates.

---

## Step 3: Building our CNN from scratch

### Theory: why a CNN and not a normal network?

A fully connected layer from a 256x256x3 image to just 64 neurons needs 196,608 x 64 = **12.6 million** weights, and it treats "Wally at (40, 80)" and "Wally at (41, 80)" as unrelated things. A convolution slides the **same** small filter over the whole image instead:

- **Weight sharing:** a 3x3 filter over 3 colours has 27 weights, however big the image is.
- **Works anywhere:** a stripe detector that works in one place works everywhere, because it's the same filter.
- **Small to big:** early layers see small patches (edges, colours). After each pooling, later layers see bigger areas (stripes, a face, a whole Wally).

### Part 1: the feature extractor

```python
nn.Conv2d(3, 16, kernel_size = 3, padding = 1), nn.ReLU(), nn.MaxPool2d(2)
```

- **`Conv2d(3, 16, ...)`**: 16 filters, each 3x3 across all 3 colour channels, giving 16 **feature maps** ("how strongly does my pattern appear at each position?"). `padding=1` adds a border of zeros so the output stays 256x256 and pixels at the edge get a full turn.
- **`ReLU`**: negatives become 0. Without it, stacking layers would be no more powerful than a single layer.
- **`MaxPool2d(2)`**: keep the biggest number in each 2x2 square. Halves the width and height, cuts the work by 4x, and lets later layers see a bigger part of the picture.

| After | Shape | Weights and biases |
|---|---|---|
| input | 3 x 256 x 256 | |
| block 1: Conv 3→16, pool | 16 x 128 x 128 | 3·3·3·16 + 16 = **448** |
| block 2: Conv 16→32, pool | 32 x 64 x 64 | 3·3·16·32 + 32 = **4,640** |
| block 3: Conv 32→64, pool | 64 x 32 x 32 | 3·3·32·64 + 64 = **18,496** |
| block 4: Conv 64→64, pool | 64 x 16 x 16 | 3·3·64·64 + 64 = **36,928** |
| score: 1x1 conv 64→1 | 1 x 16 x 16 | 64 + 1 = **65** |
| size: Linear 64→2 | 2 | 64·2 + 2 = **130** |
| **Total** | | **60,707** (matches the printout) |

`feature_size = image_size // 16`: four poolings each halve the size, so the grid is the image size divided by 2⁴ = 16. Working it out from `image_size` means changing the image size can't silently break the model.

**TODO answers:** `nn.Conv2d(16, 32, ...)` and `nn.Conv2d(32, 64, ...)`. A block's input channels must equal the previous block's output channels. If a student gets this wrong, the error ends with something like `expected input[32, 16, 128, 128] to have 32 channels, but got 16 channels instead`, which is a nice error to read together.

### Why not just Flatten → Linear?

The simple head would flatten the 64x16x16 grid into one long list of 16,384 numbers and use a Linear layer to output 4 numbers. **It failed**: on Kaggle the loss stayed stuck at about 0.058 for all 10 epochs, and IoU was 0.03.

**What went wrong:** that Linear layer has a separate set of weights for every square of the grid. So it had to learn "Wally in the top-left means these numbers" and "Wally in the bottom-right means those numbers" as unrelated facts, one square at a time, and 1000 pictures weren't nearly enough. It found a shortcut instead. Wally could be anywhere, so the answer that's least wrong on average is the middle of the picture, and it guessed the middle every time.

**Why the heatmap works:** instead of one big question, we ask every square the same small question: "how Wally-like are you?" One shared rule scores all 256 squares, just like a filter sliding over an image. If it learns to recognise Wally in one corner, it recognises him in every corner for free. The scores become a heatmap, and the centre is wherever the heat is. The model only has to learn what Wally looks like; the heatmap works out where he is.

**One-line version for students:** *The first model had to learn Wally separately for every position, so it gave up and guessed the centre. The second asks every patch the same "is this Wally?" question and takes the hottest spot, so it only has to learn what Wally looks like once.*

### Part 2: the heatmap head, step by step

**`heatmap(x)`**

```python
features = self.features(x)                          # (B, 64, 16, 16)
scores = self.score(features).flatten(start_dim = 1)  # (B, 256)
probs = scores.softmax(dim = 1)                       # (B, 256), adds up to 1
heatmap = probs.view(-1, feature_size, feature_size)  # (B, 16, 16)
```

- `self.score` is a **1x1 convolution**: a tiny Linear layer (64 inputs → 1 output) applied separately to every square, with the same weights everywhere. It has only 65 weights, and every training picture teaches it, whichever square Wally is in.
- `flatten(start_dim = 1)` lays each image's 16x16 scores out as 256 numbers, keeping the batch dimension.
- **Softmax** turns scores into probabilities: all positive, adding up to 1. It's the model's belief about which square Wally is in.
- `.view(-1, 16, 16)` puts them back into a grid: the **heatmap**. `-1` means "work out the batch size yourself".

It returns the features too, because `forward` needs them for the size.

**`square_centres` and `register_buffer`**

`(torch.arange(feature_size) + 0.5) / feature_size` is the centre of each square as a fraction of the image: square 0 is centred at 0.5/16 ≈ 0.03, square 15 at 15.5/16 ≈ 0.97. `register_buffer` stores it inside the model so `.to(device)` moves it to the GPU too, but it's **not a weight**: the optimiser never changes it.

**`forward(x)`**

```python
column_probs = heatmap.sum(dim = 1)   # (B, 16)
row_probs = heatmap.sum(dim = 2)      # (B, 16)
center_x = (column_probs * self.square_centres).sum(dim=1)
center_y = (row_probs * self.square_centres).sum(dim=1)
```

The heatmap has shape `(B, rows, columns)`. Adding down the rows (`dim = 1`) leaves one number per **column**: how likely Wally is to be in that column. Multiplying by the column centres and adding up gives the **weighted average** x position. `dim = 2` does the same for rows, giving y.

Worked example for a 4x4 grid: if the columns add up to 0, 0, 0.2 and 0.8, with centres 0.125, 0.375, 0.625 and 0.875, then x = 0.2 x 0.625 + 0.8 x 0.875 ≈ 0.83.

This trick is called **soft-argmax**. Why not just pick the brightest square (argmax)? Because a tiny change to a score almost never changes which square wins, so there's no gradient and the network can't learn from it. The weighted average changes **smoothly**, so the loss can flow back into the score layer and the conv layers. It also allows positions *between* square centres, so it's more precise than 16-pixel steps.

```python
strongest = features.amax(dim = (2, 3))   # (B, 64)
size = self.size(strongest)                # (B, 2)
```

`amax(dim = (2, 3))` takes each of the 64 channels' biggest value anywhere in the grid. That keeps **what** was found and how strongly, but forgets **where**. A Linear layer turns those 64 numbers into width and height. Size doesn't depend on position, so forgetting position is fine here.

Finally, centre ± half the size gives the corners, and `torch.stack(..., dim=1)` packs them into `(B, 4)`: the same format as the labels, so the loss, training loop and IoU don't need to know anything about heatmaps.

**TODO answers:** `x2 = center_x + width / 2`, `y2 = center_y + height / 2`

### `model = WaldoFinder().to(device)`

`WaldoFinder()` builds all the layers with random starting weights. `model(images)` calls `forward` for you (never call `model.forward(images)` directly; `model(...)` also does PyTorch's bookkeeping). The parameter loop adds up `p.numel()` (number of elements) for every weight tensor: **60,707**.

### Shape check

A fake batch of zeros should give features `(2, 64, 16, 16)`, a heatmap `(2, 16, 16)` adding up to 1 (it prints 0.99999994, which is just rounding), and output `(2, 4)`. If someone changes the architecture wrongly, it fails here, before any training time is wasted.

**Extra question if a student asks: before training, where will the predicted centre be?**
The starting scores are all almost equal, so every square gets about 1/256 and the weighted average lands in the **middle** (0.5, 0.5). That's also exactly where the broken flatten model got stuck. During training you can see the moment it "gets it": the loss suddenly drops as the heatmap goes from flat to a sharp spot.

---

## Step 4: The loss function and the optimiser

### MSE

`nn.MSELoss()` averages `(predicted − true)²` over every number in the batch: 32 images x 4 corners = 128 squared errors.

- Squaring makes every error positive, so errors don't cancel out.
- It punishes big errors far more: being 0.2 off costs 4x as much as 0.1.
- It's smooth, so its gradient is simple.

There's only one output (the box), so only one loss. If you added an "is Wally there?" head, you'd add a cross-entropy loss and have to balance the two, which links back to classification.

### Adam

Plain gradient descent nudges each weight by `−learning_rate x gradient`. Adam improves on that in two ways:

- **Momentum:** it keeps a running average of recent gradients, so it keeps moving in a steady direction and smooths out noise from individual batches.
- **A step size per weight:** weights whose gradients are usually big take smaller steps, and rarely-updated ones take bigger steps.

So Adam usually works well without much tuning. `model.parameters()` hands Adam every weight it's allowed to change.

**TODO answer:** `lr = learning_rate`

**Think: off by 0.1 on every coordinate, what's the MSE? How many pixels?**
0.1² = **0.01**. In pixels, 0.1 x 256 ≈ **26 pixels**, about a whole Wally-width.

**Think: why might a learning rate that's too big get stuck?**
Big steps overshoot the bottom of the loss and bounce around. Worse, one big step can push lots of ReLU neurons into always outputting 0. Their gradient is then 0 too, so they can never recover ("dead ReLUs"). If enough die, the heatmap stays flat, the model outputs the middle box for every picture, and the loss stops improving.

---

## Step 5: Training the model

### `test_loss()`

- `model.eval()` switches to testing mode, and `torch.no_grad()` stops PyTorch recording the steps for backpropagation. It's faster and uses less memory, because we won't call `backward` here.
- `loss.item()` turns the 1-number tensor into a plain Python number.
- `loss` is the **average** over the batch, so `x len(images)` turns it back into a total. The last batch is smaller, so dividing the grand total by `len(test_dataset)` gives the true average per image.

### The five lines that are deep learning

```python
predictions = model(images)                  # 1. forward pass
loss = loss_function(predictions, boxes)     # 2. how wrong?
optimizer.zero_grad()                        # 3. clear old gradients
loss.backward()                              # 4. backpropagation
optimizer.step()                             # 5. update the weights
```

1. **Forward pass:** the batch goes through every layer, and PyTorch quietly records each step.
2. **Loss:** one number at the end of those steps.
3. **`zero_grad()`:** every weight has a `.grad` box. PyTorch **adds** each new gradient into it rather than replacing it, so we empty it before each batch.
4. **`backward()`:** works backwards through the recorded steps (the chain rule) and fills in `.grad` for every weight: "how would the loss change if I nudged this weight?".
5. **`step()`:** Adam reads every `.grad` and nudges every weight.

`model.train()` at the start of each epoch switches back to training mode after `test_loss()` switched to eval. Our model has no Dropout or BatchNorm, so it changes nothing here, but forgetting it in a model that has them is a classic bug.

**TODO answers:** `predictions = model(images)`, `loss = loss_function(predictions, boxes)`

### What students should see

On Kaggle (GPU) each epoch takes about 10 seconds, and training takes **about 1.6 minutes**:

| Epoch | Train loss | Test loss |
|---|---|---|
| 1 | 0.0573 | 0.0557 |
| 2 | 0.0570 | 0.0550 |
| 3 | 0.0461 | 0.0296 |
| 4 | 0.0099 | 0.0033 |
| 5 | 0.0024 | 0.0017 |
| 10 | 0.0009 | 0.0009 |

The shape to point out is **flat, then a cliff**. In epochs 1–2 the loss sits at about 0.057: the heatmap is still blurry and the model predicts the middle of every picture. Then the score layer locks onto Wally's features and the loss falls off a cliff.

If a student's run is still flat at 0.057 at epoch 10, they've had an unlucky start or a bug. Check their TODOs against the answers, then re-run from the model cell **including Step 4** (see [Errors students will hit](#errors-students-will-hit)).

### Learning curves

**Think: what if we delete `optimizer.zero_grad()`?**
Gradients pile up across batches: after 10 batches, `.grad` holds the sum of 10 batches' gradients. The steps keep getting bigger and point in an out-of-date direction, so the loss jumps around and often gets stuck or explodes. It's a great live demo. Remember to re-run the model and optimiser cells afterwards.

**Think: is there a gap between training and test loss?**
In the Kaggle run, no: both end at 0.0009. That means the model is **generalising**: what it learned on the training pictures works just as well on new ones. Test loss much higher than training loss would mean **overfitting** (memorising the training pictures). The fixes for that are more data, more augmentation or a smaller model. Both high would mean **underfitting**: train longer or use a bigger model.

**What does a loss of 0.0009 mean in pixels?**
√0.0009 = 0.03, and 0.03 x 256 ≈ **8 pixels** of typical error per coordinate. Compare the stuck value: √0.057 x 256 ≈ 61 pixels, which is "guessing the middle". That's why a loss on its own is hard to picture, and why Step 6 uses IoU.

---

## Step 6: How good are the boxes? IoU

### Theory

$$\text{IoU} = \frac{\text{overlap}}{\text{area}_A + \text{area}_B - \text{overlap}}$$

- 1 is a perfect match, 0 means the boxes don't touch.
- It punishes both a wrong position and a wrong size.
- It's relative to the box size: 10 pixels off is fine on a huge box but terrible on a small one. IoU shows that; MSE doesn't.
- **IoU > 0.5** is the classic rule for a correct detection (from the PASCAL VOC benchmark).

### The hand example

**TODO answer:** `pred_box = torch.tensor([[5., 0., 15., 10.]])`

Box A covers 0–10 x 0–10 (area 100). Box B covers 5–15 x 0–10 (area 100). The overlap is 5–10 x 0–10 = **50**. The union is 100 + 100 − 50 = **150**. IoU = 50/150 = **0.33**. Shifting a box by **half** its width already drops it below 0.5.

**Extra question for strong students: Wally is about 25x50. How far sideways before IoU drops below 0.5?**
Overlap = (25 − d) x 50, union = 2500 − overlap. Setting IoU = 0.5 gives 1.5 x (25 − d) x 50 = 1250, so 25 − d ≈ 16.7 and **d ≈ 8 pixels**. Only 8 pixels sideways, a third of his width, fails the test. IoU is strict for small objects.

### `box_iou` and the diagonal

The cell collects every prediction and true box (`torch.cat` joins the batches), then multiplies by `image_size` to go back to pixels. IoU is a ratio so it's the same either way, but Step 7 draws in pixels.

`box_iou(A, B)` returns an **N x N table**: every predicted box against every true box. That's built for detectors with many boxes. We only want each prediction against its **own** true box: the **diagonal**, entries (0,0), (1,1), and so on. For 1000 images it works out a million numbers and keeps 1000, which is wasteful but quick and simple to explain.

`(ious > 0.5).float().mean()` turns True/False into 1/0 and averages them: the fraction of correct detections.

**TODO answer:** `ious > 0.5`

On Kaggle: **mean IoU 0.60, 79.0% correct**.

### Histogram

The histogram shows the spread: most pictures are to the right of the red 0.5 line, with a tail of failures on the left. The mean on its own hides that tail.

---

## Step 7: Testing on unseen images

### Single examples

For 8 random test pictures, the cell draws the green (true) and red (predicted) boxes, with the IoU and the **centre error**: the straight-line distance in pixels between the two box centres, √((tx − px)² + (ty − py)²).

- `image.permute(1, 2, 0)` turns the tensor back to channels-last for matplotlib: the reverse of Step 2.
- `true_boxes[i]` and `pred_boxes[i]` line up with `test_dataset[i]` **only because the test loader doesn't shuffle**.

**TODO answers:** `draw_box(true_box, 'green')`, `draw_box(pred_box, 'red')`

### Where is the model looking?

This cell draws the 16x16 heatmap over 4 test pictures.

- `image.unsqueeze(0)` adds a batch dimension, `(3, 256, 256)` → `(1, 3, 256, 256)`, because the model always expects a batch.
- `extent = (0, image_size, image_size, 0)` stretches the 16x16 grid over the 256x256 picture. The last two numbers are bottom then top, which keeps y pointing down.
- `alpha = (heat / heat.max())` gives every square its own see-through-ness: the most likely square is solid, unlikely ones are invisible.
- `cmap = 'autumn'` colours squares from red (less likely) to yellow (most likely).

This is the most "explainable" plot in the notebook: students can literally see what the network thinks.

**If a student asks: one bright square or spread out?**
One bright square means the model is confident. A spread over 2–4 neighbouring squares is normal when Wally sits across a square boundary, and the weighted average puts the centre between them, which is correct. Bright spots in **separate** places mean the model is torn between two candidates (Wally and something stripy). The centre then lands between them, on neither, which gives a bad box. You'll often see this in the worst pictures.

### The hardest test pictures

`ious.argsort()` sorts the picture numbers from lowest IoU to highest, so `[:4]` are the 4 worst.

**Think: why did the model get these wrong?**
Usually one of these:
- **Background:** Wally is on a busy red-and-white area, so he's camouflaged.
- **Position:** he's near the edge, where the padding zeros change what the filters see.
- **Which Waldo:** the five are different shapes (two tall and thin, three nearly square), so even when the position is right, the size can be wrong.
- **Augmentation:** a dark or colour-shifted Wally blends in.

**Think: no Wally? Two Wallys?**
The model **always** outputs exactly one box, because softmax always adds up to 1. With no Wally it will confidently draw a box somewhere, often on something red and stripy. It has no way to say "he's not here", because we never gave it a confidence output or any pictures without Wally. With two Wallys it usually picks one, or puts a box **between** them (when it's unsure, MSE rewards the average of two answers). Real detectors like YOLO solve both problems by predicting many boxes, each with a confidence score.

**Think: would it work on a real Where's Wally page?**
Probably badly, for three reasons:
- **Scale:** a real page is thousands of pixels across and Wally is tiny. Squash it to 256x256 and he's a few pixels.
- **Domain gap:** real Wally is drawn into the scene, partly hidden behind people, in the same art style as everything around him. Pasted Wally has crisp edges and is never hidden.
- **Decoys:** real pages are full of red-and-white things placed on purpose to fool you.

To make it work, you'd slide the model over 256x256 tiles of the page, and train with real examples or much harder synthetic ones (hidden parts, different sizes, decoys).

---

## What's next answers

- **Train on all 4000 images or for more epochs.** IoU should go up a little, and training takes about 4x longer per epoch with 4000 images. Watch that the test loss keeps following the training loss.
- **Learning rate `1e-2` or `1e-5`.** At `1e-2` the steps are too big: the loss usually stays stuck at about 0.057 (dead ReLUs) or jumps around. At `1e-5` the steps are tiny: after 10 epochs it will most likely still be on the flat part of the curve. Both show why 0.001 is the default.
- **Put the Flatten head back.** It reproduces the failure: the loss stays at about 0.058 for all 10 epochs and every red box sits in the middle. A memorable way to show why weight sharing matters. They'll need to change `forward` to return the 4 numbers from the new head directly.
- **Stronger augmentation.** Bigger rotations make the boxes looser, so IoU drops. A smaller Wally is harder to find, especially after 4 poolings (each grid square is 16 pixels).
- **A "is Wally here?" head.** Add images without Wally, a second output with `nn.BCEWithLogitsLoss`, and only count the box loss on pictures that do have Wally. This is the bridge to real detectors.

---

## Results from the Kaggle run

| | Result |
|---|---|
| Device | Kaggle GPU (`cuda`) |
| Model size | 60,707 weights and biases |
| Generating 5000 images | 70 s |
| Training, 10 epochs x 1000 images | 1.6 min (about 10 s per epoch, including the test loss) |
| Final train / test loss | 0.0009 / 0.0009 |
| Mean IoU on 1000 test images | **0.60** |
| Correct detections (IoU > 0.5) | **79.0%** |
| Flatten → Linear head, for comparison | loss stuck at about 0.058, IoU 0.03 |

With `seed = 1`, students on a GPU should get numbers very close to these. Small differences are normal, because some GPU operations aren't perfectly repeatable.

---

## All TODO answers on one page

| Section | Blank | Answer |
|---|---|---|
| 1.1 | `backgrounds.append(...)` | `image` |
| 1.2 | `.convert(...)` | `'RGBA'` |
| 1.3 | `wally.rotate(..., expand=True)` | `angle` |
| 1.3 | `np.array(...)` | `colours` |
| 1.4 | `random.choice(...).copy()` | `backgrounds` |
| 1.4 | `augment_waldo(random.choice(...))` | `waldos` |
| 1.4 | `x = random.randint(0, image_size - ...)` | `wally_width` |
| 1.4 | `y = random.randint(0, image_size - ...)` | `wally_height` |
| 1.5 | `width = x2 - ...` | `x1` |
| 1.5 | `height = ... - y1` | `y2` |
| 1.6 | `[train_num, ...]` | `test_num` |
| 1.6 | `os.path.join(..., file_name)` and `....append(...)` | `train_directory`, `train_labels` |
| 1.6 | `save_labels(train_directory, ...)` | `train_labels` |
| 2.1 | `image.permute(2,0,1) / ...` | `255` |
| 2.1 | `box ... / ...` | `image_size` |
| 2.2 | `leftovers = len(train_dataset) - ...` | `images_per_epoch` |
| 2.3 | `shuffle = ...` (train, test) | `True`, `False` |
| 3 | `nn.Conv2d(..., 32, ...)` | `16` |
| 3 | `nn.Conv2d(..., 64, ...)` (block 3) | `32` |
| 3 | `x2 = ...`, `y2 = ...` | `center_x + width / 2`, `center_y + height / 2` |
| 4 | `lr = ...` | `learning_rate` |
| 5 | `predictions = ...` | `model(images)` |
| 5 | `loss_function(..., ...)` | `predictions, boxes` |
| 6 | `[[5., 0., ..., 10.]]` | `15.` |
| 6 | `ious > ...` | `0.5` |
| 7 | `draw_box(true_box, ...)`, `draw_box(pred_box, ...)` | `'green'`, `'red'` |

---

## Errors students will hit

| What they see | Cause | Fix |
|---|---|---|
| An error mentioning `ellipsis`, e.g. `argument 1 must be str, not ellipsis` | A TODO blank wasn't filled in | Find the `...` in that cell |
| `Using device: cpu`, training takes 10+ min | GPU not switched on | Settings → Accelerator → GPU, then **restart and run from the top** |
| `NameError: name 'backgrounds' is not defined` (or similar) | Cells run out of order, or the session restarted | Run from the top |
| Images look like they have several Wallys | `.copy()` was removed in 1.4 | Put `.copy()` back and re-run 1.1 onwards |
| `expected input[...] to have 32 channels, but got 16 channels instead` | Wrong channel number in the Step 3 TODO | Input channels = the previous block's output channels |
| Step 7 boxes don't sit on Wally even though IoU looks fine | `shuffle = ...` left blank for the test loader. `...` counts as True, so there's **no error**, but the test set gets shuffled and `pred_boxes[i]` no longer matches `test_dataset[i]` | Set `shuffle = False` for the test loader and re-run from 2.3 |
| Loss stuck at about 0.057 for all 10 epochs | A wrong TODO, or the model cell was re-run without re-running Step 4 | Re-running `WaldoFinder()` makes a **new** model, but the old `optimizer` still points at the **old** model's weights. Always re-run Step 4 after the model cell |
| Loss carries on from where it was | The training cell was run twice | Fine, it just trains more. For a clean run, re-run the model cell and Step 4 first |
| `draw_box() takes 2 positional arguments but 3 were given` | An old version of a Step 7 cell that passes `ax` | Use the Step 7 cells from the current notebook |
| The 5000 images vanish and get re-made | The 1.6 cell was re-run (`shutil.rmtree` wipes the folder) | Expected. Run it once |

---

## Small fixes for the notebook text

These are in the markdown, not the code, so nothing breaks, but they're worth tidying before the session:

- **Step 3, Part 2B:** the bullets "Column chances:", "Row chances:" and "Strongest features: for each of the 64 channels" are unfinished. Suggested text: *Column chances: add the heatmap down each column, giving how likely Wally is to be in each column. Row chances: the same across each row. Strongest features: for each of the 64 channels, its biggest value anywhere in the grid.* "Center x and y" is also missing the end of its sentence: *...where each position counts as much as its probability.*
- **Step 3, Part 2A:** "softmac can threat them" should be "softmax can treat them".
- **Step 6:** "MSE = 0.0002" came from an older test. The Kaggle run finishes at about 0.001, so "MSE = 0.001" matches what students will see.
- **1.3:** "Tweeking" should be "Tweaking", and "one number for each of R, B, G" should be "R, G, B".
- **Step 4:** "it'll takes forever" should be "it'll take forever".
