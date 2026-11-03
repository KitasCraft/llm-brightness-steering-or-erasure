# Are 'bright' and 'dark', two opposite concepts, stored the opposite way inside a language model?
One common meaning of 'bright' is "filled with much light," while one common meaning of 'dark' is "being without light or without much light," so it seems that both have the opposite meaning. But, does a language model also encode the meaning this way by encoding the vectors in the opposite direction?

## 1. Setup
I used Qwen2.5-0.5B-Instruct as the model for this experiment, with float32 as its activations' data type. The prompts used to extract the specific activation vectors are 40 matched sentences, each for the corresponding concept of 'bright', 'dark', neutral, 'hot', and 'cold'. Same subject per row; and the neutral set has boring properties such as size, material, and age as a concept set unrelated to brightness or temperature. The cleaned variant, which removes connotations and sun/winter cross-leaks, is the main prompt set; the original variant is used as a robustness check. All sentences are drafted with AI help and reviewed by hand.

To formalize the question, two vectors are facing the opposite direction when their cosine similarity is close to -1. Thus, the goal of this small project is to determine whether the cosine similarities of 'bright' vector and 'dark' vector is close to -1. If not, the goal is to also compare the cosine similarity to other concepts with opposite meaning and unrelated meaning to assess how similar 'bright' and 'dark' are relative to what's considered as 'opposite' and 'unrelated'.

## 2. Methodology

### Cosine Similarity Distribution

To get the approximate distribution of the cosine similarities per layer, there are a few steps.

1. **Concept Vector Extraction.** I extract the last-token hidden state at every layer for each sentence in a concept's set. After that, a concept vector can be calculated as the average of the concept's hidden state minus the average of the neutral's hidden state, per layer. By subtracting the neutral means, it allows the removal of most shared state between sentences such as the '.' as the very last character of each sentence and the word "the" shared across many of the sentences.
2. **Cosine Similarity Computation.** After the concept vectors are obtained, I compute the cosine similarity between bright vs dark concept vectors, as well as hot vs cold concept vectors as a control pair, per layer. I also compute the average of the cosine similarity across 4 unrelated pairings (bright-hot, bright-cold, dark-hot, dark-cold) to establish a baseline; to check how similar two unrelated concepts are.
3. **Finding The ~95% Bootstrap Interval.** To do it, I resample the 40 matched rows with replacement and recompute all concept vectors and the three cosine curves in this new sample. I resample it for 5,000 times and use the 2.5th percentile as the lower bound and the 97.5th percentile as the upper bound for the interval. For reproducibility purpose, I fixed the seed using `torch.manual_seed(42)` at the start of the bootstrap code block.

### Cosine Curves Comparison

After getting the distribution of the cosine similarities per layer, next step is to analyze the curve comparisons between the three cosine curves which are bright-dark, hot-cold, and the baseline, averaged from the unrelated pairings.

For hot-cold curve compared with the other curves: First, I calculate the mean cosine similarity across the layers from layer 2 to layer 23 for each bootstrapped curve. Then, I calculate the list of differences between the different possible pairings of 5,000 bootstrapped mean cosine similarity of each cosine curve. Finally, I use the 2.5th percentile and 97.5th percentile of that list to get the 95% interval of the difference between hot-cold curve and the other curve across layer 2 to 23. (See Layer Choice for details)

For bright-dark curve compared with the baseline curve: Similar as the above, but instead of using the mean cosine similarity across layers, I keep each layer's cosine similarity and then plotted the 95% bootstrap interval of the difference between bright-dark curve and baseline curve for each layer.

### Layer Choice

For summary calculation, layer 0 (embedding layer) is excluded from all calculations, since each sentence has the same embedding for the last token due to each sentence's last character being the dot ('.') character.

Layer 1 is excluded from the calculation of the averages because layer 1's concept vector are 3-5x smaller relative to the activations in the middle layer (See Robustness and Sanity Checks section in the notebook for detail). Layer 24 is also excluded from the calculation due to its layer having Qwen's final normalization which differs from other layers' computation.

### Robustness Checks

The analysis was also repeated over the original 40 sentences as well as the original 21 sentences before the prompt set expansion. Split half stability was also measured to ensure a non-neutral concept set has roughly similar direction. The checks are detailed in the Robustness and Sanity Checks section in the notebook.

## 3. Results

pretend that the plot is here

All of the bootstrapped hot-cold average cosine similarities are above all of bootstrapped bright-dark average cosine similarities as well as all of bootstrapped baseline average cosine similarities, with [0.098, 0.229] as the 95% interval of cosine difference between hot-cold and bright-dark and [0.140, 0.235] as the 95% interval of cosine difference between hot-cold and the unrelated baseline.
