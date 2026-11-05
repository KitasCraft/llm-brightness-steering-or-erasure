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

For hot-cold curve compared with the other curves: First, I calculate the mean cosine similarity across the layers from layer 2 to layer 23 for each bootstrapped curve. Then, I calculate the list of differences between all 5,000 bootstrapped mean cosine similarity of each cosine curve from the same bootstrapped sample. Finally, I use the 2.5th percentile and 97.5th percentile of that list to get the 95% interval of the difference between hot-cold curve and the other curve across layer 2 to 23 (See Layer Choice for details).

For bright-dark curve compared with the baseline curve: Similar as the above, but instead of using the mean cosine similarity across layers, I keep each layer's cosine similarity and then plot the 95% bootstrap interval of the difference between bright-dark curve and baseline curve for each layer. Since negative differences can cancel an average, per-layer summary is better for this comparison as to observe how the difference changes per layer.

The reason I use average instead of per-layer for hot-cold curve comparison is because most of the layers have hot-cold cosine and bright-dark cosine difference lower bound as above 0 (19 of 22 layers). Same thing can also be said between hot-cold cosine and the baseline cosine difference lower bound (22 of 22 layers) (See Robustness and Sanity Checks section in the notebook for detail). Thus, the statistic of the average is much preferred as most layers have the same signs in terms of difference.

### Layer Choice

For summary calculation, layer 0 (embedding layer) is excluded from all calculations, since each sentence has the same embedding for the last token due to each sentence's last character being the dot ('.') character.

Layer 1 is excluded from the calculation of the averages because layer 1's concept vector are 3-5x smaller relative to the activations in the middle layer (See Robustness and Sanity Checks section in the notebook for detail). Layer 24 is also excluded from the calculation due to its layer having Qwen's final normalization which differs from other layers' computation.

### Robustness Checks

The analysis was also repeated over the original 40 sentences as well as the original 21 sentences before the prompt set expansion. Split half stability was also measured to ensure the stability of a non-neutral concept set. The checks are detailed in the Robustness and Sanity Checks section in the notebook.

## 3. Results

pretend that the plot is here

We can see that from this plot that, the cosine similarity between bright concept and dark concept is hovering around 0.75, which is nowhere near -1. We can also see that the similarity between bright and dark are very close to similarity of the unrelated pairings and below the similarity between hot and cold.

Over layer 2 to 23, the mean cosine similarity of the baseline or unrelated concept pairings are 0.729. In every bootstrapped sample, the averaged hot-cold cosine similarity is above the averaged bright-dark cosine similarities as well as averaged baseline cosine similarities, with [0.098, 0.229] as the 95% interval of cosine difference between hot-cold and bright-dark and [0.140, 0.235] as the 95% interval of cosine difference between hot-cold and the unrelated baseline.

second plot for bright-dark vs baseline

We can see that when bright-dark cosine similarity is compared to the baseline by layer, it mostly hovers around the baseline. But, we can see that layer 2, 3, 22, 23 have the 95% interval of their difference entirely above 0, while layer 12 to 14 have the 95% interval of their difference entirely below 0.

## 4. Limitations

- The model I picked is very small and the sentences I picked are only template sentences. It might not reflect results using other models or sentences.
- There are only two pairs that reflect opposite concepts, so it's unclear whether bright-dark or hot-cold are the typical patterns. The baseline is also formed from only these four concepts so an actual baseline might have lower/higher mean cosine similarity.
- The bright vector is the least stable. Its split-half is only 0.80 compared to about 0.9 for the others (See Robustness and Sanity Checks section in the notebook for detail). Its noise may lower bright-dark cosines somewhat.
- A large shared "property" component dominates every cosine, which may hide finer structure.

## 5. Next Steps

While writing this README, I noticed that one usual method to approximate the direction of a concept is by doing addition and subtraction from other concept vectors (e.g. king - male + female = queen). So, my next step will be to research whether two opposite concepts are really separated by a "negation" concept and whether that concept concludes the existence of a universal "negation" concept. I'm also thinking about observing on what happens to a model when some concept vectors are amplified to trigger more activations related to those concepts.
