# Are 'bright' and 'dark', two opposite concepts, stored the opposite way inside a language model?
One common meaning of 'bright' is "filled with much light," while one common meaning of 'dark' is "being without light or without much light," so it seems that both have the opposite meaning. But, does a language model also encode the meaning this way by encoding the vectors in the opposite direction?

## 1. Setup
I used Qwen2.5-0.5B-Instruct as the model for this experiment, with float32 as its activations' data type. The prompts used to extract the specific activation vectors are 40 matched sentences, each for the corresponding concept of 'bright', 'dark', neutral, 'hot', and 'cold'. Same subject per row; and the neutral set has boring properties such as size and object existence as a concept set unrelated to brightness or temperature. A "cleaned" variant is also generated and the main prompt set used to remove connotations and sun/winter cross-leaks between brightness and temperature sets, while the original variant is used as robustness check. All sentences are drafted with AI help and reviewed by hand.

To formalize the question, two vectors are facing the opposite direction when their cosine similarity is -1. Thus, the goal of this small project is to determine whether the cosine similarities of 'bright' vector and 'dark' vector is -1

## 2. Methodology

To get the approximate distribution of the cosine similarities per layers, there are a few steps.

1. **Concept Vector Extraction.** I extract the last-token hidden state at every layer for each sentence in a concept's set. After that, a concept vector can be calculated as the average of the concept's hidden state subtracted by the average of the neutral's hidden state, per layer. By subtracting the neutral means, it allows the removal of most shared state between sentences such as the '.' as the very last character of each sentence and the word "the" shared across many of the sentences.
2. **Cosine Similarity Computation.** After the concept vectors are obtained, I compute the cosine similarity between bright vs dark concept vectors, as well as hot vs cold concept vectors as a control pair, per layer. I also compute the average of the cosine similarity across 4 unrelated pairings (bright-hot, bright-cold, dark-hot, dark-cold) to establish a baseline; to check how similar other two opposite concepts are.
3. **Finding The ~95% Distribution.** I compute the 95% bootstrap interval. To do it, I resample the 40 matched rows with replacement and recompute all concept vectors and the three cosine curves in this new sample. I resample it for 5,000 times and use the 2.5 percentile as the lower bound and the 97.5 percentile as the upper bound for the interval. For reproducibility purpose, I fixed the seed using `torch.manual_seed(42)` at the start of the bootstrap code block.

## 3. Results

pretend that the plot is here
