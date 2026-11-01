# Can a model still understand concept of brightness without knowing 'bright' or 'dark'?
One common meaning of 'bright' is "filled with much light," while one common meaning of 'dark' is "being without light or without much light," so it seems that both have the opposite meaning. But, does a language model also encode the meaning this way? Furthermore, what happens to its understanding of brightness if the activations for 'bright' or 'dark' are stripped away from the model?

## 1. Setup
I used Qwen2.5-0.5B-Instruct as the candidate model for this experiment, with float32 as its activations' data type. The prompts used to extract the specific activation vectors are 40 matched sentences, each for the corresponding concept of 'bright', 'dark', neutral, 'hot', and 'cold'. Same subject per row; and the neutral set has boring properties to mirror the neutral concept. A “cleaned” variant is also generated and used to remove connotations and sun/winter cross-leaks between brightness and temperature sets. All sentences are drafted with AI help and reviewed by hand.

## 2. Model Exploration

To extract the specific activation vector for a concept, there are a few steps. First, I extract the last-token hidden state at every layer based from the sentences inside the concept as inputs. After each concept has been extracted, the concept vector for each non-neutral concept is equal to the average of the extracted concept token subtracted by the average of the extracted neutral set token, per layer. Second, I compute the mean cosine similarity between bright vs dark and hot vs cold as to examine the control variable. I also compute the average of the cosine similarity across 4 unrelated pairings (bright/dark vs hot/cold) to establish baseline. I also compute all three cosine similarity within 5000 bootstrapped samples and a fixed seed to establish a 95% confidence area for each layer.

After plotting each layer's mean cosine similarity with the established 95% confidence interval, I've noted that 
