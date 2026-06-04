* What "stratified" means
- A random 80/20 split could, by bad luck, put most of your 277 SMS phishing messages into the training set, leaving only a handful for test. "Stratified" means you split each group separately and combine — so each class is represented in test in the same proportion as it appears in the full data. This is standard practice.
* Two layers of stratification
- Most projects stratify by label. Yours benefits from stratifying by label and channel together — so SMS phishing, email phishing, SMS spam, email spam, SMS ham, email ham each get split independently. This is the project-specific version of "make sure smishing is in the test set."

* What we're building
- A script src/data/split.py that:

- Loads data/processed/cleaned.csv
- Creates a combined stratify_key column from label + channel
- Splits 80/20 stratified on that key
- Saves train.csv and test.csv to data/processed/
- Logs the per-group counts so we verify all six cells survived in both train and test

* The split

- Train: 10,522 rows (80%)
- Test: 2,631 rows (20%)
- All six channel × label cells populated in both train and test
- Proportions preserved (e.g. test SMS phishing is 55, train SMS phishing is 222 — ratio is 4.04, basically 80/20)

* The stratification did exactly what it was supposed to.
- The number to remember
- SMS phishing in test: 55
- This is the smallest cell in your evaluation set. When you report per-class metrics later, every prediction on those 55 examples will swing your numbers significantly. One missed phishing SMS = 1.8 percentage points off your recall.

You should:


* Plan to interpret SMS phishing results cautiously — confidence intervals on 55 examples are wide.
When we get to k-fold evaluation later, that's specifically going to help with this. K-fold averages over multiple splits so the noise from "lucky" or "unlucky" test sets gets smoothed out.