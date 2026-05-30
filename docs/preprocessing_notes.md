* What preprocessing usually means

Tutorials typically list these steps:

1. Lowercase everything
2. Remove punctuation
3. Remove stopwords ("the", "is", "and"...)
4. Lemmatize ("running" → "run")
5. Remove numbers
6. Remove URLs
7. Remove HTML tags
8. Deduplicate
9. Handle outliers (too short / too long)

* My recommended preprocessing pipeline
 Conservative — does less than tutorials suggest, deliberately:

1. Strip leading/trailing whitespace (always safe)
2. Collapse multiple whitespaces into one (clean up the HTML-stripped text from Nazario/SpamAssassin)
3. Drop rows with text shorter than 10 characters (the 24 we flagged earlier)
4. Drop rows with text longer than 10,000 characters (outliers — emails with embedded base64 attachments, etc.)
5. Deduplicate (exact text duplicates within and across sources)
6. Lowercase the text (mostly — though we'll discuss whether to keep a capital_ratio feature first)



* About the preprocessing step you just ran:
- What you did — preprocessed the combined dataset using six steps (drop missing text, clean whitespace, length filter, capital ratio, lowercase, deduplication).
- The numbers — went from 19,092 messages down to 13,153. Lost 263 to length filtering, lost 5,676 to duplicates.
* The big discovery:
- Mishra and UCI overlap massively — 4,933 of the duplicates were the same message appearing in both datasets. This means Mishra's main unique contribution is its smishing class (SMS phishing), not its ham/spam.
- Decisions you made and why:
1. Kept punctuation, numbers, URLs in the text — because they're often the most predictive features in spam/phishing (e.g. "$1000", "WINNER!!", links).
2. Calculated capital ratio before lowercasing — so we don't lose the "WINNER" capitals as a feature.
3. Strict global deduplication — accepted the cost (losing 5,676 rows including a lot of smishing) because keeping duplicates would have polluted train/test split.
4. Fixed bounds 10 to 10,000 characters — defensible in plain language. Anything shorter than 10 isn't a real message; anything longer is usually attachment corruption.
5. Numbers worth remembering:
6. After preprocessing: 13,153 messages. 8,983 ham, 2,504 spam, 1,666 phishing. Crosstab still has all six channel × label cells filled. SMS phishing is the smallest class at 277 — this is now my main worry going forward.