# UCI Section

1. **Total size:** The dataset contains 5,574 messages, all in English.
2. **Class imbalance:** The dataset contains 4,827 ham (86%) and 747 spam (13%) messages. The imbalance ratio is 6.5:1.
3. **Date and provenance:** Compiled in 2011 from four sources — Grumbletext (UK forum), Caroline Tag's PhD thesis, NUS SMS Corpus (Singapore students), and SMS Spam Corpus v.0.1 Big.
4. **Cultural skew:** The majority of spam messages are UK-style (UK phone numbers, GBP prices). The ham messages are mostly in Singaporean English ("Singlish") with informal abbreviations like "oni", "lar", "wif".
5. **Format and encoding:** Tab-separated, one message per line, with label and text. Some characters are corrupted (e.g. `£` appears as `Â£` from an old encoding issue). A few ham messages contain non-English text despite the README's claim that all messages are English.



# Mishra Section

1. Total size: 5,971 SMS messages.
2. Class balance: 4,844 ham (81.1%), 638 smishing (10.7%), 489 spam (8.2%). 
3. Date and provenance: Published 2022 on Mendeley Data by Mishra & Soni. 
4. Format: Five columns — LABEL, TEXT, URL, EMAIL, PHONE. The first two are the actual data. The other three are pre-computed binary flags which contains a URL, email address, or phone number. 
5. Encoding issues: Like UCI, this dataset has encoding corruption but the formate is different. For example, £305.96 appears as ï¿½305.96 (UTF-8 replacement character), whereas UCI corrupted £ as Â£. 
6. Quality observation: The smishing examples look like real SMS phishing (e.g. tax refund messages, fake bank alerts) with shortened URLs (smsg.io, bit.do). 


# SpamAssassin Section

1. **Total size:** 6,047 messages.
2. **Class balance:** 1,897 spam (31%), 4,150 ham (69%). Better balance than the SMS datasets.
3. **Date and provenance:** Apache SpamAssassin public corpus, files dated 2003 and 2005.
4. **Format:** Compressed `.tar.bz2` archives. Each archive contains many individual raw email files (one email per file).
5. **Five subsets:**
   - `spam`: 500 spam messages
   - `easy_ham`: 2,500 non-spam messages, easy to tell apart from spam (no HTML, no spam-style signatures)
   - `hard_ham`: 250 non-spam messages that look like spam at first (HTML, coloured text, unusual markup)
   - `easy_ham_2`: 1,400 non-spam messages (added later)
   - `spam_2`: 1,397 spam messages (added later)
6. **Quality observations:** The author warns not to use these messages with a live email system, since "From" addresses are real and replies would bounce. The labels have been actively curated (e.g. one misclassified message was removed in 2005). Copyright stays with the original senders.