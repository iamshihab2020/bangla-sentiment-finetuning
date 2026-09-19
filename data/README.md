# data/

Datasets are downloaded here by scripts and are never committed to git.

| Folder | Contents | Committed |
|---|---|---|
| `raw/<dataset>/` | Original files exactly as downloaded, plus their checksums | No |
| `processed/<dataset>/` | Normalized and deduplicated text, training subsets with text | No |
| `splits/<dataset>/` | Row IDs for every split and training subset, and split checksums | Yes (IDs only, no text) |

Why: SentNoB is licensed CC BY-ND 4.0, which does not allow sharing modified copies of the data. Keeping only IDs and checksums in git lets anyone rebuild the exact same subsets from the original download.
