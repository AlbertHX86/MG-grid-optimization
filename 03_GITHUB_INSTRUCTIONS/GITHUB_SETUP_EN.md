# GitHub Setup and Draft Pull Request (English)

## ⚠️ DO NOT MERGE — OPEN A PR AND AN ISSUE FIRST

Repository: https://github.com/AlbertHX86/MG-grid-optimization  
Existing issue: https://github.com/AlbertHX86/MG-grid-optimization/issues/1

This handoff **does not push to GitHub**. The linked repository was empty when inspected on 2026-11-11. Its author needs write permission. The connected assistant account only had read permission.

### A. Before publishing any files

1. Unzip `MG-grid-optimization_COMPLETE_HANDOFF_2026-11-11.zip` locally.
2. Work only with `01_GITHUB_REVIEW_BRANCH_UPLOAD/` for GitHub. The private folder contains network and time-series inputs that **must not be publicly published without redistribution clearance**.
3. Open `01_GITHUB_REVIEW_BRANCH_UPLOAD/DATA_PROVENANCE_AND_LICENSE.md`; review the public-release status of **every** code, data, and results file. Excluding six sensitive files is not proof that the remaining content is cleared.
4. In the repository UI, keep Issue #1 open and mark its checklist as needed. Do not merge the PR before explicit review.
5. Check for credentials or sensitive text before commit: `git diff --cached --check` and `git diff --cached --stat`; inspect `git status --short` and every staged file, including CSV/JSON.

### B. Git setup (recommended: Git CLI; macOS/Linux/Git Bash/WSL)

Install `git` and sign in to GitHub using the owner account **AlbertHX86** or another account with write permission. Optionally install GitHub CLI (`gh auth login`).

```bash
# Clone the currently empty repository.
git clone https://github.com/AlbertHX86/MG-grid-optimization.git
cd MG-grid-optimization

git config user.name "YOUR_NAME"
git config user.email "YOUR_GITHUB_EMAIL"
```

#### Because this repository is empty, initialize `main` with a warning-only README

**ONLY if `main` is still empty** (if GitHub now has a main commit, skip to the branch section):

```bash
git switch --orphan main
printf '# ⚠️ DO NOT MERGE — OPEN A DRAFT PR AND ISSUE FIRST\n\nIssue: https://github.com/AlbertHX86/MG-grid-optimization/issues/1\n' > README.md
git add README.md
git commit -m "chore: initialize review-only repository"
git push -u origin main
```

This is a minimal bootstrap commit: it does **not** publish the research data, and no PR can be created against an empty base branch.

#### Create the review branch; do NOT commit the research to `main`

```bash
git fetch origin
git switch main
git pull --ff-only origin main
git switch -c review/mg-planning-2035-2050
```

Now copy **the contents** of `01_GITHUB_REVIEW_BRANCH_UPLOAD/` into the cloned repository root. Include hidden files such as `.gitignore`. **Do not copy the outer ZIP, any private data directory, or original source archives.**

Example for macOS/Linux after setting the correct path:

```bash
# Replace this path with the path to the unzipped handoff directory.
HANDOFF="/absolute/path/to/MG-grid-optimization_COMPLETE_HANDOFF_2026-11-11"
cp -R "$HANDOFF/01_GITHUB_REVIEW_BRANCH_UPLOAD/." ./

# Confirm that data requiring approval has not been staged.
git status --short
git add .
git diff --cached --check
git diff --cached --stat
# OPTIONAL: inspect every staged path
# git diff --cached --name-only

git commit -m "docs/code: add MG portfolio and congestion study for review"
git push -u origin review/mg-planning-2035-2050
```

> **IMPORTANT:** `git add .` does not include ignored files unless forced; but `.gitignore` alone cannot retroactively protect previously tracked content. The 6 most obvious restricted files were **removed from the upload folder**, not merely ignored. Review the remainder before pushing publicly.

### C. Create a DRAFT PR and link Issue #1

Option 1: open https://github.com/AlbertHX86/MG-grid-optimization/pulls, choose **New pull request**, base = `main`, compare = `review/mg-planning-2035-2050`, then select **Create draft pull request**. Copy the content of `PR_DRAFT.md` into the PR description. The PR text contains **Related issue: #1**.

Option 2 (GitHub CLI, from the cloned repo):

```bash
gh auth login
gh pr create --draft --base main --head review/mg-planning-2035-2050 \
  --title "[DRAFT][DO NOT MERGE] MG portfolio and transmission study" \
  --body-file PR_DRAFT.md
```

The PR remains open as **Draft**. Do **NOT** choose 'Ready for review' or 'Merge' until all reviewers approve. Keep Issue #1 open as the tracking task.

### D. Run the research locally (with authorized private inputs)

From a terminal opened in the unzipped folder's `01_GITHUB_REVIEW_BRANCH_UPLOAD/`:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r MG_Step2_Congestion_Relief/requirements.txt
python -m pip install -r MG_Step2_Congestion_Relief/MG_Nodal_Connection_Optimization/requirements.txt
python tools/restore_local_inputs.py ../02_LOCAL_ONLY_DO_NOT_UPLOAD/PRIVATE_INPUTS_FOR_LOCAL_REPRODUCTION.zip
python tools/check_readiness.py
bash run_portfolio_and_siting.sh
bash run_congestion.sh
bash run_regional_comparison.sh
```

Use Windows Subsystem for Linux (WSL) for these bash commands. PowerShell users can use Python for environment setup, but should run the bash launchers in WSL/Git Bash. **Do not re-upload locally restored private inputs**. After running scripts, inspect outputs for rights and integrity before committing.

### E. What is and isn't validated

- Verified in this handoff: source files are syntactically valid Python, model modules import with available environment, 12 candidate areas, 1,489 buses and 1,890 branches in the local source data, and the two new-capacity totals match the saved portfolio data.
- Not claimed: a new end-to-end numerical re-run on your computer, a certified 2050 power-flow case, AC/N-1 safety, engineering-validated renewable site potential, verified new-line CAPEX, or permission to publicly redistribute source-derived tables.

### F. If the repository is no longer empty

Do not create a new orphan `main`; simply clone, `git switch main`, `git pull --ff-only`, create the review branch, copy the contents, review staged changes, and open a draft PR. If a branch with the same name already exists, choose a new unique name.
