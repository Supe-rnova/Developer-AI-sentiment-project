# GitHub setup (step by step, for beginners)

You only do Part A once. Your teammates do Part B once.

## Part A: you (the repo owner)

1. **Install Git**: https://git-scm.com/downloads (then restart your terminal).
2. **Tell Git who you are** (once per computer):
   ```
   git config --global user.name "Your Name"
   git config --global user.email "you@example.com"
   ```
   Use the same email as your GitHub account.
3. **Create the empty repo on GitHub**
   - Go to github.com -> click **+** (top right) -> **New repository**.
   - Name: `ai-sentiment-project` . Choose **Private**.
   - Do **not** tick "Add a README", ".gitignore" or "license" (this folder already has them).
   - Click **Create repository**.
4. **Upload this folder.** Open a terminal inside the unzipped `ai-sentiment-project` folder:
   ```
   git init
   git add .
   git commit -m "Initial commit: data prep script, README"
   git branch -M main
   git remote add origin https://github.com/<your-username>/ai-sentiment-project.git
   git push -u origin main
   ```
   (GitHub will ask you to sign in. If it asks for a password, use a *Personal Access Token*:
   GitHub -> Settings -> Developer settings -> Personal access tokens, or just use GitHub Desktop.)
5. **Invite teammates**: repo page -> **Settings** -> **Collaborators** -> **Add people** -> type their GitHub usernames. They get an email invite and must accept it.

Prefer clicking over typing? Install **GitHub Desktop** and use File -> Add local repository -> Publish repository.

## Part B: each teammate

```
git clone https://github.com/<owner-username>/ai-sentiment-project.git
cd ai-sentiment-project
pip install -r requirements.txt
```
Then download the dataset from Kaggle and put it at `data/raw/ai_developer_attitudes_2024.csv`
(the raw file is deliberately NOT stored in GitHub).

## Daily workflow (everyone)

```
git pull                         # get the latest changes FIRST
git checkout -b my-branch-name   # (optional but safer) work on your own branch
# ... do your work ...
git add .
git commit -m "Short message about what you did"
git push                         # first push of a new branch: git push -u origin my-branch-name
```
If you used a branch, open a **Pull Request** on GitHub and ask someone to merge it into `main`.

## Using Git inside VS Code (no typing needed)

- Open the project folder (File -> Open Folder). The **Source Control** icon in the left bar (branch symbol) shows changed files.
- Type a message in the box, click **Commit**, then **Sync Changes** (this pulls and pushes).
- To get the repo the first time: Command Palette (Ctrl/Cmd+Shift+P) -> **Git: Clone** -> paste the GitHub URL.
- Install the **Python** extension (and **Jupyter** if you want the Run Cell buttons on `# %%` lines).

## Rules that save you from pain

- **One person edits one file.** Two people editing the same file at once causes merge conflicts. Suggested split:
  - `01_data_prep.py` -> you
  - `02_visualization.py` -> visualization teammate
  - `03_modeling.py` -> modeling teammate
  - slides in a `slides/` folder -> presentation teammate
- Always `git pull` before you start and before you push.
- Never commit the raw Kaggle file (the `.gitignore` already blocks `data/raw/`).
- Re-running `01_data_prep.py` regenerates everything in `data/processed/`, so only re-run and push it if the prep logic changes, and tell the team.
