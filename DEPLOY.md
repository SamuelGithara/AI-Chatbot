# Deploying the app (GitHub + Streamlit Community Cloud)

The code lives in this GitHub repository. The fine-tuned model (about 650 MB zipped) is too big for a normal Git file, so it is attached to a GitHub Release named `model-v1` as `mbert-farmer-finetuned.zip`. When the app starts on Streamlit Cloud, `streamlit_app.py` downloads that file and also downloads base mBERT from Hugging Face.

## 1. Run locally (optional)

```bash
pip install -r requirements.txt
python prepare_data.py && python train.py   # creates models/mbert-farmer-finetuned
streamlit run streamlit_app.py              # opens http://localhost:8501
```

If `models/mbert-farmer-finetuned/` already exists, the app uses it and skips the download.

## 2. Deploy on Streamlit Community Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
2. Click **Create app** → **Deploy a public app from GitHub**.
3. Choose this repository, branch `main`, and main file `streamlit_app.py`.
4. Open **Advanced settings** and choose Python 3.12.
5. Paste these secrets:
   ```toml
   GITHUB_REPO = "SamuelGithara/kenyan-farmer-query-classifier"
   # Only needed while the repository is PRIVATE:
   GITHUB_TOKEN = "github_pat_..."
   ```
   To get a token, go to GitHub → Settings → Developer settings → Fine-grained tokens. Grant read-only **Contents** access to this repository only.
6. Click **Deploy**. The first start takes about 5–10 minutes (installing PyTorch, then downloading the models). After that it is faster.

If you make the repository public, delete `GITHUB_TOKEN` from the secrets. The release download then works without it.

## 3. Updating the model

After retraining (for example on Colab), zip the model folder and replace the release file:

```bash
cd models/mbert-farmer-finetuned
zip -r ../../mbert-farmer-finetuned.zip config.json model.safetensors tokenizer.json tokenizer_config.json
cd ../..
gh release upload model-v1 mbert-farmer-finetuned.zip --clobber
```

Then reboot the app from the Streamlit dashboard. Also commit the new `results/base_head.npz`, because the base comparison uses it.

## Notes

- Memory use is about 1.3 GB, within Community Cloud's limit of about 2.7 GB.
- Apps sleep after 12 hours without visitors. Opening the link wakes the app in about a minute.
