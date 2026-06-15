# Artwork Auction Price Data

The app can run in four ways:

1. Default Kaggle auto-download.
2. A user-uploaded or manually placed CSV.
3. Optional custom Kaggle auto-download, if configured.
4. A small built-in educational demo dataset, generated automatically only when no real data is available.

By default, the app tries to download this public Kaggle dataset:

- `amaboh/masterworks-top-10-1m-artists-20182022`
- Title: Million Dollar Art Market Dataset - Major artist
- License shown by Kaggle: MIT
- Useful columns: `title`, `artist`, `description`, `purchase_price`, `sale_price`, `image_url`

For a custom Kaggle dataset:

```bash
export KAGGLE_DATASET_SLUG="owner/dataset-name"
streamlit run app.py --server.fileWatcherType none
```

Kaggle sometimes requires login or API credentials. If Kaggle cannot be reached, the app will try other public sources and then fall back to demo data so the game still works.

Please do not directly scrape Sotheby's, Christie's, or other auction websites unless their terms clearly allow it. This project is an educational demo, not a professional artwork valuation tool.
