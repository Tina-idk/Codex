# Artwork Auction Price Data

The app is designed to run with real public artwork price data.

By default, it must download this public Kaggle dataset:

- `amaboh/masterworks-top-10-1m-artists-20182022`
- Title: Million Dollar Art Market Dataset - Major artist
- License shown by Kaggle: MIT
- Useful columns: `title`, `artist`, `description`, `purchase_price`, `sale_price`, `image_url`

The project does not generate fabricated artwork price rows. If Kaggle download fails, the app should stop and report the download problem instead of silently switching to fake data.

For a custom Kaggle dataset:

```bash
export KAGGLE_DATASET_SLUG="owner/dataset-name"
streamlit run app.py --server.fileWatcherType none
```

You can also upload or manually place a real CSV in `data/`, but it should be a public or properly licensed artwork auction/sale price dataset.

Please do not directly scrape Sotheby's, Christie's, or other auction websites unless their terms clearly allow it. This project is an educational demo, not a professional artwork valuation tool.
