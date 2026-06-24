# Can You Beat the AI Art Valuer?

一个 Streamlit 中文网页小游戏：玩家根据艺术品图片、艺术家背景和历史交易线索猜成交价，并和机器学习模型的预测结果比赛。

## 功能

- 自动下载公开 Kaggle 艺术品交易数据集
- 清洗 `$2.2K`、`$110.5M` 等价格格式
- 训练并自动比较多个回归模型
- 使用完整有效价格数据训练模型
- 游戏出题只使用真实可显示图片的记录
- 展示用户误差、AI 误差、胜负和 AI 预测逻辑
- 支持上传自己的 CSV 数据

## 默认数据源

默认使用 Kaggle 公开数据集：

- `amaboh/masterworks-top-10-1m-artists-20182022`
- Kaggle 标注许可证：MIT

应用不会爬取 Sotheby's、Christie's 等拍卖网站。

## 运行

```bash
pip install -r requirements.txt
python train_model.py
streamlit run app.py --server.fileWatcherType none
```

也可以直接运行 Streamlit，缺少模型时应用会自动训练：

```bash
streamlit run app.py --server.fileWatcherType none
```

## 文件

- `app.py`：Streamlit 游戏界面
- `train_model.py`：数据下载、清洗、特征工程、模型训练
- `data/README.md`：数据说明
- `data/artist_public_info.csv`：公开艺术家背景信息缓存

本项目仅用于教学演示，不是专业艺术品估值工具。
