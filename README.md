# SOX Apex Predator Dashboard

SOX 定期定額加碼策略 dashboard，資料來源優先使用 Yahoo Finance。

## 策略摘要

Apex Predator 使用 RMDD 判斷 SOX 從近 252 個交易日高點回撤的幅度，跌破防線時啟動標準加碼；當 VIX 高於 32 且 RMDD 跌破深層防線時，額外啟動 Sniper Shot；季節效應目前只保留 10 月 fallback buy 1.5 份，其餘月份 fallback 為 1 份。

## 本機更新

```powershell
python generate_apex_predator_dashboard.py
```

程式會同時輸出：

- `index.html`：GitHub Pages 首頁
- `apex_predator_dashboard.html`：本機瀏覽用檔案

## GitHub Pages

此 repo 使用 GitHub Actions 每天自動更新 Yahoo Finance 資料並重新產生 `index.html`。
