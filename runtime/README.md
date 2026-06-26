# Runtime 修改說明

這個資料夾只保存 GPU-main 相關的 llama.cpp runtime 修改，不包含完整 upstream llama.cpp 原始碼。

## 檔案

| 路徑 | 說明 |
|---|---|
| `RPP_GPU_RUNTIME.patch` | 對 llama.cpp 的完整 patch。 |
| `changed_files/llama.cpp/` | 本研究修改或新增的檔案快照。 |

## 套用方式

如果已經有對應版本的 llama.cpp，可在 llama.cpp repo 中執行：

```bash
git apply /path/to/RPP_GPU_RUNTIME.patch
```

若 patch 因 upstream 版本不同而無法直接套用，可以參考 `changed_files/llama.cpp/` 中的檔案逐一比對。
