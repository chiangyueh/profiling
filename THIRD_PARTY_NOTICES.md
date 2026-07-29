# Third-Party Notices

`compat/` 內的相容頭文件保留原始 Huawei CANN Open Software 版權與授權聲明。
其用途僅是補足 CANN Host Tiling 頭文件的相依項，使本工程能直接編譯。
其餘 `host/`、`runner/`、`scripts/` 與 `tools/` 為本工程實作。

MatMulV3 runtime-bank key、knowledge schema 與 split-K 限制依據 CANN
`ops-nn/matmul/mat_mul_v3` 公開 host tiling 原始碼；工程只產生 bank 記錄，
不包含或重新發布官方 kernel 原始碼。官方原始碼適用 CANN Open Software
License Agreement Version 2.0：

https://gitcode.com/cann/ops-nn/tree/8.5.0/matmul/mat_mul_v3
