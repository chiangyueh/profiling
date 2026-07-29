# NPU 快速執行

```bash
cd ascend_matmul
chmod +x run_npu.sh scripts/*.sh
./run_npu.sh --mode smoke
```

正常階段：

```text
[0/4] Setup CANN environment ... ok
[1/4] Detect NPU SoC ... ok
[2/4] Build tiling host/official runner ... ok
[3/4] Search tiling candidates ... ok
[4/4] Run NPU ACL Event profiling ... ok
```

第 3 階段必須顯示實際平台 AIC 數。第 4 階段會先顯示：

```text
bank_schema: soc=Ascend910B3 aic=<實際值> input_bytes=183 knowledge_fields=23
bank_lookup: [1/2] npu_smoke_fp16 role=bank_seed_control rank=0
bank_records_prepared: 2 lookup=runtime_kb(found=1) execution_check=NPU_preflight
profile_plan: ... resume_exact_assigned=... npu_searched_pending=...
```

這只代表候選的 runtime-bank 格式與索引可被 CANN 找到，不代表 kernel
一定可執行。之後每個候選仍須通過官方 MatMulV3 的 NPU preflight，確認
沒有 AICore 例外且輸出頭、尾與抽樣位置都被寫入。
Smoke 量測一個官方自動 baseline、一個 bank roundtrip control 與一個
官方 seed 附近的單變因候選；全部使用 `aclnnMatmul`，沒有自製 kernel 或
CPU fallback。

量測期間會顯示：

```text
WORKLOAD [1/1] npu_smoke_fp16 shape=256x256x256 dtype=fp16 official_ms=...
bank_control_start ... tpl=BASE T=... S=... C=... G=... L2=...
bank_control_done ... ms=... official_ms=... status=...
candidate_done ... rank=1 ... speedup_vs_official=...
WORKLOAD_RESULT ... best_rank=1 ... optimization_result=...
```

`speedup=official_ms/searched_ms`，大於 `1` 才表示搜尋結果較快。`T` 是
base MNK，`S` 是 kernel 實際 single-core MNK，`G` 是輸出 tile 網格，
`L2` 是 L2 排程的 tile 數與每個 tile 涵蓋的 block 數；`I/L1/DB`
用來區分 base tile 相同但 traversal、L1 depth 或 double buffer 不同的候選。
`optimization=improved` 只會在 searched 同時勝過官方 baseline 與
bank-path control，且改善超過噪聲門檻時出現。否則會明確顯示
`no_proven_improvement`，官方數據只作比較基準。

成功後 terminal 會輸出：

```text
NPU_RESULT_BEGIN
...
NPU_RESULT_END
```

並留下：

```text
results/npu_smoke_summary.csv
results/npu_smoke_candidates.csv
results/npu_smoke_resume.csv
```

確認 smoke 成功後：

```bash
./run_npu.sh --mode full
```

Full 預設讀取 `config/workloads.csv` 的 52 組 workload。51 組支援
MatMulV3；INT8 是明確 unsupported 的負向對照。active scope 共有 15 個
searched schedules，其中 12 個已有量測，3 個是 N=49/56/64 boundary
holdout。每個 workload 目前最多一筆 searched schedule。

既有 49 組已由 2026-07-29 full run 完整量測。預設 full 先讀 exact resume，
再由封裝歷史遷移 identity-complete baseline 與完整、preflight-valid
schedule。兩種來源都無法可信匹配時，才在任何 NPU 工作前停止：

```text
resume_guard: prior workloads use trusted history; only three new holdouts
need official baseline, bank control, and searched measurements
missing=0 action=no_prior_npu_remeasurement
```

遠端完成 run 已記錄在 `results/npu_full_ocr_measurements.csv` 與
`results/npu_full_resume.csv`。低 K skinny-N 與兩個 bank-seed traversal
ablation 都已被實測證偽；default full 不再量這些排程。實際待量測數量仍以
`profile_plan` 的
`npu_searched_pending` 為準；若是 0，代表目前只在重建 summary，不該重跑
既有 NPU 候選。

程式會讀取 `results/npu_full_ocr_measurements.csv`，以 SoC、shape、dtype、
模板及完整排程指紋逐筆重用已完成量測。最後會分別輸出：

```text
GENERALIZATION_RESULT skinny_n_initial ...
REFINED_SKINNY_N_RESULT k_eq_16384 ...
DETERMINISTIC_SPLIT_K_RESULT aligned_mn_k_16k_to_49k ...
PRIOR_FAILURE_RESULT ...
BROAD_VALIDATION_RESULT ...
MATMUL_WIDE_RESULT ...
```

程式也會自動讀取同一 output stem 的
`results/npu_full_candidates.csv` 與 `results/npu_full_resume.csv`。
`resume.csv` 在每批已完成結果後原子更新，因此中途失敗或 `Ctrl-C` 後再次
執行，不會重測完整 fingerprint 相同的成功項目。exact 匹配包含 SoC、
AIC、shape、dtype、transpose、模板、完整 23 欄 tiling 與 callback
SHA-256。legacy 遷移另外要求完整 schedule 與 `grid9_v1`，且只能唯一對應
現行 callback；只相同的 `T/S/C` 不算同一候選。執行前可從下列欄位確認
本次實際會送入 NPU 的數量：

```text
resume_exact_baselines_assigned=...
resume_exact_assigned=...
npu_official_baselines_pending=...
npu_controls_pending=...
npu_searched_pending=...
```

已量結果顯示 K=16384 skinny-N 在 anchor 加三個 preregistered holdout 上
改善 `1.24121x` 到 `1.59785x`；低 K=8192/12288 外推不成立。
aligned deterministic split-K 在 K=16384/32768/49152 改善，但 K=65536
退化，所以只保留 bounded positive range。generic BASE 的 L2/order/DB/core
grid broad 搜尋在 LLM、vision、attention、tail、BF16/FP32 workload 上沒有
泛化；default optimizer 不再送這些候選到 NPU。INT8 負向對照單列為
`unsupported_control`。

第一次可縮小 NPU 測量量：

```bash
TOP_K=5 RANK_LIMIT=5 WARMUP=5 REPEAT=20 SAMPLES=5 \
./run_npu.sh --mode full
```

Full 預設先完成所有候選的靜態契約與 runtime-bank lookup，再開始 NPU；
每個候選在獨立 process 執行，預設 60 秒 timeout。發生錯誤時 terminal
只輸出一個 `TILING_ERROR_BEGIN/END` 區塊，包含 workload、rank、模板、
`T/S/C/G/L2`、L1 深度、失敗階段、C 的 row/column/poison 值及三層驗證
狀態，隨即停止，不會把後續候選全部標成相同失敗。

查看伺服器與 linked official runner 的 ACL 狀態：

```bash
./run_npu.sh --check-server
```

完整輸出：

```bash
RUN_VERBOSE=1 ./run_npu.sh --mode smoke
```

腳本只設定自身和子行程環境，且只寫專案的 `build/`、`results/`；不修改
系統 CANN 路徑、symlink、driver、firmware 或 shell profile。
`results/npu_<mode>_resume.csv` 是續跑狀態，不要刪除。
