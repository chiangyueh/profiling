# 官方 MatMulV3 NPU 驗證與計時

## 1. 執行順序

```text
完整候選 CSV
  -> 驗證實際 SoC/AIC 與安裝 bank filename
  -> 驗證 CANN bank schema/hash ABI
  -> 每個候選建立獨立 runtime bank
  -> RuntimeKb::InitBank + QueryBank 預檢
  -> 官方自動 baseline
  -> 搜尋候選逐一呼叫 aclnnMatmul
  -> preflight
  -> warmup
  -> ACL Event
  -> NPU 排名與 baseline 比較
```

Runtime bank 查詢是 host 操作，不會啟動 NPU。probe 先以精確 SoC 的
`PlatFormInfos` 初始化 bank，再要求每個候選查詢結果為 `found=1`。這只驗證
bank schema、hash 與 lookup；真正的模板執行合法性仍由後續 NPU preflight
判定。

## 2. 相同官方 Kernel

baseline：

```text
TUNE_BANK_PATH=<空的專案目錄>
ASCEND_CACHE_PATH=<空的專案目錄>
aclnnMatmulGetWorkspaceSize(...)
aclnnMatmul(...)
```

searched：

```text
TUNE_BANK_PATH=<只含該候選的專案目錄>
ASCEND_CACHE_PATH=<該候選的專案目錄>
aclnnMatmulGetWorkspaceSize(...)
aclnnMatmul(...)
```

因此比較不包含自製 kernel 差異。每個候選在新 process 建立 executor，使
RuntimeBankManager cache 不會沿用上一個候選。

## 3. Correctness Preflight

若 A+B host buffer 不超過 `NUMERIC_PREFLIGHT_MAX_MIB`：

```text
A = 1
B = 1
C = poison
launch once
抽查 C 的 9 x 9 等距網格
expected = K
```

較大矩陣：

```text
A = 0
B = 0
C = poison
launch once
抽查 C 的 9 x 9 等距網格，所有位置必須被覆寫為 0
```

preflight、配置、H2D/D2H、workspace 與 executor 建立均不計入 latency。

## 4. ACL Event

每個配置：

```text
warmup 次 launch
synchronize stream

for sample:
    record start
    repeat 次 launch
    record end
    synchronize end
    latency_ms = elapsed_ms / repeat
```

記錄：

```text
min / mean / median / stddev / p95 / max
TFLOPS
preflight_passed
preflight_mode
```

最終以 `median_ms` 排序。改善判斷的噪聲門檻至少 1%，並納入 baseline 與
candidate 的 sample standard deviation。

## 5. Smoke 與 Full

Smoke：

```text
1 workload
1 baseline
1 bank roundtrip control
1 bottleneck-transition searched candidate
warmup=2 repeat=5 samples=3
```

Full：

```text
16 組 workload（14 個未見 shape + 2 個正向控制）
每 workload 最多 16 個 local/global/transfer/diverse 候選
active frontier 會排除所有已完成 fingerprint，再補入較深的新候選
searched/bank/official 對新 workload 同輪量測
warmup=3 repeat=10 samples=5
```

Full 預設使用 `config/workloads_general_search_v1.csv`，結果與 resume
寫入 `results/npu_full_general_v1_*`，不讀寫舊 boundary-family 的
`results/npu_full_*`。歷史真機資料仍可供 transfer 起點與 cost model 使用，
但新 workload 的三條比較路徑必須同輪量測。

既有完整 run 證明 K=16384 skinny-N 在
3072x17、4096x17、4096x24、4608x31、5120x29 五個獨立窄 N shape 上
改善 `1.24121x` 到 `1.59785x`；baseN=48 在 N=33/40/47 改善，
baseN=64 在 N=49/56/64 改善 `1.15176x` 到 `1.42283x`。
因此搜索器只有在至少三個獨立 anchor 同時勝過各自 bank control 10% 以上
時，才允許失準 proxy 之外的預先註冊因果動作。

舊 K=8192/12288 與 bank-seed iterateOrder 候選已實測證偽並關閉。下一輪
只量 N=48 的 baseN=64 crossover，且 searched、bank、official 必須同輪。
128x128 deterministic split-K order 在 K=16384/32768/49152 改善，但
K=65536 退化到 `0.867283x`，所以不再向更大 K 無條件外推。

`net_log1.txt` 的遠端 run 已把 137 個 exact row 寫入
`results/npu_full_resume.csv`。封裝 OCR 歷史可獨立重用 72/78 個目前
bank record；終端摘要省略的 2 個舊 candidate 只存在遠端 exact resume，
程式不會猜造其時間。更新 zip 時應覆蓋程式檔但保留該 resume；若遺失，
guard 會在 NPU 前停止，而不是重測。

公開 MatMulV3 不接受本清單中的 INT8 input path，因此該 workload 明確標記
unsupported，不拿 MatMulV2 或量化算子結果冒充 MatMulV3 tiling。

## 6. Timeout 與錯誤

每個候選有獨立 process 和 timeout。若發生：

```text
RuntimeKb query failure
aclnnMatmulGetWorkspaceSize failure
preflight failure
ACL synchronization failure
timeout
```

baseline 或 bank seed control 失敗時，程式立即輸出真正失敗階段並停止，
因為此時無法建立可信比較。單一 searched candidate 在獨立 process 中被
preflight 拒絕時，則把 `success=0` 與完整錯誤寫入 candidates CSV，淘汰
該候選後繼續；這不會把失敗候選列入排名。

`aclInit`/`507008`、binary load/registration、dynamic linker 或 parent
process orchestration 錯誤均分類為 infrastructure failure，第一筆即停止，
不會把同一環境錯誤重複套到後續候選。

候選 process 的 stdout/stderr 直接寫入該候選的普通暫存檔，再由 parent
使用 `wait()` 管理 timeout。它不使用 `PIPE + communicate()`，避免 process
逾時且已關閉 stdout 時，第二次讀取 pipe 以 `I/O operation on closed file`
遮住真正的候選 timeout。

coverage failure 會額外列出 `T/S/C/G/L2`、L1 step/depth/DB、C 的線性 index
與 row/column，以及該位置是否仍為 `0x5a` poison，藉此區分「未寫輸出」和
單純數值偏差。若 profiler 本身失敗，shell 不再對不完整的 baseline CSV
執行排名，以免次要 traceback 遮住原始錯誤。

## 7. 輸出

預設：

```text
results/npu_<mode>_summary.csv
results/npu_<mode>_candidates.csv
results/npu_<mode>_resume.csv
```

`summary` 每個 workload 一列，包含最佳搜尋候選、bank-path control、官方
自動 baseline、speedup、latency change 與 verdict。只有 searched 同時勝過
兩個 reference 且超過合併噪聲門檻時，才標記為已證明的 tiling 優化；其他
情況標記 `no_proven_improvement`。`candidates` 保存所有成功量測與重用紀錄。

`resume` 是跨執行的精確量測帳本。profiler 在每次正常結束、候選失敗或
shell 中止清理時，將已寫入的 official/profile CSV 原子合併到該檔；下一次
會自動匯入既有 `candidates` 與 `resume`。完整 fingerprint 相同的項目直接
產生 `*_history_reuse` 紀錄，不建立 RuntimeKb candidate directory，也不
啟動 NPU process。終端 `profile_plan` 顯示重用與待量測數量。

舊 OCR 若缺少 `iterateOrder`、L1 depth/step、DB 或 L2 order，不能證明是
同一個 tiling，因此不會被續跑機制誤用。只有 callback SHA 缺失但其餘完整
schedule 唯一匹配且曾通過 `grid9_v1` 時，才允許一次 legacy 遷移。

需要每次 sample：

```bash
KEEP_DETAILS=1 ./run_npu.sh --mode full
```

才會保留 `results/npu_full_details/`。其餘 profile 和 bank 暫存檔在成功或
失敗結束時清除；`results/npu_<mode>_resume.csv` 是續跑狀態，不會被清除。

## 8. 可聲明範圍

NPU 前只能聲明候選已通過：

- 精確 SoC/AIC 模型。
- 官方 `MultiCoreMatmulTiling`。
- 靜態容量與模板契約。
- CANN `RuntimeKb::InitBank + QueryBank` 查詢命中。

NPU 後才能聲明：

- 官方 kernel correctness preflight 通過。
- Top-K 內的實測最佳 latency。
- 相對原始官方自動 tiling 的改善或劣化。

Beam/LNS 不窮舉全部 tiling，因此不能把 Top-K 實測最佳描述為數學全域最優。
目前 active `bottleneck_guided_v1` scope 只保留已由 holdout 支持的
shape family 與預先註冊的單變量候選，並不執行 Tabu/LNS；後者僅保留在
全模板契約研究程式中。
