# Ascend MatMulV3 約束式 Tiling 搜尋

本工程只研究 CANN `MatMulV3` 的 tiling。搜尋結果透過 CANN RuntimeKb
注入安裝版 `MatMulV3`，baseline 與 searched candidate 都由同一個
`aclnnMatmul`、同一套官方 kernel 執行；工程沒有自製 MatMul kernel，也不會
用 CPU 或其他算子替代失敗候選。

## 目前研究範圍

2026-07-29 的 910B3 增量 run 完成 49 個 workload；active scope 的 12
個 searched schedules 全部勝過 official 與 bank control，其餘 37 個
workload 沒有重開已被否定的搜尋方向。已證實的條件式 family 是：

```text
K=16384 skinny-N, 16<N<=32
  3072x17  1.59785x       4096x17  1.50756x
  4096x24  1.47232x       4608x31  1.45775x
  5120x29  1.24121x

K=16384 boundary, 33<=N<=48
  4096x33  1.17078x       3072x40  1.31797x
  4096x47  1.22472x       5120x48  1.05084x

aligned 128x128 deterministic split-K traversal
  K=16384  1.10457x       K=32768  1.05467x
  K=49152  1.09397x       K=65536  0.867283x (反例)
```

這推翻「所有 `K >= 8192` skinny-N 都受益」以及「所有對齊
deterministic split-K 都能翻轉 traversal」兩個廣域假設。低 K、broad
L2、attention 與 bank-seed traversal 外推都已由真機否定並關閉。
下一輪只定位相鄰的 `baseN=64` 邊界：N=49/56/64 各一筆、共三筆新候選。

證據分組為：

```text
known_anchor                已知可改善的 4096x17x16384
skinny_n_initial_holdout     已量完的 K=8192/12288/16384 初始 holdout
skinny_n_k16384_holdout      3 個新的 K=16384 holdout
skinny_n_boundary_holdout    N=40/47/48，三筆均改善
skinny_n_boundary64_holdout  N=49/56/64，下一輪預註冊候選
det_split_k_positive_range   K=16384/32768/49152，三筆均改善
alignment_negative_control   已量完的 127x127x32769 負向對照
prior_regression             先前曾明顯退化的 workload
broad_validation             其餘一般形狀與模板契約 workload
unsupported_control INT8 負向對照，不計入 MatMulV3 改善率
```

先前把通用
`MultiCoreMatmulTiling` 的候選平鋪到所有 MatMulV3 BASE workload，造成大型
矩陣最高接近 `2x` 的退化；該策略已停用。active 搜尋先診斷
core-grid、L0K、L1 packet、L2 capacity/tail/reuse 或 Fixpipe 瓶頸，再只
構造能跨越該硬體轉折點的候選。只有嚴格符合 FP16/NN、
`16 < N <= 32`、`K=16384`、BASE 模板的 shape，才加入已有真機證據的
core/L1/DB/L2 因果消融。

專案仍保留 CANN 8.1.RC1、DAV C220 的 7 個模板族、12 個 suffix 契約，供
回歸驗證使用：

| 模板族 | kernel suffix |
|---|---|
| `BASE` | `0`, `1` |
| `SINGLE_CORE_SPLIT_K` | `20`, `21` |
| `DETERMINISTIC_SPLIT_K` | `30`, `31` |
| `AL1_FULL_LOAD` | `101` |
| `BL1_FULL_LOAD` | `200`, `201` |
| `BL1_FULL_LOAD_FIXPIPE` | `10200`, `10201` |
| `BL1_FULL_LOAD_VEC_NZ2ND` | `20201` |

`0/1` 等成對 suffix 分別覆蓋 unaligned/aligned kernel。程式不生成 CANN
8.1 kernel 中不存在的 split mode 4、5、6，也不把其他 CANN 版本的模板名稱
硬套進來。

支援 FP16、BF16、FP32。INT8 保留為負向 corpus，會明確標記 unsupported；
它不屬於這條公開 `MatMulV3` 輸入路徑。

## 目前工作流

```text
偵測精確 SoC 與 AIC/L0/L1/L2
  -> 官方 MatMulV3 callback 產生精確 23 欄 seed/control
  -> closed-form pipeline 診斷主瓶頸
  -> 只建立 AIC round、L0/L1 容量或 L2 traversal 的轉折點
  -> 對齊 deterministic split-K 只翻轉來源實際讀取的 iterateOrder
  -> single-core split-K 只在來源 inner-N 條件成立時啟用
  -> full-load 沒有來源支援的自由轉換時直接停止
  -> 每種瓶頸動作最多保留一個合法代表
  -> 官方 MatMulV3 callback 回讀
  -> 要求模板族、suffix、23 個 knowledge 欄位完全一致
  -> RuntimeKb::InitBank + QueryBank(found=1)
  -> 官方 aclnnMatmul correctness preflight
  -> ACL Event latency
  -> 與空 bank 的官方自動 tiling、bank seed control 比較
```

搜尋不是「先產生大量錯誤 tiling 再碰運氣」。每個 workload 的動作前沿
硬限制為 8 筆，每個機制最多 1 筆；目前 45 個支援 workload 實際選出
33 筆 searched schedule，其中 22 筆由封裝歷史精確命中。已測 fingerprint
使用同 run 的
`candidate_ms/bank_control_ms` 排名並直接重用。代理模型全部拒絕時，只有
至少三個獨立真機 anchor 支持的 family 才保留預先註冊的可證偽 holdout。

## 執行

```bash
cd ascend_matmul
./run_npu.sh --mode smoke
./run_npu.sh --mode full
```

腳本不修改 toolkit symlink、driver、firmware、shell profile 或系統
`LD_LIBRARY_PATH`。CANN 環境、`TUNE_BANK_PATH`、`ASCEND_CACHE_PATH` 只存在
於腳本子行程，暫存 bank 只建立在專案 `results/` 下。

### Smoke

Smoke 是 NPU 執行鏈快速檢查，不代表優化結果：

```text
workload       1 個 FP16 256x256x256
比較           官方自動 tiling + bank control + 1 個瓶頸轉折候選
warmup         2
repeat         5
samples        3
```

### Full

Full 是增量的來源導向驗證，不會只執行已知成功案例：

```text
workload       52 組；51 組 MatMulV3 支援，1 組 INT8 負向對照
control        51 個官方 RuntimeKb bank seed
候選           15 個 callback-valid searched schedules
候選上限       每個 workload 目前最多 1 個；hard cap 8
新舊分流       12 個既有 schedule 重用；本輪新增 3 個 N=49..64 holdout
warmup         10
repeat         50
samples        15
```

預設輸入是 `config/workloads.csv`。程式優先讀取
`results/npu_full_resume.csv`；若它是舊 schema 或只有 header，會再從封裝內
`npu_full_ocr_measurements.csv` 遷移 identity-complete baseline，以及具有
完整 T/S/C/G/L2/I/L1/DB/L2O、`grid9_v1` preflight 的 control/candidate。
既有 baseline、control 與標記為 `require_existing` 的候選若仍有缺口，
才會在 RuntimeKb/NPU 前停止。`allow_new` 只有未命中的 fingerprint 送 NPU。
`net_log1.txt` 的完整 run 已在遠端 `results/npu_full_resume.csv` 保存 137
筆 exact row；更新程式時必須保留該檔。終端簡略輸出省略了 2 個舊候選的
完整時間列，封裝歷史不會猜造它們；若刪除遠端 resume，guard 會停止而不是
重測。

## 本輪搜索的欄位

一筆 MatMulV3 RuntimeKb knowledge 有 23 個欄位。BASE workload 固定官方
seed 作為完整合法狀態，再依診斷出的瓶頸只執行下列一個具名動作：

- AIC round 邊界：聯動 `baseM/baseN`、`singleCoreM/N`、core 數與 L1。
- L0K 容量前沿：只提高到下一個合法 `baseK` 邊界，再解 M/N 因子。
- L1 packet 前沿：以搬運次數的連續最小值求解，再對齊合法
  `stepKa/stepKb`。
- L2 reuse：依較大的重用 operand 選 `l2IterateOrder=1`（row-first）或
  `2`（column-first）；`0` 保留官方 staggered mapping。
- L2 tail/capacity：只移動相鄰一個 tile-count 邊界。
- Fixpipe/Cube 重疊：容量允許且 Fixpipe 已成瓶頸時才切換 `dbL0C`。

對齊 deterministic split-K 只改來源實際讀取的 `iterateOrder`；
single-core split-K 只有滿足 inner-N 條件才生成一個 L2 traversal 候選。
full-load 模板沒有來源支援的自由轉換時直接停止。

窄 N 專用 family 不再重新枚舉四種消融。對已由五個獨立 shape 驗證的
`K=16384, 16<N<=32` family，只輸出一筆由 `baseM` 決定的 learned
schedule：`baseM<256` 使用 `L1=16x8, DB=2x2x1, L2O=0`；
`baseM=256` 使用 `L1=8x8, DB=2x2x2, L2O=1`。對 K=8192/12288 只輸出
前述 L2-only 與 L2+L1 兩筆候選，用來定位改善究竟來自 L2 分組還是 L1
搬運；不把 `208x32x64` 硬套到其他一般矩陣。

一般 BASE 搜尋不自行猜測 `baseMNK` 與 `singleCoreMNK` 的關係。候選先從
官方 callback seed 出發；只有 AIC round 動作會以同一個 closed-form 解
聯動重建它們，其餘動作保留 callback 的關係。最後仍要求官方 callback
回讀後模板、suffix 與 23 欄 knowledge 完整一致。

`usedCoreNum` 也可以大於輸出 tile 數，多出的 core 會退出。這是合法但通常
低效，因此由 active-core 與 tail 模型懲罰，不再誤判成非法。

## 硬合法性

所有候選先檢查：

```text
baseM % 16 == 0
baseN % 16 == 0
FP32 NT: baseK % 8 == 0
其他 layout: baseK % 16 == 0

baseM * baseK * inputBytes * dbL0A <= L0A
baseN * baseK * inputBytes * dbL0B <= L0B
baseM * baseN * 4 * dbL0C         <= L0C

baseM * baseK * inputBytes * depthA1
+ baseN * baseK * inputBytes * depthB1 <= effective L1
```

另外逐模板檢查：

- BASE 的 `singleCoreK == K`、單一 output task 不跨越多個 base M/N
  tile，以及完整 M/N output grid 與 L2 schedule。
- Single-Core Split-K 的 MK33、NK33、MK24、MK14 pipeline、K loop、
  FP32 workspace/atomic accumulate 與 AIV cast。
- Deterministic Split-K 的 128x128 基本塊、MK33/NK33、
  `singleCoreK=3*baseK` 與 workspace reduction。
- AL1/BL1 full-load 的 operand 常駐條件與 L1 容量。
- Fixpipe 的 32-byte output boundary。
- Vec NZ2ND 的 FP32/layout/workspace 條件。
- aligned/unaligned logical shape 與 callback 選到的 suffix。

910B3 Platform API 回報 L1=524032 B，但官方 callback 可合法產生 524288 B
的 MatMul allocation。程式只在本身容量判斷中按 KiB allocation boundary
處理，不改任何系統設定。

## 搜尋算法

目前 `bottleneck_guided_v1` 不做參數笛卡兒積。它先以官方 callback 與
RuntimeKb seed 建立完整合法 tiling，再只輸出已有真機證據支持的 family
schedule 或單一控制變量實驗：

```text
skinny-N, K=16384        一個 output M block/AIC 的實測 family schedule
skinny-N, N=33..48       四個 shape 支持的 baseN=48 family schedule
skinny-N, N=49..64       三個預先註冊的 baseN=64 boundary holdout
deterministic split-K    對齊 M=N=128 時只切換 iterateOrder
```

43 個 broad L2 候選以及 attention-score focused frontier 已在 910B3 真機
上被否定；`odd_tail_2`、`trans_ab_case` 的 bank-seed `iterateOrder`
ablation 也未改善，預設搜索不再產生它們。每個 state 在進入 Beam 前已
完整且 hard-legal；active scope 不執行 Tabu/LNS。cycle estimate 只做
預算控制，不冒充 NPU latency。

算法與研究依據見 [docs/algorithm.md](docs/algorithm.md)。

## Baseline 與結果定義

每個支援 workload 比較三條路徑：

```text
official_operator  空的專案 bank，CANN 原始自動/內建 tiling
bank_seed_control  官方 23 欄 seed 經相同 RuntimeKb 路徑重放
searched           本搜索器選出的 23 欄候選
```

只有 searched 勝過同一 RuntimeKb 路徑的 bank control，並同時列出相對
official 的結果，且差值超過合併量測噪聲，
才能稱為 tiling optimization。官方比較較快時，結果必須寫
`no_proven_improvement`，不能把「選回官方」稱為我們的優化。

NPU 前只能稱候選為 host-ranked；NPU 後只能稱為指定 CANN、SoC、workload
與 profiling 預算下的 best measured candidate。Beam/LNS 不是數學全空間
窮舉，因此不宣稱全域最優。

## 輸出與續跑

終端會在執行中顯示：

```text
template / T(baseMNK) / S(singleCoreMNK) / cores
output grid / L2 / L1 / DB
median ms / stddev / speedup / delta / noise / verdict
```

並留下：

```text
results/npu_smoke_summary.csv
results/npu_smoke_candidates.csv
results/npu_smoke_resume.csv
results/npu_full_summary.csv
results/npu_full_candidates.csv
results/npu_full_resume.csv
```

`resume.csv` 是持久化續跑帳本。每完成一個官方 baseline、bank control 或
searched candidate 都先寫入本次 profile；正常結束、後續候選失敗或使用者
中止時再原子合併到續跑帳本。下一次執行會先匯入既有 `*_candidates.csv`、
`*_resume.csv` 與可唯一匹配的封裝歷史。舊資料第一次成功遷移後會寫成新版
exact resume。終端的 `profile_plan` 會列出
`resume_exact_*_assigned` 與 `npu_*_pending`。

官方 baseline 只在 SoC、AIC、workload、shape、dtype 與 transpose 相同時
重用。新版 control/searched 必須匹配模板、完整 23 欄 `tiling_signature`
與 callback SHA-256。legacy 遷移只接受完整 schedule 欄位且曾通過
`grid9_v1` 的唯一匹配；只看到相同 `baseMNK` 或 `singleCoreMNK` 不會跳過
量測。

已由封閉 NPU 機器轉錄的結果保存在
`results/npu_full_ocr_measurements.csv`。官方 baseline 可按 SoC、AIC、
shape、dtype 與 transpose 重用。候選只有在保存完整排程且標記
`preflight_contract=grid9_v1` 時才能重用；仍缺少 iterate/L1/DB/L2 order
的舊 OCR 列只保留為研究紀錄，不會冒充有效候選。`KEEP_DETAILS=1` 才保留
samples 與 all-evaluated 中間資料。

## Host 契約驗證

不啟動 NPU、不跑 CPU simulator：

```bash
BUILD_JOBS=1 ASCENDC_SOC_VERSION=Ascend910B3 \
SOC_VERSION=Ascend910B3 ./scripts/validate_cpu.sh
```

目前 `bottleneck_guided_v1`、CANN 8.1.RC1 的完整 workload dry-run：

```text
workloads:                    52
supported bank controls:      51
selected rows:                66
searched rows:                15
measured-history rows:        12
new holdout candidates:        3
  M=3072, N=49, K=16384:       T=160x64x64
  M=4096, N=56, K=16384:       T=208x64x64
  M=5120, N=64, K=16384:       T=256x64x64
callback/RuntimeKb valid:     all found
old broad L2 candidates:       0
old attention candidates:      0
failed bank-order candidates:  0
```

每個 searched row 都通過官方 callback 的模板、suffix、23 欄位 roundtrip；
新記錄也通過 RuntimeKb lookup。
這證明搜索與注入契約完整，不等於已證明 NPU 加速；後者必須執行 full。

NPU 計時與錯誤輸出見 [docs/npu_profiling.md](docs/npu_profiling.md)，先前
環境與執行問題見 [NPU_TILING_EXECUTION_HISTORY.txt](NPU_TILING_EXECUTION_HISTORY.txt)。
