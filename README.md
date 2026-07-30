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
`baseN=64` 已在 N=49/56/64 全部改善。下一輪只重新定位 N=48
交界點，測一筆相鄰成功的 `baseN=64` 候選。

證據分組為：

```text
known_anchor                已知可改善的 4096x17x16384
skinny_n_initial_holdout     已量完的 K=8192/12288/16384 初始 holdout
skinny_n_k16384_holdout      3 個新的 K=16384 holdout
skinny_n_boundary_holdout    N=40/47 的 baseN=48 證據；N=48 待同輪重測
skinny_n_boundary64_holdout  N=49/56/64，三筆均改善
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

搜尋不是「先產生大量錯誤 tiling 再碰運氣」。通用 full 初篩對每個
workload 最多保留 12 筆 hard-legal、callback-valid 且結構去重的候選；
候選由 local、global、transfer、diverse 四個起點輪詢選出，避免單一
cost-model 排名或歷史成功區域佔滿預算。

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

Full 是通用搜索的廣域初篩：14 個未用於人工規則的新 shape，加 2 個
已知控制；每個 workload 最多 12 個候選，共最多 192 筆。使用
`warmup=3, repeat=10, samples=5` 擴大 shape 與候選覆蓋；明顯改善者要在
後續高精度輪次確認。只要出現新候選，searched、bank control 與 official
baseline 就強制同輪量測。結果獨立寫入
`results/npu_full_general_v1_*`，不覆蓋既有 boundary-family 證據。
terminal 每個 workload 只額外輸出一行
`SOURCE_RESULT`，比較 local/global/transfer/diverse 各來源最佳點；所有
候選的完整量測仍以 candidates CSV 為準。

再次執行相同的 `--mode full` 時，搜索器會先讀取
`npu_full_general_v1_resume.csv`。已量 fingerprint 只用來校正每個
candidate source 的模型誤差，不再佔 NPU 候選名額；每個仍有空間的 source
至少保留一個探索點，其餘預算依真機校正後分數分配。第一輪 187 筆完成後，
host 契約演練會提出 145 筆不重複的第二輪候選。

`--mode general` 只保留為相容別名，與 `--mode full` 使用相同設定；正常
工作流只需執行 `full`。

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
skinny-N, N=33..47       N=33/40/47 支持的 baseN=48 schedule
skinny-N, N=48           只測相鄰成功的 baseN=64 crossover
skinny-N, N=49..64       N=49/56/64 支持的 baseN=64 schedule
deterministic split-K    對齊 M=N=128 時只切換 iterateOrder
```

43 個 broad L2 候選以及 attention-score focused frontier 已在 910B3 真機
上被否定；`odd_tail_2`、`trans_ab_case` 的 bank-seed `iterateOrder`
ablation 也未改善，預設搜索不再產生它們。每個 state 在進入 Beam 前已
完整且 hard-legal；active scope 不執行 Tabu/LNS。cycle estimate 只做
預算控制，不冒充 NPU latency。

`general_search_v1` 是預設 full scope。它不看 workload 名稱，從四個獨立
起點建立最多 32 個 callback 候選：

```text
local       官方 RuntimeKb seed 周圍的耦合變換
global      由 Cube alignment、容量與模板契約生成的全域結構
transfer    從強真機結果轉移 partition/L1/DB 策略並重建目標 L2
diverse     在代理模型合理性能帶內，距離 seed 最遠的合法結構
```

最終以來源輪詢保留 12 筆，不讓單一 cost-model 排名吃掉全部名額。所有
候選仍須通過 hard legality、官方 callback roundtrip 與 NPU preflight。

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
results/npu_full_general_v1_summary.csv
results/npu_full_general_v1_candidates.csv
results/npu_full_general_v1_resume.csv
```

`resume.csv` 是持久化續跑帳本。每完成一個官方 baseline、bank control 或
searched candidate 都先寫入本次 profile；正常結束、後續候選失敗或使用者
中止時再原子合併到續跑帳本。下一次執行會先匯入既有 `*_candidates.csv`、
`*_resume.csv` 與可唯一匹配的封裝歷史。舊資料第一次成功遷移後會寫成新版
exact resume。終端的 `profile_plan` 會列出
`resume_exact_*_assigned` 與 `npu_*_pending`。

general campaign 已完成的完整 fingerprint 另外保存在
`config/general_search_v1_round{1,2,3_partial}_fingerprints.csv`。因此即使
重新 clone 而沒有 Git 忽略的本地 resume，full 也不會靜默重跑前 380 筆。
搜尋階段會輸出
`search_feedback`，分別列出 exact profile、可用於校準的穩定 profile，
以及 campaign manifest 排除的數量。

量測標準差超過 median 5% 時，official/bank reference 會自動重試最多兩次；
仍不穩定才延後該 workload 的新候選。兩個 reference 即使各自穩定，若
latency 相差超過 15%，也標記 `incoherent_baselines` 並延後候選，避免把
系統性 baseline 漂移當成加速。高變異資料仍保留為 exact history 以避免
重測，但不會用於 cost-model 校準或 transfer seed。

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

目前 `general_search_v1`、CANN 8.1.RC1 的 broad-screen dry-run：

```text
workloads:                    16
bank controls:                16
searched schedules:          187
callback/RuntimeKb rejected:    0
```

每個 searched row 都通過官方 callback 的模板、suffix、23 欄位 roundtrip；
新記錄也通過 RuntimeKb lookup。這證明搜索與注入契約完整，不等於已證明
NPU 加速；後者必須執行 full。

NPU 計時與錯誤輸出見 [docs/npu_profiling.md](docs/npu_profiling.md)，先前
環境與執行問題見 [NPU_TILING_EXECUTION_HISTORY.txt](NPU_TILING_EXECUTION_HISTORY.txt)。
