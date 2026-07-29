# MatMulV3 Tiling 搜尋算法

> Current scope (2026-07-30): `general_search_v1` is the default full run. It
> constructs legal candidates from local, global, structurally diverse and
> measured-transfer starts without workload-name rules. Two completed NPU
> rounds contain 332 unique candidate measurements. Their exact fingerprints
> and stable source-best feedback are versioned so a clean clone advances the
> active-search frontier instead of replaying prior work. The older
> `bottleneck_guided_v1` implementation remains available as historical
> reference, but is not the default research method.

## 1. 問題定義

本專案要解的是：

> 在指定 CANN 8.1.RC1、Ascend910B3、合法 MatMulV3 kernel 模板集合與
> NPU profiling 預算下，找出相對官方自動 tiling 有統計顯著加速的 tiling。

這不是把任意 23 個整數寫入 RuntimeKb，再把崩潰與超時稱為搜索。候選先由
kernel 契約構造，再依序通過官方 callback、RuntimeKb 與 NPU correctness
preflight。

同樣也不能只靠 host 成本模型宣稱最優。最終結論只能是：

```text
指定 CANN + SoC + workload + 搜尋預算下，
已通過 correctness 的實測候選中 latency 最低者。
```

## 2. CANN 8.1 的完整 kernel 空間

安裝版來源：

```text
$CANN_ROOT/opp/built-in/op_impl/ai_core/tbe/impl/ascendc/
  mat_mul_v3/mat_mul_v3.cpp
```

DAV C220 分支有 12 個實際 dispatch key：

| suffix | variant | family |
|---:|---|---|
| 0 | BASE_UNALIGNED | BASE |
| 1 | BASE_ALIGNED | BASE |
| 20 | SINGLE_CORE_SPLIT_K_UNALIGNED | SINGLE_CORE_SPLIT_K |
| 21 | SINGLE_CORE_SPLIT_K_ALIGNED | SINGLE_CORE_SPLIT_K |
| 30 | DETERMINISTIC_SPLIT_K_UNALIGNED | DETERMINISTIC_SPLIT_K |
| 31 | DETERMINISTIC_SPLIT_K_ALIGNED | DETERMINISTIC_SPLIT_K |
| 101 | AL1_FULL_LOAD_ALIGNED | AL1_FULL_LOAD |
| 200 | BL1_FULL_LOAD_UNALIGNED | BL1_FULL_LOAD |
| 201 | BL1_FULL_LOAD_ALIGNED | BL1_FULL_LOAD |
| 10200 | BL1_FULL_LOAD_FIXPIPE_UNALIGNED | BL1_FULL_LOAD_FIXPIPE |
| 10201 | BL1_FULL_LOAD_FIXPIPE_ALIGNED | BL1_FULL_LOAD_FIXPIPE |
| 20201 | BL1_FULL_LOAD_VEC_NZ2ND | BL1_FULL_LOAD_VEC_NZ2ND |

Runtime bank 的 `tilingEnable` 由 suffix 解碼：

```text
split = suffix / 10    % 10
full  = suffix / 100   % 10
fix   = suffix / 10000 % 10

tilingEnable = split + 10*full + 1000*fix
```

這個版本沒有 suffix 40，也沒有可直接注入的 split mode 4。舊版程式曾建立
一條 FP32 NT mode-4 路徑；它不對應 CANN 8.1 的 kernel dispatch，已完全移除。

## 3. 23 維離散表示

RuntimeKb knowledge record 是：

```text
usedCoreNum
singleCoreM/N/K
baseM/N/K
depthA1/B1
stepM/N
iterateOrder
stepKa/Kb
dbL0A/B/C
l2MTileCnt/NTileCnt
l2MTileBlock/NTileBlock
l2IterateOrder
tilingEnable
```

欄位不是獨立變數。例如：

```text
L0A = baseM * baseK * inputBytes * dbL0A
L0B = baseN * baseK * inputBytes * dbL0B
L0C = baseM * baseN * 4          * dbL0C

A1 = baseM * baseK * inputBytes * depthA1
B1 = baseN * baseK * inputBytes * depthB1

depthA1 % (stepM * stepKa) == 0
depthB1 % (stepN * stepKb) == 0
```

因此普通 GA 的逐基因 crossover、普通 PSO 的連續座標取整都會產生大量沒有
模板語義的點。它們可以在特製 constraint-preserving encoding 下作為基線，
但不是本空間的首選。

## 4. 模板硬約束

### 4.1 共通容量與對齊

```text
baseM % 16 == 0
baseN % 16 == 0
```

FP32 `transA=false, transB=true` 可使用 K0=8；其他 dtype/layout 使用 K0=16。
L0A、L0B、L0C 分別使用自己的 DB，不能以單一 DB 值代替。

910B3 Platform API 回報 L1=524032 B，但官方 callback 會產生合法的
524288 B A1+B1 配置。程式以 MatMul allocator 的 KiB 邊界得到 effective
L1=524288 B；這只是候選檢查，不修改 CANN 或系統記憶體。

### 4.2 BASE

一般 BASE 的 `singleCoreK` 必須等於完整 K。一個 output task 對應一個
Cube base tile：

```text
singleCoreK = K
singleCoreM <= baseM <= align16(singleCoreM)
singleCoreN <= baseN <= align16(singleCoreN)
```

最後兩個不等式只允許 logical tail 小於對齊後的 base。依據是 CANN 8.1
`MatmulV3BaseTiling::CalL1Tiling()` 與 `CalL1TilingV200()` 均令
`singleCoreM/N=baseM/N`，且已安裝的一般 BASE RuntimeKb 記錄也遵守此
關係。`singleCoreM/N > baseM/N` 只出現在 Split-K、AL1/BL1 full-load 等
具有不同 kernel 語意的模板，不能由一般 BASE 候選任意組合。

`usedCoreNum` 可以大於 output tile 數；多出的 AIC 退出。它是合法配置，但
成本模型只把實際工作的 core 計為 active core。

CANN 8.1 kernel 定義 `ROW_FIRST=1`、`COL_FIRST=2`；`0` 是原始交錯 core
mapping，不是布林 false。`mat_mul_base_block.h` 明確指出 B 大於 A 時，
column-first 可減少替換。因此 BASE 的 L2 reuse 候選按 operand footprint
選 `1` 或 `2`，而不是用 `1-l2IterateOrder` 翻轉。

### 4.3 Single-Core Split-K

CANN 8.1 suffix 20/21 使用 `mat_mul_sc_splitk_kernel.h`。每個 output-owning
AIC 依序跑多個 K slice；後續 slice 對 FP32 workspace 做 atomic accumulate，
AIV 最後轉回輸出 dtype。

候選構造涵蓋來源中可接受的四種 pipeline：

```text
MK33  stepM=3 stepN=1 stepK=3 depthA/B=9/6
NK33  stepM=1 stepN=3 stepK=3 depthA/B=6/9
MK24  stepM=2 stepN=1 stepK=4 depthA/B=8/8
MK14  stepM=1 stepN=1 stepK=4 depthA/B=8/8
```

並要求：

```text
baseM=baseN=128
baseK=256/inputBytes
singleCoreK=stepK*baseK < K
stepKa=stepKb
K slice count >= 2
```

四種 pipeline 已分別以官方 callback 驗證 suffix 21 與 23 欄 roundtrip。
unaligned workload 另外驗證 suffix 20。

### 4.4 Deterministic Split-K

suffix 30/31 沿 K 把 partial C 寫入 FP32 workspace，再由 AIV 做固定順序
reduction/cast。候選要求：

```text
baseM=baseN=128
baseK=256/inputBytes
singleCoreK=3*baseK
MK33 或 NK33
usedCoreNum <= ceil(K/singleCoreK)
```

它適合 M/N 很小而 K 很大、BASE 無法使用足夠 AIC 的形狀。模型另計 partial
workspace、reduction bytes 與 AIV 工作，不把它當成 BASE。

### 4.5 AL1 Full Load

suffix 101 將完整 A 常駐每個 active AIC 的 L1，再重用於多個 N partition。
目前 CANN 8.1 可構造域為 FP32 NT、小 M、N 不超過 AIC 可分範圍、K 至少
4096 且滿足 full-load 對齊。候選同步搜索 `baseN/baseK`、N partition、
B1 depth 與 C DB。

### 4.6 BL1 Full Load

suffix 200/201 將完整 B 常駐 L1，適合 M 遠大於 K/N 且 B 可放入 L1 的形狀。
候選搜索：

```text
baseM/N/K
singleCoreM
A1 buffer count
dbL0C
aligned/unaligned callback branch
```

### 4.7 Fixpipe 與 Vec NZ2ND

suffix 10200/10201 處理普通 Fixpipe 無法直接覆蓋的短 N output boundary；
邊界依實際輸出 32-byte transaction 判斷，不是舊版錯用的 256-byte條件。

suffix 20201 使用 AIV 做 NZ2ND，並有額外 A/workspace movement。這兩類模板
的 `l2MTileBlock/l2NTileBlock=0` 是 kernel 契約的一部分，不能套用 BASE
「block 必須大於零」的規則。

## 5. 候選構造

搜索器先取得兩個 seed：

```text
default seed  不注入 runtime bank 的官方 callback 結果
bank seed     同一 23 欄經 RuntimeKb 路徑重放後的 callback 結果
```

全模板 reference scope 另外使用 `MultiCoreMatmulTiling` 的輸出作為中心，
七個模板族再按上述契約建立有限離散池。候選池不是所有正整數的笛卡兒積，
而是：

- Cube K0、L0/L1 allocation unit 對齊值。
- 接近官方 seed 的 base 值。
- 能形成 1 至數個完整 AIC round 的 M/N partition。
- kernel 來源定義的 Split-K pipeline。
- full-load operand 確實能常駐 L1 的值。
- 完整覆蓋 output grid 的 L2 schedule。

每個進入搜索的 state 已是完整且 hard-legal 的候選，沒有「搜尋後 repair」。

完整模型仍會診斷：

```text
core_grid    active-core 或 AIC round 尾部利用率
l0_k         淺 K 是否能跨到下一個合法 L0 capacity frontier
mte1/mte2    L1 packet 數與 L2/HBM movement
l2_capacity  每 L2 tile 工作集是否跨過來源中的安全容量
l2_tail      每 L2 tile 的 AIC 尾輪利用率
fixpipe      output transaction 是否接近 Cube critical path
```

但 910B3 實測已否定把這些診斷廣泛轉成預設候選：43 個新 L2
tile-count/order 候選沒有產生新 winner。因此 active scope 只保留有
holdout 證據或明確單變量對照的規則。符合 FP16/NN、
`16 < N <= 32`、`K=16384` 且官方模板為 BASE 的 shape 已有五個獨立
真機改善證據，因此只輸出一筆 learned schedule：`baseM<256` 使用
`L1=16x8, DB=2x2x1, L2O=0`；`baseM=256` 使用
`L1=8x8, DB=2x2x2, L2O=1`。

K=8192 與 K=12288 的 lower-K 外推已被真機結果否定，預設搜索不再輸出。
attention-score 的 L2 frontier 同樣未勝官方，亦已關閉。

`odd_tail_2` 的 BASE `iterateOrder=1` 實測退化到 `0.132863x`，
`trans_ab_case` 也沒有勝過 bank control，因此該方向已關閉。

`skinny_n_boundary_n33`、N=40 與 N=47 已確認 baseN=48 改善：

```text
3072x40x16384  T=160x48x64
4096x47x16384  T=208x48x64
```

它們固定 `L1=8x40, DB=2x2x1, L2=1x1(20x1)`；差異只由 M 尺寸
推導出恰好 20 個 M blocks。

下一個 16-column 邊界由 CANN 8.1 callback 的 `baseN=64` seed 定義。
三個預註冊 shape 已全部改善：

```text
3072x49x16384  T=160x64x64
4096x56x16384  T=208x64x64
5120x64x16384  T=256x64x64
```

三者固定 `L1=8x32, DB=2x2x1, L2=1x1(20x1)`，改善分別為
`1.42283x`、`1.25969x`、`1.15176x`，因此 N=49..64 的實測範圍成立。

N=48 的舊候選為 `0.0977752 ms`，但最新報告將它與另一輪新量到的
`0.0894544 ms` 官方基準比較；舊候選所在輪的官方基準其實是
`0.102746 ms`。這是跨輪 absolute latency 混用，不能判定 tiling
退化。下一輪只測 `5120x48x16384, T=256x64x64, L1=8x32` 一筆交界
候選，並強制 searched、bank、official 同輪量測。

DETERMINISTIC_SPLIT_K 只在 M/N/K 對齊時切換來源直接讀取的
`iterateOrder`。SINGLE_CORE_SPLIT_K 只在
`singleCoreN > stepN*baseN` 時啟用來源的 inner-N path。full-load family
沒有可證明不破壞 operand residency 的自由轉換時直接停止，不製造候選。

### 5.1 預設 general search

`general_search_v1` 不使用上述 shape 名稱或人工 N/K 區間。每個 workload
由四個起點來源建立完整合法 schedule：

```text
local     官方 RuntimeKb seed 的耦合鄰域
global    Cube 對齊、L0/L1 容量與 output partition 推導出的離散格點
diverse   與官方 seed 結構距離較遠、但代理模型未判為災難的合法點
transfer  從其他強實測點轉移 partition count，再重建目標 L1/L2 欄位
```

Host 候選上限分別為 12/16/16/12，去重後每 workload 最多 60 筆並全部經過
官方 callback。這是供模型選擇的較深前沿，不是 NPU 預算；每輪 NPU 最多
量 16 筆，且先保留每個仍有未測候選的來源 leader。

完成一輪後，完整 23-field fingerprint 進入版本化 manifest。下一輪仍會
構造相同受約束空間，但 exact fingerprint 被 active frontier 排除，因而會
向每個來源的下一層移動。穩定的 source-best 同輪
`candidate/bank/official` 結果另作模型殘差；目前採各來源 upper-quartile
的保守倍率，只修正模型低估，不把歷史好結果當成所有未測點都會加速。

## 6. Constraint-Aware Beam Search

完整 state 按依賴分成五層 prefix：

```text
1. template + usedCore + singleCoreMNK
2. baseMNK
3. step/depth + iterateOrder
4. L0 DB
5. L2 schedule
```

對每個 prefix，程式取其所有合法 completion 中模型分數最低者作代表，每層
只保留前 `BEAM_WIDTH` 個 prefix。模板各自擁有 beam，避免數量大的 BASE
空間擠掉 Split-K/full-load。

這一做法採用 ROLLER 的 construction-based、hardware-aligned tile 思路：
先以執行單元、transaction 與 memory capacity 限定 shape，再做搜索，而不是
對完整整數空間盲抽樣。

在 general scope 中，Beam 只排序上述完整合法候選，不自行生成新欄位組合。
`general_source_frontier` 先保留各起點來源，再依真機校正後分數與結構距離
填滿固定預算。因此 Beam width 不會把候選數擴張為 `width^layers`。

## 7. Tabu 與 Large Neighborhood Search

Beam 前四個 seed 分別做離散鄰域搜索。語義欄位分成：

```text
core/singleCore
base/L1/DB
iterateOrder
L2 schedule
```

Tabu move 只改一組，32-entry tabu queue 防止反覆往返。LNS 每輪放鬆一組或
兩組，其餘組固定，可跨過單步鄰域的局部障礙。所有 move 都從已建立的
hard-legal pool 中取值。

這與 Ansor 的 hierarchical representation + fine-tuning 精神相同，但此處
不是生成任意程式，而是在 CANN 已編譯好的 MatMulV3 模板內搜索 23 欄
schedule。由於目前 NPU 資料不足，沒有假裝使用已訓練好的 learned model。

## 8. 模板感知代理模型

模型不配置大矩陣、不跑 CPU simulator，使用 closed-form axis sum，因此不會
因 8192/65536 形狀建立數百萬 tile 物件。

### 8.1 Active core 與尾輪

```text
logicalTiles = output partitions 或 split-K partitions
activeCores  = min(usedCoreNum, logicalTiles)
roundBalance = ceil(logicalTiles/activeCores)
               / (logicalTiles/activeCores)
critical(stage) = total(stage)/activeCores * roundBalance
```

HBM/L2 bandwidth 分母也使用 active core，不會讓空閒 core 虛構額外帶寬。

### 8.2 Pipeline stages

模型分別估算：

- Cube：M0=N0=16、dtype K0、base subtile 與 fill/drain。
- MTE2：L2->L1 bytes、L1 reuse window 與 transaction startup。
- MTE1：L1->L0 bytes、base subtile 數與 startup。
- Fixpipe：輸出 transaction、tail 與 `dbL0C` overlap。
- L2/HBM：operand 重讀、L2 tile 工作集與尾 tile parallelism。
- ND2NZ：callback blob 的 `baseAN/baseAD/baseBN/baseBD`。
- Split-K：workspace、atomic/reduction、AIV cast。
- AL1/BL1：完整 operand 每 active AIC 載入一次的 reuse。
- Vec NZ2ND：額外 workspace 與 vector movement。

輸入與 Cube 在雙緩衝時以 overlap critical path 建模；未雙緩衝時相加。
模型 cycle 只在同一 workload 內排序，不轉成虛假的毫秒。

### 8.3 模型可信度

早期 OCR 候選資料的 log-latency Pearson 約 0.14，中位絕對百分比誤差約
8.9%。後續雖已補入成功 full run 的 baseline/control 與部分完整候選，
這個結果仍說明廣域代理模型不足以證明最佳值。

所以目前 general search 策略是：

1. 先由 kernel/template 與硬體容量建立每 workload 最多 60 筆合法前沿。
2. 同 run 有 bank control 的穩定歷史使用
   `candidate_ms/bank_control_ms` 校準，exact fingerprint 不再量測。
3. 跨 clone 的穩定 source-best 殘差對模型低估加保守懲罰；每個來源仍保留
   leader，避免模型偏差完全關閉全域或遠距探索。
4. 每輪每 workload 最多量 16 筆，candidate、bank、official 必須同輪且
   coefficient of variation 不高於 5% 才進入模型回饋。
5. 最終只相信 correctness 後的 NPU latency。

目前 44 筆 source-best 只足以做低維保守校正，還不足以訓練聲稱可泛化的
Bayesian/TPE/神經 cost model。後續每輪保留完整候選量測後，才評估以
template、dtype/layout 與硬體衍生特徵訓練 residual model。

`bottleneck_guided_v1` 將 `TABU_ITERS=LNS_ROUNDS=0`，不進入本節流程。這是
對先前廣域模型失準的修正；Tabu/LNS 程式只供全模板 reference scope 使用。

## 9. 三層合法性證據

每個輸出到 NPU 的 searched candidate 必須通過：

### 層 1：硬約束

對齊、L0/L1、DB、step/depth、partition、L2 coverage 與各模板契約。

### 層 2：官方 callback

以 23 欄 knowledge 呼叫安裝版 MatMulV3 callback，要求：

```text
callback family == requested family
callback suffix 屬於 CANN 8.1 的 12 個 dispatch
callback 回傳的 23 欄 == 候選 23 欄
```

並保存完整 blob、tiling key、block dim、workspace 與 SHA-256。callback
在 bank lookup 後仍會產生 ND2NZ metadata，因此不能只保存 23 欄。

### 層 3：RuntimeKb 與 NPU

每筆候選建立獨立 runtime bank，`RuntimeKb::QueryBank` 必須 `found=1`。
這只證明 schema/hash/query 成立。真正可執行性由官方 `aclnnMatmul` 的輸出
coverage/numeric preflight 證明，通過後才允許 ACL Event 計時。

## 10. Baseline、歷史資料與決策

每個 workload 都有：

```text
official_operator
bank_seed_control
searched Top-K
```

bank control 用來隔離 RuntimeKb 路徑本身造成的 ND2NZ metadata 差異。只有
searched 同時顯著勝過兩個 reference 才標記 `improved`。

`results/npu_full_ocr_measurements.csv` 以以下 exact fingerprint 重用：

```text
SoC + AIC + workload + shape + dtype + transpose
+ family + baseMNK + singleCoreMNK + cores
+ L1/DB + output grid + L2 schedule
```

相同成功候選不重測。不同任何一個排程欄位都視為新候選，不能把舊時間複製
過去。bank seed control 只有完整 fingerprint 相同時才可重用。

## 11. 已完成驗證

`scripts/validate_cpu.sh` 不啟動 NPU 或 simulator。active campaign 的
CANN 8.1.RC1、Ascend910B3 contract validation 結果：

```text
unit tests                       passed
workloads                       52
supported bank controls         51
selected candidates             66
searched candidates             15
callback/RuntimeKb validation   all found
measured searched schedules     14
new crossover schedules          1
max candidates/workload          1 in active scope
template contract               95 rows / 12 suffixes / 7 families
exact-resume                    fingerprint + explicit resume policy
```

這只證明候選符合 callback 與 RuntimeKb 注入契約。既有 NPU 結果證實
K=16384 skinny-N 在五個獨立窄 N shape 上泛化，baseN=48 在
N=33/40/47 改善，baseN=64 在 N=49/56/64 改善；同時 K=65536 否定
deterministic split-K traversal 的無界外推。active search 現在只送出
一個尚未量測的 N=48 baseN=64 crossover fingerprint。

## 12. 研究依據

- 安裝版 CANN 8.1 `mat_mul_v3.cpp` 與同目錄 kernel headers：本版本實際
  dispatch、core mapping、Split-K、full-load、Fixpipe/Vec 的首要證據。
- [公開 MatMulV3 來源樹（CANN ops-nn 8.5）](https://gitcode.com/cann/ops-nn/tree/8.5.0/matmul/mat_mul_v3)：
  `matmul_v3_base_tiling.cpp` 與 `matmul_v3_l2_cache.cpp` 用來判斷
  traversal 與模板控制流；版本不同處仍以本機 8.1 callback/RuntimeKb
  roundtrip 為準。
- [Ascend C MultiCoreMatmulTiling 使用說明](https://www.hiascend.com/document/detail/zh/canncommercial/700/operatordev/Ascendcopdevg/atlasascendc_api_07_0220.html)
- [Ascend C Matmul tiling API](https://www.hiascend.com/document/detail/en/canncommercial/850/API/ascendcopapi/atlasascendc_api_07_0671.html)
- [ROLLER: Fast and Efficient Tensor Compilation](https://www.usenix.org/conference/osdi22/presentation/zhu)
- [Ansor: Generating High-Performance Tensor Programs](https://www.usenix.org/system/files/osdi20-zheng.pdf)
- [Learning to Optimize Tensor Programs / AutoTVM](https://papers.nips.cc/paper/2018/file/8b5700012be65c9da25f49408d959ca0-Paper.pdf)
- `DaVinci_AIC_V220_ISA_User_Guide2-910B.pdf`：Cube/MTE/Vector/Fixpipe
  pipeline 與記憶體層級參照。
- 使用者提供的 `MultiCoreMatmulTiling.cpp`：API 行為旁證；若來源是反組譯
  或重建檔，不把它宣稱為華為官方原始碼。
