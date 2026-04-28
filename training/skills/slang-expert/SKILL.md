---
name: slang-expert
description: 針對 Slang 著色器語言、TMA (Tensor Memory Accelerator) 轉型與 RTX 50 系列硬體優化的專家技能。適用於將 GLSL/HLSL 遷移至 Slang，並實作極限效能的 Cooperative Matrix 與非同步記憶體搬運。
---

# Slang Expert | CHIMERA-V 專用

## 概述
本技能旨在引導工程師利用 Slang 語言的強大特性（如泛型、介面、Inline PTX），將 CHIMERA-V 的運算核心從傳統 GLSL 轉型至 Slang-TMA 架構，以實現 RTX 5060 Ti 的極限效能。

## 核心工作流：TMA 轉型期

### 1. 著色器遷移 (GLSL -> Slang)
- **進入點標記**：使用 `[shader("compute")]` 與 `[numthreads(x, y, z)]`。
- **類型映射**：將 GLSL 的 `readonly buffer` 轉換為 Slang 的 `StructuredBuffer<T>` 或 `ByteAddressBuffer`。

### 2. Cooperative Matrix (NVIDIA 12.0+)
- **定義**：`CooperativeMatrix<T, rows, cols, layout>`。
- **優勢**：在 Slang 中使用原生 `half` 與 `float` 進行混和精度運算，編譯器會自動生成優化的 `mma.sync` 指令。

### 3. Slang-TMA (Tensor Memory Accelerator)
- **非同步搬運**：利用 Slang 的 `__asm` 塊或內建函式調用 `cp.async.bulk`。
- **目標**：繞過 L1/Shared Memory 的手動搬運循環，由硬體直接填充 Shared Memory。

## 程式碼範例

### Cooperative Matrix 宣告
```slang
// 16x16 Half-Precision Matrix
CooperativeMatrix<half, 16, 16> matA;
CooperativeMatrix<half, 16, 16> matB;
CooperativeMatrix<float, 16, 16> matC; // Accumulator
```

### 進入點與資源綁定
```slang
struct Params {
    uint2 dims;
};

ParameterBlock<Params> gParams;
StructuredBuffer<half> gInput;
RWStructuredBuffer<float> gOutput;

[shader("compute")]
[numthreads(128, 1, 1)]
void main(uint3 threadId : SV_DispatchThreadID) {
    // 實作邏輯...
}
```

## 資源索引
- **references/tma_spec.md**: 詳細的 TMA 指令集與 PTX 嵌入規範。
- **references/coopmat_layout.md**: 針對 GDDR7 頻寬優化的記憶體佈局指南。
- **scripts/compile_slang.cjs**: 自動化 Slang 到 SPIR-V 的編譯工具，支援 `-target spirv`。
