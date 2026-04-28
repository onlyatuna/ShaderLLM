# Slang-TMA 轉型技術規範 (Draft v1.0)

## 1. 核心目標
將 CHIMERA-V 的運算核心從手動記憶體搬運切換為 **Tensor Memory Accelerator (TMA)**，利用 RTX 50 系列硬體的非同步搬運功能，預期將 VRAM 到 Shared Memory 的頻寬利用率提升至 90% 以上。

## 2. Slang 嵌入 PTX 語法
針對 Slang 2026.7，建議使用 `__asm` 區塊直接操作 TMA 暫存器：

```slang
void loadTMA_Async(void* sharedDest, void* globalSrc, uint size) {
    __asm {
        "cp.async.bulk.tensor.default.global.shared::cluster [%0], [%1], %2;" 
        : 
        : "r"(sharedDest), "r"(globalSrc), "n"(size)
    };
}
```

## 3. Cooperative Matrix 配對
TMA 搬運後的數據應直接與 `linalg::CoopMat` 對齊。

- **佈局要求**：Shared Memory 必須維持 `Stride + 8` 錯位以消除 Bank Conflict。
- **類型對齊**：
    - `MatrixA`: 16x16 `half`
    - `MatrixB`: 16x16 `half`
    - `Accumulator`: 16x16 `float`

## 4. 軟體流水線 (Double Buffering)
實作 TMA 時必須配合雙緩衝區：
1. 發射 TMA 搬運 `Tile N+1` 到 `SMEM_Buf_1`。
2. 同時執行 Tensor Core 運算於 `SMEM_Buf_0` (Tile N)。
3. 使用 `cp.async.wait_group 0` 確保下一輪搬運完成。

## 5. 轉型期驗證清單
- [ ] `slangc` 編譯無誤 (Target: SPIR-V / NVPTX)
- [ ] `OpCooperativeMatrixMulAdd` 出現在 SPIR-V 彙編中
- [ ] 執行時間低於 10ms (針對 2560-dim NCA 演化步)
