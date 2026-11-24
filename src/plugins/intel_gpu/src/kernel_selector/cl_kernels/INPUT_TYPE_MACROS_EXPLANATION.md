# OpenCL Kernel Input Type Macros Explanation

## 问题 (Question)
在 `src/plugins/intel_gpu/src/kernel_selector/cl_kernels/matrix_nms_ref.cl` 文件中使用了 `INPUT1_TYPE` 宏，它是在哪里定义的？类型是由什么决定的？

## 回答 (Answer)

### INPUT1_TYPE 的定义位置
`INPUT1_TYPE` **不是**在任何 `.cl` 或 `.h` 头文件中静态定义的。它是由 OpenVINO 的 **JIT (Just-In-Time)** 编译系统在运行时动态生成的宏定义。

### 定义过程

#### 1. 入口点：MatrixNmsKernelRef 类
文件位置: `src/plugins/intel_gpu/src/kernel_selector/kernels/matrix_nms/matrix_nms_kernel_ref.cpp`

当创建 Matrix NMS 内核时，`GetKernelsData` 方法会被调用：
```cpp
JitConstants MatrixNmsKernelRef::GetJitConstants(const matrix_nms_params& params) const {
    JitConstants jit = MakeBaseParamsJitConstants(params);
    // ... 其他常量定义
    return jit;
}
```

#### 2. 基础参数 JIT 常量生成
文件位置: `src/plugins/intel_gpu/src/kernel_selector/kernel_base.cpp`

`MakeBaseParamsJitConstants` 函数为所有输入和输出张量创建 JIT 常量：
```cpp
JitConstants KernelBase::MakeBaseParamsJitConstants(const base_params& params, ...) const {
    // ...
    for (size_t i = 0; i < params.inputs.size(); i++) {
        jit.AddConstant(MakeJitConstant("INPUT" + toCodeString(i), params.inputs[i]));
    }
    // ...
}
```

对于 Matrix NMS：
- `INPUT0` 对应 `params.inputs[0]` - boxes（边界框）
- `INPUT1` 对应 `params.inputs[1]` - scores（分数）

#### 3. 张量 JIT 常量
文件位置: `src/plugins/intel_gpu/src/kernel_selector/jitter.cpp`

`MakeJitConstant` 为 DataTensor 创建 `DataTensorJitConstant` 对象：
```cpp
std::shared_ptr<JitConstant> MakeJitConstant(const std::string& name, const DataTensor& value) {
    return std::static_pointer_cast<JitConstant>(std::make_shared<DataTensorJitConstant>(name, value));
}
```

`DataTensorJitConstant::GetDefinitions()` 生成所有张量相关的宏定义，包括：
```cpp
JitDefinitions DataTensorJitConstant::GetDefinitions() const {
    JitDefinitions baseDefinitions = TensorBaseTJitConstant::GetDefinitions(_tensor);
    // 生成维度相关的宏
    // 生成索引访问宏
    // ...
}
```

#### 4. 类型宏生成 - 核心功能
在 `TensorBaseTJitConstant::GetDefinitions()` 中调用：
```cpp
auto type_defs = MakeTypeJitConstants(t.GetDType(), _name).GetDefinitions();
definitions.insert(definitions.end(), type_defs.begin(), type_defs.end());
```

`MakeTypeJitConstants` 函数根据数据类型生成所有类型相关的宏：

```cpp
JitConstants MakeTypeJitConstants(Datatype dataType, const std::string& macroName) {
    std::string type = "undefined";
    std::string max_val, min_val, val_one, val_zero;
    // ... 其他变量
    
    switch (dataType) {
        case Datatype::F16:
            type = "half";
            max_val = "HALF_MAX";
            min_val = "-" + macroName + "_VAL_MAX";
            val_one = "1.0h";
            val_zero = "0.0h";
            // ...
            break;
        case Datatype::F32:
            type = "float";
            max_val = "FLT_MAX";
            min_val = "-" + macroName + "_VAL_MAX";
            val_one = "1.0f";
            val_zero = "0.0f";
            // ...
            break;
        case Datatype::INT32:
            type = "int";
            max_val = "INT_MAX";
            min_val = "INT_MIN";
            val_one = "(int) 1";
            val_zero = "(int) 0";
            // ...
            break;
        // ... 其他类型
    }
    
    return JitConstants{
        MakeJitConstant(macroName + "_TYPE", type),
        MakeJitConstant(macroName + "_VAL_MAX", max_val),
        MakeJitConstant(macroName + "_VAL_MIN", min_val),
        MakeJitConstant(macroName + "_VAL_ONE", val_one),
        MakeJitConstant(macroName + "_VAL_ZERO", val_zero),
        MakeJitConstant("TO_" + macroName + "_TYPE(v)", to_type),
        // ... 其他宏
    };
}
```

### INPUT1_TYPE 的实际值由什么决定？

**INPUT1_TYPE 的类型由 Matrix NMS 操作的第二个输入张量（scores）的数据类型决定。**

支持的数据类型（在 `matrix_nms_kernel_ref.cpp` 中定义）：
```cpp
ParamsKey MatrixNmsKernelRef::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);  // 16位浮点
    k.EnableInputDataType(Datatype::F32);  // 32位浮点
    k.EnableInputDataType(Datatype::INT32); // 32位整数
    // ...
}
```

### 生成的宏示例

当 `params.inputs[1]` 的数据类型是 `Datatype::F32` 时，会生成：
```c
#define INPUT1_TYPE float
#define INPUT1_VAL_MAX FLT_MAX
#define INPUT1_VAL_MIN -INPUT1_VAL_MAX
#define INPUT1_VAL_ONE 1.0f
#define INPUT1_VAL_ZERO 0.0f
#define TO_INPUT1_TYPE(v) convert_float(v)
#define TO_INPUT1_TYPE_SAT(v) convert_float(v)
#define AS_INPUT1_TYPE(v) as_float(v)
#define INPUT1_MAX_FUNC fmax
#define INPUT1_MIN_FUNC fmin
#define INPUT1_ABS_FUNC fabs
#define INPUT1_TYPE_SIZE 4
#define INPUT1_IS_FP 1
```

当数据类型是 `Datatype::F16` 时，会生成：
```c
#define INPUT1_TYPE half
#define INPUT1_VAL_MAX HALF_MAX
#define INPUT1_VAL_MIN -INPUT1_VAL_MAX
#define INPUT1_VAL_ONE 1.0h
#define INPUT1_VAL_ZERO 0.0h
#define TO_INPUT1_TYPE(v) convert_half(v)
// ... 等等
```

### 在 matrix_nms_ref.cl 中的使用

在 `matrix_nms_ref.cl` 中，这些宏被用于：

1. **定义结构体成员类型**：
```c
typedef struct {
    int batch_idx;
    int class_idx;
    int box_idx;
    INPUT1_TYPE score;  // 分数的类型由输入张量决定
} FUNC(BoxInfo);
```

2. **函数参数和返回类型**：
```c
inline INPUT1_TYPE FUNC(decay_gaussian)(INPUT1_TYPE iou, INPUT1_TYPE max_iou) {
    return exp((max_iou * max_iou - iou * iou) * GAUSSIAN_SIGMA);
}
```

3. **常量值**：
```c
if (score > INPUT1_VAL_ZERO) { ... }
if (min_decay == INPUT1_VAL_ONE) { ... }
```

### 总结

1. **定义位置**：`INPUT1_TYPE` 在运行时由 JIT 系统生成，不在任何静态文件中定义
2. **生成代码位置**：`src/plugins/intel_gpu/src/kernel_selector/jitter.cpp` 中的 `MakeTypeJitConstants` 函数
3. **类型决定因素**：由 Matrix NMS 算子的第二个输入张量（scores）的数据类型决定
4. **支持的类型**：F16 (half)、F32 (float)、INT32 (int)
5. **相关文件**：
   - 宏使用：`cl_kernels/matrix_nms_ref.cl`
   - 内核定义：`kernels/matrix_nms/matrix_nms_kernel_ref.cpp`
   - JIT 系统：`jitter.cpp` 和 `kernel_base.cpp`

这种 JIT 方法的优势是可以为不同的数据类型生成优化的 OpenCL 代码，而无需为每种类型编写单独的内核。
