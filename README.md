# snn-brain

**脉冲神经网络（SNN）训练测试 APK** —— 在手机上用摄像头与麦克风实时训练一个脉冲大脑模型。

## 用途

本应用用于**测试 SNN 网络的训练**：真实世界的视觉（摄像头视频流）与听觉（麦克风）信号直接编码为神经脉冲，驱动一个多尺度脉冲大脑学习识别物体与声音。全部学习、记忆、成长由大脑自主调度，无手动控制按钮。

## 功能特性

- 📷 **视觉脉冲编码**：摄像头预览帧 → 视网膜感受野拮抗（DoG）→ V1 方向选择性（4 方向 × 4 尺度金字塔）→ **5120 维神经信号**
- 🎤 **听觉脉冲编码**：麦克风 → 耳蜗 128 对数频带 → 听神经发放率
- 🧠 **脉冲神经网络**：LIF / HH / ALIF / PLIF / 多腔室（树突 NMDA 整合）神经元；Hebbian / pp-prop / D-RTRL / SparseRTRL / STDP 在线学习；皮层局部连接 + 稳态可塑性
- 🧩 **认知模块**：联想记忆、工作记忆（4 槽）、意识（GWT + Φ）、睡眠巩固重放、多巴胺奖赏、预测引擎、跨模态绑定（视觉↔听觉）、递归元认知
- 🔁 **自然成长**：模仿期（复读）→ 理解期 → 自主期；咿呀 → 模仿 → 自主发声（学习进化的模拟声带）
- 📊 **可视化**：大脑视角（5120 维电信号解码）、脑图、音波、连接拓扑
- 🔄 **自主调度**：自动观察、自动听环境、自动睡眠、自动备份（.brainx 快照跨会话记忆）

## 构建

```bash
export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
export ANDROID_HOME=/opt/android-sdk

# JVM 测试 (303 项全绿)
gradle :brainx-core:test

# 打包 APK
gradle :brainx-android:assembleDebug
# 产物: brainx-android/build/outputs/apk/debug/brainx-android-debug.apk
```

## 项目结构

```
brainx-core/     纯 Java 17 算法库（无 Android 依赖，JVM 可测）
  io/brainx/core/
    neuron/      LIF ALIF Izhikevich HH PLIF SpikingGamma 多腔室神经元
    learning/    PPProp DRTRL SparseRTRL DualNumber(自动微分)
    mass/        JansenRit WongWang WilsonCowan Kuramoto 神经群
    encoding/    σ-δ 编码 / TTFS / ITD 双耳
    synapse/     STDP / Synapse
    ...          工作记忆 意识 分层记忆 多巴胺 预测引擎 跨模态 虚拟神经元层 等
brainx-android/  APK 模块（摄像头/麦克风/声带/可视化）
```

## 技术说明

- 神经元/突触/学习算法对应开源脑仿真生态论文实现（BrainTrace / SpikingGamma / STEP / BrainFuse / fWBM 等），全部算法有 JVM 测试对照论文数值
- 物理神经元 5120+ 视觉 + 128 听觉 + 联想 + 中枢；虚拟神经元层以神经群抽象达到 2000 亿等效规模
- 模型状态可导出 .brainx 纯文本快照，跨平台/跨会话恢复

## 许可

GPLv3
