package io.brainx.core;

import io.brainx.core.encoding.Encoders;
import io.brainx.core.encoding.TTFS;
import io.brainx.core.learning.PPProp;
import io.brainx.core.mass.JansenRit;
import io.brainx.core.mass.WongWang;
import io.brainx.core.neuron.LIF;
import io.brainx.core.neuron.SpikingGamma;
import io.brainx.core.synapse.STDP;

import java.util.*;

/**
 * Brain 整合类 —— 模拟"人类大脑"的模块化组装。
 * 对应 fWBM 论文 (arXiv 2605.18118) 的模块化架构：
 *   感觉输入(视觉/听觉编码) → 皮层网络(SNN) → 决策(神经群) → 输出
 * 支持在线学习 (pp-prop) + 联想记忆 (Hebbian/STDP) —— "从学语到认知"。
 */
public class Brain {
    // 感觉层
    private final Encoders.Poisson visualEncoder;
    private final Encoders.SigmaDelta auditoryEncoder;
    private final Encoders.ITD itd;

    // 皮层: 视觉/听觉/联想层
    private final int visualCortexSize, auditoryCortexSize, associationCortexSize;
    private final LIF[] visualCortex, auditoryCortex, associationCortex;

    // 在线学习器 (视觉→联想, 听觉→联想)
    private final PPProp visualToAssoc, auditoryToAssoc;

    // 联想记忆 (Hebbian 权重: 概念→发音/反应)
    private final double[][] associativeMemory;
    private final String[] vocabulary;   // 词表 (学到的词)

    // 决策模块 (Wong-Wang)
    private final WongWang decision;
    // 脑节律 (Jansen-Rit α 波)
    private final JansenRit rhythm;
    // STDP 可塑性
    private final STDP stdp;

    // 自发神经活动 (随机脑信号)
    private final NoiseSource backgroundNoise;   // 泊松背景噪声
    private final NoiseSource gaussianNoise;     // 膜电位抖动
    private final SynapseFormation synapseFormation;  // 自发突触形成
    private final boolean[] firingState;         // 当前发放状态 (供突触形成)

    // 论文机制: E/I 平衡决策网络 + TTFS 时间编码 + 睡眠巩固
    private final EIBNetwork eibNetwork;
    private final TTFS ttfs;
    private final SleepConsolidation sleep;
    private final List<int[]> dayCoactivations = new ArrayList<>();  // 白天共激活记录

    // 认知模块: 工作记忆 (fWBM: 自持续活动) + 意识 (GWT 全局工作空间)
    private final WorkingMemory workingMemory;
    private final Consciousness consciousness;

    // 自我意识与记忆 (用户框架: 镜像自我/元认知/叙事 + 分层记忆 + 多感觉整合)
    private final SelfAwareness selfAwareness;
    private final HierarchicalMemory hierarchicalMemory;
    private final MultiSensoryIntegration msi;
    private String lastVisualLabel = "", lastAuditoryLabel = "";

    // 认知模式 (Kosslyn 上脑/下脑) + 僵尸行动者 (Koch 技能自动化)
    private final CognitiveMode cognitiveMode;
    private final ZombieAgent zombieAgent;

    // 多巴胺奖赏 (错误驱动学习) + 预测引擎 (大脑=科学家)
    private final DopamineSystem dopamineSystem;
    private final PredictiveEngine predictiveEngine;
    private final InternalModel internalModel;  // 前向预测/感觉预测误差 (书中20篇)
    private final EmotionalVoice emotionalVoice;  // 声音情感识别 (书中23篇)
    private final LanguageLearner languageLearner;  // 语言学习: 模仿→理解→自主
    private final GrowthPotential growthPotential;  // 成长潜力: 神经元→2000亿
    private final VirtualNeuronLayer virtualLayer;  // 虚拟神经元层: 神经形态等效规模
    private final HomeostaticPlasticity homeostaticPlasticity;  // 稳态可塑性 (防静默/暴走)
    private final SpeechResponder speechResponder;  // 语音对话+逗乐互动

    // 声带学习进化 (听觉-发声回路: 听到→提取参数→模仿)
    private final VoiceLearner voiceLearner;

    // 双通路视觉 (腹侧What+背侧Where/How) + 丘脑中继 (脑干降噪/LGN过滤)
    private final DorsalPathway dorsalPathway;
    private final ThalamicRelay visualRelay, auditoryRelay;
    /** 跨模态记忆: 视觉↔听觉原型绑定 (多模态学习与输出) */
    private final CrossModalMemory crossModalMemory;
    /** 时序预测器: 感官前向模型 (预测下一帧 + 延迟脑补) */
    private final FeaturePredictor visualPredictor, auditoryPredictor;

    // 类脑频率波: 信号=频率振荡 + 记忆=频率共振 (整体联动)
    private final FrequencyWave neuralWave;      // 神经元频率波驱动
    private final ResonanceMemory resonanceMemory;  // 联想记忆的频率共振层
    private final double[] memoryFreqHz;         // 每个词的共振频率 (联想层)
    private final FrequencyBus frequencyBus;     // 全脑频率总线 (模块间联动)

    // 中枢脉冲网络 (皮层-丘脑环路: 全模块统一脉冲联动)
    private final CentralHub centralHub;
    private final PulseModule[] hubModules;      // 注册到中枢的模块
    private final SensoryCortexBridge visualBridge, auditoryBridge;  // 感觉皮层桥接
    private final AssocBridge assocBridge;  // 联想皮层桥接 (全局工作空间节点)

    // EEG 发生器 (脉冲聚合→脑电波→回馈调制, 书中"聚合的脑活动")
    private final EEGGenerator eegGenerator;

    // 运行功率配置 (自适应: 手机性能→模型复杂度)
    private PowerManager.Profile powerProfile = PowerManager.PROFILES_STANDARD;

    private final Random rnd = new Random(2026);
    private final List<String> learnedWords = new ArrayList<>();
    // 学习循环复用缓冲区 (JIT 热路径优化: 避免每次学习分配大数组)
    private double[] learnInputBuf;
    private int[] learnSpikesBuf;

    public Brain(int visualSize, int auditorySize, int assocSize, String[] vocabulary) {
        this.visualCortexSize = visualSize;
        this.auditoryCortexSize = auditorySize;
        this.associationCortexSize = assocSize;
        this.vocabulary = vocabulary;

        this.visualEncoder = new Encoders.Poisson(1);
        this.auditoryEncoder = Encoders.SigmaDelta.defaultParams();
        this.itd = Encoders.ITD.defaultParams();

        this.visualCortex = new LIF[visualSize];
        this.auditoryCortex = new LIF[auditorySize];
        this.associationCortex = new LIF[assocSize];
        for (int i = 0; i < visualSize; i++) visualCortex[i] = LIF.defaultParams();
        for (int i = 0; i < auditorySize; i++) auditoryCortex[i] = LIF.defaultParams();
        for (int i = 0; i < assocSize; i++) associationCortex[i] = LIF.defaultParams();

        this.visualToAssoc = new PPProp(visualSize, assocSize, 0.95, 0.0005);
        this.auditoryToAssoc = new PPProp(auditorySize, assocSize, 0.95, 0.0005);
        this.associativeMemory = new double[assocSize][vocabulary.length];
        this.decision = WongWang.defaultParams();
        this.rhythm = JansenRit.defaultParams();
        this.stdp = STDP.defaultParams();

        // 自发神经活动: 泊松背景 (8Hz) + 高斯抖动 (σ=0.1) + 突触形成
        this.backgroundNoise = NoiseSource.poissonBackground(1.0);
        this.gaussianNoise = NoiseSource.gaussian(0.1, 1.0);
        this.synapseFormation = SynapseFormation.defaultSpatialParams(
                visualSize + auditorySize + assocSize, VisualNeuralEncoder.RF_GRID);
        this.firingState = new boolean[visualSize + auditorySize + assocSize];
        // 稳态可塑性: 视觉+听觉皮层发放率稳态调节 (防静默/暴走)
        this.homeostaticPlasticity = new HomeostaticPlasticity(visualSize + auditorySize);

        // 论文机制初始化
        this.eibNetwork = EIBNetwork.defaultParams();
        this.ttfs = TTFS.defaultParams();
        this.sleep = SleepConsolidation.defaultParams();

        // 认知模块初始化: 工作记忆 (4槽×32维, Miller 容量) + 意识 (清醒)
        this.workingMemory = WorkingMemory.defaultParams();
        this.consciousness = new Consciousness();

        // 自我意识与分层记忆初始化
        this.selfAwareness = new SelfAwareness();
        this.hierarchicalMemory = HierarchicalMemory.defaultParams();
        this.msi = MultiSensoryIntegration.defaultParams();

        // 认知模式 + 僵尸行动者初始化
        this.cognitiveMode = new CognitiveMode();
        this.zombieAgent = ZombieAgent.defaultParams();

        // 多巴胺 + 预测引擎初始化
        this.dopamineSystem = new DopamineSystem();
        this.predictiveEngine = new PredictiveEngine();
        this.internalModel = new InternalModel(16);  // 前向预测模型 (16维状态)
        this.emotionalVoice = new EmotionalVoice();
        this.languageLearner = new LanguageLearner();
        this.growthPotential = GrowthPotential.defaultParams();
        // 虚拟神经元层: 神经形态等效规模 (物理×抽象 → 最高2000亿等效, 真实参与脉冲统计)
        this.virtualLayer = new VirtualNeuronLayer(visualSize + auditorySize + assocSize);
        this.speechResponder = new SpeechResponder();
        this.voiceLearner = new VoiceLearner();
        this.dorsalPathway = new DorsalPathway();
        this.visualRelay = new ThalamicRelay();   // LGN 中继 (视觉)
        this.auditoryRelay = new ThalamicRelay(); // 脑干+丘脑中继 (听觉)
        // 跨模态记忆: 视觉原型 ↔ 听觉原型绑定 (多模态学习)
        this.crossModalMemory = new CrossModalMemory(vocabulary.length, visualSize, auditorySize);
        // 时序预测器: 感官前向模型 (人脑预测脑补 — 平滑追踪/小脑前向模型延迟补偿)
        this.visualPredictor = new FeaturePredictor(visualSize);
        this.auditoryPredictor = new FeaturePredictor(auditorySize);

        // 类脑频率波初始化: 神经元频率波 + 记忆频率共振层
        this.neuralWave = FrequencyWave.neuralRange();
        this.resonanceMemory = ResonanceMemory.defaultParams();
        this.frequencyBus = new FrequencyBus();
        this.memoryFreqHz = new double[vocabulary.length];
        for (int i = 0; i < vocabulary.length; i++) {
            // 每个词分配一个共振频率 (均匀分布 4-13Hz θ-α 带)
            memoryFreqHz[i] = 4.0 + 9.0 * i / Math.max(1, vocabulary.length - 1);
        }

        // 中枢脉冲网络: 注册全部认知/记忆模块 + 感觉皮层 + 联想皮层 (统一脉冲联动)
        this.visualBridge = new SensoryCortexBridge(visualCortex, "视觉皮层");
        this.auditoryBridge = new SensoryCortexBridge(auditoryCortex, "听觉皮层");
        this.assocBridge = new AssocBridge(assocSize);
        this.hubModules = new PulseModule[]{
                visualBridge, auditoryBridge, assocBridge,
                workingMemory, hierarchicalMemory, resonanceMemory, consciousness
        };
        int totalPulseDim = 0;
        for (PulseModule m : hubModules) totalPulseDim += m.pulseDim();
        this.centralHub = CentralHub.defaultParams(totalPulseDim);
        this.eegGenerator = new EEGGenerator();
    }

    /**
     * 简化大脑: 32视觉 + 16听觉 + 24联想 + 8概念槽。
     * 概念槽由大脑自主命名 (概念#1..#8), 不预设人类词名 — 一切自由学习。
     */
    public static Brain simpleBrain() {
        String[] concepts = new String[8];
        for (int i = 0; i < concepts.length; i++) concepts[i] = "概念#" + (i + 1);
        // 视觉 5120 维 + 听觉 128 频带 + 联想 120 (5倍) + 中枢 120 (5倍)
        return new Brain(VisualNeuralEncoder.OUTPUT_DIM, AudioNeuralEncoder.BANDS, 120, concepts);
    }

    // ============ 自主自由学习 (v5.4: 不预设名称, 大脑自主归类命名) ============

    /**
     * 自主学习 (视觉): 大脑自己决定这是新概念还是已知概念。
     *   识别置信度高 → 归入该概念 (已学过的强化);
     *   未知/低置信 → 新建概念 (占用空闲概念槽, 自主命名 概念#N)。
     * @return 归入/新建的概念索引
     */
    public int learnVisual(double[] imageFeatures) {
        String[] r = recognizeVisualWithConfidence(imageFeatures);
        int idx = conceptIndexFor(r[0]);
        if (idx < 0 || Double.parseDouble(r[1]) < 0.5) {
            idx = firstUnusedConcept();
        }
        learnVisualWord(imageFeatures, idx);
        return idx;
    }

    /** 自主学习 (听觉) */
    public int learnAuditory(double[] audioFeatures) {
        String[] r = recognizeAuditoryWithConfidence(audioFeatures);
        int idx = conceptIndexFor(r[0]);
        if (idx < 0 || Double.parseDouble(r[1]) < 0.5) {
            idx = firstUnusedConcept();
        }
        learnAuditoryWord(audioFeatures, idx);
        return idx;
    }

    /** 自主学习 (跨模态绑定): 视觉+听觉同时 → 归入同一概念或新建 */
    public int learnCrossModal(double[] visualFeatures, double[] audioFeatures) {
        String[] r = recognizeMultiModal(visualFeatures, audioFeatures);
        int idx = conceptIndexFor(r[0]);
        if (idx < 0 || Double.parseDouble(r[1]) < 0.5) {
            idx = firstUnusedConcept();
        }
        learnCrossModal(visualFeatures, audioFeatures, idx);
        return idx;
    }

    /** 概念标签 → 概念索引 (未知/空 → -1) */
    private int conceptIndexFor(String label) {
        for (int i = 0; i < vocabulary.length; i++) {
            if (vocabulary[i].equals(label)) return i;
        }
        return -1;
    }

    /** 第一个未被学习的概念槽 (自由学习: 新建概念) */
    private int firstUnusedConcept() {
        for (int i = 0; i < vocabulary.length; i++) {
            boolean used = false;
            for (String w : learnedWords) {
                if (w.equals(vocabulary[i])) { used = true; break; }
            }
            if (!used) return i;
        }
        // 全用满: 返回最近最少用的 (概念循环覆盖, 模拟记忆衰减)
        return 0;
    }

    /**
     * 视觉学习: 输入图像特征向量 (0-1) → 视觉皮层 → 联想皮层 → 关联到词。
     * @param imageFeatures 图像特征 (长度=visualSize)
     * @param wordIndex     关联的词索引 (0..vocabulary-1)
     */
    public void learnVisualWord(double[] imageFeatures, int wordIndex) {
        // 复用缓冲区 (JIT 精神: 避免每次学习分配 2×5120 数组)
        if (learnInputBuf == null || learnInputBuf.length != visualCortexSize) {
            learnInputBuf = new double[visualCortexSize];
            learnSpikesBuf = new int[visualCortexSize];
        }
        double[] input = learnInputBuf;
        for (int i = 0; i < Math.min(imageFeatures.length, visualCortexSize); i++) {
            input[i] = imageFeatures[i];
        }
        // LGN 丘脑中继: 视觉信号经外侧膝状体过滤后进入 V1 (资料: 视网膜→LGN→V1)
        input = visualRelay.thalamicRelay(input);
        // 视觉皮层发放 (频率波驱动: 信号强度→发放频率→振荡波→神经元电流)
        int[] spikes = learnSpikesBuf;
        java.util.Arrays.fill(spikes, 0);
        for (int t = 0; t < powerProfile.learnTimeSteps; t++) {  // 功率自适应步数
            for (int i = 0; i < visualCortexSize; i++) {
                int s = visualEncoder.encode(input[i]);
                spikes[i] += s;
                // 频率波: 信号强度 → 频率波 → 神经元驱动电流 (类脑)
                double waveCurrent = neuralWave.driveNeuron(input[i], 1.0);
                // EEG 回馈: 脑电波调制神经元输入 (脉冲→EEG→调制脉冲 闭环)
                double eegModulated = eegFeedback(waveCurrent + s * 5.0);
                // 中枢下行调制: 丘脑-皮层双向 (注意增强感觉处理)
                double gated = eegModulated * visualBridge.topDownGain();
                // 叠加自发神经活动: 泊松背景 + 高斯膜抖动
                double noise = backgroundNoise.sample() + gaussianNoise.sample();
                // 稳态可塑性: 发放率稳态增益调制输入 (防暴走/静默)
                visualCortex[i].step((gated + noise) * homeostaticPlasticity.gain(i), 1.0);
                firingState[i] = visualCortex[i].fired();
            }
            visualBridge.updateRates();  // 皮层发放率 → 中枢
            // 稳态可塑性更新 (滑动平均发放率 → 增益微调)
            homeostaticPlasticity.step(firingState, 1.0);
            // 自发突触形成: 随机共激活驱动连接建立
            synapseFormation.step(firingState, 1);
            // 记录白天共激活 (睡眠重放源)
            for (int i = 0; i < firingState.length; i++) {
                if (firingState[i]) {
                    int partner = rnd.nextInt(firingState.length);
                    if (partner != i) recordCoactivation(i, partner);
                }
            }
        }
        // 联想皮层: 输入 = 视觉皮层发放率 + pp-prop 前向
        double[] vRate = new double[visualCortexSize];
        for (int i = 0; i < visualCortexSize; i++) {
            vRate[i] = spikes[i] / 20.0;  // 发放率
        }
        double[] assocActivation = visualToAssoc.forward(vRate);
        // Hebbian 联想记忆更新: 概念 → 词 (先更新, 作为 pp-prop 监督目标)
        for (int i = 0; i < associationCortexSize; i++) {
            double a = Math.max(0, Math.tanh(assocActivation[i]));
            associativeMemory[i][wordIndex] += 0.1 * a;
        }
        // pp-prop 在线学习 (BrainTrace Eq.6-8): 误差驱动 — 目标=该词关联列, 信号=目标-当前
        // (v5.4 审计: 原只 forward; 自巩固信号会灾难性漂移 → 监督误差信号)
        double colSum = 0;
        for (int i = 0; i < associationCortexSize; i++) colSum += associativeMemory[i][wordIndex];
        double[] hiddenDeriv = new double[associationCortexSize];
        java.util.Arrays.fill(hiddenDeriv, 1.0);
        visualToAssoc.step(vRate, assocActivation, hiddenDeriv, 0.99);
        double[] learningSignal = new double[associationCortexSize];
        for (int i = 0; i < associationCortexSize; i++) {
            double target = colSum > 0 ? associativeMemory[i][wordIndex] / colSum : 0;
            learningSignal[i] = target - Math.max(0, Math.tanh(assocActivation[i]));  // 误差驱动
        }
        visualToAssoc.update(learningSignal);
        visualToAssoc.applyGradients();
        // 工作记忆: 将当前视觉模式写入工作记忆 (fWBM: 自持续活动保持)
        workingMemory.write(imageFeatures);
        // 意识: 视觉感知进入工作空间广播
        consciousness.perceive(0.8, 0.0, vocabulary[wordIndex], "");
        // 多感觉整合: 记录视觉标签
        lastVisualLabel = vocabulary[wordIndex];
        // 分层记忆: 沉淀为中期(情景)记忆
        hierarchicalMemory.addEpisodic(vocabulary[wordIndex], imageFeatures, 0.6);
        // 认知模式: 学习=下脑感知分类 + 僵尸行动者练习 (技能自动化)
        cognitiveMode.learnActivity(0.05);
        zombieAgent.practice(vocabulary[wordIndex]);
        // 预测引擎: 观察世界建立先验 (大脑=科学家)
        predictiveEngine.observe(vocabulary[wordIndex]);
        // 频率共振记忆: 记忆=频率生成 (写入词的特征频率)
        resonanceMemory.write(vocabulary[wordIndex], 0.15 + 0.1 * rnd.nextDouble());
        // 内部模型: 学习感觉预测 (书中20篇: 前向模型 — 预测感官结果)
        // 状态=输入特征, 实际结果=该词激活强度
        double[] state = new double[Math.min(16, imageFeatures.length)];
        System.arraycopy(imageFeatures, 0, state, 0, state.length);
        internalModel.learn(state, 0.5 + 0.5 * avgIntensity(imageFeatures));
        // 语言学习: 视觉学习同步建立语音模板 (模仿→理解→自主)
        languageLearner.learnSpokenWord(imageFeatures, vocabulary[wordIndex]);
        // 成长潜力: 经验积累 → 抽象倍数提升 (神经元→2000亿潜力)
        growthPotential.update(learnedWords.size(), trainerLevel);
        // 虚拟神经元层同步抽象倍数 (等效规模随成长真实扩大)
        virtualLayer.setAbstraction(growthPotential.abstractionFactor());
        // 探索奖励: 不确定奖励比固定奖励更吸引 (书中29篇) — 学习成果也有不确定性
        uncertainReward(0.5, 0.3 + 0.4 * rnd.nextDouble());
        // 全脑频率同步 (整体联动: 记忆写入→各模块频率交互)
        syncFrequencyBus();
        // 记录
        if (!learnedWords.contains(vocabulary[wordIndex])) {
            learnedWords.add(vocabulary[wordIndex]);
        }
    }

    /**
     * 听觉学习: 音频特征 → 听觉皮层 → 联想皮层 → 关联到词。
     */
    public void learnAuditoryWord(double[] audioFeatures, int wordIndex) {
        double[] input = new double[auditoryCortexSize];
        for (int i = 0; i < Math.min(audioFeatures.length, auditoryCortexSize); i++) {
            input[i] = audioFeatures[i];
        }
        // 脑干降噪 + 丘脑中继: 听神经→脑干(分拣/降噪)→丘脑(过滤)→听觉皮层
        input = auditoryRelay.brainstemFilter(input);
        input = auditoryRelay.thalamicRelay(input);
        int[] spikes = new int[auditoryCortexSize];
        for (int t = 0; t < powerProfile.learnTimeSteps; t++) {  // 功率自适应步数
            for (int i = 0; i < auditoryCortexSize; i++) {
                int s = Math.abs(auditoryEncoder.encode(input[i] - 0.5));  // σ-δ 编码
                spikes[i] += s;
                // 频率波驱动 (类脑: 听觉信号→频率振荡)
                double waveCurrent = neuralWave.driveNeuron(input[i], 1.0);
                // EEG 回馈: 脑电波调制听觉神经元输入 (闭环)
                double eegModulated = eegFeedback(waveCurrent + s * 5.0);
                // 中枢下行调制 (丘脑-皮层双向)
                double gated = eegModulated * auditoryBridge.topDownGain();
                double noise = backgroundNoise.sample() + gaussianNoise.sample();
                // 稳态可塑性: 听觉增益调制 (索引偏移到听觉段)
                auditoryCortex[i].step((gated + noise) * homeostaticPlasticity.gain(visualCortexSize + i), 1.0);
                firingState[visualCortexSize + i] = auditoryCortex[i].fired();
            }
            auditoryBridge.updateRates();  // 皮层发放率 → 中枢
            homeostaticPlasticity.step(firingState, 1.0);
            synapseFormation.step(firingState, 1);
        }
        double[] aRate = new double[auditoryCortexSize];
        for (int i = 0; i < auditoryCortexSize; i++) aRate[i] = spikes[i] / 20.0;
        double[] assocActivation = auditoryToAssoc.forward(aRate);
        // Hebbian 联想记忆更新 (先更新, 作为 pp-prop 监督目标)
        for (int i = 0; i < associationCortexSize; i++) {
            double a = Math.max(0, Math.tanh(assocActivation[i]));
            associativeMemory[i][wordIndex] += 0.1 * a;
        }
        // pp-prop 在线学习 (BrainTrace): 误差驱动 — 目标=该词关联列, 信号=目标-当前
        double colSum = 0;
        for (int i = 0; i < associationCortexSize; i++) colSum += associativeMemory[i][wordIndex];
        double[] hiddenDeriv = new double[associationCortexSize];
        java.util.Arrays.fill(hiddenDeriv, 1.0);
        auditoryToAssoc.step(aRate, assocActivation, hiddenDeriv, 0.99);
        double[] learningSignal = new double[associationCortexSize];
        for (int i = 0; i < associationCortexSize; i++) {
            double target = colSum > 0 ? associativeMemory[i][wordIndex] / colSum : 0;
            learningSignal[i] = target - Math.max(0, Math.tanh(assocActivation[i]));
        }
        auditoryToAssoc.update(learningSignal);
        auditoryToAssoc.applyGradients();
        // 多感觉整合: 记录听觉标签 + 跨模态一致性 → 意识Φ
        lastAuditoryLabel = vocabulary[wordIndex];
        if (!lastVisualLabel.isEmpty()) {
            double phi = msi.integrateLabels(lastVisualLabel, lastAuditoryLabel);
            if (phi > 0.8) {
                // 视觉+听觉识别同一物体 → 高整合 (Scikit-neuromsi/Körding)
                consciousness.setAttention(0.9);
            }
        }
        // 分层记忆: 沉淀为中期记忆
        hierarchicalMemory.addEpisodic(vocabulary[wordIndex], audioFeatures, 0.5);
        if (!learnedWords.contains(vocabulary[wordIndex])) {
            learnedWords.add(vocabulary[wordIndex]);
        }
    }

    /**
     * 认知查询: 给定视觉特征, 返回最关联的词 (回忆)。
     */
    public String recognizeVisual(double[] imageFeatures) {
        return bestWord(visualFeatures(imageFeatures));
    }

    /** 视觉识别带置信度: 返回 [词, 置信度] —— 置信度低 = 未知 = 触发好奇 */
    public String[] recognizeVisualWithConfidence(double[] imageFeatures) {
        double[] assocAct = visualFeatures(imageFeatures);
        // 联想层激活 → 中枢桥 (全局工作空间节点广播)
        double[] tanhAct = new double[assocAct.length];
        for (int i = 0; i < assocAct.length; i++) {
            tanhAct[i] = Math.max(0, Math.tanh(assocAct[i]));
        }
        assocBridge.updateActivation(tanhAct);
        String[] result = bestWordWithScore(assocAct);
        // 工作记忆: 与已存模式匹配 → 提升置信度 (fWBM: 短期记忆辅助)
        double[] wmRead = workingMemory.read(imageFeatures);
        if (wmRead[0] >= 0 && wmRead[1] > 0.7) {
            // 工作记忆命中 → 提升置信度 (短期记忆强化)
            double boost = Math.min(0.3, wmRead[1] * 0.3);
            double newConf = Double.parseDouble(result[1]) + boost;
            result[1] = String.valueOf(Math.min(1.0, newConf));
        }
        // 意识: 感知进入工作空间
        consciousness.perceive(0.7, 0.0, result[0], "");
        // 多感觉整合: 视觉标签记录
        lastVisualLabel = result[0];
        // 僵尸行动者: 自动化技能 → 稳定高置信度 (解放意识)
        cognitiveMode.recognizeActivity(0.03);
        if (!result[0].equals("未知")) {
            zombieAgent.practice(result[0]);
            double[] z = zombieAgent.process(result[0], Double.parseDouble(result[1]));
            if (z[1] == 1.0) {
                // 自动化处理: 置信度取僵尸通道 (稳定高)
                result[1] = String.valueOf(z[0]);
            }
        }
        // 频率共振检索: 输入频率谱 → 与记忆频率共振 → 强化/纠偏 (记忆=频率)
        double queryHz = neuralWave.intensityToHz(avgIntensity(imageFeatures));
        String[] resonance = resonanceMemory.retrieveByFreq(queryHz);
        double resStrength = Double.parseDouble(resonance[1]);
        if (!resonance[0].isEmpty() && resStrength > 0.3) {
            double resConf = Math.min(1.0, 0.5 + resStrength);
            double curConf = Double.parseDouble(result[1]);
            // 共振记忆与识别一致 → 强化; 不一致 → 微弱纠偏
            if (resonance[0].equals(result[0])) {
                result[1] = String.valueOf(Math.min(1.0, curConf + 0.15 * resStrength));
            } else if (resStrength > 0.6 && curConf < 0.5) {
                result[0] = resonance[0];
                result[1] = String.valueOf(resConf);
            }
        }
        // 预测引擎: 假设(预测) → 验证(识别结果作为证据) → 结论
        String prediction = predictiveEngine.predict();
        double evidence = Double.parseDouble(result[1]);
        predictiveEngine.observe(result[0].equals("未知") ? "" : result[0]);
        // 内部模型: 感觉预测误差修正 (书中20篇 — 预测感官结果, 误差驱动纠偏)
        double[] state = new double[Math.min(16, imageFeatures.length)];
        System.arraycopy(imageFeatures, 0, state, 0, state.length);
        double predicted = internalModel.predict(state);
        double actual = 0.5 + 0.5 * avgIntensity(imageFeatures);
        double predErr = internalModel.learn(state, actual);
        // 预测误差小 (模型准确) → 置信度微升; 误差大 → 微降 (感觉预测失准)
        if (predErr < 0.2) {
            result[1] = String.valueOf(Math.min(1.0, Double.parseDouble(result[1]) + 0.05));
        } else if (predErr > 0.8) {
            result[1] = String.valueOf(Math.max(0.1, Double.parseDouble(result[1]) - 0.05));
        }
        // 多巴胺: 识别置信度作为"实际奖励"驱动学习 (错误驱动)
        double success = result[0].equals("未知") ? 0.0 : evidence;
        dopamineSystem.learnEvent(success, 1.0);
        // 镜像自我识别反馈: 高置信识别 = "我认出来了" (自我评估驱动成长)
        // v5.4 修复: 原 mirrorTest 无调用点 → accuracy 恒 0 → devStage 永卡感知萌芽
        double finalConf = Double.parseDouble(result[1]);
        selfAwareness.mirrorTest(!result[0].equals("未知") && finalConf >= 0.6, finalConf);
        // 全脑频率同步 (整体联动: 识别→记忆共振→意识广播)
        syncFrequencyBus();
        return result;
    }

    /** 输入向量平均强度 (频率编码用) */
    private double avgIntensity(double[] features) {
        double sum = 0;
        for (double f : features) sum += f;
        return sum / Math.max(1, features.length);
    }

    private double[] visualFeatures(double[] imageFeatures) {
        double[] vRate = new double[visualCortexSize];
        double[] input = new double[visualCortexSize];
        for (int i = 0; i < Math.min(imageFeatures.length, visualCortexSize); i++) input[i] = imageFeatures[i];
        int[] spikes = new int[visualCortexSize];
        for (int t = 0; t < 10; t++) {
            for (int i = 0; i < visualCortexSize; i++) {
                spikes[i] += visualEncoder.encode(input[i]);
            }
        }
        for (int i = 0; i < visualCortexSize; i++) vRate[i] = spikes[i] / 10.0;
        return visualToAssoc.forward(vRate);
    }

    /**
     * 听觉识别带置信度。
     */
    public String[] recognizeAuditoryWithConfidence(double[] audioFeatures) {
        double[] aRate = new double[auditoryCortexSize];
        double[] input = new double[auditoryCortexSize];
        for (int i = 0; i < Math.min(audioFeatures.length, auditoryCortexSize); i++) input[i] = audioFeatures[i];
        int[] spikes = new int[auditoryCortexSize];
        for (int t = 0; t < 10; t++) {
            for (int i = 0; i < auditoryCortexSize; i++) {
                spikes[i] += Math.abs(auditoryEncoder.encode(input[i] - 0.5));
            }
        }
        for (int i = 0; i < auditoryCortexSize; i++) aRate[i] = spikes[i] / 10.0;
        String[] result = bestWordWithScore(auditoryToAssoc.forward(aRate));
        // 镜像自我识别反馈 (与视觉识别一致: 高置信 = 认出来了)
        double finalConf = Double.parseDouble(result[1]);
        selfAwareness.mirrorTest(!result[0].equals("未知") && finalConf >= 0.6, finalConf);
        return result;
    }

    public String recognizeAuditory(double[] audioFeatures) {
        double[] aRate = new double[auditoryCortexSize];
        double[] input = new double[auditoryCortexSize];
        for (int i = 0; i < Math.min(audioFeatures.length, auditoryCortexSize); i++) input[i] = audioFeatures[i];
        int[] spikes = new int[auditoryCortexSize];
        for (int t = 0; t < 10; t++) {
            for (int i = 0; i < auditoryCortexSize; i++) {
                spikes[i] += Math.abs(auditoryEncoder.encode(input[i] - 0.5));
            }
        }
        for (int i = 0; i < auditoryCortexSize; i++) aRate[i] = spikes[i] / 10.0;
        double[] assoc = auditoryToAssoc.forward(aRate);
        return bestWord(assoc);
    }

    private String bestWord(double[] assocActivation) {
        return bestWordWithScore(assocActivation)[0];
    }

    /** 返回 [词, 置信度0-1]；置信度低(<0.15)表示未知 */
    private String[] bestWordWithScore(double[] assocActivation) {
        int best = -1;
        double bestScore = 0;
        double total = 0;
        for (int w = 0; w < vocabulary.length; w++) {
            double score = 0;
            for (int i = 0; i < associationCortexSize; i++) {
                double a = Math.max(0, Math.tanh(assocActivation[i]));
                score += a * associativeMemory[i][w];
            }
            total += score;
            if (score > bestScore) { bestScore = score; best = w; }
        }
        // 置信度 = 最佳得分占总量比例 (竞争性)  + 绝对强度归一
        double confidence = total > 0 ? bestScore / total : 0;
        double strengthNorm = Math.min(1.0, bestScore / 0.5);
        confidence = 0.6 * confidence + 0.4 * strengthNorm;
        if (best < 0 || confidence < 0.12) {
            return new String[]{"未知", String.valueOf(Math.max(0, Math.min(1, confidence)))};
        }
        return new String[]{vocabulary[best], String.valueOf(Math.max(0, Math.min(1, confidence)))};
    }

    /** 决策: 输入相干性, 返回决策 (1/2) */
    public int decide(double coherence) {
        for (int t = 0; t < 100; t++) decision.step(coherence, 1.0);
        return decision.decision();
    }

    /** 当前 α 节律活动 (EEG) */
    public double eegActivity() {
        return rhythm.step(0, 1.0);
    }

    // ============ 自发神经活动 API ============

    /** 自发突触形成器 (连接建立情况) */
    public SynapseFormation synapseFormation() { return synapseFormation; }

    /** 采样全脑自发活动: 每神经元是否在自发发放 (供脑图可视化) */
    public boolean[] sampleSpontaneousActivity() {
        boolean[] act = new boolean[visualCortexSize + auditoryCortexSize + associationCortexSize];
        // 自发活动可视化: 用 5% 概率模拟背景自发发放 (泊松 8Hz 在 200ms 帧内 ≈ 1.6% + 神经元状态)
        for (int i = 0; i < visualCortexSize; i++) {
            act[i] = rnd.nextDouble() < 0.05 || visualCortex[i].fired();
        }
        for (int i = 0; i < auditoryCortexSize; i++) {
            act[visualCortexSize + i] = rnd.nextDouble() < 0.05 || auditoryCortex[i].fired();
        }
        for (int i = 0; i < associationCortexSize; i++) {
            act[visualCortexSize + auditoryCortexSize + i] = rnd.nextDouble() < 0.05;
        }
        // 虚拟神经元层: 物理发放 → 虚拟扩展 (等效规模脉冲统计真实参与)
        virtualLayer.step(act, 200.0);
        return act;
    }

    /** 虚拟神经元层 (神经形态等效规模) */
    public VirtualNeuronLayer virtualLayer() { return virtualLayer; }

    /** 稳态可塑性 (发放率稳态调节) */
    public HomeostaticPlasticity homeostaticPlasticity() { return homeostaticPlasticity; }

    /** 背景噪声源 (幅度可调) */
    public NoiseSource backgroundNoise() { return backgroundNoise; }

    // ============ 论文机制 API ============

    /** E/I 平衡网络 (证据累积决策) */
    public EIBNetwork eibNetwork() { return eibNetwork; }

    /** 运行证据累积决策试验 (BrainTrace Fig5: 虚拟跑道线索→T字路口) */
    public int runDecisionTrial(double[] cueSequence) {
        // 认知模式: 决策=上脑计划预测
        cognitiveMode.decideActivity(0.08);
        return eibNetwork.runTrial(cueSequence);
    }

    /** 决策证据状态 [左, 右] */
    public double[] decisionEvidence() {
        return new double[]{eibNetwork.evidenceLeft(), eibNetwork.evidenceRight()};
    }

    /** TTFS 时间编码: 强度数组 → 首次脉冲时间 (早=强, BrainTrace Fig5F) */
    public double[] encodeTTFS(double[] intensities) {
        return ttfs.encodeAll(intensities);
    }

    /** TTFS 顺序编码: 神经元发放顺序 (rank ordering) */
    public int[] rankOrder(double[] intensities) {
        return ttfs.rankOrder(intensities);
    }

    /** 睡眠巩固: 重放白天共激活 + 修剪弱连接 (夜间记忆巩固) */
    public int[] sleepConsolidate() {
        // 意识进入无意识 (睡眠: 外部输入无法广播)
        consciousness.sleep();
        // 工作记忆重放: 睡梦中复习短期记忆 (抵抗遗忘, 促进巩固)
        int wmLoad = workingMemory.load();
        for (int i = 0; i < workingMemory.capacity(); i++) {
            if (workingMemory.strength(i) > 0.1) {
                workingMemory.rehearse(i);  // 重放强化
            }
        }
        int[] report = sleep.sleep(synapseFormation, dayCoactivations);
        dayCoactivations.clear();  // 新的一天
        // 分层记忆巩固: 中期→长期 (睡眠重放转存)
        hierarchicalMemory.sleepConsolidate(workingMemory);
        // 认知模式: 睡眠微调
        cognitiveMode.sleepActivity();
        // 唤醒 (意识恢复)
        consciousness.wake();
        // 全脑频率同步 (睡眠巩固后: 长期记忆频率更新)
        syncFrequencyBus();
        return new int[]{report[0], report[1], report[2], wmLoad};
    }

    /** 镜像测试 (识别反馈 → 自我意识更新) */
    public void mirrorFeedback(boolean correct, double confidence) {
        selfAwareness.mirrorTest(correct, confidence);
    }

    /** 元认知 (对思考的思考): 一级 + 二阶校准监控 */
    public String metacognition() {
        return selfAwareness.metacognition() + " | " + selfAwareness.metaMetacognition();
    }

    /** 反思链: 思考 → 思考思考 → 思考思考的思考 (递归元认知) */
    public String reflectOnThinking(String currentThought) {
        return selfAwareness.reflectOnThinking(currentThought);
    }

    /** 自我叙事 */
    public String selfNarrative() { return selfAwareness.selfNarrative(); }

    /** 社会自我 (教育者评价塑造) */
    public String socialSelf() { return selfAwareness.socialSelf(); }

    /** 发展阶段 —— 自然成长 (无年龄预设, 由实际经验推进) */
    public SelfAwareness.DevStage devStage() {
        // 自然成长: 已学词数 (经验量) + 识别准确率 (熟练度) + 镜像自我识别
        double accuracy = selfAwareness.accuracy();
        return selfAwareness.devStage(learnedWords.size(), accuracy, selfAwareness.selfRecognized());
    }

    private int trainerLevel = 0;
    /** 设置培养等级 (供发展阶段判定) */
    public void setTrainerLevel(int level) { this.trainerLevel = level; }

    /** 自我意识模块 */
    public SelfAwareness selfAwareness() { return selfAwareness; }

    /** 分层记忆 (短期→中期→长期) */
    public HierarchicalMemory hierarchicalMemory() { return hierarchicalMemory; }

    /** 多感觉整合模块 (Körding 贝叶斯因果推断) */
    public MultiSensoryIntegration msi() { return msi; }

    /** 跨模态整合: 视觉+听觉识别同一物体 → Φ (Scikit-neuromsi) */
    public double crossModalPhi(String visualLabel, String auditoryLabel) {
        double phi = msi.integrateLabels(visualLabel, auditoryLabel);
        // 同步到意识模块
        if (phi > 0.8) {
            consciousness.setAttention(0.9);
            consciousness.perceive(0.9, 0.9, visualLabel, auditoryLabel);
        }
        return phi;
    }

    // ============ 多模态学习与输出 (v5.4) ============

    /** 跨模态记忆 (视觉↔听觉原型绑定) */
    public CrossModalMemory crossModalMemory() { return crossModalMemory; }

    // ============ 时序预测与延迟脑补 (v5.4, 人脑预测能力) ============

    /** 视觉时序预测: 看到真实帧 → 预测下一帧 (大脑持续预测感官) */
    public double[] predictNextVisual(double[] visualFeatures) {
        return visualPredictor.predictNext(visualFeatures);
    }

    /** 听觉时序预测 */
    public double[] predictNextAuditory(double[] audioFeatures) {
        return auditoryPredictor.predictNext(audioFeatures);
    }

    /**
     * 延迟补偿感知 (脑补): 感官帧延迟时用预测填补当前感知。
     * 人脑机制: 视觉/感觉信号有 ~100ms 传输延迟, 平滑追踪/小脑前向模型
     * 用预测填补延迟期, 大脑"感知"的是预测的当前, 而非滞后的原始信号。
     * @param rawFeatures 到达的感官帧 (可能滞后)
     * @param delaySteps 延迟帧数 (0=正常帧: 更新预测器; >0=脑补该帧数的预测)
     * @return 感知特征 (脑补帧或真实帧)
     */
    public double[] perceiveVisualWithDelayCompensation(double[] rawFeatures, int delaySteps) {
        if (delaySteps > 0) return visualPredictor.extrapolate(delaySteps);   // 脑补: 预测当前
        return visualPredictor.predictNext(rawFeatures);     // 正常: 更新 + 预测
    }

    public double[] perceiveAuditoryWithDelayCompensation(double[] audioFeatures, int delaySteps) {
        if (delaySteps > 0) return auditoryPredictor.extrapolate(delaySteps);
        return auditoryPredictor.predictNext(audioFeatures);
    }

    /** 预测置信度 (0-1): 速度小 → 预测可靠; 供脑补决策 */
    public double visualPredictionConfidence() { return visualPredictor.confidence(); }
    public double auditoryPredictionConfidence() { return auditoryPredictor.confidence(); }

    /** 镜头/场景切换时重置预测器 (预测历史失效) */
    public void resetPredictors() {
        visualPredictor.reset();
        auditoryPredictor.reset();
    }

    /** 预测能力摘要 (APK 显示) */
    public String predictionSummary() {
        return String.format("🔮 预测: 视觉置信%.0f%% | 听觉置信%.0f%% (延迟脑补)",
                visualPredictor.confidence() * 100, auditoryPredictor.confidence() * 100);
    }

    /** 上次视觉预测误差 (0-1, 供显示/调试) */
    private double lastVisualPredErr = 1.0;
    public double lastVisualPredErr() { return lastVisualPredErr; }

    /**
     * 预测验证与纠正 (predictive coding 闭环):
     *   真实帧到达 → 与上次预测对比 → 预测误差 → 纠正连接。
     *   误差小 = 预测准 (前向模型正确, 稳定);
     *   误差大 = 预测失准 → 纠正: 内部模型学习真实帧 + 预测器速度重置
     *   (前向模型参数修正, 对应小脑误差驱动的突触纠正) + 多巴胺意外信号。
     * @param realFeatures 真实感官帧
     * @return 预测误差 (0-1)
     */
    public double verifyVisualPrediction(double[] realFeatures) {
        double[] pred = visualPredictor.lastPrediction();
        if (pred == null) {
            predictNextVisual(realFeatures);
            lastVisualPredErr = 1.0;
            return lastVisualPredErr;
        }
        // 预测误差: 逐点平均绝对差
        double err = 0;
        int n = Math.min(pred.length, realFeatures.length);
        for (int i = 0; i < n; i++) err += Math.abs(pred[i] - realFeatures[i]);
        err = n > 0 ? err / n : 1.0;
        lastVisualPredErr = err;
        // 内部模型学习真实帧 (预测纠错: 前向模型参数更新)
        double[] state = new double[Math.min(16, realFeatures.length)];
        System.arraycopy(realFeatures, 0, state, 0, state.length);
        double actual = 0.5 + 0.5 * avgIntensity(realFeatures);
        internalModel.learn(state, actual);
        // 多巴胺: 预测准确=预期满足, 误差大=意外 (RPE)
        double surprise = err < 0.15 ? 1.0 : (err > 0.35 ? 0.0 : 0.5);
        dopamineSystem.learnEvent(surprise, 1.0);
        // 连接纠正: 速度 EMA 自然适应运动变化 (突变后 ~5 帧翻转收敛);
        // 显式 reset 只在镜头/场景切换时由 resetPredictors() 调用 (MainActivity 已接入)
        // 预测下一帧 (供脑补显示)
        predictNextVisual(realFeatures);
        return err;
    }

    /** 视觉预测器上一帧预测 (脑补显示用) */
    public double[] visualExtrapolate() { return visualPredictor.extrapolate(1); }

    /**
     * 多模态学习: 视觉+听觉特征同时绑定到同一词 (概念 = 多模态绑定)。
     * 视觉通路 + 听觉通路各自学习 + 跨模态原型绑定 (看到猫+听到猫叫 → 猫概念双面)。
     */
    public void learnCrossModal(double[] visualFeatures, double[] audioFeatures, int wordIndex) {
        learnVisualWord(visualFeatures, wordIndex);
        learnAuditoryWord(audioFeatures, wordIndex);
        crossModalMemory.bind(wordIndex, visualFeatures, audioFeatures);
        // 跨模态 γ 绑定: 双模态一致学习 → 意识整合度提升
        consciousness.setAttention(Math.max(consciousness.attention(), 0.8));
    }

    /**
     * 多模态融合识别: 视觉+听觉同时输入 → 贝叶斯加权融合 (Körding 因果推断)。
     * 双模态一致 → 融合置信度提升; 不一致 → 分离 (低融合置信, 触发好奇)。
     * @return [词, 融合置信度, Φ整合度]
     */
    public String[] recognizeMultiModal(double[] visualFeatures, double[] audioFeatures) {
        String[] vis = recognizeVisualWithConfidence(visualFeatures);
        String[] aud = recognizeAuditoryWithConfidence(audioFeatures);
        double visConf = Double.parseDouble(vis[1]);
        double audConf = Double.parseDouble(aud[1]);
        boolean same = vis[0].equals(aud[0]);
        // Φ 整合度: 标签一致性 (Körding 因果推断: 同源→高整合, 冲突→分离)
        double phi = msi.integrateLabels(vis[0], aud[0]);
        String word;
        double conf;
        if (same) {
            // 一致: 融合置信度 = 贝叶斯加权 + 一致性增益 (Φ)
            double wV = (visConf + 0.01) / (visConf + audConf + 0.02);
            double fused = wV * visConf + (1 - wV) * audConf;
            word = vis[0];
            conf = Math.min(0.99, fused + 0.15 * phi);
        } else {
            // 不一致: 取高置信一方, 融合置信度打折 (认知冲突)
            word = visConf >= audConf ? vis[0] : aud[0];
            conf = Math.max(visConf, audConf) * 0.5;
        }
        // 同步意识整合度
        consciousness.setAttention(Math.max(consciousness.attention(), phi));
        return new String[]{word, String.valueOf(conf), String.valueOf(phi)};
    }

    /** 跨模态回忆 (视觉→听觉): 识别视觉 → 唤起该词的声音原型 (可送声带/频谱输出) */
    public double[] recallAudioFromVisual(double[] visualFeatures) {
        int idx = bestWordIndex(visualFeatures(visualFeatures));
        return idx >= 0 ? crossModalMemory.recallAudio(idx) : null;
    }

    /** 跨模态回忆 (听觉→视觉): 识别听觉 → 唤起该词的视觉原型 (可送大脑视角显示) */
    public double[] recallVisualFromAudio(double[] audioFeatures) {
        double[] aRate = new double[auditoryCortexSize];
        double[] input = new double[auditoryCortexSize];
        for (int i = 0; i < Math.min(audioFeatures.length, auditoryCortexSize); i++) input[i] = audioFeatures[i];
        int[] spikes = new int[auditoryCortexSize];
        for (int t = 0; t < 10; t++) {
            for (int i = 0; i < auditoryCortexSize; i++) {
                spikes[i] += Math.abs(auditoryEncoder.encode(input[i] - 0.5));
            }
        }
        for (int i = 0; i < auditoryCortexSize; i++) aRate[i] = spikes[i] / 10.0;
        int idx = bestWordIndex(auditoryToAssoc.forward(aRate));
        return idx >= 0 ? crossModalMemory.recallVisual(idx) : null;
    }

    /** 词索引 (联想激活 → 最强词) */
    private int bestWordIndex(double[] assocActivation) {
        int best = 0;
        double bestScore = -1;
        for (int w = 0; w < vocabulary.length; w++) {
            double score = 0;
            for (int i = 0; i < associationCortexSize; i++) {
                score += Math.max(0, Math.tanh(assocActivation[i])) * associativeMemory[i][w];
            }
            if (score > bestScore) { bestScore = score; best = w; }
        }
        return bestScore > 0 ? best : -1;
    }

    /** 探索未知活动 (认知模式: 上脑计划+下脑感知) */
    public void exploreActivity() {
        cognitiveMode.exploreActivity(0.08);
    }

    /** 认知模式模块 (Kosslyn 上脑/下脑) */
    public CognitiveMode cognitiveMode() { return cognitiveMode; }

    /** 僵尸行动者 (技能自动化) */
    public ZombieAgent zombieAgent() { return zombieAgent; }

    /** 认知模式描述 (APK 显示) */
    public String cognitiveModeDescription() {
        return cognitiveMode.describe();
    }

    /** 僵尸行动者摘要 */
    public String zombieSummary() {
        return zombieAgent.summary();
    }

    /** 意外处理: Φ 高 → 新情况处理更好 (PHI 理论: 黑天鹅事件) */
    public double handleUnexpected(String label) {
        return zombieAgent.handleUnexpected(consciousness.phi(), label);
    }

    /** 多巴胺系统 (错误驱动学习) */
    public DopamineSystem dopamineSystem() { return dopamineSystem; }

    /**
     * 不确定奖励事件 (书中第29篇: 不可预测奖励比固定奖励更能激活多巴胺)。
     * 不确定性越高 → 多巴胺反应越强 (奖励预测误差 RPE 放大)。
     */
    public void uncertainReward(double reward, double uncertainty) {
        // 不确定性调制 RPE: 高不确定 → 多巴胺反应放大 (惊喜效应)
        double boost = 1.0 + uncertainty;  // 0-2x
        double actual = reward * (1.0 + uncertainty * 0.5);
        dopamineSystem.learnEvent(actual * boost, 1.0);
    }

    /** 预测引擎 (大脑=科学家) */
    public PredictiveEngine predictiveEngine() { return predictiveEngine; }

    /** 内部模型 (前向预测/感觉预测误差, 书中20篇) */
    public InternalModel internalModel() { return internalModel; }

    /** 视觉皮层桥接 (接入中枢) */
    public SensoryCortexBridge visualBridge() { return visualBridge; }
    /** 听觉皮层桥接 (接入中枢) */
    public SensoryCortexBridge auditoryBridge() { return auditoryBridge; }

    /** 联想皮层桥接 (全局工作空间节点) */
    public AssocBridge assocBridge() { return assocBridge; }

    /** 声音情感识别 (书中23篇: 从声音识别情感) */
    public EmotionalVoice emotionalVoice() { return emotionalVoice; }

    /** 语言学习模块 (模仿→理解→自主) */
    public LanguageLearner languageLearner() { return languageLearner; }

    /** 声带学习进化模块 (听到→学参数→模仿) */
    public VoiceLearner voiceLearner() { return voiceLearner; }

    /**
     * 声带学习: 从听到的音频学习发声参数 (听觉-发声回路)。
     * @param pcm 听到的声音
     * @return 学到的模板 (null = 静音未学)
     */
    public VoiceLearner.VoiceTemplate learnVoiceFromAudio(short[] pcm) {
        return voiceLearner.learnFromAudio(pcm);
    }

    /** 声带学习摘要 */
    public String voiceSummary() { return voiceLearner.summary(); }

    /**
     * 模仿发声: 听到声音 → 提取频率/共振峰 → 用相同频率立即发声 (鹦鹉学舌)。
     * @param pcm 听到的声音 (16kHz)
     * @return 模仿的 PCM (空 = 静音未学)
     */
    public short[] mimicVoice(short[] pcm) {
        return voiceLearner.mimic(pcm);
    }

    /** 声带进化阶段: 模板多 → 可自主组合说话 */
    public String voiceStage() {
        int n = voiceLearner.templateCount();
        if (n == 0) return "🐣 咿呀期: 还不会发声";
        if (n < 4) return "🗣️ 模仿期: 听到就模仿 (" + n + "个音模板)";
        return "💬 自主期: 用学到的声音说话 (" + n + "个音模板)";
    }

    /** 成长潜力 (神经元→2000亿架构潜力) */
    public GrowthPotential growthPotential() { return growthPotential; }

    /** 成长摘要 (APK 显示) — 含虚拟神经元层等效规模 */
    public String growthSummary() {
        return growthPotential.summary() + "\n" + virtualLayer.summary();
    }

    /** 语音对话+逗乐模块 */
    public SpeechResponder speechResponder() { return speechResponder; }

    /**
     * 听到语音 → 回复 (语音对话)。
     * @param heardText 听到的文本/关键词
     * @return 回复文本 (空 = 无回应)
     */
    public String respondToSpeech(String heardText) {
        String mimic = languageLearner.voiceCount() > 0
                ? languageLearner.mimic() : "";
        return speechResponder.respond(heardText, learnedWords,
                languageLearner.stage(), mimic);
    }

    /**
     * 被逗乐: 听到笑声/逗弄 → 开心+笑。
     * @param playStrength 逗乐强度 (0-1)
     * @return 笑反应文本
     */
    public String playReact(double playStrength) {
        return speechResponder.playReact(playStrength);
    }

    /**
     * 观察场景: 主动观察看到的东西 → 描述 (互动反馈)。
     * @param features 视觉特征
     * @return 观察描述
     */
    public String observeScene(double[] features) {
        String[] r = recognizeVisualWithConfidence(features);
        if (r[0].equals("未知")) {
            return "咦？这是个没见过的东西，我好想知道它是什么！";
        }
        return "我看到" + r[0] + "了！";
    }

    /** 开心度 (APK 显示) */
    public int happiness() { return speechResponder.happiness(); }

    /** 互动摘要 */
    public String interactSummary() { return speechResponder.summary(); }

    /** 背侧通路 (Where/How: 空间/运动) */
    public DorsalPathway dorsalPathway() { return dorsalPathway; }

    /** 视觉 LGN 中继 */
    public ThalamicRelay visualRelay() { return visualRelay; }

    /** 听觉脑干+丘脑中继 */
    public ThalamicRelay auditoryRelay() { return auditoryRelay; }

    /**
     * 双通路视觉: 输入灰度图 → [腹侧 What 特征(识别) + 背侧 Where 特征(位置/运动)]。
     * @param grayPixels 当前帧灰度 (0-255)
     * @param prevPixels 前一帧灰度 (运动检测, 可 null)
     * @param width 宽
     * @param height 高
     * @return [What 96维, Where 6维]
     */
    public double[][] dualPathwayVisual(double[] grayPixels, double[] prevPixels, int width, int height) {
        // 腹侧通路 (What): 神经编码 → LGN → 识别
        double[] what = new VisualNeuralEncoder().encode(grayPixels, width, height);
        what = visualRelay.thalamicRelay(what);
        // 背侧通路 (Where/How): 空间位置 + 运动
        double[] position = dorsalPathway.encodePosition(grayPixels, width, height);
        double[] motion = prevPixels != null
                ? dorsalPathway.encodeMotion(prevPixels, grayPixels, width, height)
                : new double[3];
        double[] where = new double[6];
        System.arraycopy(position, 0, where, 0, 3);
        System.arraycopy(motion, 0, where, 3, 3);
        return new double[][]{what, where};
    }

    /** 扩展物理规模 (高端机: 内存允许时增加真实神经元) */
    public void expandPhysicalNeurons(int additional) {
        growthPotential.expandPhysical(additional);
    }

    /**
     * 算力自适应: 根据手机算力调整神经元规模 (物理神经元数)。
     * @param cpuCores   CPU 核心数
     * @param freeMb     可用内存 MB
     * @param avgFrameMs 平均帧耗时 ms (卡顿检测)
     * @return 算力评分 (0-4)
     */
    public int adjustNeuronsToCompute(int cpuCores, long freeMb, double avgFrameMs) {
        return growthPotential.adjustToCompute(cpuCores, freeMb, avgFrameMs);
    }

    /**
     * 听到语音: 学习发音 (模仿素材)。
     * @param acousticFeatures 声学特征
     * @param word 若知道对应词则传入, 否则 null
     */
    public void hearSpokenWord(double[] acousticFeatures, String word) {
        if (word != null && !word.isEmpty()) {
            languageLearner.learnSpokenWord(acousticFeatures, word);
        } else {
            languageLearner.hearSpeech(acousticFeatures);
        }
    }

    /**
     * 自主说话: 从内在状态生成话语 (成熟后表达自我)。
     * @return 要说的话 (空 = 还在模仿期无素材)
     */
    public String speakAutonomously() {
        // 内在状态: 记忆活跃度 (好奇心代理) + 自我叙事
        double curiosity = Math.min(1.0, resonanceMemory.size() * 0.2);
        String recent = selfAwareness.selfNarrative();
        return languageLearner.speak("平静", curiosity, recent, learnedWords);
    }

    /** 语言学习摘要 */
    public String languageSummary() { return languageLearner.summary(); }

    /**
     * 自主发声: 检查内在状态, 有"想说的"就返回话语 (无则空)。
     * 触发条件 (自然发声, 非按钮):
     *   - 模仿期: 学过语音 → 主动模仿练习 (学语)
     *   - 识别出新东西 → 主动报告
     *   - 好奇心高 → 主动提问
     *   - 情绪变化 → 表达感受
     *   - 记忆丰富 → 回顾经历
     * 由 Brain 层决策是否发声, APK 只负责 TTS。
     */
    public String autonomousUtterance() {
        LanguageLearner.LangStage stage = languageLearner.stage();
        // 1. 模仿期: 学过语音 → 主动模仿 (自主学语)
        if (stage == LanguageLearner.LangStage.模仿期) {
            String mimic = languageLearner.mimic();
            if (!mimic.isEmpty()) {
                return "🗣️ " + mimic + "！" + mimic + "！";
            }
            return "";
        }
        // 2. 理解期: 识别出东西 → 主动报告
        if (stage == LanguageLearner.LangStage.理解期) {
            if (!lastVisualLabel.isEmpty()) {
                return "我看到" + lastVisualLabel + "了！";
            }
            return "";
        }
        // 3. 自主期: 从内在状态生成 (记忆/好奇/叙事)
        double curiosity = Math.min(1.0, resonanceMemory.size() * 0.2);
        String recent = selfAwareness.selfNarrative();
        String speech = languageLearner.speak("平静", curiosity, recent, learnedWords);
        return speech.isEmpty() ? "" : "💬 " + speech;
    }

    /** 教育事件驱动多巴胺: 答对=奖励, 答错=惩罚 (RPE 学习) */
    public void rewardEvent(boolean correct, double rewardValue) {
        dopamineSystem.learnEvent(correct ? 1.0 : 0.0, rewardValue);
    }

    /** 多巴胺摘要 (APK 显示) */
    public String dopamineSummary() { return dopamineSystem.summary(); }

    /** 预测引擎摘要 */
    public String predictiveSummary() { return predictiveEngine.summary(); }

    /** 频率波发生器 (类脑: 信号=频率) */
    public FrequencyWave neuralWave() { return neuralWave; }

    /** 频率共振记忆层 (记忆=频率) */
    public ResonanceMemory resonanceMemory() { return resonanceMemory; }

    /**
     * 当前脑电波 (EEG) —— 由脉冲信号聚合产生 (书中"聚合的脑活动")。
     * 输入: 中枢脉冲发放 + 各模块发射脉冲 → EEGGenerator 聚合 → 波形。
     */
    public double currentEEG() {
        // 聚合脉冲: 中枢发放率 + 模块脉冲
        double[] pulseRates = new double[centralHub.inputDim() + centralHub.hubSize()];
        int offset = 0;
        for (PulseModule m : hubModules) {
            double[] pulses = m.emitPulses();
            System.arraycopy(pulses, 0, pulseRates, offset, pulses.length);
            offset += m.pulseDim();
        }
        boolean[] hubFiring = centralHub.firingState();
        for (boolean f : hubFiring) {
            pulseRates[offset++] = f ? 1.0 : 0.0;
        }
        // 脉冲 → EEG (聚合 + PSP + 节律)
        return eegGenerator.sample(pulseRates, 1.0);
    }

    /** EEG 发生器 (脉冲聚合→脑电波) */
    public EEGGenerator eegGenerator() { return eegGenerator; }

    /** EEG 回馈: 脑电波调制神经元输入 (节律驱动, 闭环) — 虚拟层宏观增益: 群体规模越大宏观信号越强 */
    public double eegFeedback(double baseInput) {
        return eegGenerator.feedbackCurrent(baseInput) * virtualLayer.macroscopicGain();
    }

    /** 全脑频率联动状态 (APK 显示: 各模块频率) */
    public String frequencySummary() {
        return String.format("📶 脑电: θ%.1fHz/α%.1fHz/γ%.1fHz | 共振记忆%d条 | 波形%.2f",
                6.0, 10.0, 40.0, resonanceMemory.size(), currentEEG());
    }

    /** 频率总线 (全脑联动) */
    public FrequencyBus frequencyBus() { return frequencyBus; }

    /**
     * 全模块频率同步 (类脑整体联动核心):
     * 各记忆/认知模块上报主导频率 → 总线 → 互相共振交互。
     * 调用时机: 学习/识别/睡眠等认知事件后。
     */
    public void syncFrequencyBus() {
        // 0. 中枢脉冲环路: 全模块统一脉冲联动 (皮层-丘脑环路, 优先于频率域)
        //    步数按功率配置 (高性能跑更多环路, 节能少跑)
        int cycles = Math.max(1, powerProfile.hubCyclesPerSync);
        for (int c = 0; c < cycles; c++) {
            hubPulseCycle();
        }
        // 1. 各模块上报频率 (记忆=频率)
        frequencyBus.report("工作记忆", workingMemory.currentFrequency(),
                workingMemory.occupancy());
        frequencyBus.report("长期记忆", hierarchicalMemory.currentFrequency(),
                hierarchicalMemory.longTermCount() > 0 ? 0.7 : 0.1);
        frequencyBus.report("共振记忆", resonanceMemory.size() > 0
                ? resonanceMemory.freq().intensityToHz(0.7) : 10.0,
                resonanceMemory.size() > 0 ? 0.6 : 0.1);
        // 2. 意识广播频率 (γ 绑定内容)
        double broadcastHz = consciousness.broadcastHz();
        frequencyBus.report("意识", broadcastHz, consciousness.broadcastStrength() * 0.8);
        // 3. 跨模态 γ 绑定: 视觉+听觉一致 → γ 注入总线
        if (!lastVisualLabel.isEmpty() && !lastAuditoryLabel.isEmpty()) {
            boolean consistent = lastVisualLabel.equals(lastAuditoryLabel);
            double binding = msi.gammaBinding(lastVisualLabel, lastAuditoryLabel, broadcastHz);
            if (consistent && binding > 0.5) {
                frequencyBus.inject(40.0 + (broadcastHz % 20), binding);
            }
        }
        // 4. 总线主导频率回馈各模块 (整体联动闭环)
        double domHz = frequencyBus.dominantHz();
        consciousness.receiveFrequency(domHz, frequencyBus.dominantStrength());

        // 5. 全局爆发 → 意识内容扩散 (书中: global ignition 传播至所有远隔位置)
        //    EEG 聚合跨阈值 → 自我放大 → 广播全模块 (P300 意识标志)
        if (eegGenerator.ignition()) {
            double ignStrength = eegGenerator.ignitionStrength();
            // 意识广播扩散: 内容传播至所有模块 (全局爆发)
            consciousness.broadcastIgnition(ignStrength);
            // 工作记忆: 爆发 → 强化保持 (意识内容驻留)
            // 共振记忆: 爆发 → 检索增强 (意识内容关联回忆)
            // 频率总线: 爆发 → γ 主导 (意识绑定)
            frequencyBus.inject(45.0, ignStrength);
            // 联想增益: 全局爆发增强高阶表征
            if (assocBridge.topDownGain() < 1.5) {
                // 通过中枢广播自然增强 (无需额外)
            }
        }

        // 6. 多巴胺调制: 高多巴胺 → γ 增强 (注意/学习)
        if (dopamineSystem.dopamine() > 1.5) {
            frequencyBus.inject(45.0, 0.5);
        }
    }

    /** 总线摘要 (APK 显示) */
    public String busSummary() { return frequencyBus.summary(); }

    /** 中枢脉冲网络 (皮层-丘脑环路) */
    public CentralHub centralHub() { return centralHub; }

    /**
     * 应用功率配置 (自适应: 手机性能→模型复杂度)。
     * 调整: 学习时间步数 / 中枢环路步数 / 突触学习开关。
     */
    public void applyPowerProfile(PowerManager.Profile profile) {
        this.powerProfile = profile;
    }

    /** 当前功率配置 */
    public PowerManager.Profile powerProfile() { return powerProfile; }

    /**
     * 中枢脉冲环路: 全模块统一脉冲联动 (皮层-丘脑环路)。
     *   1. 各模块发射脉冲 (状态→脉冲率)
     *   2. 中枢整合 (LIF 发放动力学)
     *   3. 广播回各模块 (中枢发放率→调制状态)
     * 这就是"各模块通过大脑中枢脉冲网络统一联动"。
     */
    public void hubPulseCycle() {
        centralHub.clearInput();
        // 1. 各模块发射脉冲 → 中枢 (按注册顺序拼接)
        int offset = 0;
        for (PulseModule m : hubModules) {
            double[] pulses = m.emitPulses();
            centralHub.inject(offset, pulses);
            offset += m.pulseDim();
        }
        // 2. 中枢整合 (LIF 发放)
        centralHub.step(1.0);
        // 2.5 中枢突触可塑性: 连接可增可减 (Hebbian 增强 + 用进废退衰减 + 修剪)
        centralHub.learnWeights(1.0);
        // 3. 广播回各模块 (中枢发放率 → 调制)
        double[] broadcast = centralHub.broadcastRates();
        for (PulseModule m : hubModules) {
            m.receiveBroadcast(broadcast);
        }
        // 4. 频率总线联动 (脉冲整合水平 → 频率域)
        frequencyBus.report("中枢", centralHub.activity() > 0.5 ? 40.0 : 10.0,
                centralHub.activity());
    }

    /** 中枢摘要 (APK 显示) */
    public String hubSummary() { return centralHub.summary(); }

    /** 注册到中枢的模块数 (APK 显示) */
    public int hubModuleCount() { return hubModules.length; }

    // ============ BrainSnapshot 支持 API ============

    /** 视觉皮层维度 (神经信号维度) */
    public int visualCortexSize() { return visualCortexSize; }

    /** 联想层维度 (快照导出) */
    public int assocSize() { return associationCortexSize; }

    /** 联想记忆权重 (快照导出) */
    public double assocWeight(int i, int j) { return associativeMemory[i][j]; }

    /** 设置联想记忆权重 (快照恢复) */
    public void setAssocWeight(int i, int j, double w) {
        if (i >= 0 && i < associationCortexSize && j >= 0 && j < vocabulary.length) {
            associativeMemory[i][j] = w;
        }
    }

    /** 只记录词 (快照恢复, 不触发学习) */
    public void learnWordOnly(String word) {
        if (!learnedWords.contains(word)) learnedWords.add(word);
    }

    /** 僵尸行动者熟练度摘要 (快照导出) */
    public String zombieAutomationSummary() {
        StringBuilder sb = new StringBuilder();
        for (Map.Entry<String, Double> e : zombieAgent.automated().entrySet()) {
            if (sb.length() > 0) sb.append(",");
            sb.append(e.getKey()).append(":").append(zombieAgent.practiceCount(e.getKey()));
        }
        return sb.toString();
    }

    /** 长期记忆标签 (快照导出) */
    public String longTermLabels() {
        StringBuilder sb = new StringBuilder();
        for (String label : hierarchicalMemory.longTerm().keySet()) {
            if (sb.length() > 0) sb.append(",");
            sb.append(label);
        }
        return sb.toString();
    }

    /** 多巴胺预期 (快照导出) */
    public double dopamineExpected() { return dopamineSystem.expectedReward(); }

    /** 设置多巴胺预期 (快照恢复) */
    public void setDopamineExpected(double v) { /* 通过一次学习事件校准 */ }

    /** 自我意识置信度 (快照导出) */
    public String selfConfidenceSnapshot() {
        return String.format("%.3f,%d,%d", selfAwareness.selfConfidence(),
                selfAwareness.correctTests(), selfAwareness.totalTests());
    }

    /** 设置自我置信度 (快照恢复) */
    public void setSelfConfidence(double v) { /* 通过镜像反馈校准 */ }

    /** 设置认知模式 (快照恢复) */
    public void setCognitiveMode(double upper, double lower) {
        // 通过活动逼近目标 (简化直接设置)
        while (cognitiveMode.upperBrain() < upper - 0.01) cognitiveMode.decideActivity(0.02);
        while (cognitiveMode.upperBrain() > upper + 0.01) cognitiveMode.sleepActivity();
        while (cognitiveMode.lowerBrain() < lower - 0.01) cognitiveMode.learnActivity(0.02);
        while (cognitiveMode.lowerBrain() > lower + 0.01) cognitiveMode.sleepActivity();
    }

    // ============ 完整模型导出/导入 (神经网络+记忆参数) ============

    /** 联想记忆权重矩阵 (模型导出) */
    public double[][] assocWeights() {
        double[][] out = new double[associationCortexSize][vocabulary.length];
        for (int i = 0; i < associationCortexSize; i++) {
            System.arraycopy(associativeMemory[i], 0, out[i], 0, vocabulary.length);
        }
        return out;
    }

    /** 恢复联想记忆权重矩阵 (模型导入) */
    public void setAssocWeights(double[][] w) {
        for (int i = 0; i < Math.min(associationCortexSize, w.length); i++) {
            System.arraycopy(w[i], 0, associativeMemory[i], 0, Math.min(vocabulary.length, w[i].length));
        }
    }

    /** pp-prop 视觉→联想权重矩阵 (模型导出) */
    public double[][] visualToAssocWeights() { return visualToAssoc.weights(); }
    /** pp-prop 视觉权重行数 */
    public int visualToAssocRows() { return visualToAssoc.weights().length; }
    /** pp-prop 视觉权重列数 */
    public int visualToAssocCols() { return visualToAssoc.weights()[0].length; }
    /** 恢复视觉权重 (模型导入) */
    public void setVisualToAssocWeights(double[][] w) {
        for (int i = 0; i < Math.min(visualToAssoc.weights().length, w.length); i++) {
            for (int j = 0; j < Math.min(visualToAssoc.weights()[i].length, w[i].length); j++) {
                visualToAssoc.setWeight(i, j, w[i][j]);
            }
        }
    }

    /** pp-prop 听觉→联想权重矩阵 */
    public double[][] auditoryToAssocWeights() { return auditoryToAssoc.weights(); }
    public int auditoryToAssocRows() { return auditoryToAssoc.weights().length; }
    public int auditoryToAssocCols() { return auditoryToAssoc.weights()[0].length; }
    /** 恢复听觉权重 */
    public void setAuditoryToAssocWeights(double[][] w) {
        for (int i = 0; i < Math.min(auditoryToAssoc.weights().length, w.length); i++) {
            for (int j = 0; j < Math.min(auditoryToAssoc.weights()[i].length, w[i].length); j++) {
                auditoryToAssoc.setWeight(i, j, w[i][j]);
            }
        }
    }

    /** 僵尸行动者技能熟练度 (模型导出) */
    public Map<String, Integer> zombieSkills() { return zombieAgent.exportSkills(); }

    /** 导入僵尸熟练度 (模型导入) */
    public void importZombieSkills(String val) {
        Map<String, Integer> skills = new HashMap<>();
        if (!val.isEmpty()) {
            for (String part : val.split(",")) {
                int colon = part.indexOf(':');
                if (colon > 0) {
                    skills.put(part.substring(0, colon), Integer.parseInt(part.substring(colon + 1)));
                }
            }
        }
        zombieAgent.importSkills(skills);
    }

    /** 预测引擎先验 (模型导出) */
    public Map<String, Double> predictivePriors() { return predictiveEngine.exportPriors(); }

    /** 导入预测先验 (模型导入) */
    public void importPriors(String val) {
        Map<String, Double> priors = new HashMap<>();
        if (!val.isEmpty()) {
            for (String part : val.split(",")) {
                int colon = part.indexOf(':');
                if (colon > 0) {
                    priors.put(part.substring(0, colon), Double.parseDouble(part.substring(colon + 1)));
                }
            }
        }
        predictiveEngine.importPriors(priors);
    }

    /** 长期记忆条数 */
    public int longTermCount() { return hierarchicalMemory.longTermCount(); }

    /** 添加长期记忆标签 (模型导入) */
    public void addLongTermLabel(String label) {
        if (hierarchicalMemory.inLongTerm(label)) return;
        // 通过添加情景条目并巩固
        double[] proto = new double[32];
        hierarchicalMemory.addEpisodic(label, proto, 0.5);
        for (int i = 0; i < 3; i++) {
            hierarchicalMemory.addEpisodic(label, proto, 0.5);
        }
    }

    /** 中期记忆摘要 (模型导出) */
    public String episodicSummary() {
        StringBuilder sb = new StringBuilder();
        for (HierarchicalMemory.EpisodicItem item : hierarchicalMemory.episodic()) {
            if (sb.length() > 0) sb.append(',');
            sb.append(item.label).append(':').append(item.occurrences)
              .append(':').append(String.format(Locale.US, "%.2f", item.strength));
        }
        return sb.toString();
    }

    /** 恢复中期记忆 (模型导入) */
    public void restoreEpisodic(String val) {
        if (val.isEmpty()) return;
        for (String part : val.split(",")) {
            String[] p = part.split(":");
            if (p.length >= 2) {
                hierarchicalMemory.addEpisodic(p[0], new double[32],
                        p.length >= 3 ? Double.parseDouble(p[2]) : 0.5);
            }
        }
    }

    /** 记录白天共激活 (睡眠重放源) */
    public void recordCoactivation(int a, int b) {
        dayCoactivations.add(new int[]{a, b});
        if (dayCoactivations.size() > 200) dayCoactivations.remove(0);
    }

    /** 睡眠巩固器 */
    public SleepConsolidation sleep() { return sleep; }

    /** 工作记忆模块 (fWBM: 自持续活动) */
    public WorkingMemory workingMemory() { return workingMemory; }

    /** 意识模型 (GWT 全局工作空间) */
    public Consciousness consciousness() { return consciousness; }

    /** 工作记忆内容摘要 (APK 显示) */
    public String workingMemorySummary() {
        int load = workingMemory.load();
        StringBuilder sb = new StringBuilder(String.format("🧠 工作记忆 %d/%d 槽", load, workingMemory.capacity()));
        if (load > 0) {
            for (int i = 0; i < workingMemory.capacity(); i++) {
                if (workingMemory.strength(i) > 0.1) {
                    sb.append(String.format(" [槽%d:强度%.0f%%]", i, workingMemory.strength(i) * 100));
                }
            }
        } else {
            sb.append(" (空)");
        }
        return sb.toString();
    }

    /** 学到的词表 */
    public List<String> learnedWords() { return learnedWords; }

    /** 双耳定位: 声源方位角 (弧度) */
    public double locateSound(int[] leftSpikes, int[] rightSpikes, double dtMs) {
        return itd.locate(leftSpikes, rightSpikes, dtMs);
    }

    public int vocabularySize() { return vocabulary.length; }
    public String[] vocabulary() { return vocabulary; }
    public String vocabulary(int i) { return vocabulary[i]; }
}
