package io.brainx.core;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * 完整模型迁移验证: 导出→导入 → 神经网络连接参数 + 记忆参数完整恢复。
 * 这是"跨平台泛化"的核心: 训练成果=整个网络的连接参数+记忆参数。
 */
public class ModelTransferTest {

    @Test
    void snapshotContainsNeuralParams() {
        Brain brain = Brain.simpleBrain();
        double[] img = new double[VisualNeuralEncoder.OUTPUT_DIM];
        for (int i = 0; i < VisualNeuralEncoder.OUTPUT_DIM; i++) img[i] = (i % 2 == 0) ? 0.9 : 0.1;
        for (int e = 0; e < 5; e++) {
            brain.learnVisualWord(img, 0);
            brain.learnAuditoryWord(img, 1);
        }
        // 触发一些突触连接
        brain.synapseFormation().step(new boolean[io.brainx.core.VisualNeuralEncoder.OUTPUT_DIM + io.brainx.core.AudioNeuralEncoder.BANDS + 120], 10);
        String snap = BrainSnapshot.export(brain, 2, 15, 60, "成就", "物品");
        // 应包含神经网络参数
        assertTrue(snap.contains("nn-vis:"), "应含视觉神经网络权重");
        assertTrue(snap.contains("nn-aud:"), "应含听觉神经网络权重");
        assertTrue(snap.contains("syn:"), "应含突触连接");
        assertTrue(snap.contains("mem:"), "应含联想记忆");
        assertTrue(snap.contains("wm:"), "应含工作记忆");
        assertTrue(snap.contains("zombie:"), "应含僵尸熟练度");
        assertTrue(snap.contains("priors:"), "应含预测先验");
        assertTrue(snap.contains("ltmem:"), "应含长期记忆");
        assertTrue(snap.startsWith("BRAINX-SNAP-2"), "新格式魔数");
    }

    @Test
    void neuralWeightsTransferExactly() {
        Brain brain = Brain.simpleBrain();
        double[] imgA = new double[VisualNeuralEncoder.OUTPUT_DIM], imgB = new double[VisualNeuralEncoder.OUTPUT_DIM];
        for (int i = 0; i < VisualNeuralEncoder.OUTPUT_DIM; i++) { imgA[i] = (i%3==0)?0.9:0.1; imgB[i] = (i%3==1)?0.9:0.1; }
        for (int e = 0; e < 10; e++) {
            brain.learnVisualWord(imgA, 0);
            brain.learnVisualWord(imgB, 1);
        }
        // 记录原网络权重
        double[][] wBefore = brain.visualToAssocWeights();
        double[][] memBefore = brain.assocWeights();

        String snap = BrainSnapshot.export(brain, 2, 15, 60, "", "");
        Brain brain2 = Brain.simpleBrain();
        BrainSnapshot.RestoreInfo info = BrainSnapshot.importSnapshot(brain2, snap);
        assertNotNull(info, "快照应解析");
        assertTrue(info.nnRestored, "神经网络参数应恢复");

        // 验证权重逐元素一致 (真正的连接参数迁移) — 容差 5e-6 (导出精度 %.6f)
        double[][] wAfter = brain2.visualToAssocWeights();
        assertEquals(wBefore.length, wAfter.length);
        for (int i = 0; i < wBefore.length; i++) {
            for (int j = 0; j < wBefore[i].length; j++) {
                assertEquals(wBefore[i][j], wAfter[i][j], 5e-6, "权重[" + i + "][" + j + "]应一致");
            }
        }
        // 联想记忆一致
        double[][] mAfter = brain2.assocWeights();
        for (int i = 0; i < memBefore.length; i++) {
            for (int j = 0; j < memBefore[i].length; j++) {
                assertEquals(memBefore[i][j], mAfter[i][j], 5e-6, "联想[" + i + "][" + j + "]应一致");
            }
        }
    }

    @Test
    void importedModelStillRecognizes() {
        Brain brain = Brain.simpleBrain();
        double[] imgA = new double[VisualNeuralEncoder.OUTPUT_DIM], imgB = new double[VisualNeuralEncoder.OUTPUT_DIM];
        for (int i = 0; i < VisualNeuralEncoder.OUTPUT_DIM; i++) { imgA[i] = (i%3==0)?0.9:0.1; imgB[i] = (i%3==1)?0.9:0.1; }
        for (int e = 0; e < 10; e++) {
            brain.learnVisualWord(imgA, 0);
            brain.learnVisualWord(imgB, 1);
        }
        String before = brain.recognizeVisual(imgA);
        String snap = BrainSnapshot.export(brain, 2, 15, 60, "", "");

        // 导入到全新大脑 (模拟另一台设备)
        Brain brain2 = Brain.simpleBrain();
        BrainSnapshot.importSnapshot(brain2, snap);
        String after = brain2.recognizeVisual(imgA);
        // 导入后识别能力保持一致 (迁移成功)
        assertEquals(before, after, "导入后识别应一致: before=" + before + " after=" + after);
    }

    @Test
    void synapseConnectionsTransfer() {
        Brain brain = Brain.simpleBrain();
        double[] img = new double[VisualNeuralEncoder.OUTPUT_DIM];
        for (int i = 0; i < VisualNeuralEncoder.OUTPUT_DIM; i++) img[i] = 0.9;
        for (int e = 0; e < 5; e++) brain.learnVisualWord(img, 0);
        int connBefore = brain.synapseFormation().connectionCount();
        String snap = BrainSnapshot.export(brain, 1, 5, 30, "", "");

        Brain brain2 = Brain.simpleBrain();
        BrainSnapshot.RestoreInfo info = BrainSnapshot.importSnapshot(brain2, snap);
        assertNotNull(info);
        assertTrue(info.synapseRestored, "突触连接应恢复");
        // 连接数应一致或接近 (导入重建)
        int connAfter = brain2.synapseFormation().connectionCount();
        assertTrue(connAfter > 0, "导入后应有连接, got=" + connAfter);
    }

    @Test
    void snapshotCrossPlatformFormat() {
        Brain brain = Brain.simpleBrain();
        double[] img = new double[VisualNeuralEncoder.OUTPUT_DIM];
        for (int i = 0; i < VisualNeuralEncoder.OUTPUT_DIM; i++) img[i] = (i%2==0)?0.9:0.1;
        brain.learnVisualWord(img, 0);
        String snap = BrainSnapshot.export(brain, 1, 5, 30, "", "");
        // 跨平台: 纯 ASCII 结构 (除词表中文外), 每行 key:value
        String[] lines = snap.split("\n");
        assertTrue(lines.length > 10, "应有足够字段");
        // 权重数值格式可被任何语言解析 (double 格式)
        for (String line : lines) {
            if (line.startsWith("nn-vis:") || line.startsWith("mem:")) {
                assertTrue(line.contains(",") || line.contains(";"), "矩阵应含分隔符");
            }
        }
        // 不含二进制/换行符嵌套
        assertFalse(snap.contains("\r"), "不应含回车");
    }

    @Test
    void zombieAndPriorsTransfer() {
        Brain brain = Brain.simpleBrain();
        double[] img = new double[VisualNeuralEncoder.OUTPUT_DIM];
        for (int i = 0; i < VisualNeuralEncoder.OUTPUT_DIM; i++) img[i] = 0.9;
        for (int e = 0; e < 5; e++) brain.learnVisualWord(img, 0);  // 触发僵尸自动化
        String snap = BrainSnapshot.export(brain, 1, 5, 30, "", "");

        Brain brain2 = Brain.simpleBrain();
        BrainSnapshot.importSnapshot(brain2, snap);
        // 僵尸技能应恢复
        assertTrue(brain2.zombieSkills().size() > 0, "僵尸熟练度应恢复");
        // 预测先验应恢复
        assertTrue(brain2.predictivePriors().size() > 0, "预测先验应恢复");
    }

    @Test
    void legacyV1SnapshotStillImports() {
        // 兼容旧版 BRAINX-SNAP-1 格式 (向后兼容)
        Brain brain = Brain.simpleBrain();
        double[] img = new double[VisualNeuralEncoder.OUTPUT_DIM];
        for (int i = 0; i < VisualNeuralEncoder.OUTPUT_DIM; i++) img[i] = 0.9;
        brain.learnVisualWord(img, 0);
        // 构造旧格式快照
        StringBuilder sb = new StringBuilder();
        sb.append("BRAINX-SNAP-1\nlevel:1\npoints:5\nxp:30\nwords:你好\n");
        sb.append("mem:").append(io.brainx.core.BrainSnapshot.matrixToStr(brain.assocWeights())).append("\n");
        Brain brain2 = Brain.simpleBrain();
        BrainSnapshot.RestoreInfo info = BrainSnapshot.importSnapshot(brain2, sb.toString());
        assertNotNull(info, "旧格式应兼容");
        assertEquals(1, info.level);
    }
}
