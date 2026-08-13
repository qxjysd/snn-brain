package io.brainx.core;

import io.brainx.core.mass.*;

public class DebugMain {
    public static void main(String[] args) {
        // Jansen-Rit 调试
        System.out.println("=== Jansen-Rit (0.5s, dt=0.1ms) ===");
        JansenRit jr = JansenRit.defaultParams();
        double min = 1e9, max = -1e9;
        int crossings = 0;
        double prev = 0;
        for (int t = 0; t < 5000; t++) {
            double y = jr.step(0.1, 0.1);
            if (t > 1000) {
                min = Math.min(min, y); max = Math.max(max, y);
                if (prev * y < 0) crossings++;
            }
            prev = y;
        }
        System.out.println("min=" + min + " max=" + max + " crossings=" + crossings);

        // 检查是否只是幅度小
        System.out.println("\n=== Jansen-Rit 幅度检查 (打印最后20步) ===");
        jr.reset();
        for (int t = 0; t < 5000; t++) {
            double y = jr.step(0.1, 0.1);
            if (t >= 4980) System.out.printf("t=%d y=%.4f\n", t, y);
        }

        // Wong-Wang 调试
        System.out.println("\n=== Wong-Wang 决策 (coh=0.5, 500步) ===");
        WongWang ww = WongWang.defaultParams();
        for (int t = 0; t < 500; t++) {
            ww.step(0.5, 1.0);
            if (t % 100 == 0) System.out.printf("t=%d s1=%.4f s2=%.4f act=%.4f\n",
                    t, ww.s1(), ww.s2(), ww.activity());
        }
        System.out.println("decision=" + ww.decision());

        System.out.println("\n=== Wong-Wang 决策 (coh=-0.5) ===");
        ww.reset();
        for (int t = 0; t < 500; t++) {
            ww.step(-0.5, 1.0);
            if (t % 100 == 0) System.out.printf("t=%d s1=%.4f s2=%.4f\n", t, ww.s1(), ww.s2());
        }
        System.out.println("decision=" + ww.decision());
    }
}
