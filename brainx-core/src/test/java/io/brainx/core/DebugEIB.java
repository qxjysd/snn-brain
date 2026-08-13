package io.brainx.core;

public class DebugEIB {
    public static void main(String[] args) {
        EIBNetwork net = EIBNetwork.defaultParams();
        // 持续左线索 200 步
        int totalFire = 0;
        for (int t = 0; t < 200; t++) {
            net.step(1.0, 0.0, 1.0);
            boolean[] firing = net.firingState();
            int f = 0;
            for (boolean b : firing) if (b) f++;
            totalFire += f;
            if (t % 50 == 0) {
                double[] rates = net.firingRates();
                System.out.printf("t=%d firing=%d E_rate=%.1fHz I_rate=%.1fHz evL=%.4f evR=%.4f%n",
                        t, f, rates[0], rates[1], net.evidenceLeft(), net.evidenceRight());
            }
        }
        System.out.println("totalFire=" + totalFire + " avg/step=" + (totalFire/200.0));
    }
}
