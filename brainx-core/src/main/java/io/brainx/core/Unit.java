package io.brainx.core;

/**
 * 物理单位系统 (对应 brain 生态 brainunit, NC 2025: 物理单位感知计算)。
 *
 * 量纲感知: 每个单位 = 量纲 (基维度幂次) + 标度 (相对 SI 基准倍数)。
 *   - 运算自动推导: 乘=量纲相加+标度相乘, 除=相减+标度相除, 幂=标度乘幂
 *   - 加减校验量纲一致 (防 ms+mV 错误), 标度换算后相加
 *   - 换算: to(target) = v × scale/scale_target
 *   - 复合维度: 电导 = 电流/电压 (I·V⁻¹), 膜电容 = 电流×时间 (I·T)
 *     → Cm/gL 量纲自动推出时间, V=I/G 推出电压 (物理关系内建)
 *
 * 算力友好: 量纲 6 个 int + 标度 1 个 double, 校验 O(1)。
 */
public final class Unit {
    private Unit() {}

    /** 量纲: [时间, 电压, 电流, 频率, 长度] 幂次 (电导=电流/电压复合) */
    public static final class Dim {
        public final int t, v, i, f, l;
        public Dim(int t, int v, int i, int f, int l) {
            this.t = t; this.v = v; this.i = i; this.f = f; this.l = l;
        }
        public Dim mul(Dim o) { return new Dim(t+o.t, v+o.v, i+o.i, f+o.f, l+o.l); }
        public Dim div(Dim o) { return new Dim(t-o.t, v-o.v, i-o.i, f-o.f, l-o.l); }
        public Dim pow(int p) { return new Dim(t*p, v*p, i*p, f*p, l*p); }
        public boolean isDimensionless() { return t==0 && v==0 && i==0 && f==0 && l==0; }
        @Override public boolean equals(Object o) {
            if (!(o instanceof Dim)) return false;
            Dim d = (Dim) o;
            return t==d.t && v==d.v && i==d.i && f==d.f && l==d.l;
        }
        @Override public int hashCode() { return (((t*7+v)*7+i)*7+f)*7+l; }
        @Override public String toString() {
            StringBuilder sb = new StringBuilder();
            if (t!=0) sb.append("T^").append(t).append(" ");
            if (v!=0) sb.append("V^").append(v).append(" ");
            if (i!=0) sb.append("I^").append(i).append(" ");
            if (f!=0) sb.append("F^").append(f).append(" ");
            if (l!=0) sb.append("L^").append(l).append(" ");
            return sb.length()==0 ? "1" : sb.toString().trim();
        }
    }

    public static final Dim DIMLESS = new Dim(0,0,0,0,0);
    public static final Dim TIME = new Dim(1,0,0,0,0);
    public static final Dim VOLTAGE = new Dim(0,1,0,0,0);
    public static final Dim CURRENT = new Dim(0,0,1,0,0);
    public static final Dim FREQUENCY = new Dim(-1,0,0,0,0);      // 1/T
    public static final Dim LENGTH = new Dim(0,0,0,0,1);
    /** 电导 = 电流/电压 (I·V⁻¹) */
    public static final Dim CONDUCTANCE = CURRENT.div(VOLTAGE);

    /** 单位 = 量纲 + 标度 (相对 SI 基准倍数) */
    public static final class UnitDef {
        public final Dim dim;
        public final double scale;
        public UnitDef(Dim dim, double scale) { this.dim = dim; this.scale = scale; }
        public UnitDef mul(UnitDef o) { return new UnitDef(dim.mul(o.dim), scale * o.scale); }
        public UnitDef div(UnitDef o) { return new UnitDef(dim.div(o.dim), scale / o.scale); }
        public UnitDef pow(int p) { return new UnitDef(dim.pow(p), Math.pow(scale, p)); }
    }

    // 常用单位 (SI 基准: s, V, A, Hz, m)
    public static final UnitDef MS = new UnitDef(TIME, 1e-3);
    public static final UnitDef S = new UnitDef(TIME, 1);
    public static final UnitDef MV = new UnitDef(VOLTAGE, 1e-3);
    public static final UnitDef NA = new UnitDef(CURRENT, 1e-9);
    public static final UnitDef NS = new UnitDef(CONDUCTANCE, 1e-9);
    public static final UnitDef HZ = new UnitDef(FREQUENCY, 1);
    public static final UnitDef M = new UnitDef(LENGTH, 1);
    public static final UnitDef DIMLESS_UNIT = new UnitDef(DIMLESS, 1);

    /** 带单位数值 */
    public static final class Value {
        public final double v;
        public final UnitDef unit;
        public Value(double v, UnitDef unit) { this.v = v; this.unit = unit; }

        private void requireSameDim(Value o) {
            if (!unit.dim.equals(o.unit.dim)) {
                throw new IllegalArgumentException("量纲不一致: " + unit.dim + " vs " + o.unit.dim);
            }
        }

        /** 加/减: 量纲必须一致, 标度换算后相加 (10ms + 1s = 1.01s) */
        public Value add(Value o) {
            requireSameDim(o);
            return new Value(v + o.v * o.unit.scale / unit.scale, unit);
        }
        public Value sub(Value o) {
            requireSameDim(o);
            return new Value(v - o.v * o.unit.scale / unit.scale, unit);
        }

        /** 乘/除/幂: 量纲与标度自动推导 */
        public Value mul(Value o) { return new Value(v * o.v, unit.mul(o.unit)); }
        public Value div(Value o) {
            if (o.v == 0) throw new ArithmeticException("除零");
            return new Value(v / o.v, unit.div(o.unit));
        }
        public Value pow(int p) { return new Value(Math.pow(v, p), unit.pow(p)); }

        /** 换算到目标单位 (同量纲): ms→s 等 */
        public double to(UnitDef target) {
            if (!unit.dim.equals(target.dim)) throw new IllegalArgumentException("换算量纲不一致");
            return v * unit.scale / target.scale;
        }

        public boolean isDimensionless() { return unit.dim.isDimensionless(); }
        public double value() { return v; }
        public UnitDef unit() { return unit; }
        @Override public String toString() { return v + " [" + unit.dim + "]"; }
    }

    // ===== 便捷构造 =====
    public static Value ms(double v) { return new Value(v, MS); }
    public static Value s(double v) { return new Value(v, S); }
    public static Value mV(double v) { return new Value(v, MV); }
    public static Value nA(double v) { return new Value(v, NA); }
    public static Value nS(double v) { return new Value(v, NS); }
    public static Value Hz(double v) { return new Value(v, HZ); }
    public static Value m(double v) { return new Value(v, M); }
    public static Value unitless(double v) { return new Value(v, DIMLESS_UNIT); }

    /** 膜时间常数 τ = Cm/gL (Cm 电荷量纲 I·T, gL 电导 I·V⁻¹ → τ 时间量纲自动推导) */
    public static Value tauFromCmG(Value cm, Value gl) {
        return cm.div(gl);
    }

    /** 欧姆定律: V = I/G */
    public static Value voltageFromCurrentConductance(Value i, Value g) {
        return i.div(g);
    }
}
