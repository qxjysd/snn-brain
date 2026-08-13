package io.brainx.app;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.view.View;

import java.util.List;

/**
 * 突触连接拓扑可视化 —— Canvas 绘制神经元网络拓扑。
 * 节点 = 神经元 (按皮层分色: 视觉蓝/听觉绿/联想紫)
 * 边 = 成熟突触连接 (粗细 = 权重)
 * 对应: 连接组拓扑可视化 (fWBM 结构约束; 突触形成可视化)
 */
public class TopologyView extends View {
    private final Paint nodePaint = new Paint();
    private final Paint edgePaint = new Paint();
    private final Paint bgPaint = new Paint();
    private final Paint labelPaint = new Paint();

    // 节点布局: 三列 (视觉/听觉/联想)
    private final int visualCount, auditoryCount, assocCount;
    private float[] nodeX, nodeY;
    private int[] nodeColor;
    private List<double[]> connections;  // [pre, post, weight]

    public TopologyView(Context ctx, int visualCount, int auditoryCount, int assocCount) {
        super(ctx);
        this.visualCount = visualCount;
        this.auditoryCount = auditoryCount;
        this.assocCount = assocCount;
        bgPaint.setColor(Color.parseColor("#0B1026"));
        labelPaint.setColor(Color.parseColor("#80CBC4"));
        labelPaint.setTextSize(22f);
        layoutNodes();
    }

    /** 布局节点位置 (三列环形) */
    private void layoutNodes() {
        int total = visualCount + auditoryCount + assocCount;
        nodeX = new float[total];
        nodeY = new float[total];
        nodeColor = new int[total];
        // 视觉 (左列)
        placeColumn(0, visualCount, 0.25f, Color.parseColor("#00E5FF"));
        // 听觉 (右列)
        placeColumn(visualCount, auditoryCount, 0.75f, Color.parseColor("#4CAF50"));
        // 联想 (中列)
        placeColumn(visualCount + auditoryCount, assocCount, 0.5f, Color.parseColor("#CE93D8"));
    }

    private void placeColumn(int start, int count, float cx, int color) {
        for (int i = 0; i < count; i++) {
            nodeX[start + i] = cx;
            nodeY[start + i] = 0.15f + 0.7f * i / Math.max(1, count - 1);
            nodeColor[start + i] = color;
        }
    }

    public void setConnections(List<double[]> conns) {
        this.connections = conns;
        postInvalidate();
    }

    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        int w = getWidth(), h = getHeight();
        canvas.drawRect(0, 0, w, h, bgPaint);

        // 列标签
        labelPaint.setColor(Color.parseColor("#00E5FF"));
        canvas.drawText("视觉", w * 0.25f - 20, 30, labelPaint);
        labelPaint.setColor(Color.parseColor("#4CAF50"));
        canvas.drawText("听觉", w * 0.75f - 20, 30, labelPaint);
        labelPaint.setColor(Color.parseColor("#CE93D8"));
        canvas.drawText("联想", w * 0.5f - 20, 30, labelPaint);

        // 边 (连接)
        if (connections != null) {
            for (double[] c : connections) {
                int pre = (int) c[0], post = (int) c[1];
                double weight = c[2];
                if (pre >= nodeX.length || post >= nodeX.length) continue;
                float stroke = (float) (1 + weight * 4);
                edgePaint.setColor(weight > 0.3 ? Color.parseColor("#FFD54F")
                        : Color.parseColor("#4FC3F7"));
                edgePaint.setStrokeWidth(stroke);
                edgePaint.setAlpha((int) (100 + weight * 155));
                canvas.drawLine(nodeX[pre] * w, nodeY[pre] * h,
                        nodeX[post] * w, nodeY[post] * h, edgePaint);
            }
        }

        // 节点
        for (int i = 0; i < nodeX.length; i++) {
            nodePaint.setColor(nodeColor[i]);
            nodePaint.setAlpha(220);
            canvas.drawCircle(nodeX[i] * w, nodeY[i] * h, 7, nodePaint);
        }
    }
}
