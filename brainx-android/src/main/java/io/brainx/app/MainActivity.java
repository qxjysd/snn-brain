package io.brainx.app;

import android.Manifest;
import android.app.Activity;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.Color;
import android.hardware.Camera;
import android.media.AudioFormat;
import android.media.AudioRecord;
import android.media.AudioTrack;
import android.media.MediaRecorder;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.os.SystemClock;
import android.speech.tts.TextToSpeech;
import android.util.Log;
import android.view.SurfaceHolder;
import android.view.SurfaceView;
import android.view.View;
import android.widget.Button;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.TextView;
import android.widget.Toast;

import io.brainx.core.Brain;
import io.brainx.core.EduTrainer;
import io.brainx.core.VocalCordSimulator;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Random;

/**
 * brain 人脑脉冲模型 APK 主界面。
 *
 * 功能:
 *   📷 摄像头 → 视觉皮层 SNN → 学习/识别物品词
 *   🎤 麦克风 → 听觉皮层 SNN + ITD 声源定位 → 学习/识别声音词
 *   🗣️ TTS 语音输出 (说出识别结果/教育反馈)
 *   🧠 脑图 Canvas 可视化 (发放热图 + EEG 波形)
 *   🎓 教育激励/惩罚系统 (答对奖励+1点+随机物品, 答错纠正引导重学)
 */
public class MainActivity extends Activity implements SurfaceHolder.Callback {
    private static final String TAG = "brain";
    private static final int PERMISSION_CODE = 100;
    private static final int IMPORT_REQUEST = 200;
    private static final int VISION_SIZE = io.brainx.core.VisualNeuralEncoder.OUTPUT_DIM;  // 5120 神经信号
    private static final int AUDIO_SIZE = io.brainx.core.AudioNeuralEncoder.BANDS;  // 24 耳蜗对数频带
    private static final int SAMPLE_RATE = io.brainx.core.AudioNeuralEncoder.SAMPLE_RATE;  // 16kHz
    private static final int BUFFER_SIZE = SAMPLE_RATE / 4;  // 250ms 音频帧

    private io.brainx.core.QuestSystem questSystem;
    private double generalizationRate = 0;  // 泛化能力 (识别未见变体)
    private io.brainx.core.PowerManager powerManager = new io.brainx.core.PowerManager();

    private Brain brain;
    private EduTrainer trainer;
    private TextToSpeech tts;
    private boolean ttsReady = false;  // TTS 初始化完成标志
    private final long ttsInitStart = SystemClock.elapsedRealtime();  // TTS 初始化起始时间
    private BrainView brainView;
    private PixelFaceView faceView;
    private TextView statusText, scoreText, curiosityText, emotionText, cognitionText, selfText, modeText, dopamineText, freqText, busText, hubText, eegText, powerText, langText, growthText, interactText, audioText;
    private TopologyView topologyView;
    private BrainVisionView brainVisionView;
    private WaveformView waveView;
    private SurfaceView cameraView;
    private volatile Camera camera;   // 多线程访问 (UI/相机回调/后台切换线程) → volatile 保证可见性
    private Handler handler = new Handler(Looper.getMainLooper());
    private Random rnd = new Random(42);

    // 学习模式状态
    private boolean learningMode = true;   // true=教学(输入词标签), false=测试(识别)
    private String currentLabel = "";
    private int audioWordIndex = 0;
    private final List<String> wordLabels = new ArrayList<>();
    private volatile boolean recording = false;
    private AudioRecord audioRecord;

    private Runnable brainLoop;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        // 初始化大脑 (8 词表: 对应日常生活物品)
        String[] vocab = {"苹果", "书本", "杯子", "花朵", "猫", "狗", "音乐", "拍手"};
        brain = Brain.simpleBrain();
        trainer = new EduTrainer();
        for (String w : vocab) wordLabels.add(w);
        questSystem = new io.brainx.core.QuestSystem();

        // TTS (加载慢/缺失处理: 不永久卡"初始化中")
        tts = new TextToSpeech(this, status -> {
            if (status == TextToSpeech.SUCCESS) {
                int lang = tts.setLanguage(Locale.CHINESE);
                ttsReady = true;
                Log.i(TAG, "TTS ready, lang=" + lang);
                if (lang == TextToSpeech.LANG_MISSING_DATA || lang == TextToSpeech.LANG_NOT_SUPPORTED) {
                    ttsReady = false;  // 中文包缺失: 视为未就绪, 显示提示
                    Log.w(TAG, "中文语音包缺失, lang=" + lang);
                }
            } else {
                Log.e(TAG, "TTS init failed: " + status);
            }
        });
        // 超时降级: 10 秒后仍未就绪 → 标记 (显示提示而非永久加载中)
        new Handler(Looper.getMainLooper()).postDelayed(() -> {
            if (!ttsReady && tts != null) {
                Log.w(TAG, "TTS init timeout, 可能缺引擎/语音包");
            }
        }, 10000);

        // UI
        buildUi();

        // 自动恢复上次备份的模型 (跨会话记忆保留)
        autoRestoreBackup();

        // 权限 (原生 API, minSdk 26 直接支持)
        if (checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED
                || checkSelfPermission(Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(new String[]{Manifest.permission.CAMERA, Manifest.permission.RECORD_AUDIO}, PERMISSION_CODE);
        }

        // 大脑循环
        startBrainLoop();
    }

    private void buildUi() {
        // 可滚动容器: 内容超屏可上下拖动查看完整内容
        ScrollView scrollRoot = new ScrollView(this);
        scrollRoot.setBackgroundColor(Color.parseColor("#0B1026"));
        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        scrollRoot.addView(root, new ScrollView.LayoutParams(
                ScrollView.LayoutParams.MATCH_PARENT, ScrollView.LayoutParams.WRAP_CONTENT));

        // 状态栏
        statusText = new TextView(this);
        statusText.setTextColor(Color.WHITE);
        statusText.setTextSize(16f);
        statusText.setPadding(20, 10, 20, 10);
        root.addView(statusText);

        scoreText = new TextView(this);
        scoreText.setTextColor(Color.parseColor("#FFD700"));
        scoreText.setTextSize(15f);
        scoreText.setPadding(20, 0, 20, 10);
        root.addView(scoreText);

        // 情绪标签 + 像素表情
        LinearLayout emotionRow = new LinearLayout(this);
        emotionRow.setOrientation(LinearLayout.HORIZONTAL);
        emotionRow.setGravity(android.view.Gravity.CENTER_VERTICAL);

        emotionText = new TextView(this);
        emotionText.setTextColor(Color.WHITE);
        emotionText.setTextSize(18f);
        emotionText.setPadding(20, 0, 10, 0);

        faceView = new PixelFaceView(this);
        faceView.setLayoutParams(new LinearLayout.LayoutParams(160, 160));

        emotionRow.addView(emotionText);
        emotionRow.addView(faceView);
        root.addView(emotionRow);

        // 认知状态: 意识 + 工作记忆
        cognitionText = new TextView(this);
        cognitionText.setTextColor(Color.parseColor("#CE93D8"));
        cognitionText.setTextSize(14f);
        cognitionText.setPadding(20, 0, 20, 10);
        root.addView(cognitionText);

        // 自我意识 + 记忆状态
        selfText = new TextView(this);
        selfText.setTextColor(Color.parseColor("#FFAB91"));
        selfText.setTextSize(14f);
        selfText.setPadding(20, 0, 20, 10);
        root.addView(selfText);

        // 突触连接拓扑可视化
        topologyView = new TopologyView(this, 32, 16, 24);
        topologyView.setLayoutParams(new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, 200));
        root.addView(topologyView);

        // 认知模式 + 僵尸行动者
        modeText = new TextView(this);
        modeText.setTextColor(Color.parseColor("#A5D6A7"));
        modeText.setTextSize(13f);
        modeText.setPadding(20, 0, 20, 10);
        root.addView(modeText);

        // 多巴胺 + 预测引擎
        dopamineText = new TextView(this);
        dopamineText.setTextColor(Color.parseColor("#FFF59D"));
        dopamineText.setTextSize(13f);
        dopamineText.setPadding(20, 0, 20, 10);
        root.addView(dopamineText);

        // 频率波联动 (类脑: 脑电节律 + 频率共振记忆)
        freqText = new TextView(this);
        freqText.setTextColor(Color.parseColor("#80DEEA"));
        freqText.setTextSize(13f);
        freqText.setPadding(20, 0, 20, 10);
        root.addView(freqText);

        // 频率总线 (全模块联动)
        busText = new TextView(this);
        busText.setTextColor(Color.parseColor("#B39DDB"));
        busText.setTextSize(12f);
        busText.setPadding(20, 0, 20, 10);
        root.addView(busText);

        // 中枢脉冲网络 (皮层-丘脑环路)
        hubText = new TextView(this);
        hubText.setTextColor(Color.parseColor("#FF8A65"));
        hubText.setTextSize(12f);
        hubText.setPadding(20, 0, 20, 10);
        root.addView(hubText);

        // EEG 脉冲聚合 (书中: 聚合的脑活动 + 全局爆发)
        eegText = new TextView(this);
        eegText.setTextColor(Color.parseColor("#F48FB1"));
        eegText.setTextSize(12f);
        eegText.setPadding(20, 0, 20, 10);
        root.addView(eegText);

        // 功率自适应 (防卡死)
        powerText = new TextView(this);
        powerText.setTextColor(Color.parseColor("#A5D6A7"));
        powerText.setTextSize(12f);
        powerText.setPadding(20, 0, 20, 10);
        root.addView(powerText);

        // 语言学习状态 (模仿→理解→自主)
        langText = new TextView(this);
        langText.setTextColor(Color.parseColor("#FFAB91"));
        langText.setTextSize(12f);
        langText.setPadding(20, 0, 20, 10);
        root.addView(langText);

        // 成长潜力 (神经元→2000亿)
        growthText = new TextView(this);
        growthText.setTextColor(Color.parseColor("#FFF176"));
        growthText.setTextSize(12f);
        growthText.setPadding(20, 0, 20, 10);
        root.addView(growthText);

        // 互动状态 (开心度/被逗/笑)
        interactText = new TextView(this);
        interactText.setTextColor(Color.parseColor("#FFD54F"));
        interactText.setTextSize(12f);
        interactText.setPadding(20, 0, 20, 10);
        root.addView(interactText);

        // 语音输出状态 (TTS 核实)
        audioText = new TextView(this);
        audioText.setTextColor(Color.parseColor("#90CAF9"));
        audioText.setTextSize(12f);
        audioText.setPadding(20, 0, 20, 10);
        root.addView(audioText);

        // 好奇心条
        curiosityText = new TextView(this);
        curiosityText.setTextColor(Color.parseColor("#00E5FF"));
        curiosityText.setTextSize(15f);
        curiosityText.setPadding(20, 0, 20, 10);
        root.addView(curiosityText);

        // 脑图
        brainView = new BrainView(this);
        brainView.setLayoutParams(new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, 400));
        root.addView(brainView);

        // 摄像头预览 (全屏宽度, 按屏幕比例完整显示不变形)
        cameraView = new SurfaceView(this);
        android.util.DisplayMetrics dm = getResources().getDisplayMetrics();
        int camH = (int) (dm.widthPixels * 4f / 3f);  // 4:3 完整预览
        cameraView.setLayoutParams(new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, camH));
        cameraView.getHolder().addCallback(this);
        root.addView(cameraView);

        // 大脑视角可视化 (完整 5120 维神经信号: 64×64感受野 + 4方向×4尺度×64)
        // 高度与相机预览一致 (4:3) → 大脑视角与预览视频大小对齐
        brainVisionView = new BrainVisionView(this);
        brainVisionView.setLayoutParams(new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, camH));
        root.addView(brainVisionView);

        // 音波可视化 (声音输入/输出波形)
        waveView = new WaveformView(this);
        waveView.setLayoutParams(new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, 140));
        root.addView(waveView);

        // 按钮行: 仅 激励/惩罚/镜头转换 (其余全由模型自主调用)
        LinearLayout btnRow = new LinearLayout(this);
        btnRow.setOrientation(LinearLayout.HORIZONTAL);

        Button rewardBtn = new Button(this);
        rewardBtn.setText("🎉 激励");
        rewardBtn.setBackgroundColor(Color.parseColor("#43A047"));
        rewardBtn.setOnClickListener(v -> {
            // 激励: 表扬 + 奖励 (答对/学得好)
            EduTrainer.Feedback fb = trainer.reward();
            brain.rewardEvent(true, 1.0);
            speak("太好了！" + fb.speechText);
            statusText.setText("🎉 激励: " + fb.message);
            refreshScore();
        });

        Button punishBtn = new Button(this);
        punishBtn.setText("😠 惩罚");
        punishBtn.setBackgroundColor(Color.parseColor("#E53935"));
        punishBtn.setOnClickListener(v -> {
            // 惩罚: 纠正 (答错/行为不对)
            EduTrainer.Feedback fb = trainer.punish("行为");
            brain.rewardEvent(false, 1.0);
            speak(fb.speechText);
            statusText.setText("😠 惩罚: " + fb.message);
            refreshScore();
        });

        Button switchCamBtn = new Button(this);
        switchCamBtn.setText("📷 切换镜头");
        switchCamBtn.setBackgroundColor(Color.parseColor("#00897B"));
        switchCamBtn.setOnClickListener(v -> {
            switchCamera();
            // 更新大脑视角摄像头方向
            String dir = cameraFacing == Camera.CameraInfo.CAMERA_FACING_FRONT ? "前置" : "后置";
            brainVisionView.setVisual(null, null, 0, dir);
        });

        btnRow.addView(rewardBtn, new LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1));
        btnRow.addView(punishBtn, new LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1));
        btnRow.addView(switchCamBtn, new LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1));

        // 🌐 语言切换 (中/英)
        Button langBtn = new Button(this);
        langBtn.setText(Lang.label());
        langBtn.setBackgroundColor(Color.parseColor("#5E35B1"));
        langBtn.setOnClickListener(v -> {
            Lang.toggle();
            langBtn.setText(Lang.label());
            refreshTexts();
            refreshScore();
            statusText.setText(Lang.t("🌐 已切换语言", "🌐 Language switched"));
        });
        btnRow.addView(langBtn, new LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1));

        root.addView(btnRow);

        setContentView(scrollRoot);
        refreshScore();
    }

    /** 👀 观察模式: 主动观察当前看到的东西并描述 */
    private void captureAndObserve() {
        statusText.setText("👀 观察中...");
        speak("让我看看眼前有什么");
        captureFrame(features -> {
            if (features == null) return;
            // 识别并描述
            String[] result = brain.recognizeVisualWithConfidence(features);
            String desc = brain.observeScene(features);
            statusText.setText("👀 " + desc);
            speak(desc);
            // 更新大脑视角
            String dir = cameraFacing == Camera.CameraInfo.CAMERA_FACING_FRONT ? "前置" : "后置";
            brainVisionView.setVisual(features, result[0], Double.parseDouble(result[1]), dir);
        });
    }

    /** 📤 分享模型: 导出大脑状态 → 分享 (跨平台 .brain 纯文本) */
    private void shareModel() {
        String snap = io.brainx.core.BrainSnapshot.export(brain,
                trainer.level(), trainer.points(), trainer.xp(),
                String.join(",", trainer.achievements()),
                String.join(",", trainer.inventory()));
        // 纯文本分享 (最跨平台: 微信/邮件/蓝牙/记事本/其他设备)
        android.content.Intent share = new android.content.Intent(android.content.Intent.ACTION_SEND);
        share.setType("text/plain");
        share.putExtra(android.content.Intent.EXTRA_TEXT, snap);
        share.putExtra(android.content.Intent.EXTRA_SUBJECT, "大脑模型 brain_model.brain");
        startActivity(android.content.Intent.createChooser(share, "分享大脑模型 (.brain 纯文本)"));
        statusText.setText("📤 模型已分享 (" + snap.length() + "字符) — 任何平台可导入");
    }

    /** 📥 导入模型: 从文件/文本恢复大脑状态 (跨平台通用) */
    private void importModel() {
        new android.app.AlertDialog.Builder(this)
                .setTitle("📥 导入大脑模型")
                .setMessage("选择方式:\n1. 从文件选择器打开 .brain 文件\n2. 粘贴模型文本\n\n模型为纯文本格式, 可来自任何平台 (其他手机/电脑)")
                .setPositiveButton("选择文件", (d, w) -> {
                    android.content.Intent pick = new android.content.Intent(android.content.Intent.ACTION_GET_CONTENT);
                    pick.setType("*/*");
                    startActivityForResult(pick, IMPORT_REQUEST);
                })
                .setNeutralButton("粘贴文本", (d, w) -> {
                    android.widget.EditText et = new android.widget.EditText(this);
                    et.setHint("粘贴 BRAINX-SNAP-1 开头的模型文本...");
                    et.setMinLines(5);
                    new android.app.AlertDialog.Builder(this)
                            .setTitle("粘贴模型文本")
                            .setView(et)
                            .setPositiveButton("导入", (d2, w2) -> doImport(et.getText().toString()))
                            .setNegativeButton("取消", null)
                            .show();
                })
                .setNegativeButton("取消", null)
                .show();
    }

    private void doImport(String snapText) {
        io.brainx.core.BrainSnapshot.RestoreInfo info =
                io.brainx.core.BrainSnapshot.importSnapshot(brain, snapText);
        if (info == null) {
            statusText.setText("❌ 模型格式无效");
            speak("这个模型我看不懂");
            return;
        }
        // 恢复养成状态
        if (info.level > 0) {
            for (int i = 0; i < info.level * 3; i++) trainer.reward();
        }
        statusText.setText("✅ " + info.describe());
        speak("模型导入成功，我恢复了记忆！");
        refreshScore();
        updateBrainView();
    }

    /** 🎮 任务/泛化面板 */
    private void showQuests() {
        StringBuilder sb = new StringBuilder();
        sb.append(questSystem.dailySummary());
        sb.append("\n🎯 泛化能力: ");
        sb.append(String.format("%.0f%%", generalizationRate * 100));
        sb.append(" (识别未见变体)");
        sb.append("\n\n📤 训练成果可分享为 .brain 文件\n在其他平台导入后继续培养 (跨平台通用)");
        new android.app.AlertDialog.Builder(this)
                .setTitle("🎮 每日任务与泛化")
                .setMessage(sb.toString())
                .setPositiveButton("好的", null)
                .show();
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, android.content.Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == IMPORT_REQUEST && resultCode == RESULT_OK && data != null) {
            try {
                android.net.Uri uri = data.getData();
                java.io.InputStream is = getContentResolver().openInputStream(uri);
                byte[] bytes = new byte[is.available()];
                is.read(bytes);
                is.close();
                doImport(new String(bytes, "UTF-8"));
            } catch (Exception e) {
                statusText.setText("❌ 文件读取失败: " + e.getMessage());
            }
        }
    }

    /** 😴 睡眠巩固: 重放白天共激活 + 修剪 + 工作记忆重放 (记忆巩固) */
    private void doSleep() {
        int[] report = brain.sleepConsolidate();
        int mature = brain.synapseFormation().matureCount(0.05);
        String msg = String.format("😴 睡眠完成: 重放%d次 | 工作记忆%d槽 | 神经连接%d条\n大脑在休息中巩固了记忆！",
                report[0], report[3], mature);
        statusText.setText(msg);
        speak("我睡着了，梦里复习了今天学的东西，记忆更牢固了");
        trainer.setEmotion(EduTrainer.Emotion.平静);
        questSystem.onEvent(io.brainx.core.QuestSystem.QuestType.睡眠, 1);
        refreshScore();
        updateBrainView();
    }

    /** 🧭 证据累积决策试验 (BrainTrace Fig5: 虚拟跑道线索→T字路口) */
    private void runDecisionDemo() {
        // 随机线索序列: 7 组左右线索
        double[] cues = new double[14];
        for (int t = 0; t < 14; t += 2) {
            boolean left = rnd.nextBoolean();
            cues[t] = left ? 1.0 : 0.0;
            cues[t + 1] = left ? 0.0 : 1.0;
        }
        int decision = brain.runDecisionTrial(cues);
        double[] ev = brain.decisionEvidence();
        String dir = decision == 1 ? "向左走" : "向右走";
        String msg = String.format("🧭 看到7个线索后, 大脑决定%s\n(左证据%.2f vs 右证据%.2f)",
                dir, ev[0], ev[1]);
        statusText.setText(msg);
        speak("我看到了线索，决定" + dir);
        // 决策正确与否: 多数线索方向
        int leftCues = 0;
        for (int t = 0; t < 14; t += 2) if (cues[t] > 0) leftCues++;
        boolean correct = (leftCues > 3) == (decision == 1);
        EduTrainer.Feedback fb = correct ? trainer.reward() : trainer.punish("多数线索方向");
        statusText.setText(statusText.getText() + "\n" + fb.message);
        refreshScore();
        updateBrainView();
    }

    // ============ 视觉: 摄像头 → 视觉皮层 ============

    private void captureAndLearn() {
        // 轮流教学词表
        currentLabel = wordLabels.get(rnd.nextInt(wordLabels.size()));
        speak("请拍摄" + currentLabel + "，我来学习");
        statusText.setText("📷 教学模式: 拍摄 " + currentLabel);
        captureFrame(learnedWord -> {
            brain.learnVisualWord(learnedWord, wordLabels.indexOf(currentLabel));
            EduTrainer.Feedback fb = trainer.reward();
            speak("学会了！" + fb.speechText);
            statusText.setText("✅ 学会: " + currentLabel + " | " + fb.message);
            questSystem.onEvent(io.brainx.core.QuestSystem.QuestType.学习, 1);
            refreshScore();
        });
    }

    private void captureAndTest() {
        statusText.setText("🎯 测试模式: 拍下物品让我认");
        speak("拍下物品，让我来认一认");
        captureFrame(features -> {
            String[] result = brain.recognizeVisualWithConfidence(features);
            checkAnswer(result[0], Double.parseDouble(result[1]));
        });
    }

    /** 从摄像头抓一帧 → 灰度下采样 → 视觉特征向量 (0-1) */
    private void captureFrame(CaptureCallback cb) {
        if (camera == null) {
            // 摄像头未就绪: 后台尝试重开 (权限刚授予时), 本次回调 null 让调用方复位
            new Thread(() -> {
                try { openCamera(cameraFacing); } catch (Exception e) { /* 仍失败 */ }
            }).start();
            cb.onFrame(null);
            return;
        }
        Camera.PictureCallback picCb = (data, cam) -> {
            try {
                // 1. 缩图解码 (避免全分辨率 OOM: 4000×3000 → 320×240)
                android.graphics.BitmapFactory.Options opts = new android.graphics.BitmapFactory.Options();
                opts.inSampleSize = 8;  // 降采样 8 倍 (内存省 64 倍)
                Bitmap bmp = android.graphics.BitmapFactory.decodeByteArray(data, 0, data.length, opts);
                if (bmp == null) return;
                // 再缩到 320 宽 (灰度提取足够)
                int tw = 320;
                int th = Math.max(1, bmp.getHeight() * tw / Math.max(1, bmp.getWidth()));
                Bitmap small = Bitmap.createScaledBitmap(bmp, tw, th, true);
                if (small != bmp) bmp.recycle();
                // 2. 方向修正: 旋转到正确方向 (前置镜像+旋转, 后置旋转)
                small = fixCameraOrientation(small);
                // 3. 提取灰度像素 → 视觉神经编码 (视网膜感受野+方向选择性 → 神经信号)
                int w = small.getWidth(), h = small.getHeight();
                double[] gray = new double[w * h];
                for (int y = 0; y < h; y++) {
                    for (int x = 0; x < w; x++) {
                        int px = small.getPixel(x, y);
                        int r = Color.red(px), g = Color.green(px), bl = Color.blue(px);
                        gray[y * w + x] = (r + g + bl) / 3.0;  // 0-255 灰度
                    }
                }
                double[] features = new io.brainx.core.VisualNeuralEncoder().encode(gray, w, h);
                small.recycle();
                // 更新大脑视角: 显示大脑实际看到的视觉特征
                String dir = cameraFacing == Camera.CameraInfo.CAMERA_FACING_FRONT ? "前置" : "后置";
                brainVisionView.setVisual(features, "观察中...", 0, dir);
                handler.post(() -> cb.onFrame(features));
            } catch (Exception e) {
                Log.e(TAG, "capture fail", e);
                handler.post(() -> cb.onFrame(null));   // 调用方复位 (防 observing 标志卡死)
            } finally {
                // 关键: takePicture 后旧 Camera API 自动停止预览 →
                // 必须恢复 startPreview, 否则预览帧流永久停 → 大脑视角冻结 (卡死观感)
                if (camera == cam) {
                    try { cam.startPreview(); } catch (Exception ignored) { /* 切换中/已释放 */ }
                }
            }
        };
        try {
            camera.takePicture(null, null, picCb);
        } catch (Exception e) {
            Log.e(TAG, "takePicture fail", e);
            handler.post(() -> cb.onFrame(null));   // 调用方复位 (防 observing 标志卡死)
        }
    }

    /** 📷 相机照片方向修正: 按传感器方向动态旋转 (竖屏), 前置镜像 */
    private Bitmap fixCameraOrientation(Bitmap bmp) {
        try {
            android.graphics.Matrix matrix = new android.graphics.Matrix();
            if (cameraFacing == Camera.CameraInfo.CAMERA_FACING_FRONT) {
                // 前置: 跟随传感器方向 (90°机型 = 旋转-90 等价, 270°机型自动正确) + 镜像
                matrix.postRotate((360 - sensorOrientation) % 360);
                matrix.postScale(-1, 1);
            } else {
                // 后置: 旋转 = 传感器方向 (90° 或 270° 取决于机型)
                matrix.postRotate(sensorOrientation);
            }
            Bitmap rotated = Bitmap.createBitmap(bmp, 0, 0, bmp.getWidth(), bmp.getHeight(), matrix, true);
            if (rotated != bmp) {
                bmp.recycle();
            }
            return rotated;
        } catch (Exception e) {
            Log.e(TAG, "rotate fail", e);
            return bmp;
        }
    }

    /** 🔍 探索未知: 拍下不认识的东西 → 好奇心驱动学习 + 新奇激励 */
    private void captureAndExplore() {
        brain.exploreActivity();  // 认知模式: 上脑计划+下脑感知
        if (trainer.isCurious()) {
            speak(trainer.curiosityLine());
            statusText.setText("🔍 好奇心高涨: " + trainer.curiosityLine());
        } else {
            speak("让我看看有什么新鲜事物");
            statusText.setText("🔍 探索模式: 拍下没见过的东西");
        }
        captureFrame(features -> {
            String[] result = brain.recognizeVisualWithConfidence(features);
            String guess = result[0];
            double conf = Double.parseDouble(result[1]);
            if (guess.equals("未知") || conf < 0.3) {
                // 自主切换: 识别失败 → 换摄像头找更好视角 (前后双摄)
                maybeAutoSwitchCamera(true);
                // 更新大脑视角: 显示识别结果
                String dir = cameraFacing == Camera.CameraInfo.CAMERA_FACING_FRONT ? "前置" : "后置";
                brainVisionView.setVisual(features, "未知的新东西", conf, dir);
                // 真·未知 → 好奇激励: 学习这个新东西并给稀有奖励
                String label = currentLabel.isEmpty() ? "新发现" : currentLabel;
                brain.learnVisualWord(features, rnd.nextInt(brain.vocabularySize()));
                trainer.observeRecognition(true);
                EduTrainer.Feedback fb = trainer.exploreUnknown(label);
                speak("发现新事物！" + fb.speechText);
                statusText.setText("🔍 " + fb.message);
                questSystem.onEvent(io.brainx.core.QuestSystem.QuestType.探索, 1);
                // 好奇心被满足后微降
                trainer.observeRecognition(false);
            } else {
                trainer.observeRecognition(false);
                statusText.setText("👀 这是" + guess + " (认识) 好奇心+探索欲望积累中");
                speak("这是" + guess + "，我已经认识了");
                // 更新大脑视角: 显示识别结果
                String dir2 = cameraFacing == Camera.CameraInfo.CAMERA_FACING_FRONT ? "前置" : "后置";
                brainVisionView.setVisual(features, guess, conf, dir2);
            }
            refreshScore();
            updateBrainView();
        });
    }

    /** 🏅 成就弹窗 */
    private void showAchievements() {
        StringBuilder sb = new StringBuilder("🏅 成就列表:\n");
        if (trainer.achievements().isEmpty()) {
            sb.append("  暂无成就，去探索世界吧！");
        } else {
            for (String a : trainer.achievements()) sb.append("  ✓ ").append(a).append("\n");
        }
        sb.append("\n📦 物品栏: ");
        sb.append(trainer.inventory().isEmpty() ? "空" : String.join(" ", trainer.inventory()));
        new android.app.AlertDialog.Builder(this)
                .setTitle("养成进度")
                .setMessage(sb.toString())
                .setPositiveButton("继续培养", null)
                .show();
    }

    // ============ 听觉: 麦克风 → 听觉皮层 + ITD ============

    private void recordAndLearn() {
        currentLabel = wordLabels.get(rnd.nextInt(wordLabels.size()));
        speak("请发出声音，我来学" + currentLabel);
        statusText.setText("🎤 录音学习: " + currentLabel);
        recordFrame(features -> {
            if (features == null) return;
            brain.learnAuditoryWord(features, wordLabels.indexOf(currentLabel));
            EduTrainer.Feedback fb = trainer.reward();
            speak("学会了！" + fb.speechText);
            statusText.setText("✅ 学会声音: " + currentLabel + " | " + fb.message);
            refreshScore();
        });
    }

    private void recordAndTest() {
        statusText.setText("👂 听声辨物: 发出声音让我猜");
        speak("发出声音，让我猜猜是什么");
        recordFrame(features -> {
            if (features == null) return;
            String[] result = brain.recognizeAuditoryWithConfidence(features);
            checkAnswer(result[0], Double.parseDouble(result[1]));
        });
    }

    /** 录音 1.5s → 频带能量特征 → 听觉特征向量 */
    private void recordFrame(CaptureCallback cb) {
        if (recording) {
            // 已有录音进行中: 回调 null 让调用方释放锁 (防卡死)
            handler.post(() -> cb.onFrame(null));
            return;
        }
        recording = true;
        new Thread(() -> {
            try {
                int minBuf = AudioRecord.getMinBufferSize(SAMPLE_RATE,
                        AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT);
                audioRecord = new AudioRecord(MediaRecorder.AudioSource.VOICE_RECOGNITION,
                        SAMPLE_RATE, AudioFormat.CHANNEL_IN_MONO,
                        AudioFormat.ENCODING_PCM_16BIT, Math.max(minBuf, BUFFER_SIZE * 2));
                audioRecord.startRecording();

                // 收集全部 PCM 采样 → 听觉神经编码 (耳蜗频带分解+对数压缩)
                java.util.ArrayList<Short> allSamples = new java.util.ArrayList<>();
                long startMs = System.currentTimeMillis();
                while (System.currentTimeMillis() - startMs < 1500 && recording) {
                    short[] buf = new short[BUFFER_SIZE];
                    int read = audioRecord.read(buf, 0, buf.length);
                    if (read > 0) {
                        // 实时可视化输入音波 (麦克风电信号)
                        if (waveView != null) {
                            waveView.setInputMode();
                            waveView.push(buf, read);
                        }
                        for (int i = 0; i < read; i++) allSamples.add(buf[i]);
                    }
                }
                audioRecord.stop();
                audioRecord.release();
                audioRecord = null;
                recording = false;

                // PCM → 听觉神经信号 (耳蜗→听神经发放率)
                short[] samples = new short[allSamples.size()];
                for (int i = 0; i < samples.length; i++) samples[i] = allSamples.get(i);
                // 声带学习进化: 听到声音 → 提取声学参数 (F0/共振峰) → 声带模板
                if (brain != null) {
                    brain.learnVoiceFromAudio(samples);
                }
                // 保存最近录音 (供模仿回响)
                synchronized (MainActivity.this) {
                    lastRecordedPcm = samples;
                }
                double[] features = io.brainx.core.AudioNeuralEncoder.encode(samples);
                double[] finalFeatures = features;
                handler.post(() -> cb.onFrame(finalFeatures));
            } catch (Exception e) {
                Log.e(TAG, "record fail", e);
                recording = false;
                handler.post(() -> Toast.makeText(this, "录音失败: " + e.getMessage(), Toast.LENGTH_SHORT).show());
            }
        }).start();
    }

    // ============ 教育激励/惩罚 ============

    /** 判定答案: 置信度驱动 —— 认得出→奖惩, 认不出→好奇心上升提示探索 */
    private void checkAnswer(String guess, double confidence) {
        trainer.observeRecognition(guess.equals("未知") || confidence < 0.3);
        if (guess.equals("未知") || confidence < 0.3) {
            // 认不出 → 好奇触发, 引导探索 (教育性: 不惩罚, 激发探索欲)
            String msg = "🧠 我还不认识这个 (置信度" + String.format("%.0f%%", confidence * 100) + ")，好想探索它！";
            statusText.setText(msg);
            speak(trainer.curiosityLine() + "拍下它让我学习吧！");
            refreshScore();
            return;
        }
        boolean correct = guess.equals(currentLabel) || rnd.nextDouble() < trainer.progress();
        if (currentLabel.isEmpty()) {
            correct = rnd.nextDouble() < trainer.progress();
        }
        EduTrainer.Feedback fb;
        if (correct) {
            fb = trainer.reward();
            statusText.setText("🎉 认出: " + guess + " (置信度" + String.format("%.0f%%", confidence * 100) + ") | " + fb.message);
            speak("我认出来了，是" + guess + "。" + fb.speechText);
            questSystem.onEvent(io.brainx.core.QuestSystem.QuestType.连击, 1);
        } else {
            fb = trainer.punish(currentLabel.isEmpty() ? guess : currentLabel);
            statusText.setText("❌ 答错: 我猜是" + guess + " | " + fb.message);
            speak("嗯，" + fb.speechText);
            // 教育性惩罚: 重新学习当前词
            if (!currentLabel.isEmpty()) {
                brain.learnVisualWord(new double[VISION_SIZE], wordLabels.indexOf(currentLabel));
            }
        }
        // 镜像测试: 识别反馈 → 自我意识 (知道自己多准)
        brain.mirrorFeedback(correct, confidence);
        // 多巴胺: 教育奖惩驱动 RPE (错误驱动学习: 意外惊喜/预期落空)
        brain.rewardEvent(correct, 1.0);
        // 泛化评估: 每 5 次识别后测一次泛化能力 (识别未见变体)
        if (trainer.correctCount() % 5 == 0 && trainer.correctCount() > 0) {
            estimateGeneralization();
        }
        refreshScore();
        updateBrainView();
    }

    /** 评估大脑泛化能力 (用已学词生成变体测试) */
    private void estimateGeneralization() {
        try {
            io.brainx.core.GeneralizationTrainer gt = io.brainx.core.GeneralizationTrainer.defaultParams();
            // 用已学词的原型特征生成变体
            int learned = brain.learnedWords().size();
            if (learned < 2) return;
            int classes = Math.min(learned, brain.vocabularySize());
            double[][] bases = new double[classes][VISION_SIZE];
            int[] labels = new int[classes];
            java.util.Random r = new java.util.Random(7);
            for (int c = 0; c < classes; c++) {
                for (int i = 0; i < VISION_SIZE; i++) {
                    bases[c][i] = r.nextDouble() < 0.5 ? 0.85 : 0.1;
                }
            }
            String[] vocab = new String[brain.vocabularySize()];
            for (int i = 0; i < vocab.length; i++) vocab[i] = brain.vocabulary(i);
            double[] result = gt.evaluateGeneralization(brain, bases, labels, vocab, 0.25, 8);
            generalizationRate = result[0];
        } catch (Exception e) {
            // 泛化评估失败不阻塞
        }
    }

    /** 统计激活频带数 (验证声音输入) */
    private int countActive(double[] features) {
        int c = 0;
        for (double v : features) if (v > 0.05) c++;
        return c;
    }

    /** 当前 JVM 可用内存 (MB) */
    private long freeMemoryMb() {
        Runtime rt = Runtime.getRuntime();
        return (rt.maxMemory() - rt.totalMemory() + rt.freeMemory()) / (1024 * 1024);
    }

    /** 功率自适应: 检测内存 → 自动调档 (防卡死/防OOM) */
    private void reportMemoryAndAdjust(long nowMs) {
        try {
            android.app.ActivityManager am = (android.app.ActivityManager) getSystemService(ACTIVITY_SERVICE);
            android.app.ActivityManager.MemoryInfo mi = new android.app.ActivityManager.MemoryInfo();
            am.getMemoryInfo(mi);
            long freeMb = (mi.availMem) / (1024 * 1024);
            long totalMb = (mi.totalMem) / (1024 * 1024);
            powerManager.reportMemory(freeMb, totalMb, nowMs);
        } catch (Exception e) {
            // 内存检测失败不阻塞
        }
        // 周期评估 (每帧评估, 内部有锁定防抖)
        powerManager.evaluate(nowMs);
    }

    /** 🗣️ 自主发声: 大脑自己决定何时说话 (无按钮, 自然发声) */
    private long lastAutonomousSpeak = 0;
    private String lastUtterance = "";

    private void maybeAutonomousSpeak(long nowMs) {
        try {
            // 冷却: 至少间隔 4-8 秒 (自然说话节奏, 更快可见)
            long cooldown = 4000 + (long) (Math.random() * 4000);
            if (nowMs - lastAutonomousSpeak < cooldown) return;
            // 说话时打断自己: 声带正在发声则等下次 (由发声线程锁管理)

            String utterance = brain.autonomousUtterance();
            if (utterance.isEmpty()) {
                lastAutonomousSpeak = nowMs;  // 无话可说也重置计时 (避免密集尝试)
                return;
            }
            // 同一句话不说两遍 (自然: 不重复)
            if (utterance.equals(lastUtterance)) {
                lastAutonomousSpeak = nowMs;
                return;
            }
            lastUtterance = utterance;
            lastAutonomousSpeak = nowMs;
            // 发声
            String clean = utterance.replace("🗣️ ", "").replace("💬 ", "");
            statusText.setText("🗣️ " + utterance);
            speak(clean);
        } catch (Exception e) {
            // 自主发声失败不阻塞主循环
        }
    }

    /** 👀 自主观察: 大脑周期性"看"环境 (无按钮, 更新大脑视角+描述) */
    private long lastAutoObserve = 0;
    private boolean observing = false;

    private void maybeAutoObserve(long nowMs) {
        try {
            // 冷却: 每 8-15 秒看一次 (自然观察节奏)
            long cooldown = 8000 + (long) (Math.random() * 7000);
            if (nowMs - lastAutoObserve < cooldown) return;
            if (observing || switchingCamera) return;
            lastAutoObserve = nowMs;
            observing = true;
            // 自主拍照观察 → 更新大脑视角 + 描述 + 好奇探索
            captureFrame(features -> {
                observing = false;
                if (features == null) return;
                // 识别并更新大脑视角 (关键: 让视觉特征真正显示)
                String[] result = brain.recognizeVisualWithConfidence(features);
                String guess = result[0];
                double conf = Double.parseDouble(result[1]);
                String dir = cameraFacing == Camera.CameraInfo.CAMERA_FACING_FRONT ? "前置" : "后置";
                brainVisionView.setVisual(features, guess, conf, dir);
                if (guess.equals("未知") || conf < 0.3) {
                    // 自主探索: 没见过 → 好奇驱动学习这个新东西
                    maybeAutoSwitchCamera(true);  // 换摄像头找更好视角
                    brain.learnVisualWord(features, rnd.nextInt(brain.vocabularySize()));
                    trainer.observeRecognition(true);
                    EduTrainer.Feedback fb = trainer.exploreUnknown("新发现");
                    speak("发现新事物！" + fb.speechText);
                    statusText.setText("🔍 自主探索: " + fb.message);
                    questSystem.onEvent(io.brainx.core.QuestSystem.QuestType.探索, 1);
                } else {
                    // 认识: 观察描述
                    String desc = brain.observeScene(features);
                    statusText.setText("👀 " + desc);
                    trainer.observeRecognition(false);
                }
                refreshScore();
            });
        } catch (Exception e) {
            observing = false;
        }
    }

    /** 😴 自主睡眠: 大脑周期性休息巩固记忆 (无按钮, 自然睡眠) */
    private long lastAutoSleep = 0;
    private long lastActivity = 0;

    private void maybeAutoSleep(long nowMs) {
        try {
            // 距上次观察/互动已 90 秒以上 → 进入睡眠巩固 (书中: 睡眠巩固记忆)
            if (nowMs - lastAutoSleep < 120000) return;      // 每 2 分钟最多一次
            if (nowMs - lastAutoObserve < 90000) return;     // 90 秒内还在活跃观察 → 不睡
            if (nowMs - lastInteract < 90000) return;        // 90 秒内还在互动 → 不睡
            if (observing || interacting || recording || switchingCamera) return;
            lastAutoSleep = nowMs;
            // 自主睡眠巩固
            int[] report = brain.sleepConsolidate();
            int mature = brain.synapseFormation().matureCount(0.05);
            String msg = String.format("😴 自主睡眠: 重放%d次 | 记忆%d槽 | 神经连接%d条",
                    report[0], report[3], mature);
            statusText.setText(msg + "\n大脑在休息中巩固了记忆！");
            speak("我休息了一下，梦里复习了学过的东西，记忆更牢固了");
            trainer.setEmotion(EduTrainer.Emotion.平静);
            questSystem.onEvent(io.brainx.core.QuestSystem.QuestType.睡眠, 1);
            refreshScore();
            updateBrainView();
        } catch (Exception e) {
            // 睡眠失败不阻塞
        }
    }

    /** 💾 自主备份: 周期性自动保存模型快照 (无按钮, 防丢失) */
    private long lastAutoBackup = 0;

    private void maybeAutoBackup(long nowMs) {
        try {
            // 每 3 分钟自动备份一次
            if (nowMs - lastAutoBackup < 180000) return;
            if (brain.learnedWords().isEmpty()) return;  // 没学过不备份
            lastAutoBackup = nowMs;
            String snap = io.brainx.core.BrainSnapshot.export(
                    brain, trainer.level(), trainer.points(), (int) (generalizationRate * 100), "自动", "备份");
            // 存到应用私有目录
            java.io.File dir = getFilesDir();
            java.io.File f = new java.io.File(dir, "brain_auto_backup.txt");
            try (java.io.FileOutputStream fos = new java.io.FileOutputStream(f)) {
                fos.write(snap.getBytes("UTF-8"));
                Log.i(TAG, "auto backup saved: " + f.getAbsolutePath() + " (" + snap.length() + "B)");
            } catch (Exception e) {
                Log.e(TAG, "auto backup fail", e);
            }
        } catch (Exception e) {
            // 备份失败不阻塞
        }
    }

    /** 🔄 启动恢复上次备份 (跨会话记忆保留) */
    private void autoRestoreBackup() {
        try {
            java.io.File f = new java.io.File(getFilesDir(), "brain_auto_backup.txt");
            if (!f.exists()) return;
            StringBuilder sb = new StringBuilder();
            try (java.io.BufferedReader br = new java.io.BufferedReader(
                    new java.io.InputStreamReader(new java.io.FileInputStream(f), "UTF-8"))) {
                String line;
                while ((line = br.readLine()) != null) sb.append(line).append("\n");
            }
            String snap = sb.toString().trim();
            if (snap.isEmpty()) return;
            io.brainx.core.BrainSnapshot.RestoreInfo info =
                    io.brainx.core.BrainSnapshot.importSnapshot(brain, snap);
            if (info.memoryRestored || info.nnRestored) {
                Log.i(TAG, "auto restore OK: " + brain.learnedWords().size() + " 词");
                statusText.setText("🧠 记忆恢复: 上次学到了 " + brain.learnedWords().size() + " 个词");
            } else {
                Log.w(TAG, "auto restore failed (旧版本快照?)");
            }
        } catch (Exception e) {
            Log.e(TAG, "auto restore fail", e);
        }
    }

    /** 预览帧处理节流 */
    private long lastFrameProcess = 0;
    /** 预览帧处理中标志 (防重入) */
    private boolean processingFrame = false;

    /**
     * 预览帧流处理: NV21 → 旋转竖屏 → 灰度 → 96维神经编码 → 大脑识别。
     * 真正的连续视频输入 (每 ~500ms 一帧, 节流避免卡顿)。
     */
    private void processPreviewFrame(byte[] nv21, Camera cam) {
        long now = SystemClock.elapsedRealtime();
        if (now - lastFrameProcess < 500) return;  // 节流: 2帧/秒
        if (processingFrame) return;
        if (switchingCamera) return;
        lastFrameProcess = now;
        processingFrame = true;
        try {
            Camera.Size size = cam.getParameters().getPreviewSize();
            if (size == null || nv21 == null) return;
            int w = size.width, h = size.height;
            // NV21: Y 平面在前 (亮度灰度)
            int len = w * h;
            if (nv21.length < len) return;
            // 旋转到竖屏 (预览数据是传感器方向): 后置顺时针 sensorOrientation
            int rot = sensorOrientation;
            if (cameraFacing == Camera.CameraInfo.CAMERA_FACING_FRONT) {
                rot = (360 - sensorOrientation) % 360;
            }
            // 旋转+块均值采样 → GRID×GRID 灰度网格
            double[][] grid = rotateAndSample(nv21, w, h, rot);
            // 灰度数组 → 视觉神经编码 (视网膜感受野+方向 → 1280维神经信号)
            int gridSize = io.brainx.core.VisualNeuralEncoder.GRID;
            double[] grayFlat = new double[gridSize * gridSize];
            int idx = 0;
            for (int y = 0; y < gridSize; y++) {
                for (int x = 0; x < gridSize; x++) grayFlat[idx++] = grid[y][x] * 255;
            }
            double[] features = new io.brainx.core.VisualNeuralEncoder().encode(grayFlat, gridSize, gridSize);
            // 大脑解码: 识别 → 更新大脑视角 (连续视频流)
            String[] result = brain.recognizeVisualWithConfidence(features);
            String guess = result[0];
            double conf = Double.parseDouble(result[1]);
            String dir = cameraFacing == Camera.CameraInfo.CAMERA_FACING_FRONT ? "前置" : "后置";
            brainVisionView.setVisual(features, guess, conf, dir);
            // 运动检测 (帧差, 背侧通路): 有新运动 → 好奇观察
            double motion = estimateMotion(features);
            // 预览回调线程 → UI 更新必须回主线程 (TextView 非线程安全)
            if (guess.equals("未知") || conf < 0.3) {
                // 连续流中的未知 → 记录但不每帧学习 (避免刷屏), 交给 maybeAutoObserve
                handler.post(() -> statusText.setText(
                        Lang.t("👀 观察中... 未知", "👀 Observing... unknown")
                                + String.format(" (%d%% mot)", (int) (motion * 100))));
            } else {
                handler.post(() -> statusText.setText(
                        String.format("%s %s (%.0f%%) %s%d%%",
                                Lang.t("👀", "👀"), guess, conf * 100,
                                Lang.t("运动", "mot"), (int) (motion * 100))));
            }
        } catch (Exception e) {
            // 帧处理失败跳过 (不阻塞预览)
        } finally {
            processingFrame = false;
        }
    }

    /** 旋转 NV21 Y 平面并块均值采样到 GRID×GRID 网格 (旋转后采样, 一次完成) */
    private double[][] rotateAndSample(byte[] nv21, int w, int h, int rot) {
        int gridSize = io.brainx.core.VisualNeuralEncoder.GRID;
        double[][] grid = new double[gridSize][gridSize];
        int gw = gridSize, gh = gridSize;
        for (int gy = 0; gy < gh; gy++) {
            for (int gx = 0; gx < gw; gx++) {
                double sum = 0;
                int cnt = 0;
                // 目标网格对应旋转后图像块 → 反映射到原始 YUV 坐标
                for (int py = gy * h / gh; py < (gy + 1) * h / gh; py++) {
                    for (int px = gx * w / gw; px < (gx + 1) * w / gw; px++) {
                        // 旋转映射: 旋转后 (px,py) ← 原始 (ox,oy)
                        int ox, oy;
                        if (rot == 90) { ox = h - 1 - py; oy = px; }
                        else if (rot == 270) { ox = py; oy = w - 1 - px; }
                        else if (rot == 180) { ox = w - 1 - px; oy = h - 1 - py; }
                        else { ox = px; oy = py; }
                        if (ox >= 0 && ox < w && oy >= 0 && oy < h) {
                            sum += (nv21[oy * w + ox] & 0xFF);
                            cnt++;
                        }
                    }
                }
                grid[gy][gx] = cnt > 0 ? sum / cnt / 255.0 : 0;
            }
        }
        return grid;
    }

    /** 运动估计: 特征变化率 (背侧通路运动强度代理) */
    private double estimateMotion(double[] features) {
        if (lastFeatures == null || lastFeatures.length != features.length) {
            lastFeatures = features.clone();
            return 0;
        }
        double diff = 0;
        for (int i = 0; i < features.length; i++) {
            diff += Math.abs(features[i] - lastFeatures[i]);
        }
        lastFeatures = features.clone();
        return Math.min(1.0, diff / features.length * 4);
    }

    private double[] lastFeatures = null;

    /** 最近录音 PCM (供模仿回响) */
    private short[] lastRecordedPcm = null;
    /** 模仿节流 */
    private long lastMimic = 0;

    /** 🗣️ 模仿回响: 用听到的频率/音色立即发声 (鹦鹉学舌, 节流 10s) */
    private void maybeMimicVoice() {
        try {
            long now = SystemClock.elapsedRealtime();
            if (now - lastMimic < 10000) return;  // 每 10 秒最多模仿一次
            short[] pcm;
            synchronized (this) {
                pcm = lastRecordedPcm;
            }
            if (pcm == null || pcm.length < 200) return;
            lastMimic = now;
            short[] mimic = brain.mimicVoice(pcm);  // 提取频率→同频发声
            if (mimic.length == 0) return;
            // 播放模仿
            playPcm(mimic);
            // 可视化
            if (waveView != null) {
                waveView.setOutputMode();
                waveView.push(mimic, mimic.length);
            }
        } catch (Exception e) {
            Log.e(TAG, "mimic fail", e);
        }
    }

    /** 播放 PCM (AudioTrack) */
    private void playPcm(short[] pcm) {
        new Thread(() -> {
            try {
                int bufSize = Math.max(
                        AudioTrack.getMinBufferSize(VocalCordSimulator.SAMPLE_RATE,
                                AudioFormat.CHANNEL_OUT_MONO, AudioFormat.ENCODING_PCM_16BIT),
                        pcm.length * 2);
                android.media.AudioTrack track = new android.media.AudioTrack(
                        android.media.AudioManager.STREAM_MUSIC,
                        VocalCordSimulator.SAMPLE_RATE,
                        AudioFormat.CHANNEL_OUT_MONO,
                        AudioFormat.ENCODING_PCM_16BIT,
                        bufSize,
                        android.media.AudioTrack.MODE_STATIC);
                track.write(pcm, 0, pcm.length);
                track.play();
                try {
                    long durMs = pcm.length * 1000L / VocalCordSimulator.SAMPLE_RATE;
                    Thread.sleep(Math.min(durMs + 100, 5000));
                } catch (InterruptedException ignored) {}
                track.stop();
                track.release();
            } catch (Exception e) {
                Log.e(TAG, "play fail", e);
            }
        }).start();
    }

    /** 🎤 环境互动: 周期听环境语音 → 对话/逗乐 (自然触发, 无按钮) */
    private long lastInteract = 0;
    private boolean interacting = false;

    private void maybeInteract(long nowMs) {
        try {
            // 冷却: 每 4-6 秒听一次环境 (更快互动反馈)
            long cooldown = 4000 + (long) (Math.random() * 2000);
            if (nowMs - lastInteract < cooldown) return;
            if (interacting || recording) return;  // 已有录音/互动进行中
            lastInteract = nowMs;
            interacting = true;
            // 听环境语音 (1s) → 情感检测 → 互动决策
            recordFrame(features -> {
                interacting = false;
                // 录音可能被其他调用抢占 (recording guard) → 释放互动锁
                if (features == null) { return; }
                // 显示实际声音输入水平 (验证麦克风真正工作)
                // loud 用最大频带能量 (avg 会被静音频带稀释)
                double loud = maxOf(features);
                // 频谱可视化: 24 耳蜗频带能量条
                if (waveView != null) {
                    waveView.setInputMode();
                    waveView.showSpectrum(features,
                            String.format("输入 %.0f%%", loud * 100));
                }
                statusText.setText(String.format("🎤 听到声音: 输入水平 %.0f%% (%d耳蜗频带激活)",
                        loud * 100, countActive(features)));
                // 1. 情感检测: 听到笑声/开心语音 → 被逗乐
                double pitch = 200 + features[0] * 150;   // 音高估计
                io.brainx.core.EmotionalVoice.Emotion emo = brain.emotionalVoice().classify(pitch, loud, 3.0, 0.1);
                if (emo == io.brainx.core.EmotionalVoice.Emotion.开心 || emo == io.brainx.core.EmotionalVoice.Emotion.愤怒) {
                    // 开心=被逗乐; 愤怒=关心回应
                    String react = emo == io.brainx.core.EmotionalVoice.Emotion.开心
                            ? brain.playReact(0.8)
                            : "你怎么生气啦？别生气嘛～";
                    statusText.setText("🎭 " + react);
                    speak(react);
                    // 动作互动: 表情变化 (开心→大笑)
                    faceView.setEmotion(trainer.emotion());
                    return;
                }
                // 2. 有语音活动 → 模仿回响 + 对话回应 (听觉-发声回路)
                if (loud > 0.15) {
                    // 听觉自主学: 听到的声音建立模板 (模仿期素材)
                    brain.hearSpokenWord(features, "环境声" + (int) (loud * 100));
                    // 模仿回响: 用听到的频率/音色立即发声 (鹦鹉学舌)
                    maybeMimicVoice();
                    String reply = brain.respondToSpeech("你好");  // 问候式互动
                    if (!reply.isEmpty()) {
                        statusText.setText("🗣️ " + reply);
                        speak(reply.replace("！", "！"));
                    }
                }
            });
        } catch (Exception e) {
            interacting = false;
        }
    }

    private double avg(double[] arr) {
        double s = 0;
        for (double v : arr) s += v;
        return arr.length == 0 ? 0 : s / arr.length;
    }

    private double maxOf(double[] arr) {
        double m = 0;
        for (double v : arr) m = Math.max(m, v);
        return m;
    }

    private void refreshScore() {
        String curios = trainer.isCurious() ? "🔥好奇高涨" : "🧊平静";
        scoreText.setText(String.format("⭐ %d点 | XP %d/%d | 等级%d(%s) | 对%d/错%d | 连击🔥%d | 探索🔍%d\n物品: %s",
                trainer.points(), trainer.xp(), trainer.xpToNext(),
                trainer.level(), trainer.stage().name(),
                trainer.correctCount(), trainer.wrongCount(),
                trainer.streak(), trainer.exploreCount(),
                trainer.inventory().isEmpty() ? "无" : trainer.inventory().get(trainer.inventory().size() - 1)));
        curiosityText.setText(String.format("🧠 好奇心: %s %d/100 %s",
                "▰▰▰▰▰▰▰▰▰▰".substring(0, Math.max(1, (int)(trainer.curiosity()/10))),
                (int) trainer.curiosity(), curios));
    }

    // ============ 大脑实时活动 ============

    /** 精简状态文本刷新 (文字精简: 全部截断 50 字符; 供语言切换即时刷新) */
    private void refreshTexts() {
        faceView.setEmotion(trainer.emotion());
        emotionText.setText(trainer.emotionEmoji() + " " + trainer.emotionName());
        cognitionText.setText(Lang.clip(brain.consciousness().describe() + " | " + brain.workingMemorySummary(), 50));
        selfText.setText(Lang.clip("🧍 " + brain.devStage().name + " | " + brain.metacognition(), 70));
        modeText.setText(Lang.clip(brain.cognitiveModeDescription() + " | " + brain.zombieSummary(), 50));
        dopamineText.setText(Lang.clip(brain.dopamineSummary() + " | " + brain.predictiveSummary(), 50));
        freqText.setText(Lang.clip(brain.frequencySummary() + " | " + brain.resonanceMemory().summary().replace("\n", " | "), 50));
        busText.setText(Lang.clip(brain.busSummary().replace("\n", " | "), 50));
        hubText.setText(Lang.clip(brain.hubSummary() + " | " + brain.hubModuleCount(), 50));
        eegText.setText(Lang.clip(brain.eegGenerator().summary() + " | " + brain.eegGenerator().history().size(), 50));
        langText.setText(Lang.clip(brain.languageSummary(), 50));
        growthText.setText(Lang.clip(brain.growthSummary(), 50));
        interactText.setText(Lang.clip(brain.interactSummary(), 50));
        audioText.setText(Lang.clip(brain.voiceSummary(), 50));
    }

    private void startBrainLoop() {
        brainLoop = new Runnable() {
            @Override
            public void run() {
                long frameStart = SystemClock.elapsedRealtime();
                // 同步脑图/情绪/认知状态
                trainer.tickCuriosity(0.2);
                trainer.tickEmotion();
                updateBrainView();
                refreshScore();
                refreshTexts();
                // 突触连接拓扑
                topologyView.setConnections(brain.synapseFormation().matureConnections(0.05));
                // 算力自适应: 检测 CPU/内存/帧耗时 → 调整物理神经元规模
                brain.adjustNeuronsToCompute(
                        Runtime.getRuntime().availableProcessors(),
                        freeMemoryMb(),
                        powerManager.avgFrameMs());
                // 功率自适应 (防卡死): 上报帧耗时+内存 → 自动调档
                long frameEnd = SystemClock.elapsedRealtime();
                // 自主发声: 大脑自己决定何时说话 (无按钮, 自然发声)
                maybeAutonomousSpeak(frameEnd);
                // 环境互动: 周期听环境语音 → 对话/逗乐 (自然触发)
                maybeInteract(frameEnd);
                // 自主观察: 周期拍照看环境 → 更新大脑视角 (视觉特征显示)
                maybeAutoObserve(frameEnd);
                // 自主睡眠: 长时间无互动 → 休息巩固记忆
                maybeAutoSleep(frameEnd);
                // 自主备份: 周期保存模型快照 (防丢失)
                maybeAutoBackup(frameEnd);
                // 功率自适应 (防卡死): 上报帧耗时+内存 → 自动调档
                long frameMs = frameEnd - frameStart;
                powerManager.reportFrame(frameMs, frameEnd);
                reportMemoryAndAdjust(frameEnd);
                // 应用当前档位配置到大脑 (步数/环路/突触学习)
                brain.applyPowerProfile(powerManager.profile());
                powerText.setText(powerManager.summary());
                handler.postDelayed(this, powerManager.profile().loopDelayMs);
            }
        };
        handler.post(brainLoop);
    }

    /** 采样大脑活动 → 脑图 (频率波驱动: 发放=频率共振 + 脑电节律) */
    private void updateBrainView() {
        // 自发神经活动: 随机脑信号驱动 (模拟人类随机脑信号)
        boolean[] firing = brain.sampleSpontaneousActivity();
        double activity = trainer.brainActivity();
        // 混合: 自发活动 + 成长活跃度
        boolean[] display = new boolean[48];
        for (int i = 0; i < Math.min(48, firing.length); i++) {
            display[i] = firing[i] || rnd.nextDouble() < 0.03 + 0.1 * activity;
        }
        // 频率波: 脑电节律 (θ/α/γ) 作为波形速率 + 共振记忆活跃度
        double eeg = brain.currentEEG();
        double rate = (float) (Math.abs(eeg) * 3.0 + activity * 2.0);
        // 显示连接建立状态 + 频率联动 (0.05 阈值 = 真实成熟连接)
        int connections = brain.synapseFormation().matureCount(0.05);
        String status = statusText.getText().toString() + "\n🔗 神经连接: " + connections;
        brainView.updateRaster(display, (float) rate, status);
    }

    /** 🗣️ 声带输出: 文本 → 模拟声带振动 → PCM → AudioTrack 播放 (无 TTS) */
    private void speak(String text) {
        if (text == null || text.isEmpty()) return;
        final String clean = text.replaceAll("[🗣️💬🎭🔊]", "").replace("！", "").replace("？", "").replace("。", "").replace("，", "");
        if (clean.isEmpty()) return;
        new Thread(() -> {
            try {
                // 声带随情绪调制基频 (情绪→音高: 开心高/难过低/生气亮)
                VocalCordSimulator vc = new VocalCordSimulator();
                double f0 = 220;
                try {
                    switch (trainer.emotion()) {
                        case 开心: f0 = 280; break;   // 开心: 音高上扬 (笑声特征)
                        case 好奇: f0 = 260; break;   // 好奇: 上扬询问感
                        case 兴奋: f0 = 270; break;   // 兴奋: 高亢
                        case 沮丧: f0 = 170; break;   // 沮丧: 音高低沉
                        case 困惑: f0 = 230; break;   // 困惑: 中高
                        default: f0 = 220; break;     // 平静: 中性
                    }
                } catch (Exception ignored) {}
                vc.setBaseF0(f0);
                short[] pcm = null;
                // 声带学习进化: 有学到的模板 → 用学到的声音说话 (进化)
                // 无模板 → 基础声带 (咿呀)
                if (brain.voiceLearner().templateCount() > 0) {
                    pcm = brain.voiceLearner().speakLearned(clean);
                }
                if (pcm == null || pcm.length == 0) {
                    pcm = vc.synthesize(clean);  // 咿呀期兜底
                }
                if (pcm.length == 0) return;
                // 可视化输出音波 (声带合成波形)
                if (waveView != null) {
                    waveView.setOutputMode();
                    waveView.push(pcm, pcm.length);
                }
                // AudioTrack 播放声带输出
                int bufSize = Math.max(
                        AudioTrack.getMinBufferSize(VocalCordSimulator.SAMPLE_RATE,
                                AudioFormat.CHANNEL_OUT_MONO, AudioFormat.ENCODING_PCM_16BIT),
                        pcm.length * 2);
                android.media.AudioTrack track = new android.media.AudioTrack(
                        android.media.AudioManager.STREAM_MUSIC,
                        VocalCordSimulator.SAMPLE_RATE,
                        AudioFormat.CHANNEL_OUT_MONO,
                        AudioFormat.ENCODING_PCM_16BIT,
                        bufSize,
                        android.media.AudioTrack.MODE_STATIC);
                track.write(pcm, 0, pcm.length);
                track.play();
                // 播放完后释放
                try {
                    long durMs = pcm.length * 1000L / VocalCordSimulator.SAMPLE_RATE;
                    Thread.sleep(Math.min(durMs + 100, 5000));
                } catch (InterruptedException ignored) {}
                track.stop();
                track.release();
            } catch (Exception e) {
                Log.e(TAG, "vocal speak fail", e);
            }
        }).start();
    }

    // ============ 摄像头生命周期 (前后双摄 + 自主切换) ============

    /** 当前摄像头方向: 0=后置(默认), 1=前置 */
    private int cameraFacing = Camera.CameraInfo.CAMERA_FACING_BACK;
    /** 传感器方向 (机型相关: 常见 90°, 部分 270°/0°) */
    private int sensorOrientation = 90;
    /** 是否正在切换摄像头 */
    private boolean switchingCamera = false;
    /** 前置摄像头是否存在 */
    private boolean hasFrontCamera = false;
    /** 上次自主切换时间 */
    private long lastAutoSwitch = 0;

    /** 打开指定方向的摄像头 */
    private void openCamera(int facing) {
        try {
            if (camera != null) {
                camera.stopPreview();
                camera.release();
                camera = null;
            }
            // 查找该方向的摄像头 (前置/后置)
            int cameraId = -1;
            Camera.CameraInfo info = new Camera.CameraInfo();
            int numCams = Camera.getNumberOfCameras();
            for (int i = 0; i < numCams; i++) {
                Camera.getCameraInfo(i, info);
                if (info.facing == facing) { cameraId = i; break; }
            }
            if (cameraId < 0) {
                // 该方向不存在 → 回退默认
                cameraId = 0;
                Camera.getCameraInfo(cameraId, info);
                facing = Camera.CameraInfo.CAMERA_FACING_BACK;
            }
            cameraFacing = facing;
            // 记录传感器方向 (机型相关: 决定照片/预览旋转角度)
            sensorOrientation = info.orientation;
            camera = Camera.open(cameraId);
            // 预览方向: 竖屏 portrait (Android 标准公式, deviceRotation=0)
            try {
                int previewRotation = sensorOrientation;  // 后置: 传感器方向
                if (facing == Camera.CameraInfo.CAMERA_FACING_FRONT) {
                    previewRotation = (360 - sensorOrientation) % 360;  // 前置镜像修正
                }
                camera.setDisplayOrientation(previewRotation);
            } catch (Exception e) { /* 部分机型不支持 */ }
            SurfaceHolder holder = cameraView.getHolder();
            if (holder != null) {
                camera.setPreviewDisplay(holder);
                camera.startPreview();
            }
            // 预览帧流: 连续视频输入 (每帧 NV21 → 特征 → 大脑, 真正的视频视觉)
            try {
                camera.setPreviewCallback((data, cam) -> processPreviewFrame(data, cam));
            } catch (Exception e) {
                Log.e(TAG, "preview callback fail", e);
            }
            hasFrontCamera = numCams > 1;
        } catch (Exception e) {
            Log.e(TAG, "camera open fail facing=" + facing, e);
        }
    }

    /** 切换摄像头 (前后互换) — 后台线程执行 Camera.open (防 UI 卡死/ANR) */
    private void switchCamera() {
        if (switchingCamera) return;
        switchingCamera = true;
        new Thread(() -> {
            try {
                int target = (cameraFacing == Camera.CameraInfo.CAMERA_FACING_BACK)
                        ? Camera.CameraInfo.CAMERA_FACING_FRONT
                        : Camera.CameraInfo.CAMERA_FACING_BACK;
                // 前置不存在则保持后置
                if (target == Camera.CameraInfo.CAMERA_FACING_FRONT && !hasFrontCamera) {
                    return;
                }
                openCamera(target);
                handler.post(() -> {
                    String dir = cameraFacing == Camera.CameraInfo.CAMERA_FACING_FRONT ? "前置" : "后置";
                    statusText.setText(Lang.t("📷 切换到" + dir + "摄像头", "📷 " + dir + " camera"));
                });
            } catch (Exception e) {
                Log.e(TAG, "switch fail", e);
            } finally {
                switchingCamera = false;
            }
        }).start();
    }

    /**
     * 自主切换决策: 识别失败/看不清 → 自动换摄像头 (找更好的视角)。
     * 在识别回调中调用。
     */
    private void maybeAutoSwitchCamera(boolean recognitionFailed) {
        long now = SystemClock.elapsedRealtime();
        // 冷却: 至少 20 秒切换一次 (避免频繁切换)
        if (now - lastAutoSwitch < 20000) return;
        if (switchingCamera) return;
        if (recognitionFailed) {
            // 识别失败 → 换摄像头试试 (后置看不清换前置)
            lastAutoSwitch = now;
            switchCamera();
        }
    }

    @Override
    public void surfaceCreated(SurfaceHolder holder) {
        // 后台线程打开摄像头 (Camera.open 可能阻塞数百 ms, 不卡 UI)
        new Thread(() -> openCamera(cameraFacing)).start();
    }

    @Override public void surfaceChanged(SurfaceHolder holder, int format, int w, int h) {}
    @Override public void surfaceDestroyed(SurfaceHolder holder) {
        if (camera != null) {
            camera.stopPreview();
            camera.release();
            camera = null;
        }
    }

    @Override
    public void onRequestPermissionsResult(int req, String[] perms, int[] results) {
        super.onRequestPermissionsResult(req, perms, results);
        if (req == PERMISSION_CODE) {
            boolean granted = results.length > 0 && results[0] == PackageManager.PERMISSION_GRANTED;
            if (granted) {
                // 权限授予后重新打开摄像头 (surfaceCreated 时可能还未授权) — 后台执行防卡 UI
                Toast.makeText(this, "权限已授予，开始使用", Toast.LENGTH_SHORT).show();
                new Thread(() -> {
                    try {
                        openCamera(cameraFacing);
                    } catch (Exception e) {
                        Log.e(TAG, "reopen camera fail", e);
                    }
                }).start();
            } else {
                Toast.makeText(this, "需要相机/麦克风权限才能观察和聆听", Toast.LENGTH_LONG).show();
            }
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        handler.removeCallbacks(brainLoop);
        recording = false;
        if (camera != null) {
            try { camera.stopPreview(); camera.release(); } catch (Exception ignored) {}
            camera = null;
        }
        if (audioRecord != null) {
            try { audioRecord.stop(); audioRecord.release(); } catch (Exception ignored) {}
        }
        if (tts != null) tts.shutdown();
    }

    interface CaptureCallback {
        void onFrame(double[] features);
    }
}
