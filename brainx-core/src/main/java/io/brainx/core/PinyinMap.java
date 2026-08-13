package io.brainx.core;

/**
 * 汉字→拼音映射表 (常用字, 人声合成用)。
 * 格式: 汉字 → [声母, 韵母, 声调(1-4, 0=轻声)]
 * 声母 "" = 零声母 (直接韵母)
 */
public class PinyinMap {
    /** 返回 [声母, 韵母, 声调]; 未知字返回 null */
    public static String[] lookup(char c) {
        String s = MAP.get(c);
        return s == null ? null : s.split(",");
    }

    /** 常用字拼音表 */
    private static final java.util.Map<Character, String> MAP = new java.util.HashMap<>();
    static {
        // 你 我 他 她 它 们
        put('你', "n,i,3"); put('我', "w,o,3"); put('他', "t,a,1"); put('她', "t,a,1"); put('它', "t,a,1");
        put('们', "m,en,0");
        // 好 是 这 那 什 么 的 了
        put('好', "h,ao,3"); put('是', "sh,i,4"); put('这', "zh,e,4"); put('那', "n,a,4");
        put('什', "sh,en,2"); put('么', "m,e,0"); put('的', "d,e,0"); put('了', "l,e,0");
        // 看 到 见 听 说 想 知 道
        put('看', "k,an,4"); put('到', "d,ao,4"); put('见', "j,ian,4"); put('听', "t,ing,1");
        put('说', "sh,uo,1"); put('想', "x,iang,3"); put('知', "zh,i,1"); put('道', "d,ao,4");
        // 苹果 猫 狗 水 花 书 本 杯子 红 大 小
        put('苹', "p,ing,2"); put('果', "g,uo,3"); put('猫', "m,ao,1"); put('狗', "g,ou,3");
        put('水', "sh,ui,3"); put('花', "h,ua,1"); put('书', "sh,u,1"); put('本', "b,en,3");
        put('杯', "b,ei,1"); put('子', "z,i,0"); put('红', "h,ong,2"); put('大', "d,a,4"); put('小', "x,iao,3");
        // 新 事 物 学 习 认 识 记 忆 忘
        put('新', "x,in,1"); put('事', "sh,i,4"); put('物', "w,u,4"); put('学', "x,ue,2");
        put('习', "x,i,2"); put('认', "r,en,4"); put('识', "sh,i,2"); put('记', "j,i,4");
        put('忆', "y,i,4"); put('忘', "w,ang,4");
        // 睡 觉 梦 里 复 习 天 东 西
        put('睡', "sh,ui,4"); put('觉', "j,ue,4"); put('梦', "m,eng,4"); put('里', "l,i,3");
        put('复', "f,u,4"); put('习', "x,i,2"); put('天', "t,ian,1"); put('东', "d,ong,1"); put('西', "x,i,1");
        // 感 谢 喜 欢 明 白 帮 助 请 问 吗 吧
        put('感', "g,an,3"); put('谢', "x,ie,4"); put('喜', "x,i,3"); put('欢', "h,uan,1");
        put('明', "m,ing,2"); put('白', "b,ai,2"); put('帮', "b,ang,1"); put('助', "zh,u,4");
        put('请', "q,ing,3"); put('问', "w,en,4"); put('吗', "m,a,0"); put('吧', "b,a,0");
        // 一起 玩 笑 高兴 生气 哭 怕 爱
        put('一', "y,i,1"); put('起', "q,i,3"); put('玩', "w,an,2"); put('笑', "x,iao,4");
        put('高', "g,ao,1"); put('兴', "x,ing,4"); put('生', "sh,eng,1"); put('气', "q,i,4");
        put('哭', "k,u,1"); put('怕', "p,a,4"); put('爱', "a,i,4");
        // 发现 探索 好奇 环境 声音 视频 视觉 记忆
        put('发', "f,a,1"); put('现', "x,ian,4"); put('探', "t,an,4"); put('索', "s,uo,3");
        put('好', "h,ao,3"); put('奇', "q,i,2"); put('环', "h,uan,2"); put('境', "j,ing,4");
        put('声', "sh,eng,1"); put('音', "y,in,1"); put('视', "sh,i,4"); put('频', "p,in,2");
        put('颜', "y,an,2"); put('色', "s,e,4"); put('光', "g,uang,1"); put('亮', "l,iang,4");
        put('黑', "h,ei,1"); put('白', "b,ai,2");
        // 数字/基础
        put('零', "l,ing,2"); put('一', "y,i,1"); put('二', "e,r,4"); put('三', "s,an,1");
        put('四', "s,i,4"); put('五', "w,u,3"); put('六', "l,iu,4"); put('七', "q,i,1");
        put('八', "b,a,1"); put('九', "j,iu,3"); put('十', "sh,i,2");
        // 其他常用
        put('吃', "ch,i,1"); put('喝', "h,e,1"); put('走', "z,ou,3"); put('来', "l,ai,2");
        put('去', "q,u,4"); put('回', "h,ui,2"); put('家', "j,ia,1"); put('人', "r,en,2");
        put('中', "zh,ong,1"); put('国', "g,uo,2"); put('真', "zh,en,1"); put('美', "m,ei,3");
        put('漂', "p,iao,1"); put('亮', "l,iang,4"); put('快', "k,uai,4"); put('乐', "l,e,4");
        put('累', "l,ei,4"); put('饿', ",e,4"); put('渴', "k,e,3"); put('困', "k,un,4");
        // 环境/互动 (零声母统一 ",韵母,声调" 3段格式)
        put('你', "n,i,3"); put('好', "h,ao,3"); put('呀', "y,a,0"); put('啊', ",a,1");
        put('嗯', ",en,0"); put('哦', ",o,2"); put('哟', "y,o,0"); put('哈', "h,a,1");
        put('爱', ",ai,4"); put('饿', ",e,4");
    }

    private static void put(char c, String v) { MAP.put(c, v); }
}
