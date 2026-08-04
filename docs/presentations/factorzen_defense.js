/**
 * FactorZen 答辩讲解 PPT
 * 受众：懂代码、不熟悉量化
 * 重点：多 Agent 因子挖掘（4 角色 + Evaluator）
 * 时长：≤10 分钟（含演示）
 */
const pptxgen = require("pptxgenjs");
const React = require("react");
const ReactDOMServer = require("react-dom/server");
const sharp = require("sharp");
const path = require("path");
const {
  FaLightbulb, FaCogs, FaCode, FaDatabase, FaShieldAlt,
  FaChartLine, FaRobot, FaLayerGroup, FaCheckCircle,
  FaTerminal, FaGlobe, FaFlask, FaBolt, FaLock,
  FaBalanceScale, FaServer, FaGitAlt, FaBook, FaUsers,
  FaUserEdit, FaSearch, FaComments, FaBrain, FaRedo,
  FaVial, FaBoxes, FaSitemap, FaExclamationTriangle,
  FaUserSecret, FaClipboardCheck
} = require("react-icons/fa");
const { SiPython } = require("react-icons/si");
const { MdMemory, MdOutlineScience } = require("react-icons/md");
const { HiOutlineLightBulb } = require("react-icons/hi");

function renderIconSvg(IconComponent, color = "#000000", size = 256) {
  return ReactDOMServer.renderToStaticMarkup(
    React.createElement(IconComponent, { color, size: String(size) })
  );
}
async function iconToBase64Png(IconComponent, color, size = 256) {
  const svg = renderIconSvg(IconComponent, color, size);
  const buf = await sharp(Buffer.from(svg)).png().toBuffer();
  return "image/png;base64," + buf.toString("base64");
}

const D_BG = "16181D", D_PANEL = "1E2026", D_CARD = "252830";
const D_TEXT = "EDEFF2", D_LIGHT = "C4C8D0", D_MUTED = "8B909A", D_DIM = "5C616B", D_RULE = "2E323C";
const L_BG = "F5F6F8", L_CARD = "FFFFFF", L_TEXT = "16181D", L_SUB = "2A2D35";
const L_MUTED = "6B7080", L_RULE = "E0E3E8", L_BORDER = "E8EBEF";
const ACCENT = "6B4EFF", ACCENT2 = "8E7BFF";
const HFONT = "Arial Black", SFONT = "Arial", BFONT = "Arial", CFONT = "Consolas";
const SW = 13.3, SH = 7.5, ML = 0.5, CW = 12.3;

let PRES;
const TL = { bg: L_BG, card: L_CARD, text: L_TEXT, subtext: L_SUB, muted: L_MUTED, rule: L_RULE, border: L_BORDER };

function shadow() {
  return { type: "outer", color: "000000", blur: 8, offset: 2, angle: 135, opacity: 0.08 };
}
function addRule(slide, x, y, w, color) {
  slide.addShape(PRES.shapes.RECTANGLE, { x, y, w, h: 0.016, fill: { color }, line: { type: "none" } });
}
function addHeader(slide, section, title, page) {
  slide.addText(section, {
    x: ML, y: 0.22, w: CW * 0.8, h: 0.22,
    fontSize: 10, fontFace: SFONT, bold: true, color: ACCENT, charSpacing: 1.5, margin: 0
  });
  slide.addText(title, {
    x: ML, y: 0.44, w: CW * 0.92, h: 0.48,
    fontSize: 22, fontFace: HFONT, color: L_TEXT, margin: 0
  });
  addRule(slide, ML, 0.98, CW, L_RULE);
  if (page != null) {
    slide.addText(String(page).padStart(2, "0"), {
      x: SW - ML - 0.45, y: 7.18, w: 0.4, h: 0.2,
      fontSize: 10, fontFace: SFONT, bold: true, color: ACCENT, align: "right", margin: 0
    });
  }
}
function addFooter(slide, text) {
  slide.addText(text, {
    x: ML, y: 7.18, w: CW - 0.55, h: 0.2,
    fontSize: 9, fontFace: BFONT, italic: true, color: L_MUTED, margin: 0
  });
}
function addTakeaway(slide, text, y) {
  const ty = y ?? 6.55;
  slide.addShape(PRES.shapes.RECTANGLE, {
    x: ML, y: ty, w: CW, h: 0.48, fill: { color: L_CARD },
    line: { color: L_BORDER, width: 1 }, shadow: shadow()
  });
  slide.addShape(PRES.shapes.RECTANGLE, {
    x: ML, y: ty, w: 0.06, h: 0.48, fill: { color: ACCENT }, line: { type: "none" }
  });
  slide.addText([
    { text: "一句话  ", options: { bold: true, fontSize: 11, fontFace: SFONT, color: ACCENT } },
    { text: text, options: { fontSize: 12, fontFace: BFONT, color: L_SUB } }
  ], { x: ML + 0.2, y: ty, w: CW - 0.35, h: 0.48, valign: "middle", margin: 0 });
}
function liveBadge(slide) {
  slide.addShape(PRES.shapes.ROUNDED_RECTANGLE, {
    x: SW - ML - 1.7, y: 0.28, w: 1.6, h: 0.32,
    fill: { color: ACCENT }, rectRadius: 0.05, line: { type: "none" }
  });
  slide.addText("LIVE DEMO", {
    x: SW - ML - 1.7, y: 0.28, w: 1.6, h: 0.32,
    fontSize: 11, fontFace: SFONT, bold: true, color: "FFFFFF",
    align: "center", valign: "middle", margin: 0
  });
}
function focusBadge(slide, label) {
  slide.addShape(PRES.shapes.ROUNDED_RECTANGLE, {
    x: SW - ML - 2.5, y: 0.28, w: 2.4, h: 0.32,
    fill: { color: "EDE9FF" }, rectRadius: 0.05, line: { type: "none" }
  });
  slide.addText(label, {
    x: SW - ML - 2.5, y: 0.28, w: 2.4, h: 0.32,
    fontSize: 10, fontFace: SFONT, bold: true, color: ACCENT,
    align: "center", valign: "middle", margin: 0
  });
}

async function buildDeck() {
  const pres = new pptxgen();
  PRES = pres;
  pres.layout = "LAYOUT_WIDE";
  pres.author = "rookiewu417";
  pres.title = "FactorZen 答辩讲解 · 多 Agent 因子挖掘";
  pres.subject = "面向懂代码的听众 · 重点 multi-agent mining · ≤10 分钟";

  const logoDark = path.join(__dirname, "logo-icon-dark-512.png");
  const ic = {
    idea: await iconToBase64Png(FaLightbulb, "#6B4EFF", 256),
    cogs: await iconToBase64Png(FaCogs, "#6B4EFF", 256),
    code: await iconToBase64Png(FaCode, "#6B4EFF", 256),
    db: await iconToBase64Png(FaDatabase, "#6B4EFF", 256),
    shield: await iconToBase64Png(FaShieldAlt, "#6B4EFF", 256),
    chart: await iconToBase64Png(FaChartLine, "#6B4EFF", 256),
    robot: await iconToBase64Png(FaRobot, "#6B4EFF", 256),
    layer: await iconToBase64Png(FaLayerGroup, "#6B4EFF", 256),
    check: await iconToBase64Png(FaCheckCircle, "#6B4EFF", 256),
    term: await iconToBase64Png(FaTerminal, "#6B4EFF", 256),
    globe: await iconToBase64Png(FaGlobe, "#6B4EFF", 256),
    flask: await iconToBase64Png(FaFlask, "#6B4EFF", 256),
    bolt: await iconToBase64Png(FaBolt, "#6B4EFF", 256),
    lock: await iconToBase64Png(FaLock, "#6B4EFF", 256),
    bal: await iconToBase64Png(FaBalanceScale, "#6B4EFF", 256),
    server: await iconToBase64Png(FaServer, "#6B4EFF", 256),
    git: await iconToBase64Png(FaGitAlt, "#6B4EFF", 256),
    book: await iconToBase64Png(FaBook, "#6B4EFF", 256),
    users: await iconToBase64Png(FaUsers, "#6B4EFF", 256),
    edit: await iconToBase64Png(FaUserEdit, "#6B4EFF", 256),
    search: await iconToBase64Png(FaSearch, "#6B4EFF", 256),
    chat: await iconToBase64Png(FaComments, "#6B4EFF", 256),
    brain: await iconToBase64Png(FaBrain, "#6B4EFF", 256),
    redo: await iconToBase64Png(FaRedo, "#6B4EFF", 256),
    vial: await iconToBase64Png(FaVial, "#6B4EFF", 256),
    boxes: await iconToBase64Png(FaBoxes, "#6B4EFF", 256),
    site: await iconToBase64Png(FaSitemap, "#6B4EFF", 256),
    warn: await iconToBase64Png(FaExclamationTriangle, "#6B4EFF", 256),
    secret: await iconToBase64Png(FaUserSecret, "#6B4EFF", 256),
    clip: await iconToBase64Png(FaClipboardCheck, "#6B4EFF", 256),
    mem: await iconToBase64Png(MdMemory, "#6B4EFF", 256),
    py: await iconToBase64Png(SiPython, "#6B4EFF", 256),
    bulb: await iconToBase64Png(HiOutlineLightBulb, "#6B4EFF", 256),
  };

  // ========== 1 封面 ==========
  {
    const s = pres.addSlide();
    s.background = { color: D_BG };
    s.addShape(PRES.shapes.RECTANGLE, {
      x: 8.4, y: 0, w: 4.9, h: SH, fill: { color: D_PANEL }, line: { type: "none" }
    });
    s.addShape(PRES.shapes.RECTANGLE, {
      x: 8.4, y: 0, w: 0.08, h: SH, fill: { color: ACCENT }, line: { type: "none" }
    });
    s.addImage({ path: logoDark, x: ML, y: 0.45, w: 0.6, h: 0.6 });
    s.addText("FACTORZEN", {
      x: ML + 0.75, y: 0.55, w: 4, h: 0.4,
      fontSize: 17, fontFace: HFONT, color: D_TEXT, margin: 0, valign: "middle"
    });
    s.addText("项目答辩讲解", {
      x: ML, y: 1.35, w: 7, h: 0.3,
      fontSize: 13, fontFace: SFONT, bold: true, color: ACCENT2, charSpacing: 2, margin: 0
    });
    addRule(s, ML, 1.75, 2.0, ACCENT);
    s.addText("多 Agent 驱动的\n因子研究平台", {
      x: ML, y: 2.0, w: 7.4, h: 1.55,
      fontSize: 36, fontFace: HFONT, color: D_TEXT, margin: 0, lineSpacingMultiple: 1.05
    });
    s.addText("重点讲解：LLM 四角色团队如何「提案 → 写表达式 → 实测 → 评审」\n闭环挖特征，并用记忆与门禁避免重复踩坑、把噪声当信号。", {
      x: ML, y: 3.75, w: 7.3, h: 0.75,
      fontSize: 14, fontFace: BFONT, color: D_LIGHT, margin: 0
    });
    s.addText("听众：懂代码即可  ·  金融词会用大白话解释  ·  ≤10 分钟（含演示）", {
      x: ML, y: 4.7, w: 7.3, h: 0.35,
      fontSize: 13, fontFace: BFONT, color: D_MUTED, margin: 0
    });
    s.addText("v0.3.0  ·  MIT  ·  Python  ·  ~57K LOC  ·  952 tests", {
      x: ML, y: 6.85, w: 7, h: 0.28,
      fontSize: 11, fontFace: SFONT, color: D_DIM, margin: 0
    });

    const side = [
      { v: "4+1", l: "角色协作\n4 个 LLM 角色 + 确定性评估器" },
      { v: "闭环", l: "否决回路\nkeep / 改代码 / 改方向 / 丢弃" },
      { v: "记忆", l: "跨 session\n挖穿区避开 · 未探索优先" },
      { v: "门禁", l: "护栏 + Lift\n实测通过才进候选库" },
    ];
    side.forEach((st, i) => {
      const y = 0.55 + i * 1.55;
      s.addText(st.v, {
        x: 8.75, y: y, w: 4.2, h: 0.4,
        fontSize: 22, fontFace: HFONT, color: ACCENT2, margin: 0
      });
      s.addText(st.l, {
        x: 8.75, y: y + 0.45, w: 4.2, h: 0.7,
        fontSize: 13, fontFace: BFONT, color: D_MUTED, margin: 0
      });
    });
  }

  // ========== 2 大纲 ==========
  {
    const s = pres.addSlide();
    s.background = { color: L_BG };
    addHeader(s, "AGENDA", "讲解大纲 · 今天重点讲「多 Agent 怎么挖因子」", 2);
    addFooter(s, "超时优先保：术语 → 为何 multi-agent → 角色流水线 → 演示；其余快翻");

    const items = [
      { n: "01", t: "对齐语境", d: "这是什么软件\n术语大白话", m: "1.5′", icon: ic.book },
      { n: "02", t: "项目意义", d: "为何不能只看\n「单指标好不好」", m: "1.5′", icon: ic.idea },
      { n: "03", t: "多 Agent 挖掘", d: "四角色 + 评估器\n记忆 · 否决 · 演示", m: "4.5′", icon: ic.users },
      { n: "04", t: "门禁与技术", d: "Lift · 栈 · 代码\n工程保障", m: "2′", icon: ic.code },
      { n: "05", t: "总结", d: "交付 · 边界 · Q&A", m: "0.5′", icon: ic.check },
    ];
    items.forEach((a, i) => {
      const x = ML + i * 2.5;
      s.addShape(PRES.shapes.RECTANGLE, {
        x, y: 1.3, w: 2.35, h: 4.9, fill: { color: L_CARD },
        line: { color: i === 2 ? ACCENT : L_BORDER, width: i === 2 ? 2 : 1 }, shadow: shadow()
      });
      if (i === 2) {
        s.addShape(PRES.shapes.RECTANGLE, {
          x, y: 1.3, w: 2.35, h: 0.08, fill: { color: ACCENT }, line: { type: "none" }
        });
      }
      s.addImage({ data: a.icon, x: x + 0.2, y: 1.65, w: 0.4, h: 0.4 });
      s.addText(a.n, {
        x: x + 0.7, y: 1.7, w: 1.4, h: 0.35,
        fontSize: 16, fontFace: HFONT, color: ACCENT, margin: 0
      });
      s.addText(a.t, {
        x: x + 0.2, y: 2.35, w: 1.95, h: 0.7,
        fontSize: 16, fontFace: HFONT, color: L_TEXT, margin: 0
      });
      s.addText(a.d, {
        x: x + 0.2, y: 3.2, w: 1.95, h: 1.4,
        fontSize: 12, fontFace: BFONT, color: L_MUTED, margin: 0
      });
      s.addShape(PRES.shapes.ROUNDED_RECTANGLE, {
        x: x + 0.2, y: 5.5, w: 1.95, h: 0.4,
        fill: { color: i === 2 ? ACCENT : "EDE9FF" }, rectRadius: 0.05, line: { type: "none" }
      });
      s.addText(a.m, {
        x: x + 0.2, y: 5.5, w: 1.95, h: 0.4,
        fontSize: 14, fontFace: SFONT, bold: true, color: i === 2 ? "FFFFFF" : ACCENT,
        align: "center", valign: "middle", margin: 0
      });
    });
  }

  // ========== 3 这是什么 ==========
  {
    const s = pres.addSlide();
    s.background = { color: L_BG };
    addHeader(s, "00  对齐语境", "先忘掉股票：这是一套「自动挖特征」的研究系统", 3);
    addTakeaway(s, "核心产品不是交易按钮，而是：能协作挖特征的 Agent 团队 + 能拦坏特征的门禁。", 6.55);
    addFooter(s, "");

    const boxes = [
      { t: "业务目标（大白话）", icon: ic.bulb, lines: [
        "想从历史行情里找出：",
        "「哪些可计算的信号，能帮我们",
        "  判断股票相对强弱？」",
        "这些信号就叫「因子」——",
        "你可以先把它当成 ML 里的特征。"
      ]},
      { t: "工程形态", icon: ic.site, lines: [
        "CLI 流水线 + 本地数据湖",
        "LLM 多角色协作写候选",
        "确定性代码负责算分与拦截",
        "通过的进「因子库」登记簿",
        "再下游组合、回测、报告"
      ]},
      { t: "今天重点", icon: ic.users, lines: [
        "fz mine team",
        "Librarian / Hypothesis /",
        "Coder / Critic + Evaluator",
        "像一个小研发团队：",
        "有人提方向、有人写代码、",
        "有人跑测、有人做 code review"
      ]},
    ];
    boxes.forEach((b, i) => {
      const x = ML + i * 4.2;
      s.addShape(PRES.shapes.RECTANGLE, {
        x, y: 1.25, w: 4.0, h: 4.95, fill: { color: L_CARD },
        line: { color: L_BORDER, width: 1 }, shadow: shadow()
      });
      s.addShape(PRES.shapes.RECTANGLE, {
        x, y: 1.25, w: 4.0, h: 0.9, fill: { color: D_BG }, line: { type: "none" }
      });
      s.addImage({ data: b.icon, x: x + 0.25, y: 1.45, w: 0.38, h: 0.38 });
      s.addText(b.t, {
        x: x + 0.75, y: 1.45, w: 3.0, h: 0.45,
        fontSize: 14, fontFace: HFONT, color: D_TEXT, valign: "middle", margin: 0
      });
      b.lines.forEach((ln, j) => {
        s.addText(ln, {
          x: x + 0.3, y: 2.45 + j * 0.5, w: 3.4, h: 0.45,
          fontSize: 14, fontFace: BFONT, color: L_SUB, margin: 0
        });
      });
    });
  }

  // ========== 4 术语大白话（重点改通俗） ==========
  {
    const s = pres.addSlide();
    s.background = { color: L_BG };
    addHeader(s, "00  术语小抄（超白话）", "后面会反复出现的词 · 先用人话记住", 4);
    addFooter(s, "不必背金融定义；听懂「它在系统里干什么」就够了");

    // Two-column card glossary - more conversational
    const terms = [
      {
        w: "因子",
        say: "一个「算得出来的分数」",
        eg: "比如：近 20 天涨得多不多、成交是否异常放大。\n就像表格里新加的一列特征，用来猜未来谁更强。"
      },
      {
        w: "挖因子 / 挖掘",
        say: "自动试很多种算分公式",
        eg: "不是人手写一个指标，而是让程序（或 AI）\n批量提出候选公式，再一个个实测。"
      },
      {
        w: "IC（相关）",
        say: "「这个分数和后来涨跌，沾不沾边」",
        eg: "分数高的股票，后面是不是更容易涨。\n只说明「单独看还行」，不说明「库里还需要它」。"
      },
      {
        w: "过拟合",
        say: "把运气当成规律",
        eg: "试了几千个公式，总会撞上几个历史碰巧好看。\n像网格搜索超参只报最好的那个分数。"
      },
      {
        w: "未来函数 / PIT",
        say: "不许偷看「当时还不知道」的信息",
        eg: "用今天的名单回算十年前，或用尚未公告的财报——\n等于考试时偷看了答案，成绩会虚高。"
      },
      {
        w: "因子库",
        say: "通过验收的「特征登记册」",
        eg: "不是无限加列。只有证明「对已有特征还有增量」\n的，才能登记进去给后面用。"
      },
      {
        w: "Lift 增量",
        say: "去掉旧信息之后，还有没有新贡献",
        eg: "新特征若几乎是旧特征的马甲，就不收。\n像：新 PR 不能只是复制粘贴已有模块。"
      },
      {
        w: "Agent 角色",
        say: "分工明确的「虚拟同事」",
        eg: "有人提方向、有人写表达式、有人跑分、\n有人挑刺；不是一个 LLM 一口气写完假装搞定。"
      },
    ];
    terms.forEach((t, i) => {
      const col = i % 2, row = Math.floor(i / 2);
      const x = ML + col * 6.25, y = 1.15 + row * 1.4;
      s.addShape(PRES.shapes.RECTANGLE, {
        x, y, w: 6.05, h: 1.28, fill: { color: L_CARD },
        line: { color: L_BORDER, width: 1 }, shadow: shadow()
      });
      s.addShape(PRES.shapes.RECTANGLE, {
        x, y, w: 0.08, h: 1.28, fill: { color: ACCENT }, line: { type: "none" }
      });
      s.addText(t.w, {
        x: x + 0.25, y: y + 0.12, w: 1.7, h: 0.32,
        fontSize: 14, fontFace: HFONT, color: ACCENT, margin: 0
      });
      s.addText(t.say, {
        x: x + 2.0, y: y + 0.12, w: 3.8, h: 0.32,
        fontSize: 13, fontFace: SFONT, bold: true, color: L_TEXT, margin: 0
      });
      s.addText(t.eg, {
        x: x + 0.25, y: y + 0.5, w: 5.55, h: 0.7,
        fontSize: 12, fontFace: BFONT, color: L_MUTED, margin: 0
      });
    });
  }

  // ========== 5 项目意义：痛点 ==========
  {
    const s = pres.addSlide();
    s.background = { color: L_BG };
    addHeader(s, "01  项目意义", "为什么「让 AI 随便生成指标」会翻车？", 5);
    addFooter(s, "这些坑决定了：必须 multi-agent + 实测门禁，而不能单次 prompt 完事");

    const pains = [
      { n: "1", t: "公式会重复", d: "新公式和旧特征高度相似，看起来都「有效」，合在一起却没增益。\n→ 需要：相对已有库的增量检验（Lift）" },
      { n: "2", t: "搜得越多越容易撞运气", d: "几千次尝试里总会有历史碰巧好看的。\n→ 需要：多重检验记账 + 样本外 holdout + Critic 评审" },
      { n: "3", t: "LLM 会胡说 / 语法错", d: "乱造叶子名、窗口非法、表达式解析失败。\n→ 需要：Coder 语法约束 + 解析失败自愈 + 确定性 Evaluator" },
      { n: "4", t: "团队没有记忆会反复踩坑", d: "上次证明无效的方向，下一轮又提。\n→ 需要：Librarian 跨 session 记忆（挖穿区 / 未探索区）" },
    ];
    pains.forEach((p, i) => {
      const y = 1.2 + i * 1.35;
      s.addShape(PRES.shapes.RECTANGLE, {
        x: ML, y, w: CW, h: 1.22, fill: { color: L_CARD },
        line: { color: L_BORDER, width: 1 }, shadow: shadow()
      });
      s.addShape(PRES.shapes.RECTANGLE, {
        x: ML, y, w: 0.7, h: 1.22, fill: { color: D_BG }, line: { type: "none" }
      });
      s.addText(p.n, {
        x: ML, y, w: 0.7, h: 1.22,
        fontSize: 22, fontFace: HFONT, color: ACCENT2, align: "center", valign: "middle", margin: 0
      });
      s.addText(p.t, {
        x: ML + 0.95, y: y + 0.15, w: 10.5, h: 0.35,
        fontSize: 16, fontFace: HFONT, color: L_TEXT, margin: 0
      });
      s.addText(p.d, {
        x: ML + 0.95, y: y + 0.55, w: 10.5, h: 0.55,
        fontSize: 13, fontFace: BFONT, color: L_MUTED, margin: 0
      });
    });
  }

  // ========== 6 核心问题 ==========
  {
    const s = pres.addSlide();
    s.background = { color: L_BG };
    addHeader(s, "01  项目意义", "系统真正优化的目标", 6);
    addTakeaway(s, "挖因子不是「生成漂亮公式」，而是「稳定地往登记册里添加真正有增量的特征」。", 6.55);
    addFooter(s, "");

    s.addShape(PRES.shapes.RECTANGLE, {
      x: ML, y: 1.25, w: 5.95, h: 4.9, fill: { color: L_CARD },
      line: { color: L_BORDER, width: 1 }, shadow: shadow()
    });
    s.addShape(PRES.shapes.RECTANGLE, {
      x: ML, y: 1.25, w: 5.95, h: 0.65, fill: { color: "3A2A2A" }, line: { type: "none" }
    });
    s.addText("不够的问题", {
      x: ML + 0.35, y: 1.25, w: 5.3, h: 0.65,
      fontSize: 16, fontFace: HFONT, color: "F0C0C0", valign: "middle", margin: 0
    });
    s.addText("「这个公式好不好？」", {
      x: ML + 0.35, y: 2.2, w: 5.3, h: 0.5,
      fontSize: 20, fontFace: HFONT, color: L_TEXT, margin: 0
    });
    s.addText("只回答了单变量好看不好看。\n\n就像 code review 只问「这段代码语法对不对」，\n不问「主干里是不是已经有同款功能」。", {
      x: ML + 0.35, y: 2.9, w: 5.3, h: 2.5,
      fontSize: 14, fontFace: BFONT, color: L_MUTED, margin: 0
    });

    s.addShape(PRES.shapes.RECTANGLE, {
      x: ML + 6.25, y: 1.25, w: 5.95, h: 4.9, fill: { color: D_BG }, line: { type: "none" }, shadow: shadow()
    });
    s.addShape(PRES.shapes.RECTANGLE, {
      x: ML + 6.25, y: 1.25, w: 5.95, h: 0.65, fill: { color: ACCENT }, line: { type: "none" }
    });
    s.addText("我们真正要问", {
      x: ML + 6.6, y: 1.25, w: 5.3, h: 0.65,
      fontSize: 16, fontFace: HFONT, color: D_TEXT, valign: "middle", margin: 0
    });
    s.addText("「相对已有特征库，\n还有没有新信息？」", {
      x: ML + 6.6, y: 2.15, w: 5.3, h: 1.0,
      fontSize: 20, fontFace: HFONT, color: D_TEXT, margin: 0
    });
    s.addText("多 Agent 负责：想方向、写公式、改 bug。\n确定性代码负责：算分、护栏、Lift 裁决。\n\nLLM 不「既当运动员又当裁判」——\n生成与验收拆开，这是刻意设计。", {
      x: ML + 6.6, y: 3.4, w: 5.3, h: 2.3,
      fontSize: 14, fontFace: BFONT, color: D_LIGHT, margin: 0
    });
  }

  // ========== 7 为何 multi-agent ==========
  {
    const s = pres.addSlide();
    s.background = { color: L_BG };
    addHeader(s, "02  多 Agent 因子挖掘", "为什么不用「一个大 prompt 包办」？", 7);
    focusBadge(s, "★ 讲解重点");
    addFooter(s, "入口命令：fz mine team（主力）· 对照：fz mine agent（单 Agent 闭环）");

    // comparison
    const cols = [
      {
        t: "单次 LLM 生成", bad: true,
        items: [
          "方向、写法、自评搅在一起",
          "容易自我辩解「我觉得很好」",
          "没有跨会话记忆",
          "语法错只能整段重来",
          "难做结构化日志与回放"
        ]
      },
      {
        t: "多角色流水线（本项目）", bad: false,
        items: [
          "职责拆分：提方向 / 写码 / 实测 / 评审",
          "Critic 专职唱反调（code review）",
          "Librarian 记住踩过的坑",
          "解析失败：错误回灌自愈",
          "每轮状态可落盘、可审计"
        ]
      },
    ];
    cols.forEach((c, i) => {
      const x = ML + i * 6.25;
      s.addShape(PRES.shapes.RECTANGLE, {
        x, y: 1.2, w: 6.05, h: 5.5, fill: { color: L_CARD },
        line: { color: c.bad ? L_BORDER : ACCENT, width: c.bad ? 1 : 2 }, shadow: shadow()
      });
      s.addShape(PRES.shapes.RECTANGLE, {
        x, y: 1.2, w: 6.05, h: 0.75,
        fill: { color: c.bad ? "3A2A2A" : D_BG }, line: { type: "none" }
      });
      s.addText(c.t, {
        x: x + 0.3, y: 1.2, w: 5.5, h: 0.75,
        fontSize: 16, fontFace: HFONT, color: c.bad ? "F0C0C0" : ACCENT2,
        valign: "middle", margin: 0
      });
      c.items.forEach((it, j) => {
        s.addText((c.bad ? "✗  " : "✓  ") + it, {
          x: x + 0.35, y: 2.25 + j * 0.75, w: 5.4, h: 0.65,
          fontSize: 15, fontFace: BFONT, color: L_SUB, margin: 0
        });
      });
    });
  }

  // ========== 8 四角色总览 ==========
  {
    const s = pres.addSlide();
    s.background = { color: L_BG };
    addHeader(s, "02  多 Agent 因子挖掘", "团队编制：4 个 LLM 角色 + 1 个确定性评估器", 8);
    focusBadge(s, "★ 讲解重点");
    addFooter(s, "口径：4 角色 + Evaluator；Evaluator 不调 LLM，避免「自己出题自己打分」");

    const roles = [
      { name: "Librarian", cn: "图书管理员", llm: "LLM", icon: ic.book,
        job: "开工前翻旧账\n收工时记新账",
        detail: "recall 已知无效方向\n挖穿区 / 未探索区\nlift 拒过的方向" },
      { name: "Hypothesis", cn: "研究员", llm: "LLM", icon: ic.bulb,
        job: "提出「为什么\n可能有效」的方向",
        detail: "读记忆再提案\n避开拥挤与挖穿\n可结构化假设" },
      { name: "Coder", cn: "程序员", llm: "LLM", icon: ic.edit,
        job: "把方向翻译成\n可执行表达式",
        detail: "受限算子/叶子语法\n按 Critic 意见改\n解析错则自愈重写" },
      { name: "Evaluator", cn: "测试机", llm: "代码", icon: ic.vial,
        job: "真算分数\n跑护栏判定",
        detail: "IC / holdout / DSR…\n确定性、可复现\n不听 LLM 自评" },
      { name: "Critic", cn: "审查员", llm: "LLM", icon: ic.secret,
        job: "读实测结果\n给处置意见",
        detail: "keep / 改表达式\n改方向 / drop\n驱动否决回路" },
    ];
    roles.forEach((r, i) => {
      const x = ML + i * 2.5;
      const isEval = r.name === "Evaluator";
      s.addShape(PRES.shapes.RECTANGLE, {
        x, y: 1.2, w: 2.35, h: 5.5, fill: { color: isEval ? D_BG : L_CARD },
        line: { color: isEval ? ACCENT : L_BORDER, width: isEval ? 2 : 1 }, shadow: shadow()
      });
      s.addImage({ data: r.icon, x: x + 0.2, y: 1.4, w: 0.38, h: 0.38 });
      s.addText(r.name, {
        x: x + 0.15, y: 1.95, w: 2.05, h: 0.35,
        fontSize: 13, fontFace: HFONT, color: isEval ? ACCENT2 : L_TEXT, margin: 0
      });
      s.addText(r.cn, {
        x: x + 0.15, y: 2.3, w: 2.05, h: 0.3,
        fontSize: 12, fontFace: BFONT, color: isEval ? D_MUTED : L_MUTED, margin: 0
      });
      s.addShape(PRES.shapes.ROUNDED_RECTANGLE, {
        x: x + 0.15, y: 2.7, w: 1.3, h: 0.32,
        fill: { color: isEval ? ACCENT : "EDE9FF" }, rectRadius: 0.04, line: { type: "none" }
      });
      s.addText(r.llm, {
        x: x + 0.15, y: 2.7, w: 1.3, h: 0.32,
        fontSize: 11, fontFace: SFONT, bold: true, color: isEval ? "FFFFFF" : ACCENT,
        align: "center", valign: "middle", margin: 0
      });
      s.addText(r.job, {
        x: x + 0.15, y: 3.25, w: 2.05, h: 1.1,
        fontSize: 13, fontFace: SFONT, bold: true, color: isEval ? D_TEXT : L_TEXT, margin: 0
      });
      s.addText(r.detail, {
        x: x + 0.15, y: 4.5, w: 2.05, h: 1.9,
        fontSize: 12, fontFace: BFONT, color: isEval ? D_MUTED : L_MUTED, margin: 0
      });
    });
  }

  // ========== 9 一轮流水线 ==========
  {
    const s = pres.addSlide();
    s.background = { color: L_BG };
    addHeader(s, "02  多 Agent 因子挖掘", "一轮战役怎么跑（team_orchestrator）", 9);
    focusBadge(s, "★ 讲解重点");
    addFooter(s, "实现：agents/team_orchestrator.py · 轮内可 llm_workers 并发 · 结构模板有配额防刷");

    const steps = [
      { n: "①", t: "Librarian\nrecall", d: "读长期记忆\n告知避开什么" },
      { n: "②", t: "Hypothesis\n提案", d: "提 1~N 个\n经济直觉方向" },
      { n: "③", t: "Coder\n写表达式", d: "方向→公式\n2~4 条/方向" },
      { n: "④", t: "Evaluator\n实测", d: "算 IC/护栏\n样本外 holdout" },
      { n: "⑤", t: "Critic\n裁决", d: "keep / 改码\n改方向 / drop" },
      { n: "⑥", t: "否决回路\n或进入下一轮", d: "revise 回灌\nCoder/Hypothesis" },
    ];
    steps.forEach((st, i) => {
      const x = ML + (i % 6) * 2.1;
      s.addShape(PRES.shapes.ROUNDED_RECTANGLE, {
        x, y: 1.25, w: 1.95, h: 2.35,
        fill: { color: i === 3 ? D_BG : L_CARD },
        rectRadius: 0.08,
        line: { color: i === 3 ? ACCENT : L_BORDER, width: i === 3 ? 2 : 1 },
        shadow: shadow()
      });
      s.addText(st.n, {
        x, y: 1.4, w: 1.95, h: 0.35,
        fontSize: 16, fontFace: HFONT, color: ACCENT, align: "center", margin: 0
      });
      s.addText(st.t, {
        x: x + 0.1, y: 1.85, w: 1.75, h: 0.75,
        fontSize: 13, fontFace: HFONT, color: i === 3 ? D_TEXT : L_TEXT, align: "center", margin: 0
      });
      s.addText(st.d, {
        x: x + 0.1, y: 2.7, w: 1.75, h: 0.7,
        fontSize: 11, fontFace: BFONT, color: i === 3 ? D_MUTED : L_MUTED, align: "center", margin: 0
      });
      if (i < 5) {
        s.addText("→", {
          x: x + 1.75, y: 2.15, w: 0.35, h: 0.35,
          fontSize: 16, fontFace: HFONT, color: ACCENT, margin: 0
        });
      }
    });

    // detail cards
    const cards = [
      { t: "Critic 四种 verdict", d: "keep — 留下候选\nrevise_expr — 让 Coder 改公式\nrevise_hypothesis — 方向有问题，重提\ndrop — 直接丢掉，别再纠缠" },
      { t: "确定性护栏（示例）", d: "holdout 样本外相关\nDeflated Sharpe（惩罚「试太多次」）\n换手等代价代理\n与库相关性 / 残差 IC（可选）" },
      { t: "Session 结束时", d: "Librarian record 写回记忆\n导出 candidates.csv\n默认可自动跑一轮组 Lift\n（与 lift-test 共用裁决函数）" },
    ];
    cards.forEach((c, i) => {
      const x = ML + i * 4.2;
      s.addShape(PRES.shapes.RECTANGLE, {
        x, y: 3.9, w: 4.0, h: 2.85, fill: { color: L_CARD },
        line: { color: L_BORDER, width: 1 }, shadow: shadow()
      });
      s.addText(c.t, {
        x: x + 0.25, y: 4.1, w: 3.5, h: 0.4,
        fontSize: 14, fontFace: HFONT, color: L_TEXT, margin: 0
      });
      s.addText(c.d, {
        x: x + 0.25, y: 4.6, w: 3.5, h: 1.9,
        fontSize: 13, fontFace: BFONT, color: L_MUTED, margin: 0
      });
    });
  }

  // ========== 10 记忆与防重复 ==========
  {
    const s = pres.addSlide();
    s.background = { color: L_BG };
    addHeader(s, "02  多 Agent 因子挖掘", "Librarian 记忆：让团队「越挖越聪明」", 10);
    focusBadge(s, "★ 讲解重点");
    addTakeaway(s, "没有记忆的 Agent 团队 = 每轮失忆的实习生；记忆把搜索变成有状态的战役。", 6.55);
    addFooter(s, "");

    const mems = [
      {
        t: "挖穿区", icon: ic.warn,
        rule: "某叶子上尝试 ≥15 次\n且 0 次过关",
        act: "后续提案避开\n别再死磕这块"
      },
      {
        t: "未探索区", icon: ic.search,
        rule: "历史唯一表达式\n≤ 2 次的叶子",
        act: "优先考虑\n提高覆盖广度"
      },
      {
        t: "库已覆盖", icon: ic.boxes,
        rule: "库内已有高 |IC|\n同类方向",
        act: "提醒 Hypothesis\n追求正交、少撞车"
      },
      {
        t: "Lift 拒绝史", icon: ic.bal,
        rule: "组合层已证\n无增量的方向",
        act: "别把废案\n重新包装再提"
      },
    ];
    mems.forEach((m, i) => {
      const x = ML + i * 3.15;
      s.addShape(PRES.shapes.RECTANGLE, {
        x, y: 1.25, w: 3.0, h: 4.95, fill: { color: L_CARD },
        line: { color: L_BORDER, width: 1 }, shadow: shadow()
      });
      s.addShape(PRES.shapes.RECTANGLE, {
        x, y: 1.25, w: 3.0, h: 1.0, fill: { color: D_BG }, line: { type: "none" }
      });
      s.addImage({ data: m.icon, x: x + 0.25, y: 1.5, w: 0.38, h: 0.38 });
      s.addText(m.t, {
        x: x + 0.75, y: 1.5, w: 2.0, h: 0.45,
        fontSize: 15, fontFace: HFONT, color: D_TEXT, valign: "middle", margin: 0
      });
      s.addText("判定规则", {
        x: x + 0.25, y: 2.5, w: 2.5, h: 0.35,
        fontSize: 12, fontFace: SFONT, bold: true, color: ACCENT, margin: 0
      });
      s.addText(m.rule, {
        x: x + 0.25, y: 2.95, w: 2.5, h: 1.2,
        fontSize: 14, fontFace: BFONT, color: L_SUB, margin: 0
      });
      s.addText("系统动作", {
        x: x + 0.25, y: 4.3, w: 2.5, h: 0.35,
        fontSize: 12, fontFace: SFONT, bold: true, color: ACCENT, margin: 0
      });
      s.addText(m.act, {
        x: x + 0.25, y: 4.75, w: 2.5, h: 1.1,
        fontSize: 14, fontFace: BFONT, color: L_SUB, margin: 0
      });
    });
  }

  // ========== 11 安全与工程细节 ==========
  {
    const s = pres.addSlide();
    s.background = { color: L_BG };
    addHeader(s, "02  多 Agent 因子挖掘", "工程上故意做的「防翻车」设计", 11);
    addFooter(s, "LLM 缺配置会直接报错退出，不会静默降级成随机搜索——避免「你以为在挖、其实没调模型」");

    const items = [
      { t: "生成与验收分离", d: "LLM 只生成/评审文本；分数由 Python 确定性算出。避免模型既当选手又当裁判。" },
      { t: "语法沙箱", d: "Coder 只能用白名单算子与叶子特征；乱造字段名会解析失败，进入 heal 自愈轮。" },
      { t: "结构模板配额", d: "同一种 AST「骨架」每轮最多进评估 3 条，防止换叶子名刷量制造假多样。" },
      { t: "多重检验记账", d: "DSR 的 N 用「历史唯一表达式 ∪ 本 session」，防止拆多次跑来稀释惩罚。" },
      { t: "holdout 隔离", d: "留出一段样本外数据打分；Critic 同时看 train / holdout，防「训练集装好看」。" },
      { t: "Session 末 Lift", d: "团队跑完默认可自动做组增量裁决，与 fz factor-library lift-test 共用同一函数，口径不漂。" },
    ];
    items.forEach((it, i) => {
      const col = i % 3, row = Math.floor(i / 3);
      const x = ML + col * 4.2, y = 1.2 + row * 2.7;
      s.addShape(PRES.shapes.RECTANGLE, {
        x, y, w: 4.0, h: 2.5, fill: { color: L_CARD },
        line: { color: L_BORDER, width: 1 }, shadow: shadow()
      });
      s.addShape(PRES.shapes.OVAL, {
        x: x + 0.25, y: y + 0.3, w: 0.4, h: 0.4,
        fill: { color: "EDE9FF" }, line: { type: "none" }
      });
      s.addText(String(i + 1), {
        x: x + 0.25, y: y + 0.3, w: 0.4, h: 0.4,
        fontSize: 14, fontFace: HFONT, color: ACCENT, align: "center", valign: "middle", margin: 0
      });
      s.addText(it.t, {
        x: x + 0.8, y: y + 0.35, w: 3.0, h: 0.35,
        fontSize: 15, fontFace: HFONT, color: L_TEXT, margin: 0
      });
      s.addText(it.d, {
        x: x + 0.25, y: y + 1.0, w: 3.5, h: 1.25,
        fontSize: 13, fontFace: BFONT, color: L_MUTED, margin: 0
      });
    });
  }

  // ========== 12 演示 A multi-agent ==========
  {
    const s = pres.addSlide();
    s.background = { color: L_BG };
    addHeader(s, "02  软件演示", "演示 A · 多 Agent 挖掘（主演示）", 12);
    liveBadge(s);
    addFooter(s, "现场勿跑满 iterations；可展示帮助、历史 session 产物、或短 dry 日志");

    const demos = [
      {
        n: "1", t: "看命令与参数",
        tip: "说明主力入口是 team",
        cmd: "pixi run fz mine team --help\n# 对照：fz mine agent（单 Agent）"
      },
      {
        n: "2", t: "典型启动（讲解用）",
        tip: "不必现场跑完",
        cmd: "pixi run fz mine team \\\n  --start 20200101 --end 20241231 --universe csi500 \\\n  --iterations 8 --structured --set llm_workers=4"
      },
      {
        n: "3", t: "看 session 产物",
        tip: "manifest / candidates",
        cmd: "ls workspace/mine_team/\n# 打开某次 run 的 manifest.json · candidates.csv"
      },
      {
        n: "4", t: "记忆与索引",
        tip: "跨 session 登记",
        cmd: "# experiment_index.jsonl\n# Librarian 读写的长期记忆痕迹"
      },
    ];
    demos.forEach((d, i) => {
      const y = 1.2 + i * 1.35;
      s.addShape(PRES.shapes.RECTANGLE, {
        x: ML, y, w: CW, h: 1.22, fill: { color: L_CARD },
        line: { color: L_BORDER, width: 1 }, shadow: shadow()
      });
      s.addShape(PRES.shapes.RECTANGLE, {
        x: ML, y, w: 0.65, h: 1.22, fill: { color: D_BG }, line: { type: "none" }
      });
      s.addText(d.n, {
        x: ML, y, w: 0.65, h: 1.22,
        fontSize: 20, fontFace: HFONT, color: ACCENT2, align: "center", valign: "middle", margin: 0
      });
      s.addText(d.t, {
        x: ML + 0.85, y: y + 0.2, w: 3.2, h: 0.35,
        fontSize: 15, fontFace: HFONT, color: L_TEXT, margin: 0
      });
      s.addText(d.tip, {
        x: ML + 0.85, y: y + 0.6, w: 3.2, h: 0.4,
        fontSize: 12, fontFace: BFONT, color: L_MUTED, margin: 0
      });
      s.addShape(PRES.shapes.RECTANGLE, {
        x: ML + 4.2, y: y + 0.15, w: 7.85, h: 0.92,
        fill: { color: D_BG }, line: { type: "none" }
      });
      s.addText(d.cmd, {
        x: ML + 4.35, y: y + 0.22, w: 7.55, h: 0.8,
        fontSize: 12, fontFace: CFONT, color: "A8F0C0", valign: "middle", margin: 0
      });
    });
  }

  // ========== 13 平台全景（简）+ Lift ==========
  {
    const s = pres.addSlide();
    s.background = { color: L_BG };
    addHeader(s, "03  门禁与下游", "挖出来以后：如何防止「垃圾进库」", 13);
    addFooter(s, "挖掘是上游；因子库 + Lift 是平台裁决中枢（与 multi-agent session 末钩子共用逻辑）");

    s.addShape(PRES.shapes.RECTANGLE, {
      x: ML, y: 1.2, w: CW, h: 1.5, fill: { color: D_BG }, line: { type: "none" }
    });
    s.addText("多 Agent 挖出候选  →  护栏筛选  →  Lift（相对库增量）  →  登记因子库  →  组合 / 回测 / 报告", {
      x: ML + 0.35, y: 1.2, w: CW - 0.7, h: 1.5,
      fontSize: 16, fontFace: HFONT, color: D_TEXT, valign: "middle", margin: 0
    });

    const cards = [
      {
        t: "Lift 大白话",
        d: "先问：新特征里有多少是旧特征已经解释过的？\n把这部分去掉，剩下的还和未来涨跌有关吗？\n有 → 才算真正增量；没有 → 只是马甲。"
      },
      {
        t: "安全默认",
        d: "lift-test 默认 dry-run（只看裁决）\n必须显式 --apply 才写库\n入库是不可逆库变更，有意设计"
      },
      {
        t: "和 multi-agent 的衔接",
        d: "team session 末可自动组 lift\n与 CLI lift-test 同一 lift_admission()\n三个消费方口径不漂移"
      },
    ];
    cards.forEach((c, i) => {
      const x = ML + i * 4.2;
      s.addShape(PRES.shapes.RECTANGLE, {
        x, y: 3.0, w: 4.0, h: 3.7, fill: { color: L_CARD },
        line: { color: L_BORDER, width: 1 }, shadow: shadow()
      });
      s.addText(c.t, {
        x: x + 0.25, y: 3.25, w: 3.5, h: 0.45,
        fontSize: 16, fontFace: HFONT, color: L_TEXT, margin: 0
      });
      s.addText(c.d, {
        x: x + 0.25, y: 3.9, w: 3.5, h: 2.5,
        fontSize: 14, fontFace: BFONT, color: L_MUTED, margin: 0
      });
    });
  }

  // ========== 14 演示 B · 前后端 WebUI ==========
  {
    const s = pres.addSlide();
    s.background = { color: L_BG };
    addHeader(s, "03  软件演示", "演示 B · Web 前后端（产物可浏览）", 14);
    liveBadge(s);
    addFooter(s, "dev extras · 默认无鉴权；时间紧可只开 http://localhost:8000/");

    // Architecture strip
    s.addShape(PRES.shapes.RECTANGLE, {
      x: ML, y: 1.15, w: CW, h: 1.05, fill: { color: D_BG }, line: { type: "none" }
    });
    s.addText("浏览器 (React SPA)   ──/api──▶   FastAPI (server/)   ──读──▶   workspace/ 产物湖", {
      x: ML + 0.3, y: 1.15, w: CW - 0.6, h: 0.55,
      fontSize: 15, fontFace: HFONT, color: D_TEXT, valign: "middle", margin: 0
    });
    s.addText("前端只展示 · 后端只读/受控写 · 与 CLI 同一 workspace，不另起一套库", {
      x: ML + 0.3, y: 1.65, w: CW - 0.6, h: 0.4,
      fontSize: 13, fontFace: BFONT, color: D_MUTED, margin: 0
    });

    // Backend card
    s.addShape(PRES.shapes.RECTANGLE, {
      x: ML, y: 2.4, w: 6.05, h: 4.35, fill: { color: L_CARD },
      line: { color: L_BORDER, width: 1 }, shadow: shadow()
    });
    s.addShape(PRES.shapes.RECTANGLE, {
      x: ML, y: 2.4, w: 6.05, h: 0.7, fill: { color: D_BG }, line: { type: "none" }
    });
    s.addImage({ data: ic.server, x: ML + 0.25, y: 2.55, w: 0.35, h: 0.35 });
    s.addText("后端  ·  FastAPI + uvicorn", {
      x: ML + 0.75, y: 2.4, w: 5.0, h: 0.7,
      fontSize: 15, fontFace: HFONT, color: D_TEXT, valign: "middle", margin: 0
    });
    s.addText("启动", {
      x: ML + 0.3, y: 3.3, w: 5.5, h: 0.3,
      fontSize: 12, fontFace: SFONT, bold: true, color: ACCENT, margin: 0
    });
    s.addShape(PRES.shapes.RECTANGLE, {
      x: ML + 0.3, y: 3.65, w: 5.45, h: 0.7, fill: { color: D_BG }, line: { type: "none" }
    });
    s.addText("pixi run serve          # → :8000\n# 或 uvicorn factorzen.server.api:app", {
      x: ML + 0.4, y: 3.7, w: 5.25, h: 0.6,
      fontSize: 12, fontFace: CFONT, color: "A8F0C0", margin: 0
    });
    s.addText("提供什么", {
      x: ML + 0.3, y: 4.5, w: 5.5, h: 0.3,
      fontSize: 12, fontFace: SFONT, bold: true, color: ACCENT, margin: 0
    });
    s.addText("• REST：/api/health · runs · library · ops · jobs\n• 扫 workspace 索引 manifest / 净值 / 因子库\n• 内置页：localhost:8000/ 与 /docs（OpenAPI）\n• 构建后挂载 webui/dist → /ui/", {
      x: ML + 0.3, y: 4.85, w: 5.5, h: 1.6,
      fontSize: 13, fontFace: BFONT, color: L_SUB, margin: 0
    });

    // Frontend card
    s.addShape(PRES.shapes.RECTANGLE, {
      x: ML + 6.25, y: 2.4, w: 6.05, h: 4.35, fill: { color: L_CARD },
      line: { color: L_BORDER, width: 1 }, shadow: shadow()
    });
    s.addShape(PRES.shapes.RECTANGLE, {
      x: ML + 6.25, y: 2.4, w: 6.05, h: 0.7, fill: { color: D_BG }, line: { type: "none" }
    });
    s.addImage({ data: ic.globe, x: ML + 6.5, y: 2.55, w: 0.35, h: 0.35 });
    s.addText("前端  ·  React + Ant Design + ECharts", {
      x: ML + 7.0, y: 2.4, w: 5.0, h: 0.7,
      fontSize: 14, fontFace: HFONT, color: D_TEXT, valign: "middle", margin: 0
    });
    s.addText("开发启动", {
      x: ML + 6.55, y: 3.3, w: 5.5, h: 0.3,
      fontSize: 12, fontFace: SFONT, bold: true, color: ACCENT, margin: 0
    });
    s.addShape(PRES.shapes.RECTANGLE, {
      x: ML + 6.55, y: 3.65, w: 5.45, h: 0.7, fill: { color: D_BG }, line: { type: "none" }
    });
    s.addText("cd webui && npm install && npm run dev\n# Vite 代理 /api → :8000 · base=/ui/", {
      x: ML + 6.65, y: 3.7, w: 5.25, h: 0.6,
      fontSize: 12, fontFace: CFONT, color: "A8F0C0", margin: 0
    });
    s.addText("页面能力（webui/src/pages）", {
      x: ML + 6.55, y: 4.5, w: 5.5, h: 0.3,
      fontSize: 12, fontFace: SFONT, bold: true, color: ACCENT, margin: 0
    });
    s.addText("• Overview / Runs · 因子库 Library\n• 报告 Reports · 运营 Ops · 任务 Jobs\n• 策略 / 文件浏览等\n• 图表：IC、净值等与 CLI 同源数据", {
      x: ML + 6.55, y: 4.85, w: 5.5, h: 1.6,
      fontSize: 13, fontFace: BFONT, color: L_SUB, margin: 0
    });
  }

  // ========== 15 技术栈 ==========
  {
    const s = pres.addSlide();
    s.background = { color: L_BG };
    addHeader(s, "04  开发技术", "技术栈（与 multi-agent 相关的优先说）", 15);
    addFooter(s, "环境：pixi  ·  质量：ruff / mypy / 952 pytest");

    const stacks = [
      { t: "Agent 与 LLM", items: [
        ["OpenAI-compatible SDK", "统一网关调模型"],
        ["roles/*.py", "四角色 prompt 与解析"],
        ["team_orchestrator", "编排 + 否决回路"],
        ["experiment_index", "跨 session 记忆"]
      ]},
      { t: "挖掘与求值", items: [
        ["expression AST", "表达式双向编译"],
        ["operators + leaves", "白名单算子特征"],
        ["evaluate + guardrails", "确定性打分"],
        ["holdout / DSR / PBO", "防过拟合原语"]
      ]},
      { t: "数据与数值", items: [
        ["polars / pyarrow", "大面板 + parquet"],
        ["Tushare 等", "行情接入"],
        ["numpy / scipy", "统计计算"],
        ["PIT 数据层", "防时间泄漏"]
      ]},
      { t: "前后端 Web", items: [
        ["FastAPI server/", "REST + 产物索引"],
        ["React webui/", "AntD · ECharts · Vite"],
        ["pixi run serve", "后端 :8000"],
        ["npm run dev / build", "前端 /ui 或 dist"]
      ]},
    ];
    stacks.forEach((st, i) => {
      const x = ML + i * 3.15;
      s.addShape(PRES.shapes.RECTANGLE, {
        x, y: 1.2, w: 3.0, h: 5.5, fill: { color: L_CARD },
        line: { color: L_BORDER, width: 1 }, shadow: shadow()
      });
      s.addShape(PRES.shapes.RECTANGLE, {
        x, y: 1.2, w: 3.0, h: 0.7, fill: { color: D_BG }, line: { type: "none" }
      });
      s.addText(st.t, {
        x: x + 0.2, y: 1.2, w: 2.6, h: 0.7,
        fontSize: 14, fontFace: HFONT, color: D_TEXT, valign: "middle", margin: 0
      });
      st.items.forEach((it, j) => {
        const y = 2.2 + j * 1.0;
        s.addText(it[0], {
          x: x + 0.2, y: y, w: 2.6, h: 0.35,
          fontSize: 13, fontFace: SFONT, bold: true, color: L_TEXT, margin: 0
        });
        s.addText(it[1], {
          x: x + 0.2, y: y + 0.35, w: 2.6, h: 0.35,
          fontSize: 12, fontFace: BFONT, color: L_MUTED, margin: 0
        });
      });
    });
  }

  // ========== 16 代码地图 ==========
  {
    const s = pres.addSlide();
    s.background = { color: L_BG };
    addHeader(s, "04  开发技术", "源码地图：讲 multi-agent 时打开这些文件", 16);
    addFooter(s, "紫色左边条 = 今天演示优先路径");

    const packs = [
      { p: "agents/team_orchestrator.py", d: "★ 多角色编排与否决回路", hot: true },
      { p: "agents/roles/hypothesis.py", d: "提方向 · 注入记忆与信号族", hot: true },
      { p: "agents/roles/coder.py", d: "方向→表达式 · 修订 · 自愈", hot: true },
      { p: "agents/roles/critic.py", d: "读指标给 keep/revise/drop", hot: true },
      { p: "agents/roles/librarian.py", d: "跨 session recall / record", hot: true },
      { p: "pipelines/factor_mine_team.py", d: "CLI 入口 pipeline 与产物落盘", hot: true },
      { p: "discovery/lift_test.py", d: "增量裁决（session 末共用）", hot: false },
      { p: "discovery/expression.py", d: "AST 解析 · 结构签名", hot: false },
      { p: "validation/*", d: "holdout / DSR / PBO 原语", hot: false },
      { p: "llm/client.py", d: "LLM 网关与配置", hot: false },
      { p: "server/api.py", d: "FastAPI 路由 · 挂载 /ui 静态", hot: false },
      { p: "webui/src/pages/*", d: "React 页面：库 / runs / ops", hot: false },
    ];
    packs.forEach((pk, i) => {
      const col = i % 3, row = Math.floor(i / 3);
      const x = ML + col * 4.2, y = 1.2 + row * 1.35;
      s.addShape(PRES.shapes.RECTANGLE, {
        x, y, w: 4.0, h: 1.2, fill: { color: L_CARD },
        line: { color: L_BORDER, width: 1 }, shadow: shadow()
      });
      s.addShape(PRES.shapes.RECTANGLE, {
        x, y, w: 0.08, h: 1.2,
        fill: { color: pk.hot ? ACCENT : "C8C4E8" }, line: { type: "none" }
      });
      s.addText(pk.p, {
        x: x + 0.25, y: y + 0.2, w: 3.55, h: 0.35,
        fontSize: 12, fontFace: CFONT, bold: true, color: pk.hot ? ACCENT : L_TEXT, margin: 0
      });
      s.addText(pk.d, {
        x: x + 0.25, y: y + 0.6, w: 3.55, h: 0.4,
        fontSize: 13, fontFace: BFONT, color: L_MUTED, margin: 0
      });
    });
  }

  // ========== 17 关键代码 ==========
  {
    const s = pres.addSlide();
    s.background = { color: L_BG };
    addHeader(s, "04  开发技术", "关键代码语义：一轮 multi-agent（伪代码）", 17);
    addFooter(s, "真实实现含并发 llm_workers、模板配额、结构化假设等；此处只保留主线");

    s.addShape(PRES.shapes.RECTANGLE, {
      x: ML, y: 1.15, w: CW, h: 5.7, fill: { color: D_BG }, line: { type: "none" }, shadow: shadow()
    });
    s.addText("agents/team_orchestrator.py  ·  主线语义", {
      x: ML + 0.35, y: 1.35, w: CW - 0.7, h: 0.35,
      fontSize: 13, fontFace: SFONT, color: ACCENT2, margin: 0
    });
    s.addText(
      [
        { text: "for round in range(n_rounds):", options: { breakLine: true } },
        { text: "    memory = Librarian.recall(index)          # 旧账：挖穿/未探索/拒过", options: { breakLine: true } },
        { text: "    hyps = Hypothesis.propose(memory)        # 提方向（可 structured）", options: { breakLine: true } },
        { text: "    exprs = Coder.write(hyps)                # 白名单语法 → 表达式", options: { breakLine: true } },
        { text: "    scored = Evaluator.run(exprs)            # 确定性：IC/holdout/护栏", options: { breakLine: true } },
        { text: "    for cand in scored:", options: { breakLine: true } },
        { text: "        v = Critic.critique(cand)            # keep|revise_*|drop", options: { breakLine: true } },
        { text: "        if v == revise_expr:   Coder.fix(...)", options: { breakLine: true } },
        { text: "        if v == revise_hypothesis: 重新提案", options: { breakLine: true } },
        { text: "        if v == keep:  进入候选池", options: { breakLine: true } },
        { text: "", options: { breakLine: true } },
        { text: "Librarian.record(session_results)              # 写回长期记忆", options: { breakLine: true } },
        { text: "optional: lift_admission(candidates, library)  # 组增量裁决", options: { breakLine: true } },
        { text: "dump_manifest(...)                             # 可回放", options: {} },
      ],
      {
        x: ML + 0.45, y: 1.9, w: CW - 0.9, h: 4.6,
        fontSize: 15, fontFace: CFONT, color: "C8D0DC", margin: 0, lineSpacingMultiple: 1.22
      }
    );
  }

  // ========== 18 三条铁律（压缩） ==========
  {
    const s = pres.addSlide();
    s.background = { color: L_BG };
    addHeader(s, "04  贯穿约束", "三条铁律如何约束 multi-agent", 18);
    addFooter(s, "Agent 再聪明也不能违反这些——它们写在数据层与筛选流程里");

    const laws = [
      { n: "01", t: "不许偷看未来", d: "Agent 生成的公式再花哨，数据层仍保证 t 日只用 ≤t 信息。财务按公告日、名单逐日快照。" },
      { n: "02", t: "护栏真拦截", d: "Evaluator / Critic / DSR / holdout 会改变去留，不是报告装饰。CI 红了就不能 merge 的同一逻辑。" },
      { n: "03", t: "可复现", d: "每轮 manifest、session 产物、git_sha、seed、窗口。LLM 有随机性，但输入配置与代码版本可追溯。" },
    ];
    laws.forEach((l, i) => {
      const y = 1.25 + i * 1.75;
      s.addShape(PRES.shapes.RECTANGLE, {
        x: ML, y, w: CW, h: 1.55, fill: { color: L_CARD },
        line: { color: L_BORDER, width: 1 }, shadow: shadow()
      });
      s.addShape(PRES.shapes.RECTANGLE, {
        x: ML, y, w: 1.2, h: 1.55, fill: { color: D_BG }, line: { type: "none" }
      });
      s.addText(l.n, {
        x: ML, y, w: 1.2, h: 1.55,
        fontSize: 22, fontFace: HFONT, color: ACCENT2, align: "center", valign: "middle", margin: 0
      });
      s.addText(l.t, {
        x: ML + 1.5, y: y + 0.25, w: 10, h: 0.4,
        fontSize: 18, fontFace: HFONT, color: L_TEXT, margin: 0
      });
      s.addText(l.d, {
        x: ML + 1.5, y: y + 0.75, w: 10, h: 0.55,
        fontSize: 14, fontFace: BFONT, color: L_MUTED, margin: 0
      });
    });
  }

  // ========== 19 总结 ==========
  {
    const s = pres.addSlide();
    s.background = { color: L_BG };
    addHeader(s, "05  总结", "交付 · 边界 · 下一步", 19);
    addFooter(s, "诚实披露边界，是工程可信度的一部分");

    const cols = [
      {
        t: "今天强调的交付", color: ACCENT,
        items: [
          "多 Agent 四角色 + 确定性评估器",
          "否决回路与跨 session 记忆",
          "生成与验收分离",
          "session 末 Lift 共用裁决函数",
          "CLI 产物可审计",
          "整体平台：库 / 组合 / 报告 / Web"
        ]
      },
      {
        t: "已知边界", color: "B8860B",
        items: [
          "依赖外部 LLM 网关与配额",
          "实盘下单未接入（纸面撮合）",
          "期货 / 美股链路未全通",
          "Web 无鉴权（dev）",
          "组合优化相对挖掘侧偏薄"
        ]
      },
      {
        t: "后续", color: "2E7D4F",
        items: [
          "记忆与 campaign 策略继续打磨",
          "lift 阈值 null 校准",
          "日链路接入 forward 确认",
          "Web 生产化",
          "券商接口（形状已预留）"
        ]
      },
    ];
    cols.forEach((c, i) => {
      const x = ML + i * 4.2;
      s.addShape(PRES.shapes.RECTANGLE, {
        x, y: 1.2, w: 4.0, h: 5.5, fill: { color: L_CARD },
        line: { color: L_BORDER, width: 1 }, shadow: shadow()
      });
      s.addShape(PRES.shapes.RECTANGLE, {
        x, y: 1.2, w: 4.0, h: 0.7, fill: { color: D_BG }, line: { type: "none" }
      });
      s.addShape(PRES.shapes.RECTANGLE, {
        x, y: 1.2, w: 0.1, h: 0.7, fill: { color: c.color }, line: { type: "none" }
      });
      s.addText(c.t, {
        x: x + 0.25, y: 1.2, w: 3.5, h: 0.7,
        fontSize: 15, fontFace: HFONT, color: D_TEXT, valign: "middle", margin: 0
      });
      c.items.forEach((item, j) => {
        s.addText("▸  " + item, {
          x: x + 0.25, y: 2.2 + j * 0.65, w: 3.5, h: 0.55,
          fontSize: 13, fontFace: BFONT, color: L_SUB, margin: 0
        });
      });
    });
  }

  // ========== 20 Q&A ==========
  {
    const s = pres.addSlide();
    s.background = { color: D_BG };
    s.addImage({ path: logoDark, x: (SW - 0.8) / 2, y: 0.7, w: 0.8, h: 0.8 });
    s.addText("谢谢 · 欢迎提问", {
      x: ML, y: 1.7, w: CW, h: 0.55,
      fontSize: 32, fontFace: HFONT, color: D_TEXT, align: "center", margin: 0
    });
    addRule(s, (SW - 2.2) / 2, 2.4, 2.2, ACCENT);
    s.addText("FactorZen  ·  用多 Agent 挖特征，用门禁保证增量", {
      x: ML, y: 2.65, w: CW, h: 0.4,
      fontSize: 15, fontFace: BFONT, color: D_LIGHT, align: "center", margin: 0
    });

    const qs = [
      { t: "核心命令", d: "fz mine team\n--structured · llm_workers" },
      { t: "核心代码", d: "agents/team_orchestrator.py\nagents/roles/*" },
      { t: "老师常问", d: "为何拆多角色？\n→ 分工 + 对抗 + 记忆" },
    ];
    qs.forEach((q, i) => {
      const x = ML + 1.3 + i * 3.7;
      s.addShape(PRES.shapes.RECTANGLE, {
        x, y: 3.5, w: 3.4, h: 2.2, fill: { color: D_CARD }, line: { type: "none" }
      });
      s.addText(q.t, {
        x: x + 0.25, y: 3.75, w: 2.9, h: 0.4,
        fontSize: 14, fontFace: HFONT, color: ACCENT2, margin: 0
      });
      s.addText(q.d, {
        x: x + 0.25, y: 4.35, w: 2.9, h: 1.1,
        fontSize: 14, fontFace: BFONT, color: D_MUTED, margin: 0
      });
    });
    s.addText("v0.3.0  ·  docs/presentations/", {
      x: ML, y: 6.6, w: CW, h: 0.3,
      fontSize: 12, fontFace: SFONT, color: D_DIM, align: "center", margin: 0
    });
  }

  const out = path.join(__dirname, "FactorZen_答辩讲解.pptx");
  await pres.writeFile({ fileName: out });
  console.log("Wrote:", out);
}

buildDeck().catch((e) => {
  console.error(e);
  process.exit(1);
});
