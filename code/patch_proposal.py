# -*- coding: utf-8 -*-
"""patch_proposal.py —— 在【你已排好版的 格式化版】上做增量修改

底板：C:\\Users\\21912\\Downloads\\开题报告_王阳_..._格式化版.docx
原则：
  1. 不重新生成整篇文档，只在你原有段落上做“删一段 / 改一句 / 插新段”；
  2. 新段落一律深拷贝你文档里的原型段落（正文 / 小标题 / 图题 / 图片），
     因此字体、字号、行距、首行缩进、对齐方式与你的排版完全一致；
  3. 表格（表1—表6）与参考文献（24 条）一个字节都不动。
输出：开题报告_王阳_..._格式化版_v2.docx（工作区根目录，不覆盖 Downloads 里的原件）

本次新增内容（对应上一轮 8 条意见）：
  二、国内外研究现状综述（新）
  三、本课题重点、难点与创新点（新）
  五、技术路线（新，含图5）
  六、可行性分析（新）
  七、预期成果（新）
  工作进度安排：7 条 → 10 条
改动内容：
  一、删去原“研究现状”段（已单列为第二节），并把下一段开头改成“针对上述工程需求”；
  二、第二节标题改为“四、设计主要内容及方案”，第三节标题改为“八、参考文献”；
  三、3.1 数据来源后补一段（43 条主口径事件与划分依据的说明）；
  四、图5 技术路线图。
"""
import copy
import os
import re
from xml.sax.saxutils import escape

from docx import Document
from docx.shared import Cm
from docx.oxml.ns import qn
from docx.oxml import OxmlElement, parse_xml
from docx.text.paragraph import Paragraph

BASE = r"C:\Users\21912\Downloads\开题报告_王阳_风电功率爬坡事件预测与预警系统设计与实现_格式化版.docx"
OUT = "开题报告_王阳_风电功率爬坡事件预测与预警系统设计与实现_格式化版_v3.docx"
FIG5 = os.path.join("output", "figs", "fig5_route.png")

W = qn("w:t")

# ---------------------------------------------------------------------------
# OMML（Word 公式）构造：把易乱码的符号转成公式对象
# ---------------------------------------------------------------------------
MATH_NS = ('xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math" '
           'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"')


def _mr(text):
    return ('<m:r><w:rPr><w:rFonts w:ascii="Cambria Math" w:hAnsi="Cambria Math"/>'
            '<w:sz w:val="21"/><w:szCs w:val="21"/></w:rPr>'
            '<m:t xml:space="preserve">%s</m:t></m:r>' % escape(text))


def _macc(base, ch):
    return ('<m:acc><m:accPr><m:chr m:val="%s"/></m:accPr><m:e>%s</m:e></m:acc>'
            % (ch, _mr(base)))


def _msup(base, sup):
    return '<m:sSup><m:e>%s</m:e><m:sup>%s</m:sup></m:sSup>' % (_mr(base), _mr(sup))


def _mwrap(inner):
    return parse_xml('<m:oMath %s>%s</m:oMath>' % (MATH_NS, inner))


MACRON = "\u0304"
HAT = "\u0302"

MATH = {
    "ΔP=F̄·ΔC+C̄·ΔF": lambda: _mwrap(_mr("ΔP=") + _macc("F", MACRON) + _mr("·ΔC+")
                                     + _macc("C", MACRON) + _mr("·ΔF")),
    "ΔP̂": lambda: _mwrap(_mr("Δ") + _macc("P", HAT)),
    "P̂": lambda: _mwrap(_macc("P", HAT)),
    "F̄": lambda: _mwrap(_macc("F", MACRON)),
    "C̄": lambda: _mwrap(_macc("C", MACRON)),
    "2.84×10⁻¹⁴": lambda: _mwrap(_mr("2.84×") + _msup("10", "−14")),
    "7×10⁻⁶": lambda: _mwrap(_mr("7×") + _msup("10", "−6")),
    "Δt": lambda: _mwrap(_mr("Δt")),
    "τ*": lambda: _mwrap(_mr("τ*")),
}


def convert_math(par):
    """把段落中的目标符号替换为 OMML 公式对象，保留段落格式与首个 run 的字符格式。"""
    txt = par.text
    keys = [k for k in MATH if k in txt]
    if not keys:
        return 0
    keys.sort(key=len, reverse=True)
    pat = re.compile("(" + "|".join(re.escape(k) for k in keys) + ")")
    parts = pat.split(txt)
    runs = par._p.findall(qn("w:r"))
    rpr = runs[0].find(qn("w:rPr")) if runs else None
    for child in list(par._p):
        if child.tag != qn("w:pPr"):
            par._p.remove(child)
    n = 0
    for part in parts:
        if not part:
            continue
        if part in MATH:
            par._p.append(MATH[part]())
            n += 1
        else:
            r = OxmlElement("w:r")
            if rpr is not None:
                r.append(copy.deepcopy(rpr))
            t = OxmlElement("w:t")
            t.set(qn("xml:space"), "preserve")
            t.text = part
            r.append(t)
            par._p.append(r)
    return n


def _insert_ordered(trPr, el, before_tags=(), after_tags=()):
    for tag in before_tags:
        node = trPr.find(qn(tag))
        if node is not None:
            node.addprevious(el)
            return
    for tag in after_tags:
        node = trPr.find(qn(tag))
        if node is not None:
            node.addnext(el)
            return
    trPr.append(el)


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------
def set_text(par_el, text):
    """把段落元素的文字清空后写入 text，保留第一个 run 的字符格式。"""
    runs = par_el.findall(qn("w:r"))
    if not runs:
        r = OxmlElement("w:r")
        t = OxmlElement("w:t")
        t.set(qn("xml:space"), "preserve")
        t.text = text
        r.append(t)
        par_el.append(r)
        return par_el
    keep = runs[0]
    for r in runs[1:]:
        par_el.remove(r)
    for child in list(keep):
        if child.tag != qn("w:rPr"):
            keep.remove(child)
    t = OxmlElement("w:t")
    t.set(qn("xml:space"), "preserve")
    t.text = text
    keep.append(t)
    return par_el


def clone(proto_par, text):
    new = copy.deepcopy(proto_par._p)
    return set_text(new, text)


def insert_after(anchor_el, new_el):
    anchor_el.addnext(new_el)
    return new_el


def fill_cell(cell, blocks, proto):
    """清空单元格并按 blocks 顺序写入；blocks: [(kind, text), ...]"""
    for p in list(cell.paragraphs):
        p._p.getparent().remove(p._p)
    anchor = None
    for kind, text in blocks:
        if kind == "img":
            el = copy.deepcopy(proto["img"]._p)
            for r in el.findall(qn("w:r")):
                el.remove(r)
            cell._tc.append(el)
            par = Paragraph(el, cell)
            par.add_run().add_picture(text, width=Cm(11.5))
            anchor = el
            continue
        el = clone(proto[kind], text)
        if anchor is None:
            cell._tc.append(el)
        else:
            anchor.addnext(el)
        anchor = el


# ---------------------------------------------------------------------------
# 新增章节内容（按你文档的排版习惯：中英文/数字之间不留空格）
# ---------------------------------------------------------------------------
SEC2 = [
    ("h2", "（1）爬坡事件的界定与识别。"),
    ("body", "国际上通常从幅度、持续时间与时间窗三个要素刻画爬坡，阈值多取装机容量的10%—20%，时间窗取10min至4h；识别算法以旋转门算法（Swinging Door Algorithm，SDA）及其改进、滑动窗双边CUSUM、分位数阈值以及基于气象过程判定的方法为主[3][6][14]，也有研究依据风险价值（VaR）自适应确定阈值[18]。这类方法原理直观、计算量小、可解释性强，便于现场部署；不足是阈值高度依赖经验与装机规模，对噪声与缺失敏感，且只能在事件发生后完成识别，属“事后检测”，难以为调度预留决策时间。"),
    ("h2", "（2）爬坡事件的预测方法。"),
    ("body", "预测研究经历了由统计模型向机器学习、深度学习演进的过程。早期以时间序列与统计回归为主；随着梯度提升树（XGBoost、LightGBM）与深度网络（LSTM、TCN、Transformer）的引入，功率点预测精度显著提升[1][4][5][10][17]；近年进一步向集成预测与误差修正[8][12]、相似爬坡事件修正[13]、区间预测[2]以及概念漂移条件下的在线预测[16]延伸，也有工作将功率与爬坡率联合建模并同时给出确定性与概率性结果[19]。优点是通过多变量与长历史窗口挖掘非线性时序依赖，小时级精度可观；不足是多数工作只输出功率点值，方向识别与事件级检出能力弱；深度学习在小样本、强噪声数据上易过拟合——本文实测LSTM与TCN的三分类准确率仅0.470与0.537，明显低于LightGBM的0.735，说明“模型越深越好”在本任务上并不成立。"),
    ("h2", "（3）概率预测与校准。"),
    ("body", "为刻画不确定性，概率预测与区间预测成为研究热点：分位数回归、集合预报以及基于扩散模型与爬坡趋势分类的自适应区间预测[2]等方法相继提出，评价多用区间覆盖率、可靠性图与Brier分数。其价值在于把“是否发生”扩展为“以多大概率发生”；不足有二：一是概率模型往往需要分位数损失或生成模型，训练成本较高；二是概率与告警阈值、告警等级之间缺少显式衔接，概率值难以直接转化为运行人员可执行的预警动作。"),
    ("h2", "（4）预警决策与运行应用。"),
    ("body", "在决策层面，代价敏感学习与代价曲线被用于阈值选择；国外独立系统运营商（如CAISO）引入灵活爬坡产品（FRP）应对净负荷爬坡；国内研究把爬坡事件纳入运行风险评估[15]、联络线功率平滑[9]与极端天气下的调度决策[11]。优点是把预测结果推进到决策层，显式权衡漏警与虚警的经济代价；不足是输入多为点预测或场景集，缺少“事件级”评价口径（命中率、漏警率、虚警率、平均提前量），也少见把识别、预测、决策与可视化集成到一套可运行系统的完整实现。"),
    ("h2", "（5）研究缺口与本课题定位。"),
    ("body", "综上，现有研究可归纳为三点不足：其一，识别与预测割裂，多数工作或只做事件检测、或只做功率点预测，缺少“识别—预测—预警”一体化的贯通设计；其二，概率输出与预警阈值、等级之间缺少代价敏感的衔接，漏警与虚警难以事前权衡；其三，成果多停留在算法验证，缺少含数据库、后端接口与可视化界面的系统工程实现。本课题正是针对这三点展开：以统一口径贯通事件识别与预测，用残差经验分布实现零额外训练成本的概率化，用代价敏感阈值与迟滞把概率转成分级预警，并落地为含SQLite数据库、REST接口与双模式前端的可运行系统。"),
]

SEC3 = [
    ("h2", "3.1研究重点"),
    ("body", "（1）统一口径下的数据与事件基础。确定Δt=1h、θ=20%、L=1h、H=2h的唯一口径，完成清洗、停机判据唯一化、事件检测与合并、三级分级与成因归因，形成可复现的事件清单。"),
    ("body", "（2）可对比的模型体系。以LightGBM为主模型，横向对比XGBoost、随机森林、逻辑回归基线与LSTM、TCN深度学习模型，并实现“深度学习隐层特征＋机器学习预测”的级联组合，保证结论可对比、可复现。"),
    ("body", "（3）事件级评价与代价敏感决策。以命中率、漏警率、虚警率、平均提前量为主指标构建事件级评价脚本；在代价比1:1、5:1、10:1下扫描最优阈值τ*，叠加事件合并与迟滞，给出推荐操作点。"),
    ("body", "（4）一体化系统落地。以SQLite存储数据与告警记录，REST接口提供数据查询与在线推理，Bootstrap+ECharts仪表盘与Streamlit双模式界面提供可视化与历史检索，形成“数据—模型—决策—界面”闭环。"),
    ("h2", "3.2主要难点"),
    ("body", "（1）SDWPF数据处理与口径统一。原始数据存在风速与功率同步缺失49,518条（1.047%）、功率负值1,263,062条（26.72%）、停机点301,909个（6.39%）、Day66/67整日缺失以及每日00:00集中缺测等问题，需要按“机组＋按天”分段插值并保留掩码；事件合并还需避免“一次长爬坡刷出多条告警”。"),
    ("body", "（2）代价敏感阈值标定。阈值τ的搜索与代价比强耦合，测试期主口径事件仅43条，指标随τ呈非单调波动，需采用全网格扫描＋归一化代价比较，避免单点阈值过拟合。"),
    ("body", "（3）概率预测校准。验证集残差存在+2.754MW的系统性低估，若不做偏置校正，超阈概率将整体偏移；验证集与测试集误差分布差异还会造成区间覆盖率0.8041与0.9009的偏差，需要在论文中如实解释。"),
    ("body", "（4）事件级指标评估。告警时刻按提前量L平移、告警与真实事件一对一匹配、提前量符号定义，任一处口径错误都会使指标失真（项目前期即出现过提前量符号错误）。"),
    ("body", "（5）小样本与概念漂移。测试期上爬坡数量随Day推进由338衰减至99，导致方向识别与长期稳定性受限，需在论文中作为独立小节讨论。"),
    ("h2", "3.3创新点"),
    ("body", "（1）识别—预测—预警一体化贯通。在同一口径下把事件检测、概率预测、代价敏感决策与可视化串成闭环，并以纯函数实现的事件级评价脚本（配套9个单元测试）先于模型交付；配套Bootstrap+ECharts仪表盘与Streamlit双模式界面，两者与SQLite、REST接口、大模型解读共用同一套按“步数”表达的决策层参数，保证离线实验与演示逻辑完全等价。"),
    ("body", "（2）成因归因分解划清责任边界。利用ΔP=F̄·ΔC+C̄·ΔF的精确分解（实测最大残差2.84×10⁻¹⁴MW）把事件分为气象型、工况型与混合型，实测气象型占95.97%，据此把预警指标建立在气象型事件上、对工况型事件降级处理，避免模型为“不可避免的漏警”背锅。"),
    ("body", "（3）零额外训练成本的概率化预警。不引入分位数模型，直接复用验证集残差经验分布给出P10/P50/P90区间与超阈概率，验证集区间覆盖率0.8041（目标0.80）、Brier分数0.028—0.052，实现“低成本、可解释、可校准”的概率输出。"),
    ("body", "（4）代价敏感的分级预警决策。以τ*=argmin(C_miss·漏警数+C_false·虚警数)标定阈值（1:1/5:1取0.60，10:1取0.20—0.25），叠加迟滞后归一化代价分别下降0.186/0.093/0.256，把“漏警虚警分析”从事后解释升级为事前优化。"),
]

SEC5 = [
    ("body", "技术路线如图5所示，按“数据—算法—决策—系统—验证”五个阶段推进，上一阶段的输出即下一阶段的输入，任一阶段的产物均可复现、可追溯。"),
    ("body", "第一阶段为数据层：读入SDWPF原始数据，完成缺失插值、负值置零、停机判据与按时间划分，随后按统一口径检测爬坡事件、合并同向触发点、标注三级等级，并用成因分解输出事件清单与逐点标签；该阶段冻结口径与事件清单，后续实验不再变动。"),
    ("body", "第二阶段为算法层：在14维特征与72步滑窗上构建378维输入，先以LightGBM建立回归与分类主模型，再横向对比XGBoost、随机森林、逻辑回归与LSTM、TCN，并实现深度学习隐层特征与手工特征拼接的级联组合；模型冻结后进入概率化，用验证集残差经验分布给出区间与超阈概率，并以可靠性图与Brier分数检验校准。"),
    ("body", "第三阶段为决策层：以代价敏感方式扫描阈值得到τ*，叠加事件合并与迟滞，输出方向、等级、概率与提前量，并用事件级指标评价决策效果。"),
    ("body", "第四阶段为系统层：把上述结果落库到SQLite（5张表），通过REST接口对外提供时序、概率、事件、告警查询与在线推理，前端以Bootstrap+ECharts仪表盘负责查询展示、以Streamlit负责离线回放与实时模拟，并接入大模型生成预警解读。"),
    ("body", "第五阶段为验证层：开展功能测试、在线与离线一致性测试、事件级回测与提前量敏感性实验，并完成论文撰写。需要强调，技术路线以软件为主线，Arduino采集只是系统层中的可选演示分支，用于验证“采集—推理—预警”链路，数据也可由串口模拟器按真实特征回放产生，不构成独立的硬件研究内容。"),
]

SEC6 = [
    ("h2", "6.1理论可行性"),
    ("body", "爬坡事件的定义沿用国际通用的“幅度—时间窗”口径，阈值取装机容量比例，具备明确的物理含义；LightGBM梯度提升树、TCN/LSTM时序网络、残差经验分布概率化与代价敏感决策均有成熟的理论与文献支撑[2][4][17][20][21][22][23]；事件级评价指标（命中率、漏警率、虚警率、平均提前量）在爬坡预测研究中被广泛采用，口径可对照。因此本课题的技术方案在理论上有据可依，结论可解释、可复核。"),
    ("h2", "6.2数据可行性"),
    ("body", "SDWPF为KDD Cup 2022公开数据集，可免费下载，含134台机组连续245天、10min分辨率的完整运行与气象记录，共472万条，无版权与隐私风险；事件量充足（主口径752条，约3.07次/天），正负样本比例可支撑训练与评价。数据规模适中，单机即可完成全部实验，不依赖外部数据接口。"),
    ("h2", "6.3技术可行性"),
    ("body", "开发所用工具链均为成熟开源方案：Python生态（pandas、NumPy、LightGBM、PyTorch）稳定，SQLite零部署，Streamlit与Bootstrap+ECharts均为成熟的Web可视化方案，REST接口可用标准库实现并可按需平滑替换为FastAPI，大模型解释模块配有规则模板兜底，无外网亦可运行。相关环境已在个人电脑（RTX 4060）上完成验证：模型训练单次数分钟级，接口与前端冒烟测试通过，在线推理与离线预测的最大偏差为7×10⁻⁶MW。Arduino部分只做串口协议与链路演示，不涉及真实风场硬件改造，技术上不构成风险。"),
    ("h2", "6.4时间可行性"),
    ("body", "按任务书14周安排，数据与事件约2周、模型与实验约4周、决策层约1周、系统开发与测试约4周、论文与答辩约3周，关键路径为“口径冻结→模型对比→系统集成”，并预留约2周缓冲。主要开发工作已提前完成，后续以完善与撰写为主，时间风险可控。"),
]

SEC7 = [
    ("body", "（1）完整可运行的风电功率爬坡事件预测与预警系统源代码。包含数据清洗与事件识别、模型训练与在线推理、SQLite数据库、REST接口、Bootstrap+ECharts仪表盘与Streamlit双模式界面，可在Windows+Python 3.12环境下直接运行与演示。"),
    ("body", "（2）数据库文件wind_ramp.db。含场站时序、爬坡事件、预测概率、预警记录、运行元数据5张表及索引，支持历史告警的条件查询与命中回填。"),
    ("body", "（3）测试与实验报告。包括功能测试（数据库、接口、前端共34个用例）、模型对比实验（LightGBM/XGBoost/随机森林/逻辑回归/LSTM/TCN/级联）、概率校准（可靠性图、Brier分数）、事件级回测（命中率、漏警率、虚警率、提前量）与提前量敏感性实验，均附可复现脚本。"),
    ("body", "（4）毕业设计说明书（论文）。约2万字，含绪论、理论基础与统一口径、数据与事件、预测与决策、系统实现、总结与展望六章，并附系统使用说明书与软件设计文档。"),
    ("body", "（5）答辩演示材料。包括仪表盘录屏、串口模拟链路演示与关键实验结果图表，用于说明系统的工程完整性与预警效果。"),
]

SCHEDULE = [
    "(1)第1周2026年3月2日-3月7日 调研与需求分析：完成文献调研与需求清单，确定统一口径初稿。",
    "(2)第2周2026年3月8日-3月14日 系统设计：完成用例图、功能模块图与数据库设计；3月10日前提交开题报告。",
    "(3)第3周-第4周2026年3月15日-3月28日 数据与事件：完成数据清洗、停机判据、事件检测与成因归因，冻结数据集与事件清单。",
    "(4)第5周-第6周2026年3月29日-4月11日 模型实验：完成LightGBM主模型、XGBoost/随机森林/逻辑回归基线、LSTM/TCN对比与级联组合，冻结模型。",
    "(5)第7周2026年4月12日-4月18日 概率化与决策层：完成残差概率化与校准检验、代价敏感阈值标定、迟滞与三级分级。",
    "(6)第8周2026年4月19日-4月25日 中期检查：提交中期报告，演示离线原型系统。",
    "(7)第9周-第10周2026年4月26日-5月9日 系统开发：完成SQLite建库、REST接口、Bootstrap+ECharts仪表盘、Streamlit双模式与大模型解读；5月10日前提交说明书初稿。",
    "(8)第11周-第12周2026年5月10日-5月23日 系统测试与论文完善：完成功能与一致性测试、事件级回测、提前量敏感性实验；5月21日完成说明书终稿。",
    "(9)第13周2026年5月24日-5月30日 学术不端行为检测（5月24日-5月26日）与指导教师、评阅人审阅。",
    "(10)第14周2026年5月31日-6月5日 答辩与材料归档（源码、数据库、说明书、演示素材）。",
]


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main():
    doc = Document(BASE)
    table = doc.tables[0]

    row4 = table.rows[4]
    row5 = table.rows[5]
    row6 = table.rows[6]
    row7 = table.rows[7]
    cell4 = row4.cells[0]
    cell5 = row5.cells[0]

    # 原型段落：全部取自你自己的文档
    proto = {
        "h1": cell4.paragraphs[0],          # 一、题目的意义、目的：
        "body": cell4.paragraphs[1],        # 正文（首行缩进 2 字、22 磅行距）
        "h2": cell5.paragraphs[1],          # 1、系统需求分析
        "figcap": cell5.paragraphs[43],     # 图4 系统总体架构图
        "img": cell5.paragraphs[42],        # 图4 图片段
    }

    # ---- 1. 一、题目的意义、目的：删去“研究现状”段，并承接下一段 ----
    p3 = cell4.paragraphs[3]
    assert p3.text.startswith("围绕风电功率爬坡事件"), p3.text[:20]
    p3._p.getparent().remove(p3._p)
    p4 = cell4.paragraphs[3]           # 删除后原第 4 段前移
    assert p4.text.startswith("针对上述不足"), p4.text[:20]
    set_text(p4._p, p4.text.replace("针对上述不足", "针对上述工程需求", 1))

    # ---- 2. 3.1 数据来源后补一段（43 条事件与划分依据）----
    anchor = None
    for par in cell5.paragraphs:
        if par.text.startswith("实验数据采用SDWPF"):
            anchor = par
            break
    assert anchor is not None
    new = clone(proto["body"], "测试段位于数据末端，爬坡事件相对稀疏：起点落在测试区间的主口径事件仅43条，事件级指标因此在各阈值下存在一定波动；该稀疏性的成因及其对指标的影响将在论文第三章结合事件分布曲线专门说明。")
    insert_after(anchor._p, new)

    # ---- 3. 标题改写 ----
    set_text(cell5.paragraphs[0]._p, "四、设计主要内容及方案：")
    set_text(row6.cells[0].paragraphs[0]._p, "八、参考文献：")

    # ---- 4. 插入新行（2 行在第一节后，3 行在第四节后）----
    tr4 = row4._tr
    tr5 = row5._tr
    anchor_tr = tr4
    for _ in range(2):
        nt = copy.deepcopy(tr4)
        anchor_tr.addnext(nt)
        anchor_tr = nt
    anchor_tr = tr5
    for _ in range(3):
        nt = copy.deepcopy(tr4)
        anchor_tr.addnext(nt)
        anchor_tr = nt

    rows = table.rows
    assert len(rows) == 14, len(rows)

    # ---- 5. 逐节填充（第 4 行=一，5=二，6=三，7=四，8=五，9=六，10=七，11=八，12=进度）----
    fill_cell(rows[5].cells[0], [("h1", "二、国内外研究现状综述：")] + SEC2, proto)
    fill_cell(rows[6].cells[0], [("h1", "三、本课题重点、难点与创新点：")] + SEC3, proto)
    fill_cell(rows[8].cells[0], [("h1", "五、技术路线："), ("body", SEC5[0][1]),
                                 ("img", FIG5), ("figcap", "图5 技术路线图")] + SEC5[1:], proto)
    fill_cell(rows[9].cells[0], [("h1", "六、可行性分析：")] + SEC6, proto)
    fill_cell(rows[10].cells[0], [("h1", "七、预期成果：")] + SEC7, proto)

    # ---- 6. 进度安排：7 条 → 10 条 ----
    cell7 = rows[12].cells[0]
    paras = cell7.paragraphs
    assert paras[0].text.startswith("工作进度安排"), paras[0].text[:20]
    proto_sched = paras[1]           # 先取原型，再删旧条目
    for p in paras[1:]:
        p._p.getparent().remove(p._p)
    anchor = paras[0]._p
    for line in SCHEDULE:
        el = clone(proto_sched, line)
        insert_after(anchor, el)
        anchor = el

    # ---- 7. 图3：把“如图3所示”的引用句移到图片之前 ----
    ps = cell5.paragraphs
    idx_ref = next(i for i, p in enumerate(ps) if p.text.startswith("系统功能模块图如图3所示"))
    idx_cap = next(i for i, p in enumerate(ps) if p.text.startswith("图3 系统功能模块图"))
    ps[idx_cap - 1]._p.addprevious(ps[idx_ref]._p)   # 引用句 → 图 → 图题

    # ---- 8. 补两句限定说明（需求 6）、大模型 7））----
    for par in cell5.paragraphs:
        if par.text.startswith("6）实时采集与联调模块："):
            set_text(par._p, par.text + "本课题的研究对象为软件预警系统，Arduino串口采集链路仅作为演示验证，不纳入本课题核心开发与评价指标。")
        elif par.text.startswith("7）大模型预警解读。"):
            set_text(par._p, par.text + "系统全部核心预测与预警逻辑完全不依赖大模型，大模型仅承担自然语言文案生成，断网状态下仍可完整运行全部预警功能。")

    # ---- 9. 易乱码符号 → Word 公式对象 ----
    n_math = 0
    for row in table.rows[4:13]:
        for par in row.cells[0].paragraphs:
            n_math += convert_math(par)
    print("formula objects inserted:", n_math)

    # ---- 10. 表格：行不跨页拆分 + 表头跨页重复 ----
    for row in table.rows:
        for tb in row.cells[0].tables:
            for r in tb.rows:
                trPr = r._tr.get_or_add_trPr()
                if trPr.find(qn("w:cantSplit")) is None:
                    _insert_ordered(trPr, OxmlElement("w:cantSplit"),
                                    before_tags=("w:trHeight", "w:tblHeader"))
            trPr0 = tb.rows[0]._tr.get_or_add_trPr()
            if trPr0.find(qn("w:tblHeader")) is None:
                _insert_ordered(trPr0, OxmlElement("w:tblHeader"),
                                before_tags=("w:tblCellSpacing", "w:jc", "w:hidden"),
                                after_tags=("w:trHeight",))

    doc.save(OUT)
    print("saved", OUT, os.path.getsize(OUT), "bytes")


if __name__ == "__main__":
    main()
