#!/usr/bin/env python3
"""
P5: Dose-response parameter annotation table.
Annotate each of the 23 industry-calibrated parameters with:
- evidence type (literature empirical / national standard / project-calibrated / industry empirical / calibrated-back-calculated)
- source literature
- credibility level
- model usage (which model version uses this parameter)
"""
import pandas as pd
import csv

INPUT = r"F:\20261011\畜牧行业热应激\02_Livestock_HeatStress\data_prep\industry_params\剂量反应参数_模型接入.csv"
OUTPUT = r"F:\20261011\project\02_Livestock_HeatStress\data_prep\剂量反应参数标注表_20261013.csv"

# Source literature mapping (S-code -> citation)
SOURCE_MAP = {
    "S05": "NY/T 2363-2013《奶牛热应激评价技术规范》",
    "S07": "项目组大规模牛场/肉牛生产数据GAM回归(2025)",
    "S09": "Bernabucci et al. 2010, J Dairy Sci (全球文献综合)",
    "S09b": "文献综述(环境温度-产奶量)",
    "S10": "Bouraoui et al. 2002, Tunisia (突尼斯)",
    "S24": "Wettemann 1976 / Stone 1982 (美国猪繁殖)",
    "S25": "MDPI 2026 温带气候猪氧化应激研究",
    "S26": "中国蛋鸡生产数据(舍内温度-产蛋率)",
    "S28": "中国家禽采食量分段线性研究",
    "S28b": "中国家禽采食量分段线性(补充)",
    "S29": "中国肉鸡试验研究(32℃×7天)",
    "S30": "中国蛋鸡生产数据(补充)",
    "S31": "中国家禽热应激致死条件研究",
    "S40": "山东家禽降温改造案例",
    "S42": "Smith 2013, DHIA记录(娟姗牛)",
    "S43": "李朝明等 2014, 中国畜牧杂志(娟姗牛)",
    "S44": "娟姗牛耐热性研究(补充)",
    "S46": "突尼斯21个农场2009-2014(塔兰泰斯牛)",
    "S61": "气候控制试验文献综述/ Akkaya 2024(奶山羊)",
    "S62": "Silanikove & Kolumnan 2015, 国际通用分级",
    "S63": "地中海奶绵羊/西班牙反应规范模型研究",
    "S64": "Akkaya 2024等(奶山羊Saanen试验)",
    "S65": "PMC 2025 综述(婆罗门×荷斯坦F1杂交)",
    "S73": "高产奶牛阈值可低至65(文献)",
    "S74": "西班牙反应规范模型(奶山羊MG/PYA年度积分)",
}

# Evidence type classification
EVIDENCE_TYPE = {
    "DR-01": "文献实证",
    "DR-02": "文献实证",
    "DR-03": "国家标准",
    "DR-04": "项目标定",
    "DR-05": "文献实证",
    "DR-06": "文献实证",
    "DR-07": "文献实证",
    "DR-08": "文献实证",
    "DR-09": "文献实证",
    "DR-10": "项目标定",
    "DR-11": "文献实证",
    "DR-12": "文献实证",
    "DR-13": "文献实证",
    "DR-14": "产业实证",
    "DR-15": "文献实证",
    "DR-16": "文献实证",
    "DR-17": "文献实证",
    "DR-18": "文献实证",
    "DR-19": "产业实证",
    "DR-20": "文献实证",
    "DR-21": "文献实证",
    "DR-22": "文献实证",
    "DR-23": "文献实证",
}

# Model usage: which model version uses this parameter
MODEL_USAGE = {
    "DR-01": "v3产业校准版(奶牛主参数), v4内生化模型",
    "DR-02": "v3敏感性分析(突尼斯参数对照)",
    "DR-03": "v3分段损失参考, A-1全球偏差计算(国标分档对照)",
    "DR-04": "v3推荐优先采用(THImax=70阈值)",
    "DR-05": "v3损失上限参考(DMI-产奶量分解)",
    "DR-06": "v3温度口径敏感性",
    "DR-07": "v3耐热品种修正(娟姗牛情景)",
    "DR-08": "v3耐热品种修正(地方品种情景)",
    "DR-09": "v3杂交改良情景参数",
    "DR-10": "v3肉牛参数(响应弱于奶牛)",
    "DR-11": "v3猪繁殖损失参数(夏季不孕)",
    "DR-12": "v3猪氧化应激阈值参考",
    "DR-13": "v3公猪精子活力阈值",
    "DR-14": "v3蛋鸡产蛋率分段参数",
    "DR-15": "v3肉鸡体重/采食量/FCR参数",
    "DR-16": "v3肉鸡增重率参数",
    "DR-17": "v3家禽采食量分段线性参数",
    "DR-18": "v3家禽死亡率致死条件",
    "DR-19": "v3适应差距计算关键参数(有/无降温对比)",
    "DR-20": "v3奶山羊产奶量参数",
    "DR-21": "v3奶山羊分级查表参数",
    "DR-22": "v3奶绵羊产奶量参数",
    "DR-23": "v3奶山羊年度积分参数(MG/PYA)",
}

# Credibility level (1-5, 5=highest)
CREDIBILITY = {
    "DR-01": 4,  # 应用最广，高产牛适用
    "DR-02": 3,  # 单国研究
    "DR-03": 5,  # 国家标准
    "DR-04": 4,  # 首次系统标定，推荐优先采用
    "DR-05": 3,  # 文献综述区间估计
    "DR-06": 2,  # 文献综述，无具体出处
    "DR-07": 4,  # DHIA记录+中文文献交叉验证
    "DR-08": 3,  # 单国21农场
    "DR-09": 3,  # 综述，非一手实验
    "DR-10": 4,  # 大规模数据GAM回归
    "DR-11": 4,  # 经典实验(Wettemann 1976)
    "DR-12": 3,  # MDPI 2026
    "DR-13": 3,  # Stone 1982经典
    "DR-14": 4,  # 中国生产数据
    "DR-15": 4,  # 中国试验研究
    "DR-16": 3,  # 中国，区间估计
    "DR-17": 4,  # 中国分段线性研究
    "DR-18": 4,  # 中国致死条件研究
    "DR-19": 4,  # 某省2024夏统计+山东案例
    "DR-20": 3,  # Akkaya 2024等
    "DR-21": 4,  # 国际通用分级(Silanikove)
    "DR-22": 3,  # 地中海研究
    "DR-23": 3,  # 西班牙模型
}


def main():
    df = pd.read_csv(INPUT, encoding='utf-8', low_memory=False)
    print(f"Loaded {len(df)} parameters")

    # Add annotation columns
    df['证据类型'] = df['参数ID'].map(EVIDENCE_TYPE)
    df['可信度等级(1-5)'] = df['参数ID'].map(CREDIBILITY)
    df['模型使用版本'] = df['参数ID'].map(MODEL_USAGE)

    # Expand source codes to full citations
    def expand_sources(s):
        if pd.isna(s):
            return ""
        codes = str(s).split('/')
        citations = []
        for code in codes:
            code = code.strip()
            if code in SOURCE_MAP:
                citations.append(f"{code}:{SOURCE_MAP[code]}")
            else:
                citations.append(code)
        return '; '.join(citations)

    df['来源文献全称'] = df['来源(见来源清单)'].apply(expand_sources)

    # Reorder columns for readability
    cols = ['参数ID', '物种', '响应变量', '驱动指标', '起效阈值',
            '剂量-反应系数/分段值', '函数形式', '证据类型', '可信度等级(1-5)',
            '模型使用版本', '来源文献全称', '适用条件/区域', '可信度备注']
    df_out = df[cols]

    # Save
    df_out.to_csv(OUTPUT, index=False, encoding='utf-8-sig')
    print(f"Annotation table saved: {OUTPUT}")

    # Summary
    print(f"\n=== Summary ===")
    print(f"Total parameters: {len(df_out)}")
    print(f"\nEvidence type distribution:")
    print(df_out['证据类型'].value_counts())
    print(f"\nCredibility distribution:")
    print(df_out['可信度等级(1-5)'].value_counts().sort_index())
    print(f"\nSpecies distribution:")
    print(df_out['物种'].value_counts())

    # Print all parameters for review
    print(f"\n=== Parameter Details ===")
    for _, row in df_out.iterrows():
        print(f"{row['参数ID']} [{row['证据类型']}|可信度{row['可信度等级(1-5)']}] "
              f"{row['物种']} - {row['响应变量']}: {row['剂量-反应系数/分段值'][:50]}")


if __name__ == "__main__":
    main()
