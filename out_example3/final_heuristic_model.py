CURRENT_VERSION = 'v0'

def predict_v0(features: dict) -> int:
    score = 0.5
    # 基础校准：无高危条件时score<0.5
    score -= 0.25
    # 高危因素加分
    # Respiratory_failure：强阳性关联，直接增加死亡风险
    if features.get('Respiratory_failure', 0) == 1:
        score += 0.2
    # sepsis：强阳性关联，脓毒症核心指标
    if features.get('sepsis', 0) == 1:
        score += 0.15
    # age：>65岁为高危，每增加10岁风险升高
    if features.get('age', 60) > 65:
        score += 0.1
    # SOFA：>6分死亡率显著升高
    if features.get('SOFA', 0) > 6:
        score += 0.15
    # mbp：平均动脉压<65mmHg提示休克
    if features.get('mbp', 80) < 65:
        score += 0.1
    # ICH：颅内血肿，死亡率极高
    if features.get('ICH', 0) == 1:
        score += 0.15
    # Cerebral_infarction：脑梗死导致神经功能障碍
    if features.get('Cerebral_infarction', 0) == 1:
        score += 0.1
    # wbc：极端值提示感染/炎症
    wbc = features.get('wbc', 10)
    if wbc < 4 or wbc > 12:
        score += 0.05
    # Urea_Nitrogen：>28 mg/dL提示肾功能不全
    if features.get('Urea_Nitrogen', 20) > 28:
        score += 0.05
    # 扣分规则：年龄<40岁（年轻，降低假阳性）
    if features.get('age', 60) < 40:
        score -= 0.05
    return 1 if score >= 0.5 else 0

TESTS = [
    {'name': "predict_returns_int", 'code': "assert predict_v0({}) in (0, 1)"},
    {'name': "low_risk_should_be_0", 'code': "assert predict_v0({'age':30, 'Respiratory_failure':0, 'sepsis':0, 'SOFA':0, 'mbp':80, 'ICH':0, 'Cerebral_infarction':0, 'wbc':7, 'Urea_Nitrogen':15}) == 0"},
    {'name': "high_risk_should_be_1", 'code': "assert predict_v0({'age':75, 'Respiratory_failure':1, 'sepsis':1, 'SOFA':8, 'mbp':55, 'ICH':0, 'Cerebral_infarction':1, 'wbc':15, 'Urea_Nitrogen':30}) == 1"},
]


ERROR_ANALYSIS_predict_v0 = "基于单变量统计摘要，选择排名靠前且医学关联明确的特征：Respiratory_failure、sepsis、SOFA、age、mbp、ICH、Cerebral_infarction、wbc、Urea_Nitrogen。这些特征在知识表中具有中高证据置信度。基础校准扣除0.25分，确保无高危条件时score<0.5，降低假阳性。同时加入扣分规则（年龄<40减去0.05）以平衡假阴性。评分阈值0.5。"


CURRENT_VERSION = "v1"

def predict_v1(features: dict) -> int:
    score = 0.5
    # 基础校准：无高危条件时score<0.5，降低假阳性
    score -= 0.25
    # 高危因素加分
    # Respiratory_failure：强阳性关联，直接增加死亡风险；上调分值以捕捉更多真阳性
    if features.get('Respiratory_failure', 0) == 1:
        score += 0.25
    # sepsis：强阳性关联，脓毒症核心指标
    if features.get('sepsis', 0) == 1:
        score += 0.15
    # age：>65岁为高危，每增加10岁风险升高
    if features.get('age', 60) > 65:
        score += 0.1
    # SOFA：>6分死亡率显著升高，但降权以减少假阳性
    if features.get('SOFA', 0) > 6:
        score += 0.1
    # mbp：平均动脉压<65mmHg提示休克
    if features.get('mbp', 80) < 65:
        score += 0.1
    # ICH：颅内血肿，死亡率极高
    if features.get('ICH', 0) == 1:
        score += 0.15
    # Cerebral_infarction：脑梗死导致神经功能障碍
    if features.get('Cerebral_infarction', 0) == 1:
        score += 0.1
    # wbc：极端值提示感染/炎症
    wbc = features.get('wbc', 10)
    if wbc < 4 or wbc > 12:
        score += 0.05
    # Urea_Nitrogen：>28 mg/dL提示肾功能不全
    if features.get('Urea_Nitrogen', 20) > 28:
        score += 0.05
    # 扣分规则：年龄<40岁（年轻，降低假阳性）
    if features.get('age', 60) < 40:
        score -= 0.05
    return 1 if score >= 0.5 else 0

ERROR_ANALYSIS_predict_v1 = "FP样本中SOFA>6是常见原因，导致假阳性；FN样本中Respiratory_failure=1但年龄较轻且SOFA不高的漏诊较多。为平衡误差，小幅提升Respiratory_failure权重至0.25，同时降低SOFA>6权重至0.1，以在减少假阳性的同时纠正部分假阴性。"

TESTS.extend([
    {'name': "predict_returns_int", 'code': "assert predict_v1({}) in (0, 1)"},
    {'name': "low_risk_should_be_0", 'code': "assert predict_v1({'age':30, 'Respiratory_failure':0, 'sepsis':0, 'SOFA':0, 'mbp':80, 'ICH':0, 'Cerebral_infarction':0, 'wbc':7, 'Urea_Nitrogen':15}) == 0"},
    {'name': "high_risk_should_be_1", 'code': "assert predict_v1({'age':75, 'Respiratory_failure':1, 'sepsis':1, 'SOFA':8, 'mbp':55, 'ICH':0, 'Cerebral_infarction':1, 'wbc':15, 'Urea_Nitrogen':30}) == 1"},
    {'name': "resp_failure_only_should_be_1", 'code': "assert predict_v1({'age':50, 'Respiratory_failure':1, 'sepsis':0, 'SOFA':3, 'mbp':80, 'ICH':0, 'Cerebral_infarction':0, 'wbc':7, 'Urea_Nitrogen':15}) == 1"},
])


CURRENT_VERSION = "v2"

def predict_v2(features: dict) -> int:
    score = 0.5
    # 基础校准：无高危条件时score<0.5，降低假阳性
    score -= 0.25
    # 高危因素加分
    # Respiratory_failure：强阳性关联，直接增加死亡风险；保持较高权重以捕捉真阳性
    if features.get('Respiratory_failure', 0) == 1:
        score += 0.25
    # sepsis：强阳性关联，脓毒症核心指标
    if features.get('sepsis', 0) == 1:
        score += 0.15
    # age：>65岁为高危，每增加10岁风险升高
    if features.get('age', 60) > 65:
        score += 0.1
    # SOFA：>6分死亡率显著升高，但降权以减少假阳性（退化样本中常见）
    if features.get('SOFA', 0) > 6:
        score += 0.05
    # mbp：平均动脉压<65mmHg提示休克
    if features.get('mbp', 80) < 65:
        score += 0.1
    # ICH：颅内血肿，死亡率极高
    if features.get('ICH', 0) == 1:
        score += 0.15
    # Cerebral_infarction：脑梗死导致神经功能障碍，降权以减少假阳性（退化样本中常见）
    if features.get('Cerebral_infarction', 0) == 1:
        score += 0.05
    # wbc：极端值提示感染/炎症
    wbc = features.get('wbc', 10)
    if wbc < 4 or wbc > 12:
        score += 0.05
    # Urea_Nitrogen：>28 mg/dL提示肾功能不全
    if features.get('Urea_Nitrogen', 20) > 28:
        score += 0.05
    # 扣分规则：年龄<40岁（年轻，降低假阳性）
    if features.get('age', 60) < 40:
        score -= 0.05
    return 1 if score >= 0.5 else 0

ERROR_ANALYSIS_predict_v2 = "本次错误样本分析显示，FP样本中常见SOFA>6和Cerebral_infarction=1导致假阳性，而FN样本中SOFA高但年龄较轻、Urea_Nitrogen高且Respiratory_failure=0的漏诊较多。退化警告表明上一版（v1）对SOFA权重降低不足，导致大量假阳性增加（退化46例）。为修复退化，优先降低SOFA和Cerebral_infarction的权重，以减少假阳性，同时保持其他特征不变以维持对FN的捕捉。"

TESTS.extend([
    {'name': "predict_returns_int", 'code': "assert predict_v2({}) in (0, 1)"},
    {'name': "low_risk_should_be_0", 'code': "assert predict_v2({'age':30, 'Respiratory_failure':0, 'sepsis':0, 'SOFA':0, 'mbp':80, 'ICH':0, 'Cerebral_infarction':0, 'wbc':7, 'Urea_Nitrogen':15}) == 0"},
    {'name': "high_risk_should_be_1", 'code': "assert predict_v2({'age':75, 'Respiratory_failure':1, 'sepsis':1, 'SOFA':8, 'mbp':55, 'ICH':0, 'Cerebral_infarction':1, 'wbc':15, 'Urea_Nitrogen':30}) == 1"},
    {'name': "resp_failure_only_should_be_1", 'code': "assert predict_v2({'age':50, 'Respiratory_failure':1, 'sepsis':0, 'SOFA':3, 'mbp':80, 'ICH':0, 'Cerebral_infarction':0, 'wbc':7, 'Urea_Nitrogen':15}) == 1"},
])


CURRENT_VERSION = "v3"

def predict_v3(features: dict) -> int:
    score = 0.5
    score -= 0.15
    if features.get('Respiratory_failure', 0) == 1:
        score += 0.25
    if features.get('sepsis', 0) == 1:
        score += 0.15
    if features.get('age', 60) > 65:
        score += 0.05
    if features.get('SOFA', 0) > 6:
        score += 0.05
    if features.get('mbp', 80) < 65:
        score += 0.1
    if features.get('ICH', 0) == 1:
        score += 0.15
    if features.get('Cerebral_infarction', 0) == 1:
        score += 0.05
    wbc = features.get('wbc', 10)
    if wbc < 4 or wbc > 12:
        score += 0.05
    if features.get('Urea_Nitrogen', 20) > 28:
        score += 0.02
    if features.get('age', 60) < 40:
        score -= 0.05
    return 1 if score >= 0.5 else 0

ERROR_ANALYSIS_predict_v3 = "本次修改主要针对退化警告中的49个退化样本。分析发现，退化样本中常见年龄>65、Urea_Nitrogen>28且Respiratory_failure=0，导致假阳性增加。为减少假阳性而不显著增加假阴性，小幅降低age>65的加分幅度（0.1→0.05）和Urea_Nitrogen>28的加分幅度（0.05→0.02），其他规则保持不变。（自动校准：score -= 0.25 -> 0.15）"

TESTS.extend([
    {'name': "predict_returns_int", 'code': "assert predict_v3({}) in (0, 1)"},
    {'name': "low_risk_should_be_0", 'code': "assert predict_v3({'age':30, 'Respiratory_failure':0, 'sepsis':0, 'SOFA':0, 'mbp':80, 'ICH':0, 'Cerebral_infarction':0, 'wbc':7, 'Urea_Nitrogen':15}) == 0"},
    {'name': "high_risk_should_be_1", 'code': "assert predict_v3({'age':75, 'Respiratory_failure':1, 'sepsis':1, 'SOFA':8, 'mbp':55, 'ICH':0, 'Cerebral_infarction':1, 'wbc':15, 'Urea_Nitrogen':30}) == 1"},
    {'name': "resp_failure_only_should_be_1", 'code': "assert predict_v3({'age':50, 'Respiratory_failure':1, 'sepsis':0, 'SOFA':3, 'mbp':80, 'ICH':0, 'Cerebral_infarction':0, 'wbc':7, 'Urea_Nitrogen':15}) == 1"},
    {'name': "degraded_fp_fixed_should_be_0", 'code': "assert predict_v3({'age':82, 'Respiratory_failure':0, 'sepsis':0, 'SOFA':4, 'mbp':94, 'ICH':0, 'Cerebral_infarction':0, 'wbc':14.51, 'Urea_Nitrogen':38.55}) == 0"},
])

FINAL_VERSION = "v3"

def predict(features: dict) -> int:
    return predict_v3(features)


if __name__ == '__main__':
    assert predict({}) in (0, 1)

