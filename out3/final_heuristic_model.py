CURRENT_VERSION = 'v0'

def predict_v0(features: dict) -> int:
    def _get_float(k: str, default: float = 0.0) -> float:
        v = features.get(k, default)
        try:
            if v is None:
                return default
            return float(v)
        except Exception:
            return default

    def _get_int(k: str, default: int = 0) -> int:
        v = features.get(k, default)
        try:
            if v is None:
                return default
            return int(float(v))
        except Exception:
            return default

    score = 0
    if _get_int("Respiratory_failure") == 1:
        score += 4

    age = _get_float("age")
    if age >= 85:
        score += 2
    elif age >= 75:
        score += 1

    if _get_int("sepsis") == 1:
        score += 2

    sofa = _get_float("SOFA")
    if sofa >= 7:
        score += 2
    elif sofa >= 5:
        score += 1

    if _get_float("wbc") < 8.0:
        score += 1
    if _get_int("Cerebral_infarction") == 1:
        score += 1
    if _get_int("ICH") == 1:
        score += 1
    if _get_float("mbp") < 85.0:
        score += 1
    if _get_float("Sodium") < 136.0:
        score += 1
    if _get_float("hemoglobin") < 9.5:
        score += 1
    if _get_float("bicarbonate") < 18.0:
        score += 1

    if _get_float("Urea_Nitrogen") < 12.0 and _get_float("bicarbonate") >= 20.0 and _get_float("SOFA") <= 6.0:
        score -= 2

    if _get_float("SOFA") >= 7.0 and _get_float("Urea_Nitrogen") < 10.0 and _get_float("bicarbonate") >= 20.0:
        score -= 1

    return 1 if score >= 5 else 0

TESTS = [
    {"name": "predict_returns_int", "code": "assert predict_v0({}) in (0, 1)"},
]


def predict_v1(features: dict) -> int:
    def _get_float(k: str, default: float = 0.0) -> float:
        v = features.get(k, default)
        try:
            if v is None:
                return default
            return float(v)
        except Exception:
            return default

    def _get_int(k: str, default: int = 0) -> int:
        v = features.get(k, default)
        try:
            if v is None:
                return default
            return int(float(v))
        except Exception:
            return default

    score = 0
    if _get_int("Respiratory_failure") == 1:
        score += 4

    age = _get_float("age")
    if age >= 85:
        score += 2
    elif age >= 75:
        score += 1

    if _get_int("sepsis") == 1:
        score += 2

    sofa = _get_float("SOFA")
    if sofa >= 7:
        score += 2
    elif sofa >= 5:
        score += 1

    if _get_float("wbc") < 8.0:
        score += 1
    if _get_int("Cerebral_infarction") == 1:
        score += 1
    if _get_int("ICH") == 1:
        score += 1
    if _get_float("mbp") < 85.0:
        score += 1
    if _get_float("Sodium") < 136.0:
        score += 1
    if _get_float("hemoglobin") < 9.5:
        score += 1
    if _get_float("bicarbonate") < 18.0:
        score += 1

    if _get_float("Urea_Nitrogen") < 12.0 and _get_float("bicarbonate") >= 20.0 and _get_float("SOFA") <= 6.0:
        score -= 2

    if _get_float("SOFA") >= 7.0 and _get_float("Urea_Nitrogen") < 10.0 and _get_float("bicarbonate") >= 20.0:
        score -= 1

    # 新增条件：高尿素氮（>50）减1分，减少假阳性
    if _get_float("Urea_Nitrogen") > 50.0:
        score -= 1

    # 新增条件：高龄（>=80）且中等SOFA（>=3）加1分，增加敏感性
    if age >= 80.0 and _get_float("SOFA") >= 3.0:
        score += 1

    return 1 if score >= 5 else 0

ERROR_ANALYSIS_predict_v1 = "当前规则在训练集上存在100个假阳性（FP）和100个假阴性（FN），整体F1值为0.602，假阳性与假阴性基本平衡但均较多。分析FP样本发现，许多FP患者具有高SOFA分数（≥7）、呼吸衰竭、高龄、脑梗死、低碳酸氢盐、低血红蛋白等特征，这些特征在规则中均被赋予较高权重，导致总分易超过5。同时，部分FP患者尿素氮异常升高（>50），提示慢性肾功能不全，但实际结局非死亡。为提高F1，需要减少FP（增加特异性）和增加真正例（增加敏感性）。建议增加针对高尿素氮的减法条件，以降低非急性死亡风险患者的分数；同时增加针对高龄合并中度器官衰竭的加分条件，以捕捉高风险患者。"

TESTS.extend([
    {'name': "predict_returns_int", 'code': "assert predict_v1({}) in (0, 1)"},
])


def predict_v2(features: dict) -> int:

    def _get_float(k: str, default: float=0.0) -> float:
        v = features.get(k, default)
        try:
            if v is None:
                return default
            return float(v)
        except Exception:
            return default

    def _get_int(k: str, default: int=0) -> int:
        v = features.get(k, default)
        try:
            if v is None:
                return default
            return int(float(v))
        except Exception:
            return default
    score = 0
    if _get_int('Respiratory_failure') == 1:
        score += 4
    age = _get_float('age')
    if age >= 85:
        score += 2
    elif age >= 75:
        score += 1
    if _get_int('sepsis') == 1:
        score += 2
    sofa = _get_float('SOFA')
    if sofa >= 7:
        score += 2
    elif sofa >= 5:
        score += 1
    if _get_float('wbc') < 8.0:
        score += 1
    if _get_int('Cerebral_infarction') == 1:
        score += 1
    if _get_int('ICH') == 1:
        score += 1
    if _get_float('mbp') < 85.0:
        score += 1
    if _get_float('Sodium') < 136.0:
        score += 1
    if _get_float('hemoglobin') < 9.5:
        score += 1
    if _get_float('bicarbonate') < 18.0:
        score += 1
    if _get_float('Urea_Nitrogen') < 12.0 and _get_float('bicarbonate') >= 20.0 and (_get_float('SOFA') <= 6.0):
        score -= 2
    if _get_float('SOFA') >= 7.0 and _get_float('Urea_Nitrogen') < 10.0 and (_get_float('bicarbonate') >= 20.0):
        score -= 1
    if _get_float('Urea_Nitrogen') > 50.0:
        score -= 1
    if age >= 80.0 and _get_float('SOFA') >= 3.0:
        score += 1
    if _get_int('Kidney_failure') == 1:
        score += 1
    return 1 if score >= 5 else 0

ERROR_ANALYSIS_predict_v2 = "自动候选改动：add_Kidney_failure"

TESTS.extend([
    {'name': "predict_returns_int", 'code': "assert predict_v2({}) in (0, 1)"},
])


def predict_v3(features: dict) -> int:

    def _get_float(k: str, default: float=0.0) -> float:
        v = features.get(k, default)
        try:
            if v is None:
                return default
            return float(v)
        except Exception:
            return default

    def _get_int(k: str, default: int=0) -> int:
        v = features.get(k, default)
        try:
            if v is None:
                return default
            return int(float(v))
        except Exception:
            return default
    score = 0
    if _get_int('Respiratory_failure') == 1:
        score += 4
    age = _get_float('age')
    if age >= 85:
        score += 2
    elif age >= 75:
        score += 1
    if _get_int('sepsis') == 1:
        score += 2
    sofa = _get_float('SOFA')
    if sofa >= 7:
        score += 2
    elif sofa >= 5:
        score += 1
    if _get_float('wbc') < 8.0:
        score += 1
    if _get_int('Cerebral_infarction') == 1:
        score += 1
    if _get_int('ICH') == 1:
        score += 1
    if _get_float('mbp') < 85.0:
        score += 1
    if _get_float('Sodium') < 136.0:
        score += 1
    if _get_float('hemoglobin') < 9.5:
        score += 1
    if _get_float('bicarbonate') < 18.0:
        score += 1
    if _get_float('Urea_Nitrogen') < 12.0 and _get_float('bicarbonate') >= 20.0 and (_get_float('SOFA') <= 6.0):
        score -= 1
    if _get_float('SOFA') >= 7.0 and _get_float('Urea_Nitrogen') < 10.0 and (_get_float('bicarbonate') >= 20.0):
        score -= 1
    if _get_float('Urea_Nitrogen') > 50.0:
        score -= 1
    if age >= 80.0 and _get_float('SOFA') >= 3.0:
        score += 1
    if _get_int('Kidney_failure') == 1:
        score += 1
    return 1 if score >= 5 else 0

ERROR_ANALYSIS_predict_v3 = "自动候选改动：weaken_penalty_-2_to_-1"

TESTS.extend([
    {'name': "predict_returns_int", 'code': "assert predict_v3({}) in (0, 1)"},
])


def predict_v4(features: dict) -> int:

    def _get_float(k: str, default: float=0.0) -> float:
        v = features.get(k, default)
        try:
            if v is None:
                return default
            return float(v)
        except Exception:
            return default

    def _get_int(k: str, default: int=0) -> int:
        v = features.get(k, default)
        try:
            if v is None:
                return default
            return int(float(v))
        except Exception:
            return default
    score = 0
    if _get_int('Respiratory_failure') == 1:
        score += 4
    age = _get_float('age')
    if age >= 85:
        score += 2
    elif age >= 75:
        score += 1
    if _get_int('sepsis') == 1:
        score += 2
    sofa = _get_float('SOFA')
    if sofa >= 7:
        score += 2
    elif sofa >= 5:
        score += 1
    if _get_float('wbc') < 8.0:
        score += 1
    if _get_int('Cerebral_infarction') == 1:
        score += 1
    if _get_int('ICH') == 1:
        score += 1
    if _get_float('mbp') < 85.0:
        score += 1
    if _get_float('Sodium') < 136.0:
        score += 1
    if _get_float('hemoglobin') < 9.5:
        score += 1
    if _get_float('bicarbonate') < 18.0:
        score += 1
    if _get_float('Urea_Nitrogen') < 12.0 and _get_float('bicarbonate') >= 20.0 and (_get_float('SOFA') <= 6.0):
        score -= 1
    if _get_float('SOFA') >= 7.0 and _get_float('Urea_Nitrogen') < 10.0 and (_get_float('bicarbonate') >= 20.0):
        score -= 1
    if _get_float('Urea_Nitrogen') > 50.0:
        score -= 1
    if age >= 80.0 and _get_float('SOFA') >= 3.0:
        score += 1
    if _get_int('Kidney_failure') == 1:
        score += 1
    if _get_int('Kidney_failure') == 1:
        score += 1
    return 1 if score >= 5 else 0

ERROR_ANALYSIS_predict_v4 = "自动候选改动：add_Kidney_failure"

TESTS.extend([
    {'name': "predict_returns_int", 'code': "assert predict_v4({}) in (0, 1)"},
])


def predict_v5(features: dict) -> int:

    def _get_float(k: str, default: float=0.0) -> float:
        v = features.get(k, default)
        try:
            if v is None:
                return default
            return float(v)
        except Exception:
            return default

    def _get_int(k: str, default: int=0) -> int:
        v = features.get(k, default)
        try:
            if v is None:
                return default
            return int(float(v))
        except Exception:
            return default
    score = 0
    if _get_int('Respiratory_failure') == 1:
        score += 4
    age = _get_float('age')
    if age >= 85:
        score += 2
    elif age >= 75:
        score += 1
    if _get_int('sepsis') == 1:
        score += 2
    sofa = _get_float('SOFA')
    if sofa >= 7:
        score += 2
    elif sofa >= 5:
        score += 1
    if _get_float('wbc') < 8.0:
        score += 1
    if _get_int('Cerebral_infarction') == 1:
        score += 1
    if _get_int('ICH') == 1:
        score += 1
    if _get_float('mbp') < 85.0:
        score += 1
    if _get_float('Sodium') < 136.0:
        score += 1
    if _get_float('hemoglobin') < 9.5:
        score += 1
    if _get_float('bicarbonate') < 18.0:
        score += 1
    if _get_float('Urea_Nitrogen') < 12.0 and _get_float('bicarbonate') >= 20.0 and (_get_float('SOFA') <= 6.0):
        score -= 1
    if _get_float('SOFA') >= 7.0 and _get_float('Urea_Nitrogen') < 10.0 and (_get_float('bicarbonate') >= 20.0):
        score -= 1
    if _get_float('Urea_Nitrogen') > 50.0:
        score -= 1
    if age >= 80.0 and _get_float('SOFA') >= 3.0:
        score += 1
    if _get_int('Kidney_failure') == 1:
        score += 1
    if _get_int('Kidney_failure') == 1:
        score += 1
    if _get_int('Kidney_failure') == 1:
        score += 1
    return 1 if score >= 5 else 0

ERROR_ANALYSIS_predict_v5 = "自动候选改动：add_Kidney_failure"

TESTS.extend([
    {'name': "predict_returns_int", 'code': "assert predict_v5({}) in (0, 1)"},
])


def predict_v6(features: dict) -> int:

    def _get_float(k: str, default: float=0.0) -> float:
        v = features.get(k, default)
        try:
            if v is None:
                return default
            return float(v)
        except Exception:
            return default

    def _get_int(k: str, default: int=0) -> int:
        v = features.get(k, default)
        try:
            if v is None:
                return default
            return int(float(v))
        except Exception:
            return default
    score = 0
    if _get_int('Respiratory_failure') == 1:
        score += 4
    age = _get_float('age')
    if age >= 85:
        score += 2
    elif age >= 75:
        score += 1
    if _get_int('sepsis') == 1:
        score += 2
    sofa = _get_float('SOFA')
    if sofa >= 7:
        score += 2
    elif sofa >= 5:
        score += 1
    if _get_float('wbc') < 8.0:
        score += 1
    if _get_int('Cerebral_infarction') == 1:
        score += 1
    if _get_int('ICH') == 1:
        score += 1
    if _get_float('mbp') < 85.0:
        score += 1
    if _get_float('Sodium') < 136.0:
        score += 1
    if _get_float('hemoglobin') < 9.5:
        score += 1
    if _get_float('bicarbonate') < 18.0:
        score += 1
    if _get_float('Urea_Nitrogen') < 12.0 and _get_float('bicarbonate') >= 20.0 and (_get_float('SOFA') <= 6.0):
        score -= 1
    if _get_float('SOFA') >= 7.0 and _get_float('Urea_Nitrogen') < 10.0 and (_get_float('bicarbonate') >= 20.0):
        score -= 1
    if _get_float('Urea_Nitrogen') > 50.0:
        score -= 1
    if age >= 80.0 and _get_float('SOFA') >= 3.0:
        score += 1
    if _get_int('Kidney_failure') == 1:
        score += 1
    if _get_int('Kidney_failure') == 1:
        score += 1
    if _get_int('Kidney_failure') == 1:
        score += 1
    if _get_int('Kidney_failure') == 1:
        score += 1
    return 1 if score >= 5 else 0

ERROR_ANALYSIS_predict_v6 = "自动候选改动：add_Kidney_failure"

TESTS.extend([
    {'name': "predict_returns_int", 'code': "assert predict_v6({}) in (0, 1)"},
])


def predict_v7(features: dict) -> int:

    def _get_float(k: str, default: float=0.0) -> float:
        v = features.get(k, default)
        try:
            if v is None:
                return default
            return float(v)
        except Exception:
            return default

    def _get_int(k: str, default: int=0) -> int:
        v = features.get(k, default)
        try:
            if v is None:
                return default
            return int(float(v))
        except Exception:
            return default
    score = 0
    if _get_int('Respiratory_failure') == 1:
        score += 4
    age = _get_float('age')
    if age >= 85:
        score += 2
    elif age >= 75:
        score += 1
    if _get_int('sepsis') == 1:
        score += 2
    sofa = _get_float('SOFA')
    if sofa >= 7:
        score += 2
    elif sofa >= 5:
        score += 1
    if _get_float('wbc') < 8.0:
        score += 1
    if _get_int('Cerebral_infarction') == 1:
        score += 1
    if _get_int('ICH') == 1:
        score += 1
    if _get_float('mbp') < 85.0:
        score += 1
    if _get_float('Sodium') < 136.0:
        score += 1
    if _get_float('hemoglobin') < 9.5:
        score += 1
    if _get_float('bicarbonate') < 18.0:
        score += 1
    if _get_float('Urea_Nitrogen') < 12.0 and _get_float('bicarbonate') >= 20.0 and (_get_float('SOFA') <= 6.0):
        score -= 1
    if _get_float('SOFA') >= 7.0 and _get_float('Urea_Nitrogen') < 10.0 and (_get_float('bicarbonate') >= 20.0):
        score -= 1
    if _get_float('Urea_Nitrogen') > 50.0:
        score -= 1
    if age >= 80.0 and _get_float('SOFA') >= 3.0:
        score += 1
    if _get_int('Kidney_failure') == 1:
        score += 1
    if _get_int('Kidney_failure') == 1:
        score += 1
    if _get_int('Kidney_failure') == 1:
        score += 1
    if _get_int('Kidney_failure') == 1:
        score += 1
    if _get_int('ARDS') == 1:
        score += 1
    return 1 if score >= 5 else 0

ERROR_ANALYSIS_predict_v7 = "自动候选改动：add_ARDS"

TESTS.extend([
    {'name': "predict_returns_int", 'code': "assert predict_v7({}) in (0, 1)"},
])


def predict_v8(features: dict) -> int:

    def _get_float(k: str, default: float=0.0) -> float:
        v = features.get(k, default)
        try:
            if v is None:
                return default
            return float(v)
        except Exception:
            return default

    def _get_int(k: str, default: int=0) -> int:
        v = features.get(k, default)
        try:
            if v is None:
                return default
            return int(float(v))
        except Exception:
            return default
    score = 0
    if _get_int('Respiratory_failure') == 1:
        score += 4
    age = _get_float('age')
    if age >= 85:
        score += 2
    elif age >= 75:
        score += 1
    if _get_int('sepsis') == 1:
        score += 2
    sofa = _get_float('SOFA')
    if sofa >= 7:
        score += 2
    elif sofa >= 5:
        score += 1
    if _get_float('wbc') < 8.0:
        score += 1
    if _get_int('Cerebral_infarction') == 1:
        score += 1
    if _get_int('ICH') == 1:
        score += 1
    if _get_float('mbp') < 85.0:
        score += 1
    if _get_float('Sodium') < 136.0:
        score += 1
    if _get_float('hemoglobin') < 9.5:
        score += 1
    if _get_float('bicarbonate') < 18.0:
        score += 1
    if _get_float('Urea_Nitrogen') < 12.0 and _get_float('bicarbonate') >= 20.0 and (_get_float('SOFA') <= 6.0):
        score -= 1
    if _get_float('SOFA') >= 7.0 and _get_float('Urea_Nitrogen') < 10.0 and (_get_float('bicarbonate') >= 20.0):
        score -= 1
    if _get_float('Urea_Nitrogen') > 50.0:
        score -= 1
    if age >= 80.0 and _get_float('SOFA') >= 3.0:
        score += 1
    if _get_int('Kidney_failure') == 1:
        score += 1
    if _get_int('Kidney_failure') == 1:
        score += 1
    if _get_int('Kidney_failure') == 1:
        score += 1
    if _get_int('Kidney_failure') == 1:
        score += 1
    if _get_int('ARDS') == 1:
        score += 1
    if _get_int('ARDS') == 1:
        score += 1
    return 1 if score >= 5 else 0

ERROR_ANALYSIS_predict_v8 = "自动候选改动：add_ARDS"

TESTS.extend([
    {'name': "predict_returns_int", 'code': "assert predict_v8({}) in (0, 1)"},
])


def predict_v9(features: dict) -> int:

    def _get_float(k: str, default: float=0.0) -> float:
        v = features.get(k, default)
        try:
            if v is None:
                return default
            return float(v)
        except Exception:
            return default

    def _get_int(k: str, default: int=0) -> int:
        v = features.get(k, default)
        try:
            if v is None:
                return default
            return int(float(v))
        except Exception:
            return default
    score = 0
    if _get_int('Respiratory_failure') == 1:
        score += 4
    age = _get_float('age')
    if age >= 85:
        score += 2
    elif age >= 75:
        score += 1
    if _get_int('sepsis') == 1:
        score += 2
    sofa = _get_float('SOFA')
    if sofa >= 7:
        score += 2
    elif sofa >= 5:
        score += 1
    if _get_float('wbc') < 8.0:
        score += 1
    if _get_int('Cerebral_infarction') == 1:
        score += 1
    if _get_int('ICH') == 1:
        score += 1
    if _get_float('mbp') < 85.0:
        score += 1
    if _get_float('Sodium') < 136.0:
        score += 1
    if _get_float('hemoglobin') < 9.5:
        score += 1
    if _get_float('bicarbonate') < 18.0:
        score += 1
    if _get_float('Urea_Nitrogen') < 12.0 and _get_float('bicarbonate') >= 20.0 and (_get_float('SOFA') <= 6.0):
        score -= 1
    if _get_float('SOFA') >= 7.0 and _get_float('Urea_Nitrogen') < 10.0 and (_get_float('bicarbonate') >= 20.0):
        score -= 1
    if _get_float('Urea_Nitrogen') > 50.0:
        score -= 1
    if age >= 80.0 and _get_float('SOFA') >= 3.0:
        score += 1
    if _get_int('Kidney_failure') == 1:
        score += 1
    if _get_int('Kidney_failure') == 1:
        score += 1
    if _get_int('Kidney_failure') == 1:
        score += 1
    if _get_int('Kidney_failure') == 1:
        score += 1
    if _get_int('ARDS') == 1:
        score += 1
    if _get_int('ARDS') == 1:
        score += 1
    if _get_int('Kidney_failure') == 1:
        score += 1
    return 1 if score >= 5 else 0

ERROR_ANALYSIS_predict_v9 = "自动候选改动：add_Kidney_failure"

TESTS.extend([
    {'name': "predict_returns_int", 'code': "assert predict_v9({}) in (0, 1)"},
])


def predict_v10(features: dict) -> int:

    def _get_float(k: str, default: float=0.0) -> float:
        v = features.get(k, default)
        try:
            if v is None:
                return default
            return float(v)
        except Exception:
            return default

    def _get_int(k: str, default: int=0) -> int:
        v = features.get(k, default)
        try:
            if v is None:
                return default
            return int(float(v))
        except Exception:
            return default
    score = 0
    if _get_int('Respiratory_failure') == 1:
        score += 4
    age = _get_float('age')
    if age >= 85:
        score += 2
    elif age >= 75:
        score += 1
    if _get_int('sepsis') == 1:
        score += 2
    sofa = _get_float('SOFA')
    if sofa >= 7:
        score += 2
    elif sofa >= 5:
        score += 1
    if _get_float('wbc') < 8.0:
        score += 1
    if _get_int('Cerebral_infarction') == 1:
        score += 1
    if _get_int('ICH') == 1:
        score += 1
    if _get_float('mbp') < 85.0:
        score += 1
    if _get_float('Sodium') < 136.0:
        score += 1
    if _get_float('hemoglobin') < 9.5:
        score += 1
    if _get_float('bicarbonate') < 18.0:
        score += 1
    if _get_float('Urea_Nitrogen') < 12.0 and _get_float('bicarbonate') >= 20.0 and (_get_float('SOFA') <= 6.0):
        score -= 1
    if _get_float('SOFA') >= 7.0 and _get_float('Urea_Nitrogen') < 10.0 and (_get_float('bicarbonate') >= 20.0):
        score -= 1
    if _get_float('Urea_Nitrogen') > 50.0:
        score -= 1
    if age >= 80.0 and _get_float('SOFA') >= 3.0:
        score += 1
    if _get_int('Kidney_failure') == 1:
        score += 1
    if _get_int('Kidney_failure') == 1:
        score += 1
    if _get_int('Kidney_failure') == 1:
        score += 1
    if _get_int('Kidney_failure') == 1:
        score += 1
    if _get_int('ARDS') == 1:
        score += 1
    if _get_int('ARDS') == 1:
        score += 1
    if _get_int('Kidney_failure') == 1:
        score += 1
    if _get_int('Kidney_failure') == 1:
        score += 1
    return 1 if score >= 5 else 0

ERROR_ANALYSIS_predict_v10 = "自动候选改动：add_Kidney_failure"

TESTS.extend([
    {'name': "predict_returns_int", 'code': "assert predict_v10({}) in (0, 1)"},
])

SELECTED_VERSION = 'v9'

def predict(features: dict) -> int:
    return predict_v9(features)


if __name__ == '__main__':
    assert predict({}) in (0, 1)

