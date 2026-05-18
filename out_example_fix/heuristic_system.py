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

    score = 0.5
    # 评分阈值：score >= 0.5 判定为 1（更偏向召回/降低漏诊），score < 0.5 判定为 0。
    score -= 0.25  # 基础风险校准：默认更偏向阴性，避免全部预测为 1

    if _get_int("Respiratory_failure") == 1:
        score += 0.25  # 呼吸衰竭通常提示更高危

    age = _get_float("age")
    if age >= 85:
        score += 0.10  # 高龄通常更高危
    elif age >= 75:
        score += 0.05  # 较高龄风险上升

    if _get_int("sepsis") == 1:
        score += 0.10  # 脓毒症提示系统性感染风险

    sofa = _get_float("SOFA")
    if sofa >= 7:
        score += 0.15  # SOFA 高分提示多器官功能障碍
    elif sofa >= 5:
        score += 0.08  # SOFA 中等升高提示风险上升

    if _get_float("wbc") < 8.0:
        score += 0.05  # 白细胞偏低可能提示免疫抑制/严重感染
    if _get_int("Cerebral_infarction") == 1:
        score += 0.03  # 既往脑梗提示基础状态较差
    if _get_int("ICH") == 1:
        score += 0.04  # 颅内出血提示更高危
    if _get_float("mbp") < 85.0:
        score += 0.04  # 平均动脉压偏低提示循环不稳定
    if _get_float("Sodium") < 136.0:
        score += 0.03  # 低钠与不良结局相关（非特异）
    if _get_float("hemoglobin") < 9.5:
        score += 0.03  # 贫血可能提示储备差/出血等
    if _get_float("bicarbonate") < 18.0:
        score += 0.06  # 低碳酸氢根提示代谢性酸中毒风险

    if _get_float("Urea_Nitrogen") < 12.0 and _get_float("bicarbonate") >= 20.0 and _get_float("SOFA") <= 6.0:
        score -= 0.08  # 若肾功能与酸碱相对正常且 SOFA 不高，风险下调

    if _get_float("SOFA") >= 7.0 and _get_float("Urea_Nitrogen") < 10.0 and _get_float("bicarbonate") >= 20.0:
        score -= 0.04  # 对“SOFA 高但关键生化较好”的情形做轻度下调

    return 1 if score >= 0.5 else 0

TESTS = [
    {"name": "predict_returns_int", "code": "assert predict_v0({}) in (0, 1)"},
]


ERROR_ANALYSIS_predict_v0 = "default_v0"
