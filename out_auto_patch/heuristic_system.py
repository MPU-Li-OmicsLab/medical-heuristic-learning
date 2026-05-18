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
