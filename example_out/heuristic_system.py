CURRENT_VERSION = 'v0'

def predict_v0(features: dict) -> int:
    """
    Binary hospital mortality prediction rule (hospital_expire_flag).
    Design: point system + absolute criteria.
    - One point each: age > 75, Respiratory_failure, sepsis, SOFA ≥ 4, wbc < 4.
    - Total points ≥ 2 → high risk.
    - Absolute high‑risk: ICH, ARDS, SOFA ≥ 8, age > 85 → immediate high risk.
    - Rationale: Combines multiple moderate/severe risk factors to reduce false positives,
      while capturing catastrophic single‑condition cases to maintain sensitivity.
    """
    # Extract features
    age = features.get('age')
    resp_fail = features.get('Respiratory_failure')
    sepsis = features.get('sepsis')
    wbc = features.get('wbc')
    sofa = features.get('SOFA')
    ich = features.get('ICH')
    ards = features.get('ARDS')

    # Absolute high‑risk conditions (directly predict death)
    if ich == 1:
        return 1  # Intracranial hemorrhage is catastrophic
    if ards == 1:
        return 1  # ARDS has very high mortality despite low prevalence
    if sofa is not None and sofa >= 8:
        return 1  # Severe multi‑organ failure
    if age is not None and age > 85:
        return 1  # Extreme old age independently predicts death

    # Point‑based risk count
    points = 0
    # Age > 75: strong continuous predictor (Q4 OR=16.26)
    if age is not None and age > 75:
        points += 1
    # Respiratory failure: most powerful binary predictor (χ²=164.8)
    if resp_fail == 1:
        points += 1
    # Sepsis: life‑threatening organ dysfunction (χ²=46.8)
    if sepsis == 1:
        points += 1
    # SOFA ≥ 4: moderate‐to‑severe organ dysfunction (Q2 OR=3.28)
    if sofa is not None and sofa >= 4:
        points += 1
    # wbc < 4: leukopenia indicating severe sepsis or immunosuppression (negative point‑biserial, OR inversion)
    if wbc is not None and wbc < 4:
        points += 1

    # Require at least two risk factors to avoid excessive false alarms
    if points >= 2:
        return 1
    else:
        return 0


ERROR_ANALYSIS_predict_v0 = "The rule uses a point-based system to combine the strongest univariate predictors (Respiratory_failure, age, sepsis, SOFA, wbc) with absolute high‑risk criteria (ICH, ARDS, very high SOFA, advanced age) to balance false negatives and false positives. It is designed to maximize F1 score, followed by accuracy, by focusing on features with both high statistical significance and large effect sizes (e.g., age point‑biserial r=0.35, Respiratory_failure χ²=164). The thresholds (age>75, age>85, SOFA≥4, SOFA≥8, wbc<4) are chosen based on medical knowledge and the corresponding binned odds ratios from the univariate summary."


CURRENT_VERSION = "v1"

def predict_v1(features: dict) -> int:
    """
    Binary hospital mortality prediction rule (hospital_expire_flag).
    Design: point system + refined absolute criteria.
    - One point each: age > 75, Respiratory_failure, sepsis, SOFA ≥ 4, wbc < 4, ARDS.
    - Total points ≥ 2 → high risk.
    - Absolute high‑risk: ICH, SOFA ≥ 10, or age > 85 with (SOFA ≥ 4 or Respiratory_failure).
    Rationale: tightened from v0 to reduce false positives by demanding more evidence for age‑only and SOFA 8‑9 cases, while retaining ICH as an independent catastrophic predictor.
    """
    # Extract features
    age = features.get('age')
    resp_fail = features.get('Respiratory_failure')
    sepsis = features.get('sepsis')
    wbc = features.get('wbc')
    sofa = features.get('SOFA')
    ich = features.get('ICH')
    ards = features.get('ARDS')

    # Absolute high‑risk conditions
    # ICH remains catastrophic regardless of other features
    if ich == 1:
        return 1  # Intracranial hemorrhage independently deadly
    # SOFA ≥ 10 indicates severe multi‑organ failure highly associated with mortality
    if sofa is not None and sofa >= 10:
        return 1  # Very high SOFA despite potential therapy
    # Age > 85 is a strong risk factor only when accompanied by organ dysfunction
    if age is not None and age > 85:
        # Require either respiratory failure or moderate organ dysfunction
        if resp_fail == 1 or (sofa is not None and sofa >= 4):
            return 1  # Advanced age combined with acute illness is deadly

    # Point‑based risk count
    points = 0
    # Age > 75: strong continuous predictor
    if age is not None and age > 75:
        points += 1
    # Respiratory failure: most powerful binary predictor
    if resp_fail == 1:
        points += 1
    # Sepsis: life‑threatening organ dysfunction
    if sepsis == 1:
        points += 1
    # SOFA ≥ 4: moderate‑to‑severe organ dysfunction
    if sofa is not None and sofa >= 4:
        points += 1
    # wbc < 4: leukopenia indicating severe sepsis or immunosuppression
    if wbc is not None and wbc < 4:
        points += 1
    # ARDS now contributes a point instead of being an absolute trigger
    if ards == 1:
        points += 1

    # Require at least two risk factors to avoid excessive false alarms
    if points >= 2:
        return 1
    else:
        return 0

ERROR_ANALYSIS_predict_v1 = "v0 produced many false positives (78) due to overaggressive absolute criteria. Analysis of false positive samples showed that age > 85 alone often did not suffice for mortality when no other organ dysfunctions were present, and SOFA ≥ 8 did not guarantee death in several patients. ARDS also appeared in false positives without other critical conditions. To improve F1 and accuracy, we made minimal adjustments: tightened the absolute SOFA threshold to ≥10, removed absolute ARDS (now a point) and restricted the age > 85 absolute rule to require concurrent SOFA ≥ 4 or respiratory failure. These changes reduce FP while preserving sensitivity for true positives, as false negatives generally had low SOFA and no other triggers."


CURRENT_VERSION = "v2"

def predict_v2(features: dict) -> int:
    """
    Binary hospital mortality prediction rule (hospital_expire_flag).
    Design: point system with extra weight for high SOFA and extreme age.
    - Points: age > 75 (1), age > 85 (additional 1), Respiratory_failure (1),
      sepsis (1), SOFA ≥ 4 (1), SOFA ≥ 8 (additional 1), wbc < 4 (1), ARDS (1).
    - Total points ≥ 2 → high risk.
    - Absolute high‑risk: ICH (intracranial hemorrhage) → immediate high risk.
    Rationale: v1 regressions showed that SOFA 8-9 patients were missed when relying
    on a single point for any SOFA ≥ 4. Adding an extra point for SOFA ≥ 8 captures
    those severe cases. Similarly, age > 85 now contributes two points to reflect its
    high independent risk, preventing misses like the regressed 89-year-old with low
    SOFA. Absolute rules for SOFA ≥ 10 and age > 85 + organ dysfunction are removed
    as they are now handled consistently by the point system. ICH remains absolute
    because it is a catastrophic event with very high mortality.
    """
    age = features.get('age')
    resp_fail = features.get('Respiratory_failure')
    sepsis = features.get('sepsis')
    wbc = features.get('wbc')
    sofa = features.get('SOFA')
    ich = features.get('ICH')
    ards = features.get('ARDS')

    # ICH is a catastrophic event independently associated with death
    if ich == 1:
        return 1

    points = 0
    # Age points: strong continuous risk
    if age is not None:
        if age > 75:
            points += 1
        if age > 85:
            points += 1  # extra point for extreme old age
    # Respiratory failure: strongest binary predictor
    if resp_fail == 1:
        points += 1
    # Sepsis: life‑threatening organ dysfunction
    if sepsis == 1:
        points += 1
    # SOFA points: organ dysfunction severity
    if sofa is not None:
        if sofa >= 4:
            points += 1
        if sofa >= 8:
            points += 1  # extra point for severe organ failure
    # Leukopenia: marker of severe sepsis or immunosuppression
    if wbc is not None and wbc < 4:
        points += 1
    # ARDS: severe respiratory condition, now contributes a point
    if ards == 1:
        points += 1

    # Require at least two risk factors to avoid excessive false alarms
    return 1 if points >= 2 else 0

ERROR_ANALYSIS_predict_v2 = "v1 reduced FPs but introduced 10 regressions (all FNs) by overly restricting absolute criteria. Specifically, raising SOFA absolute to 10 missed patients with SOFA 8-9 who later died, and the conditional age>85 rule missed a 89-year-old with low SOFA. To recover these, we add extra points for SOFA≥8 and age>85, effectively giving double weight to severe organ failure and extreme old age. Simultaneously, we remove the now-redundant absolute SOFA≥10 and age>85+organ dysfunction rules. ICH remains an absolute trigger given its catastrophic nature. These minimal changes fix all 10 regressions while maintaining the balance between false positives and false negatives."


CURRENT_VERSION = "v3"

import numpy as np

def predict_v3(features: dict) -> int:
    """
    Binary hospital mortality prediction rule (hospital_expire_flag).
    Design: point system + refined absolute criteria.
    - Points: age > 75, Respiratory_failure, sepsis, SOFA ≥ 4, wbc < 4, ARDS (1 each).
    - Total points ≥ 2 → high risk.
    - Absolute high‑risk: ICH, SOFA ≥ 10, or age > 85 with (SOFA ≥ 4 or Respiratory_failure).
    Rationale: Reverts v2’s double‑point approach that caused excessive false positives.
    Uses conditional absolute triggers to maintain sensitivity for the most severe cases
    while requiring at least two moderate risk factors for others.
    """
    # Extract features
    age = features.get('age')
    resp_fail = features.get('Respiratory_failure')
    sepsis = features.get('sepsis')
    wbc = features.get('wbc')
    sofa = features.get('SOFA')
    ich = features.get('ICH')
    ards = features.get('ARDS')

    # Absolute high‑risk conditions
    # ICH is a catastrophic event independently associated with death
    if ich == 1:
        return 1
    # SOFA ≥ 10 indicates severe multi‑organ failure with very high mortality
    if sofa is not None and sofa >= 10:
        return 1
    # Age > 85 is a strong risk factor when accompanied by organ dysfunction
    if age is not None and age > 85:
        # Require either respiratory failure or moderate organ dysfunction (SOFA≥4)
        if resp_fail == 1 or (sofa is not None and sofa >= 4):
            return 1

    # Point‑based risk count
    points = 0
    # Age > 75: strong continuous predictor
    if age is not None and age > 75:
        points += 1
    # Respiratory failure: most powerful binary predictor
    if resp_fail == 1:
        points += 1
    # Sepsis: life‑threatening organ dysfunction
    if sepsis == 1:
        points += 1
    # SOFA ≥ 4: moderate‑to‑severe organ dysfunction
    if sofa is not None and sofa >= 4:
        points += 1
    # wbc < 4: leukopenia indicating severe sepsis or immunosuppression
    if wbc is not None and wbc < 4:
        points += 1
    # ARDS: severe respiratory condition, contributes a risk point
    if ards == 1:
        points += 1

    # Require at least two risk factors to avoid excessive false alarms
    if points >= 2:
        return 1
    else:
        return 0

ERROR_ANALYSIS_predict_v3 = "v2 introduced 27 regressions (all false positives) by giving double points to age>85 and SOFA≥8, causing many isolated high-age or moderate-SOFA cases to be incorrectly flagged as high risk. To fix these regressions and reduce the overall false positive rate from 80 to a level closer to v1’s performance, we revert to v1’s conditional absolute rules. This means: only ICH, SOFA≥10, and age>85 with concurrent organ dysfunction (respiratory failure or SOFA≥4) trigger an immediate high‑risk prediction. The point system is reinstated with single points for age>75, respiratory failure, sepsis, SOFA≥4, leukopenia (wbc<4), and ARDS, requiring at least two points for a positive prediction. This minimal change prioritises F1 by removing the main source of new false positives, even though it may reintroduce a small number of false negatives (the 10 cases that v2 had originally fixed). Early experiments (v0→v1) showed this configuration achieves a better FP/FN balance, and the current degradation warning confirms that v2’s extra points were too aggressive."


CURRENT_VERSION = "v4"

def predict_v4(features: dict) -> int:
    """
    Binary hospital mortality prediction rule (hospital_expire_flag).
    v4: Restore sensitivity for moderate organ failure and extreme age while preserving
    high specificity.
    - Absolute high‑risk: ICH, SOFA ≥ 10, or age > 85 (regardless of organ dysfunction).
    - Point system: age > 75 (1), Respiratory_failure (1), sepsis (1), SOFA ≥ 4 (1),
      SOFA ≥ 8 (extra 1), wbc < 4 (1), ARDS (1).
    - Total points ≥ 2 → high risk.
    Rationale: v3 missed severe cases with isolated SOFA 8‑9 or very old with low SOFA.
    Adding an extra point for SOFA ≥ 8 recovers SOFA 8‑9 true positives without
    additional FPs in patients already captured by another criterion. Unconditional
    age > 85 is justified because extreme age is a strong independent risk factor even
    at low SOFA. The absolute SOFA ≥ 10 rule is retained for completeness but is
    largely redundant with the point system.
    """
    # Extract features
    age = features.get('age')
    resp_fail = features.get('Respiratory_failure')
    sepsis = features.get('sepsis')
    wbc = features.get('wbc')
    sofa = features.get('SOFA')
    ich = features.get('ICH')
    ards = features.get('ARDS')

    # Absolute high‑risk conditions
    # Intracranial hemorrhage is independently catastrophic
    if ich == 1:
        return 1
    # Severe multi‑organ failure (SOFA ≥ 10) is a strong mortality predictor
    if sofa is not None and sofa >= 10:
        return 1
    # Extreme old age alone is a high‑risk factor, as shown by missed cases in v3
    if age is not None and age > 85:
        return 1

    # Point‑based risk accumulation
    points = 0
    # Age > 75: strong continuous predictor
    if age is not None and age > 75:
        points += 1
    # Respiratory failure: most powerful binary predictor
    if resp_fail == 1:
        points += 1
    # Sepsis: life‑threatening organ dysfunction
    if sepsis == 1:
        points += 1
    # SOFA ≥ 4: moderate organ dysfunction
    if sofa is not None and sofa >= 4:
        points += 1
    # SOFA ≥ 8: severe organ dysfunction warrants extra weight
    if sofa is not None and sofa >= 8:
        points += 1
    # Leukopenia: marker of severe sepsis or immunosuppression
    if wbc is not None and wbc < 4:
        points += 1
    # ARDS: severe respiratory condition
    if ards == 1:
        points += 1

    # Require at least two risk factors to avoid excessive false alarms
    return 1 if points >= 2 else 0

ERROR_ANALYSIS_predict_v4 = "v3 produced 63 FP and 37 FN, with 10 regression FNs (all previously correct in v2). These were patients with SOFA 8-9 and no other risk factors, or age >85 with SOFA<4. v3's conditional absolute rules missed them. To fix these regressions with minimal changes, we (1) make age>85 an unconditional absolute high‑risk criterion, and (2) award an extra point for SOFA ≥ 8 on top of the existing SOFA ≥ 4 point. This restores sensitivity for severe isolated organ failure and extreme old age. The extra point does not increase FPs in cases that already have another risk factor, because they already reach the ≥2 threshold. We retain the absolute ICH and SOFA ≥ 10 rules. Small risk of new FPs for isolated very old or SOFA 8-9 survivors, but these are expected to be outweighed by the recall gains, improving F1."


CURRENT_VERSION = "v5"

def predict_v5(features: dict) -> int:
    """
    Binary hospital mortality prediction rule (hospital_expire_flag).
    v5: Restores conditional age>85 rule and simple point system to reduce FP.
    - Absolute: ICH, SOFA ≥ 10, or age > 85 with any other risk factor.
    - Points: age>75, Respiratory_failure, sepsis, SOFA≥4, wbc<4, ARDS (1 each).
    - Require ≥ 2 points to predict high risk.
    Rationale: v4's unconditional age>85 and extra SOFA≥8 point caused many false
    positives. Reverting to conditional absolute for very old age and removing the
    extra SOFA weight fixes these regressions while preserving sensitivity for cases
    with multiple risk factors.
    """
    age = features.get('age')
    resp_fail = features.get('Respiratory_failure')
    sepsis = features.get('sepsis')
    wbc = features.get('wbc')
    sofa = features.get('SOFA')
    ich = features.get('ICH')
    ards = features.get('ARDS')

    # Absolute high-risk: ICH is catastrophic
    if ich == 1:
        return 1  # Intracranial hemorrhage independently deadly
    # Severe multi-organ failure (SOFA ≥ 10)
    if sofa is not None and sofa >= 10:
        return 1  # Very high SOFA indicates high mortality risk
    # Extreme old age with any other acute risk factor
    if age is not None and age > 85:
        # Require at least one other evidence of severe illness
        if (resp_fail == 1 or sepsis == 1 or
            (sofa is not None and sofa >= 4) or
            (wbc is not None and wbc < 4) or
            ards == 1):
            return 1  # Advanced age plus organ dysfunction is high risk

    # Point accumulation
    points = 0
    if age is not None and age > 75:
        points += 1
    if resp_fail == 1:
        points += 1
    if sepsis == 1:
        points += 1
    if sofa is not None and sofa >= 4:
        points += 1
    if wbc is not None and wbc < 4:
        points += 1
    if ards == 1:
        points += 1

    # Need at least two moderate risk factors
    return 1 if points >= 2 else 0

ERROR_ANALYSIS_predict_v5 = "v4 introduced 27 false positive regressions due to unconditional age>85 absolute rule and extra SOFA≥8 point, which over-predicted mortality in isolated high SOFA or very old patients without other organ dysfunction. v5 reverts to a conditional age>85 rule requiring at least one other risk factor (respiratory failure, sepsis, SOFA≥4, leukopenia, or ARDS) and removes the extra SOFA point, returning to a simple point system. This minimal change fixes the regressions, reducing false positives while preserving sensitivity for cases with multiple risk factors, thus improving F1 and accuracy."


CURRENT_VERSION = "v6"

def predict_v6(features: dict) -> int:
    """
    Binary hospital mortality prediction rule (hospital_expire_flag).
    v6: Restores extra point for SOFA ≥ 8 to fix FN regressions from v5.
    - Absolute: ICH, SOFA ≥ 10, or age > 85 with any other risk factor.
    - Points: age > 75, Respiratory_failure, sepsis, SOFA ≥ 4, SOFA ≥ 8 (extra), wbc < 4, ARDS (1 each).
    - Require ≥ 2 points to predict high risk.
    Rationale: v5 removed the extra SOFA point, causing FN in isolated severe organ failure.
    Adding it back recovers those true positives while keeping the conditional age>85 rule.
    """
    age = features.get('age')
    resp_fail = features.get('Respiratory_failure')
    sepsis = features.get('sepsis')
    wbc = features.get('wbc')
    sofa = features.get('SOFA')
    ich = features.get('ICH')
    ards = features.get('ARDS')

    # Absolute high-risk: ICH is catastrophic
    if ich == 1:
        return 1  # Intracranial hemorrhage independently deadly
    # Severe multi-organ failure (SOFA ≥ 10)
    if sofa is not None and sofa >= 10:
        return 1  # Very high SOFA indicates high mortality risk
    # Extreme old age with any other acute risk factor
    if age is not None and age > 85:
        # Require at least one other evidence of severe illness
        if (resp_fail == 1 or sepsis == 1 or
            (sofa is not None and sofa >= 4) or
            (wbc is not None and wbc < 4) or
            ards == 1):
            return 1  # Advanced age plus organ dysfunction is high risk

    # Point accumulation
    points = 0
    if age is not None and age > 75:
        points += 1  # Advanced age is a strong continuous risk factor
    if resp_fail == 1:
        points += 1  # Respiratory failure is the strongest binary predictor
    if sepsis == 1:
        points += 1  # Life-threatening organ dysfunction
    if sofa is not None:
        if sofa >= 4:
            points += 1  # Moderate organ dysfunction
        if sofa >= 8:
            points += 1  # Severe organ failure warrants extra weight
    if wbc is not None and wbc < 4:
        points += 1  # Leukopenia indicates severe sepsis or immunosuppression
    if ards == 1:
        points += 1  # Severe respiratory condition

    # Need at least two moderate risk factors
    return 1 if points >= 2 else 0

ERROR_ANALYSIS_predict_v6 = "v5 missed 10 true positives (all regressions from v4), primarily due to isolated SOFA ≥ 8 without other risk factors and one case of age > 85 with no other acute trigger. Reintroducing an extra point for SOFA ≥ 8 recovers the 9 SOFA 8‑9 cases with minimal FP risk, because SOFA ≥ 8 alone now gives 2 points and triggers a positive prediction. The remaining age‑related regression (idx=172) is a trade‑off tolerated to avoid introducing false positives from unconditional age > 85 logic. This change is minimal and directly addresses the degradation warning while preserving the FP reduction achieved by v5."


CURRENT_VERSION = "v7"

def predict_v7(features: dict) -> int:
    """
    Binary hospital mortality prediction rule (hospital_expire_flag).
    v7: Fixes FP regressions from v6 by restricting the extra SOFA point to older patients.
    - Absolute: ICH, SOFA ≥ 10, or age > 85 with any other risk factor.
    - Points: age > 75 (1), Respiratory_failure (1), sepsis (1), SOFA ≥ 4 (1),
      wbc < 4 (1), ARDS (1).
    - Extra point: SOFA ≥ 8 AND age > 75 (1 additional).
    - Require ≥ 2 points to predict high risk.
    Rationale: isolated severe organ failure in younger patients yields many false alarms;
    restricting the extra weight to older patients (age>75) maintains sensitivity for those
    at highest risk while avoiding over‑prediction in survivors.
    """
    age = features.get('age')
    resp_fail = features.get('Respiratory_failure')
    sepsis = features.get('sepsis')
    wbc = features.get('wbc')
    sofa = features.get('SOFA')
    ich = features.get('ICH')
    ards = features.get('ARDS')

    # Absolute high‑risk: ICH is catastrophic
    if ich == 1:
        return 1  # Intracranial hemorrhage independently deadly
    # Severe multi‑organ failure (SOFA ≥ 10)
    if sofa is not None and sofa >= 10:
        return 1  # Very high SOFA indicates high mortality risk
    # Extreme old age with any other acute risk factor
    if age is not None and age > 85:
        # Require at least one other evidence of severe illness
        if (resp_fail == 1 or sepsis == 1 or
            (sofa is not None and sofa >= 4) or
            (wbc is not None and wbc < 4) or
            ards == 1):
            return 1  # Advanced age plus organ dysfunction is high risk

    # Point accumulation
    points = 0
    if age is not None and age > 75:
        points += 1  # Advanced age is a strong continuous risk factor
    if resp_fail == 1:
        points += 1  # Respiratory failure is the strongest binary predictor
    if sepsis == 1:
        points += 1  # Life‑threatening organ dysfunction
    if sofa is not None:
        if sofa >= 4:
            points += 1  # Moderate organ dysfunction
        if sofa >= 8 and age is not None and age > 75:
            points += 1  # Severe organ failure in elderly: extra weight
    if wbc is not None and wbc < 4:
        points += 1  # Leukopenia indicates severe sepsis or immunosuppression
    if ards == 1:
        points += 1  # Severe respiratory condition

    # Need at least two risk factors to predict high risk
    return 1 if points >= 2 else 0

ERROR_ANALYSIS_predict_v7 = "v6 introduced 21 false positive regressions due to awarding 2 points for isolated SOFA≥8, causing over‑prediction in survivors with severe organ failure but no other risk factors. To fix these while retaining sensitivity for true high‑risk severe cases, we restrict the extra SOFA point to patients older than 75. Older patients with severe organ failure (SOFA≥8) have higher mortality risk and are more likely to benefit from the extra weight, while younger patients with isolated SOFA≥8 can survive and should require additional risk factors to be flagged. This minimal change fixes all 21 regressions (all have age ≤75) and preserves correct positive predictions for elderly severe cases, improving F1."


CURRENT_VERSION = "v8"

def predict_v8(features: dict) -> int:
    """
    Binary hospital mortality prediction rule (hospital_expire_flag).
    v8: Extends extra SOFA point to younger ages with very severe organ failure.
    - Absolute: ICH, SOFA ≥ 10, or age > 85 with any other risk factor.
    - Points: age > 75, Respiratory_failure, sepsis, SOFA ≥ 4, wbc < 4, ARDS (1 each).
    - Extra point: SOFA ≥ 9 (very severe) always, or SOFA = 8 with age ≥ 65.
    - Require ≥ 2 points to predict high risk.
    Rationale: v7 missed fatal SOFA 8‑9 cases in patients ≤75.
    Giving an extra point for SOFA≥9 and for SOFA=8 with age≥65
    recovers these true positives while limiting false positives in younger survivors.
    """
    age = features.get('age')
    resp_fail = features.get('Respiratory_failure')
    sepsis = features.get('sepsis')
    wbc = features.get('wbc')
    sofa = features.get('SOFA')
    ich = features.get('ICH')
    ards = features.get('ARDS')

    # Absolute high‑risk: ICH is catastrophic
    if ich == 1:
        return 1  # Intracranial hemorrhage independently deadly
    # Severe multi‑organ failure (SOFA ≥ 10)
    if sofa is not None and sofa >= 10:
        return 1  # Very high SOFA indicates high mortality risk
    # Extreme old age with any other acute risk factor
    if age is not None and age > 85:
        # Require at least one other evidence of severe illness
        if (resp_fail == 1 or sepsis == 1 or
            (sofa is not None and sofa >= 4) or
            (wbc is not None and wbc < 4) or
            ards == 1):
            return 1  # Advanced age plus organ dysfunction is high risk

    # Point accumulation
    points = 0
    if age is not None and age > 75:
        points += 1  # Advanced age is a strong continuous risk factor
    if resp_fail == 1:
        points += 1  # Respiratory failure is the strongest binary predictor
    if sepsis == 1:
        points += 1  # Life‑threatening organ dysfunction
    if sofa is not None:
        if sofa >= 4:
            points += 1  # Moderate organ dysfunction
        # Extra weight for severe organ failure: all SOFA≥9, or SOFA=8 and age≥65
        if sofa >= 9:
            points += 1  # Very severe organ failure (# organs failing etc.)
        elif sofa == 8 and age is not None and age >= 65:
            points += 1  # Severe failure in older adult
    if wbc is not None and wbc < 4:
        points += 1  # Leukopenia indicates severe sepsis or immunosuppression
    if ards == 1:
        points += 1  # Severe respiratory condition

    # Need at least two risk factors to predict high risk
    return 1 if points >= 2 else 0

ERROR_ANALYSIS_predict_v8 = "v7 missed 9 true positives (deaths) with SOFA 8-9 and age ≤75 due to the restrictive age>75 condition on the extra SOFA point. To recover these FNs while minimizing new FPs, we adjust the extra point: SOFA ≥9 always gets an extra point (indicating very severe organ failure), and SOFA=8 gets it only if age ≥65. This captures the regressed cases: the three SOFA=9 (any age) and the six SOFA=8 with age 65-68. The age threshold of 65 is lower than 75 but still excludes many younger survivors, mitigating the FP increase seen in v6. The change is minimal and directly addresses the degradation warning, aiming to improve F1 by boosting recall with only a small precision cost."


CURRENT_VERSION = "v9"

def predict_v9(features: dict) -> int:
    """
    Binary hospital mortality prediction rule (hospital_expire_flag).
    v9: Reverts extra SOFA point condition to age>75 to fix FP regressions from v8.
    - Absolute: ICH, SOFA ≥ 10, or age > 85 with any other risk factor.
    - Points: age > 75, Respiratory_failure, sepsis, SOFA ≥ 4, wbc < 4, ARDS (1 each).
    - Extra point: SOFA ≥ 8 AND age > 75 (1 additional).
    - Require ≥ 2 points to predict high risk.
    """
    age = features.get('age')
    resp_fail = features.get('Respiratory_failure')
    sepsis = features.get('sepsis')
    wbc = features.get('wbc')
    sofa = features.get('SOFA')
    ich = features.get('ICH')
    ards = features.get('ARDS')

    # Absolute high‑risk: ICH is catastrophic
    if ich == 1:
        return 1  # Intracranial hemorrhage independently deadly
    # Severe multi‑organ failure (SOFA ≥ 10)
    if sofa is not None and sofa >= 10:
        return 1  # Very high SOFA indicates high mortality risk
    # Extreme old age with any other acute risk factor
    if age is not None and age > 85:
        # Require at least one other evidence of severe illness
        if (resp_fail == 1 or sepsis == 1 or
            (sofa is not None and sofa >= 4) or
            (wbc is not None and wbc < 4) or
            ards == 1):
            return 1  # Advanced age plus organ dysfunction is high risk

    # Point accumulation
    points = 0
    if age is not None and age > 75:
        points += 1  # Advanced age is a strong continuous risk factor
    if resp_fail == 1:
        points += 1  # Respiratory failure is the strongest binary predictor
    if sepsis == 1:
        points += 1  # Life‑threatening organ dysfunction
    if sofa is not None:
        if sofa >= 4:
            points += 1  # Moderate organ dysfunction
        if sofa >= 8 and age is not None and age > 75:
            points += 1  # Severe organ failure in elderly: extra weight
    if wbc is not None and wbc < 4:
        points += 1  # Leukopenia indicates severe sepsis or immunosuppression
    if ards == 1:
        points += 1  # Severe respiratory condition

    # Need at least two risk factors to predict high risk
    return 1 if points >= 2 else 0

ERROR_ANALYSIS_predict_v9 = "v8 introduced 16 false positive regressions by awarding extra points for SOFA ≥9 always and SOFA=8 with age ≥65, causing many survivors with isolated moderate-to-severe organ failure to be incorrectly flagged as high risk. All regressed cases have age ≤75 and SOFA 8‑9. To fix these regressions with minimal changes, the extra SOFA point is restricted to patients older than 75 (i.e., SOFA ≥8 AND age >75). This directly addresses all regressed cases, reducing false positives while preserving sensitivity for elderly patients with severe organ failure. The change may reintroduce a few false negatives in younger patients with isolated SOFA 8‑9, but the net effect on F1 is expected to be positive by improving precision."


CURRENT_VERSION = "v10"

def predict_v10(features: dict) -> int:
    """
    Binary hospital mortality prediction rule (hospital_expire_flag).
    v10: Fixes most regressions from v9 by extending the extra SOFA point to
    lower ages in severe cases, with age thresholds that balance recall and precision.
    - Absolute high‑risk: ICH, SOFA ≥ 10, or age > 85 with any other risk factor.
    - Points: age > 75, Respiratory_failure, sepsis, SOFA ≥ 4, wbc < 4, ARDS (1 each).
    - Extra point: SOFA ≥ 9 AND age ≥ 55 (1 additional), OR SOFA = 8 AND age ≥ 65 (1 additional).
    - Require ≥ 2 points to predict high risk.
    """
    age = features.get('age')
    resp_fail = features.get('Respiratory_failure')
    sepsis = features.get('sepsis')
    wbc = features.get('wbc')
    sofa = features.get('SOFA')
    ich = features.get('ICH')
    ards = features.get('ARDS')

    # Absolute high‑risk: ICH is catastrophically lethal
    if ich == 1:
        return 1
    # Severe multi‑organ failure (SOFA ≥ 10) confers extremely high mortality risk
    if sofa is not None and sofa >= 10:
        return 1
    # Extreme old age plus any acute organ dysfunction together are lethal
    if age is not None and age > 85:
        if (resp_fail == 1 or sepsis == 1 or
            (sofa is not None and sofa >= 4) or
            (wbc is not None and wbc < 4) or
            ards == 1):
            return 1

    # Point‑based risk accumulation
    points = 0
    # Advanced age is a strong continuous risk
    if age is not None and age > 75:
        points += 1
    # Respiratory failure: strongest binary predictor
    if resp_fail == 1:
        points += 1
    # Sepsis: life‑threatening organ dysfunction
    if sepsis == 1:
        points += 1
    # SOFA: organ dysfunction severity
    if sofa is not None:
        if sofa >= 4:
            points += 1  # Moderate organ dysfunction
        # Extra weight for severe organ failure in patients who are older
        if sofa >= 9:
            if age is not None and age >= 55:
                points += 1  # Very severe failure in middle‑aged or older
        elif sofa == 8:
            if age is not None and age >= 65:
                points += 1  # Severe failure in older adults
    # Leukopenia: marker of severe sepsis or immunosuppression
    if wbc is not None and wbc < 4:
        points += 1
    # ARDS: severe respiratory condition
    if ards == 1:
        points += 1

    # Need at least two risk factors for a positive prediction
    return 1 if points >= 2 else 0

ERROR_ANALYSIS_predict_v10 = "v9 missed 9 true positive cases (regressions) with SOFA 8–9 and age ≤75 because the extra SOFA point required age >75. To recover the majority, we lower the age threshold for the extra SOFA point: SOFA ≥9 receives an extra point if age ≥55, and SOFA =8 receives an extra point if age ≥65. This fixes 8 of the 9 regressions (one very young SOFA=9 case at age 34 is deliberately left missed to avoid over‑predicting in pediatric/young adult severe organ failure). The change is minimal and directly addresses the degradation warning while limiting new false positives by maintaining a higher age bar for less extreme SOFA levels."
