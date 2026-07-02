"""Core KARAT correction logic."""
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from .metrics import compute_cka, compute_l2_divergence


@dataclass
class KARATConfig:
    theta: float = 0.00279        # CKA drop threshold (from val p20)
    blend_range: float = 0.15     # CKA drop range over which blend ramps
    max_blend: float = 0.85       # Maximum teacher blend weight
    l2_theta: float = 0.000988    # L2 divergence threshold (optional)


class KARATCorrector:
    """
    KARAT: CKA-triggered teacher-student score blending.

    At inference time:
      1. Compute CKA(teacher_proba, student_proba) on a reference window.
      2. If CKA drop > theta, blend student scores toward teacher scores.
      3. Blend weight scales linearly with the CKA drop magnitude.

    Parameters
    ----------
    teacher     : fitted teacher model with predict_proba(X)
    student     : fitted student model with predict_proba(X)
    scaler      : fitted StandardScaler for student input
    features    : list of feature column names
    config      : KARATConfig with threshold parameters
    base_cka    : CKA at clean baseline (computed from val split)
    """

    def __init__(self, teacher, student, scaler, features: list,
                 config: KARATConfig = None, base_cka: float = None):
        self.teacher  = teacher
        self.student  = student
        self.scaler   = scaler
        self.features = features
        self.cfg      = config or KARATConfig()
        self.base_cka = base_cka
        self._history  = []

    def _teacher_proba(self, df):
        return self.teacher.predict_proba(df[self.features])

    def _student_proba(self, df):
        return self.student.predict_proba(
            self.scaler.transform(df[self.features].values))

    def predict_proba(self, df, return_meta: bool = False):
        """
        Return corrected probability scores for df.

        If base_cka is set, measures current CKA drop and applies
        correction. Otherwise returns raw student scores.
        """
        tp = self._teacher_proba(df)
        sp = self._student_proba(df)

        if self.base_cka is None:
            return sp

        current_cka = compute_cka(tp, sp)
        cka_drop    = self.base_cka - current_cka
        triggered   = cka_drop > self.cfg.theta

        if not triggered:
            corrected = sp.copy()
            blend     = 0.0
        else:
            blend = min(
                (cka_drop - self.cfg.theta) / self.cfg.blend_range,
                self.cfg.max_blend
            )
            corrected = (1 - blend) * sp + blend * tp
            row_sums  = corrected.sum(axis=1, keepdims=True)
            corrected = corrected / (row_sums + 1e-8)

        self._history.append({
            'cka': current_cka,
            'cka_drop': cka_drop,
            'triggered': triggered,
            'blend': blend,
        })

        if return_meta:
            return corrected, {
                'cka': current_cka,
                'cka_drop': cka_drop,
                'triggered': triggered,
                'blend': blend,
            }
        return corrected

    @classmethod
    def fit_threshold(
        cls, teacher, student, scaler, features: list,
        df_val, scenarios, timesteps, seed: int = 42,
        inject_fn=None, percentile: float = 20.0,
    ) -> 'KARATCorrector':
        """
        Fit CKA threshold from a validation split.

        Computes CKA drops across all scenario × timestep combinations
        on df_val, sets theta = p{percentile} of observed drops,
        and returns a fitted KARATCorrector.
        """
        from .attack import inject_adversarial
        attack_fn = inject_fn or inject_adversarial

        def tp(df): return teacher.predict_proba(df[features])
        def sp(df): return student.predict_proba(
                        scaler.transform(df[features].values))

        base_cka = compute_cka(tp(df_val), sp(df_val))
        drops = []
        for sc in scenarios:
            for t in timesteps[1:]:
                df_tv = attack_fn(df_val, t, sc, seed)
                drops.append(base_cka - compute_cka(tp(df_tv), sp(df_tv)))

        theta    = max(float(np.percentile(drops, percentile)), 1e-5)
        trig_r   = np.mean([d > theta for d in drops])
        pos_l2   = []
        base_l2  = compute_l2_divergence(tp(df_val), sp(df_val))
        for sc in scenarios:
            for t in timesteps[1:]:
                df_tv = attack_fn(df_val, t, sc, seed)
                d = compute_l2_divergence(tp(df_tv), sp(df_tv)) - base_l2
                if d > 0:
                    pos_l2.append(d)
        l2_theta = (float(np.percentile(pos_l2,
                                         max(0, (1 - trig_r) * 100)))
                    if len(pos_l2) >= 3 else 1e-5)
        l2_theta = max(l2_theta, 1e-5)

        cfg = KARATConfig(theta=theta, l2_theta=l2_theta)
        corrector = cls(teacher, student, scaler, features,
                        config=cfg, base_cka=base_cka)
        print(f"[KARAT] theta={theta:.5f}  l2_theta={l2_theta:.6f}  "
              f"trigger_rate={trig_r:.2f}")
        return corrector
