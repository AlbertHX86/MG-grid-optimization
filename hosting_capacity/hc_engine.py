#!/usr/bin/env python3
"""DC network engine for hosting-capacity screening on the EPE 2035 PWF.

The PWF export is an MG-centred equivalent: MG buses plus boundary buses of the
neighbouring states. Boundary buses carry large equivalent injections that are not
in the bus table's `pg`/`pd` columns (e.g. bus 4943 EST-XI-MG500 at Estreito, which
includes the Xingu-Estreito HVDC infeed). This engine therefore derives every bus's
net injection from the AC branch flows (`implied_injection`) instead of trusting
`pg - pd`.

Model: incremental DC flows anchored on the AC P_from flows of the reference case,
    f = F_ac + PTDF @ dp,      post-contingency  f_l|k = f_l + LODF_lk f_k.
This is a screening approximation (no AC voltage/reactive limits).
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INPUTS = ROOT / 'MG_Step2_Congestion_Relief' / 'MG_Nodal_Connection_Optimization' / 'inputs'
MG_STATE = 11
REF_BUS = 585          # angle reference only; every balancing case below is explicit
STAR_KV = 999.0        # fictitious star-point bus of a three-winding transformer

# Rough series reactance per km on a 100 MVA base, used ONLY to estimate the
# length of lines whose end substations are not geocoded (labelled as estimates).
X_PU_PER_KM = {500: 1.3e-4, 440: 1.7e-4, 345: 3.1e-4, 230: 9.1e-4}


@dataclass
class Rows:
    """Constraint rows for one injection pattern: base flow, sensitivity, limit."""
    f0: np.ndarray      # base flow (MW)
    g: np.ndarray       # MW per MW of pattern
    lim: np.ndarray     # thermal limit (MW proxy)
    elem: np.ndarray    # monitored branch index
    cont: np.ndarray    # contingency branch index (-1 = N-0)


class Network:
    def __init__(self, inputs: Path = INPUTS):
        B = pd.read_csv(inputs / 'pwf_2035_buses.csv')
        L = pd.read_csv(inputs / 'pwf_2035_branches.csv')
        assert len(B) == 1489 and len(L) == 1890 and B.id.is_unique
        self.B, self.L = B, L
        self.n, self.m = len(B), len(L)
        self.bus_ids = B.id.to_numpy(int)
        self.idx = {int(b): i for i, b in enumerate(self.bus_ids)}
        self.f = L.f.map(self.idx).to_numpy(int)
        self.t = L.t.map(self.idx).to_numpy(int)
        self.kv = B.kv.to_numpy(float)
        self.state = B.state.to_numpy(int)
        x, tap = L.x.to_numpy(float), L.tap.to_numpy(float)
        assert np.all(x != 0) and np.all(tap > 0)
        self.w = 1.0 / (x * tap)                       # same convention as optimize_siting.py
        self.F = L.p_from_mw.to_numpy(float)           # AC anchor flows
        self.rate = L.rate.to_numpy(float)
        em = L.emrate.to_numpy(float)
        self.emrate = np.where(em > 0, np.maximum(em, self.rate), self.rate)
        # Dense incidence and PTDF (1,890 x 1,489 is small).
        A = np.zeros((self.m, self.n))
        A[np.arange(self.m), self.f] = 1.0
        A[np.arange(self.m), self.t] = -1.0
        self.A = A
        Bbus = A.T @ (self.w[:, None] * A)
        ref = self.idx[REF_BUS]
        free = np.array([j for j in range(self.n) if j != ref])
        Binv = np.zeros((self.n, self.n))
        Binv[np.ix_(free, free)] = np.linalg.inv(Bbus[np.ix_(free, free)])
        self.PTDF = (self.w[:, None] * A) @ Binv        # flow per MW injected at bus, withdrawn at REF_BUS
        self.p_implied = A.T @ self.F                  # net injection implied by AC branch flows
        self._build_sets()

    # ------------------------------------------------------------------ sets
    def _build_sets(self):
        L, kv, st = self.L, self.kv, self.state
        kvf, kvt = kv[self.f], kv[self.t]
        mg = (st[self.f] == MG_STATE) | (st[self.t] == MG_STATE)
        rated = L.rate.between(100, 9998).to_numpy()
        is_line = (L.kind == 'line').to_numpy()
        is_tx = (L.kind == 'transformer').to_numpy()
        # Lines: identical to the 283-line set of the original study.
        lines = is_line & rated & (L.x.to_numpy() > 0) & (np.minimum(kvf, kvt) >= 230) & mg
        # Two-winding network transformers (both windings >= 138 kV, one >= 230 kV).
        real = (kvf != STAR_KV) & (kvt != STAR_KV)
        tx2 = is_tx & real & rated & mg & (np.minimum(kvf, kvt) >= 138) & (np.maximum(kvf, kvt) >= 230)
        # Three-winding units: legs to a star bus; the winding kV is the real end's kV.
        star_buses = np.flatnonzero(kv == STAR_KV)
        tx3_mon = np.zeros(self.m, bool)
        unit_hv_leg = []
        self.unit_of_leg = {}
        for s in star_buses:
            legs = np.flatnonzero(is_tx & ((self.f == s) | (self.t == s)))
            if len(legs) == 0:
                continue
            wkv = np.array([kv[self.t[b]] if self.f[b] == s else kv[self.f[b]] for b in legs])
            if (wkv >= 138).sum() < 2 or wkv.max() < 230:
                continue                                 # generator step-up / distribution unit
            if not (st[s] == MG_STATE or any(mg[b] for b in legs)):
                continue
            for b, k in zip(legs, wkv):
                self.unit_of_leg[int(b)] = int(s)
                if k >= 138 and rated[b]:
                    tx3_mon[b] = True
            unit_hv_leg.append(int(legs[int(np.argmax(wkv))]))
        self.monitored = np.flatnonzero(lines | tx2 | tx3_mon)
        self.kind_mon = np.where(lines[self.monitored], 'line',
                                 np.where(tx2[self.monitored], 'transformer', 'transformer-3w-leg'))
        cont = np.r_[np.flatnonzero(lines | tx2), np.array(unit_hv_leg, int)]
        cont = np.unique(cont)
        # Drop radial outages (bridges): they island a bus and have no LODF.
        ptdf_kk = self.PTDF[cont, self.f[cont]] - self.PTDF[cont, self.t[cont]]
        denom = 1.0 - ptdf_kk
        self.bridges = cont[np.abs(denom) < 1e-6]
        keep = np.abs(denom) >= 1e-6
        self.contingencies = cont[keep]
        Pmk = self.PTDF[np.ix_(self.monitored, self.f[self.contingencies])] - \
              self.PTDF[np.ix_(self.monitored, self.t[self.contingencies])]
        self.LODF = Pmk / denom[keep][None, :]
        # an element is never monitored under its own outage
        self.self_mask = self.monitored[:, None] == self.contingencies[None, :]
        self.n_lines = int(lines.sum())

    # ------------------------------------------------------------ injections
    def bus_vector(self, weights: dict[int, float]) -> np.ndarray:
        v = np.zeros(self.n)
        for b, x in weights.items():
            v[self.idx[int(b)]] += float(x)
        return v

    def balancing(self, case: str) -> tuple[np.ndarray, float, str]:
        """Withdrawal weights (sum 1) and energy headroom (MW) for an absorption case."""
        B = self.B
        if case == 'A1_mg_backdown':
            # MG hydro (UHE/PCH) and thermal (UTE) reduce output in proportion to output.
            sel = (B.state == MG_STATE) & B.gen_type.isin([3, 5, 7]) & (B.pg > 0)
            w = np.where(sel, B.pg, 0.0)
            label = 'MG hydro + thermal back-down (UHE/PCH/UTE, proportional to output)'
        elif case == 'A2_displace_inflow':
            # Boundary equivalents that currently inject (northern/central inflows,
            # incl. the Estreito HVDC terminal) inject less.
            sel = B.boundary.to_numpy() & (self.p_implied > 1.0)
            w = np.where(sel, self.p_implied, 0.0)
            label = 'Displace boundary inflows (BA/GO equivalents + Estreito HVDC infeed)'
        elif case == 'B_export':
            # Extra outflow at boundary equivalents that currently withdraw (SP/RJ/ES side).
            sel = B.boundary.to_numpy() & (self.p_implied < -1.0)
            w = np.where(sel, -self.p_implied, 0.0)
            label = 'Export to the rest of the SIN at current outflow boundaries (SP/RJ/ES)'
        elif case == 'legacy_load':
            sel = (B.state == MG_STATE) & (B.pd > 0)
            w = np.where(sel, B.pd, 0.0)
            label = 'LEGACY: withdraw at MG load buses in proportion to load'
        elif case == 'legacy_gen':
            sel = (B.state == MG_STATE) & (B.pg > 0)
            w = np.where(sel, B.pg, 0.0)
            label = 'LEGACY attribution test: withdraw at MG generator buses by pg'
        else:
            raise ValueError(case)
        w = np.asarray(w, float)
        headroom = float(w.sum()) if case in ('A1_mg_backdown', 'A2_displace_inflow') else np.inf
        return w / w.sum(), headroom, label

    def load_weights(self) -> np.ndarray:
        B = self.B
        w = np.where((B.state == MG_STATE) & (B.pd > 0), B.pd, 0.0).astype(float)
        return w / w.sum()

    # ------------------------------------------------------------------ rows
    def rows(self, dp: np.ndarray, n1: bool = True) -> Rows:
        """Constraint rows for a net-injection pattern dp (sum ~ 0) applied with scale 1."""
        assert abs(dp.sum()) < 1e-6 * max(1.0, np.abs(dp).sum())
        M, K = self.monitored, self.contingencies
        g_all = self.PTDF @ dp
        f0 = [self.F[M]]; g = [g_all[M]]; lim = [self.rate[M]]
        elem = [M]; cont = [np.full(len(M), -1)]
        if n1:
            F_post = self.F[M][:, None] + self.LODF * self.F[K][None, :]
            G_post = g_all[M][:, None] + self.LODF * g_all[K][None, :]
            keep = ~self.self_mask
            ee, kk = np.nonzero(keep)
            f0.append(F_post[keep]); g.append(G_post[keep])
            lim.append(self.emrate[M][ee]); elem.append(M[ee]); cont.append(K[kk])
        return Rows(*(np.concatenate(a) for a in (f0, g, lim, elem, cont)))

    @staticmethod
    def max_scale(r: Rows, dfax_cut: float, total: float = 1.0, relieved=frozenset()):
        """Largest s >= 0 with all rows feasible for flows f0 + s*g.

        Rows whose base flow already exceeds the limit may not get worse, but only when the
        pattern's distribution factor |g|/total >= dfax_cut (pre-existing problems with a tiny
        contribution are not attributed to the new resources). `relieved` = monitored branches
        treated as upgraded (their rows are dropped). Returns (s_max, row index or -1).
        """
        pre = np.abs(r.f0) > r.lim + 1e-6
        dfax = np.abs(r.g) / max(total, 1e-12)
        active = ~(pre & (dfax < dfax_cut))
        if relieved:
            active &= ~np.isin(r.elem, list(relieved))
        # An existing overload may not grow in its own direction; the opposite direction
        # keeps the normal limit (a relieved row may not overshoot past -lim).
        hi = np.maximum(r.lim, r.f0)
        lo = np.maximum(r.lim, -r.f0)
        with np.errstate(divide='ignore', invalid='ignore'):
            up = np.where(r.g > 1e-12, (hi - r.f0) / r.g, np.inf)
            dn = np.where(r.g < -1e-12, (lo + r.f0) / (-r.g), np.inf)
        cap = np.where(active, np.minimum(up, dn), np.inf)
        # Round before argmin so exact ties (e.g. the legs of one three-winding unit) resolve
        # to the first row instead of to floating-point noise.
        j = int(np.argmin(np.round(cap, 6)))
        return (float(cap[j]), j) if np.isfinite(cap[j]) else (np.inf, -1)

    # ------------------------------------------------------------- describe
    def branch_label(self, b: int) -> dict:
        b = int(b)
        L = self.L.iloc[b]
        nm = self.B.name.to_numpy()
        return {'branch': b, 'from_bus': int(L.f), 'to_bus': int(L.t), 'circuit': int(L.c),
                'from_name': nm[self.f[b]], 'to_name': nm[self.t[b]], 'kind': L.kind,
                'kv': float(min(k for k in (self.kv[self.f[b]], self.kv[self.t[b]]) if k != STAR_KV)),
                'rate_mw': float(self.rate[b]), 'emrate_mw': float(self.emrate[b])}

    def line_length_km(self, b: int, coords: dict[int, tuple[float, float]]):
        """Great-circle distance between geocoded end stations; otherwise a reactance estimate."""
        L = self.L.iloc[int(b)]
        a, c = coords.get(int(L.f)), coords.get(int(L.t))
        if a and c:
            lat1, lon1, lat2, lon2 = map(np.radians, (a[0], a[1], c[0], c[1]))
            h = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
            d = 2 * 6371.0 * np.arcsin(np.sqrt(h))
            if d >= 1.0:
                return float(d), 'geocoded straight line'
        kvb = min(k for k in (self.kv[self.f[int(b)]], self.kv[self.t[int(b)]]) if k != STAR_KV)
        per = X_PU_PER_KM.get(int(kvb))
        if per is None or L.x <= 0:
            return float('nan'), 'unknown'
        return float(L.x / per), 'estimated from reactance'
