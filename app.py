import copy
import datetime as dt
import hashlib
import io
import json
import math
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from fpdf import FPDF


# ==============================================================================
# 1. EXACT DECIMAL MONEY HELPERS & CONSTANTS (NUR-3 | AP CALCULUS | ADVANCED)
# ==============================================================================

CENT = Decimal("0.01")


def to_money(val: Any) -> Decimal:
    """Convert any numeric/string value into exact two-decimal USD Decimal."""
    if isinstance(val, Decimal):
        return val.quantize(CENT, rounding=ROUND_HALF_UP)
    return Decimal(str(val)).quantize(CENT, rounding=ROUND_HALF_UP)


def fmt_money(val: Any) -> str:
    d = to_money(val)
    sign = "-" if d < 0 else ""
    return f"{sign}${abs(d):,.2f}"


def sanitize_pdf_text(text: Any) -> str:
    """Sanitize Unicode punctuation/symbols to Latin-1 safe text for FPDF Helvetica."""
    s = str(text)
    replacements = {
        "\u2014": "--",
        "\u2013": "-",
        "\u2022": "*",
        "\u2264": "<=",
        "\u2265": ">=",
        "\u2260": "!=",
        "\u2248": "~=",
        "\u2032": "'",
        "\u222b": "Integral",
        "\u03c0": "pi",
        "\u03b1": "alpha",
        "\u2192": "->",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u00d7": "x",
    }
    for k, v in replacements.items():
        s = s.replace(k, v)
    return s.encode("latin-1", errors="replace").decode("latin-1")


NUR_FIXTURE_ROWS: List[Dict[str, Any]] = [
    {"month": "2027-01", "requested": 80, "capacity_override": 60, "ad_cost": "0.00"},
    {"month": "2027-02", "requested": 88, "capacity_override": 60, "ad_cost": "100.00"},
    {"month": "2027-03", "requested": 72, "capacity_override": 45, "ad_cost": "0.00"},
]

NUR_OFFERINGS = [
    {
        "offering_id": "OFF-NUR-01",
        "name": "General wellness education workshop (Primary)",
        "unit": "workshop seat",
        "price": "25.00",
        "demand_rule": "q(p) = max(0, 160 - 3.2p)",
        "capacity_rule": "min(floor(staff_hours / hours_per_unit), venue_seats)",
    },
    {
        "offering_id": "OFF-NUR-02",
        "name": "Caregiver information session (Secondary Module)",
        "unit": "workshop seat",
        "price": "25.00",
        "demand_rule": "Separate allocation only; cannot duplicate primary revenue",
        "capacity_rule": "Shares staff_hours and venue_seats pool when enabled",
    },
    {
        "offering_id": "OFF-NUR-03",
        "name": "First-aid awareness demonstration (Secondary Module)",
        "unit": "workshop seat",
        "price": "25.00",
        "demand_rule": "Separate allocation only; cannot duplicate primary revenue",
        "capacity_rule": "Shares staff_hours and venue_seats pool when enabled",
    },
]

AP_CALC_SKILLS = {
    "Foundation": {
        "skill_name": "Unconstrained Profit Function & First-Derivative Critical Point Analysis",
        "business_question": "Ignoring seat and educator capacity limits, at what ticket price p does marginal profit P'(p) equal zero, and what are the units of P'(p)?",
        "equation_latex": r"q(p) = a - bp, \quad P(p) = (p - v)(a - bp) - F, \quad P'(p) = a + bv - 2bp",
        "domain_text": "0 <= p <= 2*p0 (positive linear demand domain; unconstrained capacity assumption)",
        "units_text": "p in $/seat; q(p) in seats/month; P(p) in $/month; P'(p) in ($/month)/($/seat) = seats/month",
        "assumptions": "Linear demand q(p) = 2*D0 - (D0/p0)*p, constant unit variable cost v, constant monthly fixed cost F, and unconstrained capacity.",
        "required_explanation": "Compute P'(p) = 0 to find p_unc = (a + b*v)/(2*b), test endpoints p = 0 and p = 2*p0, and explain why the unconstrained critical point may exceed venue/staff capacity.",
    },
    "Applied": {
        "skill_name": "Piecewise Capacity-Constrained Optimization & Corner Non-Differentiability",
        "business_question": "When completed seats are capped by s(p) = min(cap_eff, max(0, a - bp)), where does the capacity corner occur, why is P'(p) undefined there, and what is the true constrained optimum?",
        "equation_latex": r"s(p) = \min(\text{cap}_{\text{eff}}, \max(0, a - bp)), \quad P(p) = (p - v)s(p) - F, \quad p_{\text{corner}} = \frac{a - \text{cap}_{\text{eff}}}{b}",
        "domain_text": "0 <= p <= 2*p0 with piecewise branches [0, p_corner) and (p_corner, 2*p0]",
        "units_text": "p in $/seat; s(p) in seats/month; P(p) in $/month; P'(p) in seats/month (left-hand vs. right-hand limits)",
        "assumptions": "Continuous relaxation used for derivatives; rounded whole-unit seats evaluated separately for actual ledger transactions.",
        "required_explanation": "Compare P'(p) on the capacity branch (P' = cap_eff > 0) vs. demand branch (P' = a + bv - 2bp), check the capacity corner p = (a - cap_eff)/b and endpoints, and compare continuous vs. integer seat sales.",
    },
    "Advanced": {
        "skill_name": "Constrained Piecewise Optimization, Definite-Integral Rate Accumulation & Sensitivity",
        "business_question": "How do interacting constraints (venue capacity, educator hours, no-show rate, -20% venue disruption) and definite integrals of daily campaign/marginal cost rates affect optimal pricing and accumulated net value?",
        "equation_latex": r"P_{\text{constrained}}(p) = (p - v)\min(\text{cap}_{\text{eff}}, a - bp) - F, \qquad \text{Net Accumulation} = \int_{0}^{T} \left(r(t) - c(t)\right) dt",
        "domain_text": "Price domain 0 <= p <= 2*p0; Campaign rate time domain 0 <= t <= T days (isolated from monthly ledger)",
        "units_text": "r(t), c(t) in $/day; t in days; Definite integral in $; Sensitivity deltas in $/month and $/seat",
        "assumptions": "Isolated daily rate exercise r(t)=100+20t, c(t)=60+5t over [0,10] does not overwrite monthly ledger rows; scenario multipliers (0.90/1.00/1.10) are deterministic assumptions, not confidence intervals.",
        "required_explanation": "Defend the constrained corner optimum under capacity/no-show interactions, evaluate the -20% venue disruption shift, interpret the definite integral with exact units, and state model extrapolation limits.",
    },
}


# ==============================================================================
# 2. DEFAULT STATE & DETERMINISTIC SIMULATION ENGINE
# ==============================================================================


def compute_snapshot_id(state: Dict[str, Any]) -> str:
    payload = json.dumps(
        {
            "team_id": state["team_id"],
            "pathway_id": state["pathway_id"],
            "state_version": state["state_version"],
            "simulation_date": state["simulation_date"],
            "params": state["params"],
            "ledger_len": len(state["ledger"]),
            "closing_cash": state["balances"]["cash"],
        },
        sort_keys=True,
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8].upper()
    return f"SNAP-{state['pathway_id']}-V{state['state_version']}-{digest}"


def build_default_state(team_id: str = "TEAM-NUR3-01") -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "team_id": team_id,
        "course": "AP Calculus",
        "pathway_id": "NUR-3",
        "business_name": "Nurse-Led Health Education Simulation",
        "selected_skill": "Constrained Piecewise Optimization, Definite-Integral Rate Accumulation & Sensitivity",
        "selected_level": "Advanced",
        "teacher_approved_level": "Advanced",
        "level_history": [
            {
                "version": 1,
                "level": "Advanced",
                "status": "Teacher Approved (Default)",
                "timestamp": "2027-01-01 08:00:00",
            }
        ],
        "state_version": 1,
        "simulation_date": "2027-01-31",
        "base_period": "2027-01",
        "period_unit": "Monthly",
        "working_days_per_month": 20,
        "calendar_days_per_month": 31,
        "currency": "USD",
        "random_seed": 2027,
        "params": {
            "base_price": "25.00",
            "D0": 80,
            "requested_demand": 80,
            "staff_hours": 30.0,
            "hours_per_unit": 0.5,
            "venue_seats": 60,
            "no_show_rate": 0.0,
            "satisfaction_mean": 4.0,
            "active_offering": "General wellness education workshop (Primary)",
        },
        "variable_cost_lines": {
            "educator_delivery": "2.00",
            "materials": "2.00",
            "cleaning_consumables": "1.00",
        },
        "fixed_cost_lines": {
            "venue": "250.00",
            "scheduling_software": "100.00",
            "accessibility_support": "200.00",
            "admin_insurance_assumption": "150.00",
        },
        "startup_financing": {
            "owner_contribution": "6000.00",
            "grant_funding": "0.00",
            "loan_principal": "0.00",
            "loan_annual_rate_pct": 0.0,
            "equipment_assets": "1500.00",
            "pre_opening_expense": "500.00",
            "opening_inventory": "0.00",
            "post_startup_cash": "4000.00",
            "opening_retained_earnings": "-500.00",
            "reserve_threshold": "800.00",
        },
        "balances": {
            "cash": "4000.00",
            "equipment_net": "1500.00",
            "inventory": "0.00",
            "debt": "0.00",
            "paid_in_capital": "6000.00",
            "retained_earnings": "-500.00",
        },
        "ledger": [],
        "executed_run_ids": [],
        "campaigns": [
            {
                "campaign_id": "CAMP-NUR-001",
                "offering": "General wellness education workshop (Primary)",
                "audience": "Community adults & family caregivers (fictional nonclinical outreach)",
                "approved_claim": "Evidence-based general wellness & preventive health literacy workshop led by licensed nurse educators. Educational only; no diagnosis or clinical care.",
                "channel": "Community Clinic Bulletin & Accessible Flyer",
                "dates": "2027-02-01 to 2027-02-28",
                "design_hours": 2.0,
                "design_hourly_rate": "25.00",
                "in_kind_student_labor": False,
                "licensed_assets": "10.00",
                "print_quantity": 100,
                "print_unit_cost": "0.20",
                "distribution": "10.00",
                "placement": "10.00",
                "total_cost": "100.00",
                "reach_assumed": 600,
                "inquiries": 45,
                "conversions": 20,
                "status": "Draft",
                "fact_checked": True,
                "reviewer_approved": False,
                "charged_in_ledger": False,
                "variant_label": "Variant A (Plain-Language Visual Flyer)",
            }
        ],
        "recommendations": [],
        "decisions": [],
        "cross_course": {
            "conn_1_subject": "English / Health Literacy Communication",
            "conn_1_concept": "Plain-Language Source Evaluation & Rhetorical Clarity",
            "conn_1_complexity": "Critique evidence and defend trade-offs",
            "conn_1_artifact": "CDC/MedlinePlus health-literacy checklist + revised nonclinical workshop flyer",
            "conn_1_explanation": "Evaluates readability (6th-8th grade level), removes clinical/diagnostic ambiguity, and ensures every claim on the workshop flyer is backed by teacher-approved public health education sources.",
            "conn_2_subject": "Science / Public Health & Ergonomics",
            "conn_2_concept": "Physical Venue Capacity Constraints & Survey Measurement Error",
            "conn_2_complexity": "Apply and compare",
            "conn_2_artifact": "Venue room-occupancy & sanitation turnaround log + Likert satisfaction bias note",
            "conn_2_explanation": "Verifies why venue_seats (60) and educator turnaround time (0.5 hr/seat) enforce hard physical caps, and why self-reported satisfaction (4.0/5.0) is ordinal/non-random and cannot prove clinical health outcomes.",
        },
        "student_evidence": {
            "prediction_before_slider": "Raising price from $25.00 to the capacity corner $31.25 keeps all 60 seats filled while increasing monthly profit from $500.00 to $875.00.",
            "independent_calculation": "a = 2(80) = 160, b = 80/25 = 3.2. Corner price p = (160 - 60)/3.2 = 100/3.2 = $31.25. At p = $31.25, s = 60 seats, R = 60*31.25 = $1,875.00, C = 700 + 5*60 = $1,000.00, P = $875.00.",
            "business_interpretation": "At the $25.00 baseline, demand (80) exceeds capacity (60) by 20 unserved seats. Increasing price up to $31.25 eliminates excess queue without losing a single attended seat.",
            "model_limitation": "Assumes deterministic linear demand and zero price-induced change in no-show behavior; real community demand may vary across neighborhoods and requires accessibility subsidies.",
        },
        "disruption_applied": False,
        "assumptions_log": [
            "All figures are fictional educational simulation assumptions (NUR-3).",
            "Nurse-led health education simulation models general wellness education only -- never diagnosis, treatment, or patient medical data.",
            "Simplified accrual basis with immediate payment at attended seat delivery; no taxes or depreciation unless explicitly enabled.",
            "Working days = 20/month, Calendar days = 31/month; daily views are modeled allocations, never monthly/30.",
        ],
        "sources_log": [
            "Career Venture Lab Student Handbook (Revised Nov 2026) - Pathway NUR-3 (p. 37), AP Calculus Adapter (p. 70), NUR Package (pp. 83-87).",
            "CDC Plain Language & Clear Communication Index (educational health literacy rubric).",
        ],
    }
    state["snapshot_id"] = compute_snapshot_id(state)
    return state


def get_unit_variable_cost(state: Dict[str, Any]) -> Decimal:
    return sum((to_money(v) for v in state["variable_cost_lines"].values()), Decimal("0.00"))


def get_monthly_fixed_cost(state: Dict[str, Any]) -> Decimal:
    return sum((to_money(v) for v in state["fixed_cost_lines"].values()), Decimal("0.00"))


def compute_nur_period_metrics(
    price: Decimal,
    requested: int,
    staff_hours: float,
    hours_per_unit: float,
    venue_seats: int,
    no_show_rate: float,
    unit_var_cost: Decimal,
    fixed_cost: Decimal,
    ad_cost: Decimal = Decimal("0.00"),
    interest_cost: Decimal = Decimal("0.00"),
    principal_repayment: Decimal = Decimal("0.00"),
    new_owner_cash: Decimal = Decimal("0.00"),
    grant_cash: Decimal = Decimal("0.00"),
    loan_proceeds: Decimal = Decimal("0.00"),
    additional_capex: Decimal = Decimal("0.00"),
    refund_amount: Decimal = Decimal("0.00"),
    capacity_override: Optional[int] = None,
) -> Dict[str, Any]:
    """Exact NUR deterministic monthly algorithm (Handbook p. 84)."""
    if hours_per_unit <= 0:
        raise ValueError("hours_per_unit must be > 0")
    if requested < 0 or venue_seats < 0 or staff_hours < 0:
        raise ValueError("requested, venue_seats, and staff_hours must be >= 0")
    if not (0.0 <= no_show_rate <= 1.0):
        raise ValueError("no_show_rate must be in [0, 1]")

    staff_cap = int(math.floor(staff_hours / hours_per_unit))
    derived_capacity = min(staff_cap, int(venue_seats))
    if capacity_override is not None:
        capacity = int(capacity_override)
        mode = "Fixture Override"
    else:
        capacity = derived_capacity
        mode = "Normal Resource-Derived"

    if capacity == 0:
        bottleneck = "Zero Capacity (0 seats/hours)"
    elif capacity_override is not None:
        bottleneck = f"Fixture Override ({capacity} seats)"
    elif staff_cap < venue_seats:
        bottleneck = f"Educator Staff Hours ({staff_cap} seats vs {venue_seats} venue)"
    elif venue_seats < staff_cap:
        bottleneck = f"Venue Seats ({venue_seats} seats vs {staff_cap} staff)"
    else:
        bottleneck = f"Co-Binding Staff & Venue ({capacity} seats)"

    booked = min(int(requested), capacity)
    unserved_demand = max(0, int(requested) - booked)
    # Exact handbook formula: attended = floor(booked * (1 - no_show_rate))
    attended = int(math.floor(booked * (1.0 - float(no_show_rate)) + 1e-9))
    no_shows = booked - attended
    sold = attended

    gross_revenue = to_money(price * Decimal(sold))
    net_revenue = to_money(gross_revenue - refund_amount)
    variable_expense = to_money(unit_var_cost * Decimal(sold))
    fixed_expense = to_money(fixed_cost)
    marketing_expense = to_money(ad_cost)
    total_operating_expense = to_money(variable_expense + fixed_expense + marketing_expense)
    operating_profit = to_money(net_revenue - total_operating_expense)
    net_profit = to_money(operating_profit - interest_cost)

    net_cash_flow = to_money(
        net_revenue
        - variable_expense
        - fixed_expense
        - marketing_expense
        - interest_cost
        - principal_repayment
        - additional_capex
        + new_owner_cash
        + grant_cash
        + loan_proceeds
    )

    used_hours = round(booked * hours_per_unit, 2)
    educator_utilization = round((used_hours / staff_hours * 100.0), 1) if staff_hours > 0 else 0.0
    seat_utilization = round((booked / venue_seats * 100.0), 1) if venue_seats > 0 else 0.0

    return {
        "mode": mode,
        "staff_cap": staff_cap,
        "derived_capacity": derived_capacity,
        "capacity": capacity,
        "bottleneck": bottleneck,
        "requested": int(requested),
        "booked": booked,
        "no_shows": no_shows,
        "attended": attended,
        "sold": sold,
        "unserved_demand": unserved_demand,
        "price": to_money(price),
        "unit_var_cost": to_money(unit_var_cost),
        "gross_revenue": gross_revenue,
        "refunds": to_money(refund_amount),
        "revenue": net_revenue,
        "variable_expense": variable_expense,
        "fixed_expense": fixed_expense,
        "marketing_expense": marketing_expense,
        "total_expense": total_operating_expense,
        "operating_profit": operating_profit,
        "interest_expense": to_money(interest_cost),
        "net_profit": net_profit,
        "principal_repayment": to_money(principal_repayment),
        "new_owner_cash": to_money(new_owner_cash),
        "grant_cash": to_money(grant_cash),
        "loan_proceeds": to_money(loan_proceeds),
        "additional_capex": to_money(additional_capex),
        "net_cash_flow": net_cash_flow,
        "used_hours": used_hours,
        "educator_utilization_pct": educator_utilization,
        "seat_utilization_pct": seat_utilization,
    }


def record_period_in_ledger(
    state: Dict[str, Any],
    run_id: str,
    period_label: str,
    sim_date: str,
    metrics: Dict[str, Any],
    note: str = "Approved Monthly Operating Step",
) -> Tuple[bool, str]:
    """Commit one simulated period to the dated ledger with duplicate-run protection."""
    if run_id in state["executed_run_ids"]:
        return False, f"Duplicate run_id '{run_id}' rejected: transaction already committed."

    opening_cash = to_money(state["balances"]["cash"])
    closing_cash = to_money(opening_cash + to_money(metrics["net_cash_flow"]))
    new_debt = to_money(
        to_money(state["balances"]["debt"])
        + to_money(metrics["loan_proceeds"])
        - to_money(metrics["principal_repayment"])
    )
    new_equip = to_money(to_money(state["balances"]["equipment_net"]) + to_money(metrics["additional_capex"]))
    new_paid_in = to_money(
        to_money(state["balances"]["paid_in_capital"])
        + to_money(metrics["new_owner_cash"])
        + to_money(metrics["grant_cash"])
    )
    new_re = to_money(to_money(state["balances"]["retained_earnings"]) + to_money(metrics["net_profit"]))

    next_ver = state["state_version"] + 1
    tx_entry = {
        "transaction_id": f"TX-{len(state['ledger']) + 1:04d}",
        "run_id": run_id,
        "date": sim_date,
        "period": period_label,
        "mode": metrics["mode"],
        "offering_id": "OFF-NUR-01",
        "offering": state["params"]["active_offering"],
        "requested": metrics["requested"],
        "capacity": metrics["capacity"],
        "booked": metrics["booked"],
        "no_shows": metrics["no_shows"],
        "sold": metrics["sold"],
        "unserved_demand": metrics["unserved_demand"],
        "unit_price": f"{to_money(metrics['price']):.2f}",
        "revenue": f"{to_money(metrics['revenue']):.2f}",
        "variable_expense": f"{to_money(metrics['variable_expense']):.2f}",
        "fixed_expense": f"{to_money(metrics['fixed_expense']):.2f}",
        "marketing_expense": f"{to_money(metrics['marketing_expense']):.2f}",
        "total_expense": f"{to_money(metrics['total_expense']):.2f}",
        "operating_profit": f"{to_money(metrics['operating_profit']):.2f}",
        "interest_expense": f"{to_money(metrics['interest_expense']):.2f}",
        "net_profit": f"{to_money(metrics['net_profit']):.2f}",
        "loan_proceeds": f"{to_money(metrics['loan_proceeds']):.2f}",
        "principal_repayment": f"{to_money(metrics['principal_repayment']):.2f}",
        "new_owner_cash": f"{to_money(metrics['new_owner_cash']):.2f}",
        "grant_cash": f"{to_money(metrics['grant_cash']):.2f}",
        "additional_capex": f"{to_money(metrics['additional_capex']):.2f}",
        "opening_cash": f"{opening_cash:.2f}",
        "closing_cash": f"{closing_cash:.2f}",
        "state_version": next_ver,
        "note": note,
    }

    state["ledger"].append(tx_entry)
    state["executed_run_ids"].append(run_id)
    state["balances"]["cash"] = f"{closing_cash:.2f}"
    state["balances"]["debt"] = f"{new_debt:.2f}"
    state["balances"]["equipment_net"] = f"{new_equip:.2f}"
    state["balances"]["paid_in_capital"] = f"{new_paid_in:.2f}"
    state["balances"]["retained_earnings"] = f"{new_re:.2f}"
    state["state_version"] = next_ver
    state["simulation_date"] = sim_date
    state["snapshot_id"] = compute_snapshot_id(state)
    return True, f"Recorded {tx_entry['transaction_id']} ({period_label}) -> State v{next_ver}."


# ==============================================================================
# 3. AP CALCULUS ANALYTIC & PIECEWISE ENGINE (HANDBOOK P. 37 & P. 70)
# ==============================================================================


def analyze_ap_calculus(
    D0: int,
    p0: float,
    v: float,
    F: float,
    raw_capacity: int,
    no_show_rate: float = 0.0,
) -> Dict[str, Any]:
    """Exact AP Calculus adapter equations for Foundation, Applied & Advanced."""
    a = 2.0 * float(D0)
    b = float(D0) / float(p0) if p0 > 0 else 1.0
    p_max = 2.0 * float(p0)

    # Effective delivered seat capacity accounting for attendance factor (1 - no_show_rate)
    eff_cap = float(math.floor(raw_capacity * (1.0 - float(no_show_rate)) + 1e-9))

    # Unconstrained critical point: P'(p) = a + b*v - 2*b*p = 0
    p_unc = (a + b * v) / (2.0 * b)
    p_unc_clamped = min(max(p_unc, 0.0), p_max)
    q_unc = max(0.0, a - b * p_unc_clamped)
    prof_unc = (p_unc_clamped - v) * q_unc - F

    # Break-even prices on unconstrained quadratic: (p - v)(a - b*p) - F = 0
    # -b*p^2 + (a + b*v)*p - (a*v + F) = 0
    disc = (a + b * v) ** 2 - 4.0 * b * (a * v + F)
    be_prices: List[float] = []
    if disc >= 0 and b > 0:
        r1 = ((a + b * v) - math.sqrt(disc)) / (2.0 * b)
        r2 = ((a + b * v) + math.sqrt(disc)) / (2.0 * b)
        for r in sorted([r1, r2]):
            if 0.0 <= r <= p_max:
                be_prices.append(round(r, 4))

    # Capacity corner: a - b*p = eff_cap  =>  p_corner = (a - eff_cap) / b
    p_corner_raw = (a - eff_cap) / b if b > 0 else 0.0
    has_corner = 0.0 < p_corner_raw < p_max
    p_corner = min(max(p_corner_raw, 0.0), p_max)

    # Candidate points for constrained continuous optimum: endpoints, corner, feasible critical point
    candidates = [0.0, p_max]
    if has_corner:
        candidates.append(p_corner)
    if p_unc_clamped >= p_corner:
        candidates.append(p_unc_clamped)

    best_p = 0.0
    best_prof = -1e18
    for cp in candidates:
        q_c = max(0.0, a - b * cp)
        s_c = min(eff_cap, q_c)
        pr_c = (cp - v) * s_c - F
        if pr_c > best_prof:
            best_prof = pr_c
            best_p = cp

    # Constrained break-even price on capacity branch: eff_cap*(p - v) - F = 0 => p = F/eff_cap + v
    be_cap_branch = (F / eff_cap + v) if eff_cap > 0 else None

    # Integer whole-seat evaluation
    int_rows = []
    best_int_prof = -1e18
    best_int_p = p0
    best_int_s = 0
    for s_int in range(0, int(eff_cap) + 1):
        # Highest price supporting at least s_int seats: a - b*p >= s_int => p <= (a - s_int)/b
        p_for_s = min(p_max, max(0.0, (a - s_int) / b))
        pr_int = (p_for_s - v) * s_int - F
        int_rows.append({"integer_seats": s_int, "max_supporting_price": round(p_for_s, 2), "profit": round(pr_int, 2)})
        if pr_int > best_int_prof:
            best_int_prof = pr_int
            best_int_p = p_for_s
            best_int_s = s_int

    return {
        "a": a,
        "b": b,
        "p_max": p_max,
        "eff_cap": eff_cap,
        "p_unc": p_unc_clamped,
        "q_unc": q_unc,
        "prof_unc": prof_unc,
        "unconstrained_feasible": q_unc <= eff_cap + 1e-6,
        "has_corner": has_corner,
        "p_corner": p_corner,
        "p_opt_constrained": best_p,
        "s_opt_constrained": min(eff_cap, max(0.0, a - b * best_p)),
        "prof_opt_constrained": best_prof,
        "left_deriv_at_corner": eff_cap,
        "right_deriv_at_corner": a + b * v - 2.0 * b * p_corner,
        "be_prices_unconstrained": be_prices,
        "be_price_capacity_branch": be_cap_branch,
        "best_int_seats": best_int_s,
        "best_int_price": best_int_p,
        "best_int_profit": best_int_prof,
        "int_table": pd.DataFrame(int_rows),
    }


def build_calc_grid(
    D0: int,
    p0: float,
    v: float,
    F: float,
    raw_capacity: int,
    no_show_rate: float = 0.0,
    num_points: int = 101,
) -> pd.DataFrame:
    """Generate synchronized price-demand-revenue-cost-profit-derivative grid."""
    info = analyze_ap_calculus(D0, p0, v, F, raw_capacity, no_show_rate)
    a, b, p_max, eff_cap, p_corner = info["a"], info["b"], info["p_max"], info["eff_cap"], info["p_corner"]

    prices = np.linspace(0.0, p_max, num_points)
    if info["has_corner"]:
        prices = np.unique(np.sort(np.append(prices, [p_corner - 1e-4, p_corner, p_corner + 1e-4])))

    rows = []
    for p in prices:
        q_dem = max(0.0, a - b * p)
        s_del = min(eff_cap, q_dem)
        rev_con = p * s_del
        cost_con = F + v * s_del
        prof_con = rev_con - cost_con
        prof_unc = (p - v) * q_dem - F

        if info["has_corner"] and abs(p - p_corner) < 5e-5:
            deriv = np.nan
            branch = "Corner (P' Undefined)"
        elif p < p_corner and info["has_corner"]:
            deriv = eff_cap
            branch = "Capacity-Constrained (s = cap)"
        else:
            deriv = a + b * v - 2.0 * b * p
            branch = "Demand-Constrained (s = q(p))"

        rows.append(
            {
                "Price ($/seat)": round(float(p), 2),
                "Unconstrained Demand q(p)": round(float(q_dem), 2),
                "Delivered Seats s(p)": round(float(s_del), 2),
                "Revenue R(p) ($)": round(float(rev_con), 2),
                "Total Cost C(p) ($)": round(float(cost_con), 2),
                "Constrained Profit P(p) ($)": round(float(prof_con), 2),
                "Unconstrained Profit ($)": round(float(prof_unc), 2),
                "Derivative P'(p) (seats)": round(float(deriv), 2) if not np.isnan(deriv) else np.nan,
                "Active Branch": branch,
                "Feasible (q <= cap)": "Yes" if q_dem <= eff_cap + 1e-6 else "No (Capped)",
            }
        )
    return pd.DataFrame(rows)


def compute_integral_exercise(T: float = 10.0) -> Dict[str, Any]:
    """AP Calculus Advanced isolated rate exercise: r(t)=100+20t, c(t)=60+5t over [0, T]."""
    # Integral_0^T (100 + 20t) dt = 100*T + 10*T^2
    acc_rev = 100.0 * T + 10.0 * (T ** 2)
    # Integral_0^T (60 + 5t) dt = 60*T + 2.5*T^2
    acc_cost = 60.0 * T + 2.5 * (T ** 2)
    acc_profit = acc_rev - acc_cost
    t_vals = np.linspace(0.0, T, 100)
    df = pd.DataFrame(
        {
            "Day t": t_vals,
            "Revenue Rate r(t) ($/day)": 100.0 + 20.0 * t_vals,
            "Cost Rate c(t) ($/day)": 60.0 + 5.0 * t_vals,
            "Net Rate r(t)-c(t) ($/day)": 40.0 + 15.0 * t_vals,
        }
    )
    return {
        "T": T,
        "acc_rev": round(acc_rev, 2),
        "acc_cost": round(acc_cost, 2),
        "acc_profit": round(acc_profit, 2),
        "df": df,
    }


# ==============================================================================
# 4. REPORT GENERATOR & PRINT-READY PDF EXPORT (ALL 8 HANDBOOK REPORTS)
# ==============================================================================


def generate_all_reports(state: Dict[str, Any], current_metrics: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
    """Generate all 8 required Career Venture Lab reports from the unified ledger & snapshot."""
    ledger = state["ledger"]
    v_unit = get_unit_variable_cost(state)
    f_month = get_monthly_fixed_cost(state)
    p0 = to_money(state["params"]["base_price"])
    D0 = int(state["params"]["D0"])

    # Aggregate actuals if ledger has rows; otherwise show current baseline preview clearly labeled
    if ledger:
        tot_rev = sum(to_money(r["revenue"]) for r in ledger)
        tot_var = sum(to_money(r["variable_expense"]) for r in ledger)
        tot_fix = sum(to_money(r["fixed_expense"]) for r in ledger)
        tot_mkt = sum(to_money(r["marketing_expense"]) for r in ledger)
        tot_int = sum(to_money(r["interest_expense"]) for r in ledger)
        tot_op = sum(to_money(r["operating_profit"]) for r in ledger)
        tot_net = sum(to_money(r["net_profit"]) for r in ledger)
        tot_req = sum(int(r["requested"]) for r in ledger)
        tot_cap = sum(int(r["capacity"]) for r in ledger)
        tot_booked = sum(int(r["booked"]) for r in ledger)
        tot_noshows = sum(int(r["no_shows"]) for r in ledger)
        tot_sold = sum(int(r["sold"]) for r in ledger)
        tot_unserved = sum(int(r["unserved_demand"]) for r in ledger)
        tot_loan_in = sum(to_money(r["loan_proceeds"]) for r in ledger)
        tot_prin_out = sum(to_money(r["principal_repayment"]) for r in ledger)
        tot_owner_in = sum(to_money(r["new_owner_cash"]) for r in ledger)
        tot_grant_in = sum(to_money(r["grant_cash"]) for r in ledger)
        tot_capex = sum(to_money(r["additional_capex"]) for r in ledger)
        status_label = f"ACTUAL LEDGER ({len(ledger)} committed period(s))"
    else:
        tot_rev = to_money(current_metrics["revenue"])
        tot_var = to_money(current_metrics["variable_expense"])
        tot_fix = to_money(current_metrics["fixed_expense"])
        tot_mkt = to_money(current_metrics["marketing_expense"])
        tot_int = to_money(current_metrics["interest_expense"])
        tot_op = to_money(current_metrics["operating_profit"])
        tot_net = to_money(current_metrics["net_profit"])
        tot_req = current_metrics["requested"]
        tot_cap = current_metrics["capacity"]
        tot_booked = current_metrics["booked"]
        tot_noshows = current_metrics["no_shows"]
        tot_sold = current_metrics["sold"]
        tot_unserved = current_metrics["unserved_demand"]
        tot_loan_in = to_money(current_metrics["loan_proceeds"])
        tot_prin_out = to_money(current_metrics["principal_repayment"])
        tot_owner_in = to_money(current_metrics["new_owner_cash"])
        tot_grant_in = to_money(current_metrics["grant_cash"])
        tot_capex = to_money(current_metrics["additional_capex"])
        status_label = "PROJECTED BASELINE PREVIEW (0 ledger periods committed yet)"

    gross_profit = to_money(tot_rev - tot_var)
    gross_margin = f"{(float(gross_profit) / float(tot_rev) * 100.0):.1f}%" if tot_rev > 0 else "N/A (Zero Sales)"
    net_margin = f"{(float(tot_net) / float(tot_rev) * 100.0):.1f}%" if tot_rev > 0 else "N/A (Zero Sales)"

    # 1. Income Statement
    df_income = pd.DataFrame(
        [
            {"Line Item": "Net Workshop Sales Revenue", "Amount ($)": f"{tot_rev:.2f}", "Status": status_label},
            {"Line Item": "Cost of Sales (Attended Seat Variable Costs)", "Amount ($)": f"-{tot_var:.2f}", "Status": status_label},
            {"Line Item": "Gross Profit", "Amount ($)": f"{gross_profit:.2f}", "Status": f"Gross Margin: {gross_margin}"},
            {"Line Item": "Operating Fixed Expenses (incl. Accessibility $200/mo)", "Amount ($)": f"-{tot_fix:.2f}", "Status": status_label},
            {"Line Item": "Own-Business Marketing Expense", "Amount ($)": f"-{tot_mkt:.2f}", "Status": status_label},
            {"Line Item": "Operating Profit (EBIT)", "Amount ($)": f"{tot_op:.2f}", "Status": status_label},
            {"Line Item": "Interest Expense", "Amount ($)": f"-{tot_int:.2f}", "Status": status_label},
            {"Line Item": "Net Result (Net Profit)", "Amount ($)": f"{tot_net:.2f}", "Status": f"Net Margin: {net_margin}"},
        ]
    )

    # 2. Cash-Flow Report
    open_cash = to_money(state["startup_financing"]["post_startup_cash"])
    op_cash_flow = to_money(tot_rev - tot_var - tot_fix - tot_mkt - tot_int)
    inv_cash_flow = to_money(-tot_capex)
    fin_cash_flow = to_money(tot_owner_in + tot_grant_in + tot_loan_in - tot_prin_out)
    close_cash = to_money(open_cash + op_cash_flow + inv_cash_flow + fin_cash_flow) if ledger else to_money(state["balances"]["cash"])
    reserve_req = to_money(state["startup_financing"]["reserve_threshold"])
    reserve_status = "PASS (>= $800 Reserve)" if close_cash >= reserve_req else "ALERT (< $800 Reserve)"

    df_cash = pd.DataFrame(
        [
            {"Cash Flow Section": "Post-Startup Opening Cash", "Amount ($)": f"{open_cash:.2f}", "Notes": "Starting capital $6,000 - $1,500 equip - $500 setup"},
            {"Cash Flow Section": "Operating Cash Inflows & Outflows", "Amount ($)": f"{op_cash_flow:.2f}", "Notes": "Net revenue - variable - fixed - marketing - interest"},
            {"Cash Flow Section": "Investing Cash Flows (CapEx / Equipment)", "Amount ($)": f"{inv_cash_flow:.2f}", "Notes": "Capitalized equipment (not expensed twice)"},
            {"Cash Flow Section": "Financing Cash Flows (Owner / Grant / Loan / Principal)", "Amount ($)": f"{fin_cash_flow:.2f}", "Notes": "Loan & owner funding excluded from revenue"},
            {"Cash Flow Section": "Closing Spendable Cash", "Amount ($)": f"{close_cash:.2f}", "Notes": reserve_status},
        ]
    )

    # 3. Budget Variance Report (Actual vs. Single-Period Baseline Budget)
    periods_n = max(1, len(ledger))
    bud_rev = to_money(Decimal("25.00") * Decimal(60 * periods_n))
    bud_var = to_money(Decimal("5.00") * Decimal(60 * periods_n))
    bud_fix = to_money(Decimal("700.00") * Decimal(periods_n))
    bud_mkt = Decimal("0.00")
    bud_net = to_money(bud_rev - bud_var - bud_fix - bud_mkt)

    def var_row(name: str, act: Decimal, bud: Decimal, driver: str) -> Dict[str, Any]:
        diff = to_money(act - bud)
        pct = f"{(float(diff) / float(bud) * 100.0):+.1f}%" if bud != 0 else "N/A (Zero Budget)"
        return {
            "Category": name,
            "Actual/Current ($)": f"{act:.2f}",
            "Baseline Budget ($)": f"{bud:.2f}",
            "Variance ($)": f"{diff:+.2f}",
            "Variance (%)": pct,
            "Primary Driver": driver,
        }

    df_variance = pd.DataFrame(
        [
            var_row("Workshop Seat Revenue", tot_rev, bud_rev, "Attended seats & ticket price vs. $25 x 60"),
            var_row("Variable Seat Expenses", tot_var, bud_var, "Attended seats x $5/unit baseline"),
            var_row("Monthly Fixed Expenses", tot_fix, bud_fix, "Venue, software, accessibility ($200), admin"),
            var_row("Marketing Campaign Spend", tot_mkt, bud_mkt, "Approved campaign launches"),
            var_row("Net Profit", tot_net, bud_net, "Combined volume, price, no-show & campaign variance"),
        ]
    )

    # 4. Sales Report
    df_sales = pd.DataFrame(
        [
            {
                "Offering": state["params"]["active_offering"],
                "Channel": "Direct Community Registration",
                "Requested Seats": tot_req,
                "Booked Seats": tot_booked,
                "No-Shows": tot_noshows,
                "Completed/Attended (Sold)": tot_sold,
                "Unserved Demand": tot_unserved,
                "Net Revenue ($)": f"{tot_rev:.2f}",
            }
        ]
    )

    # 5. Operations Report
    df_ops = pd.DataFrame(
        [
            {"Metric": "Total Seat Capacity", "Value": f"{tot_cap} seats", "Rule / Bottleneck": current_metrics["bottleneck"]},
            {"Metric": "Requested vs. Booked", "Value": f"{tot_req} req -> {tot_booked} booked", "Rule / Bottleneck": f"Unserved demand: {tot_unserved} seats"},
            {"Metric": "Booked vs. Attended (Sold)", "Value": f"{tot_booked} booked -> {tot_sold} attended", "Rule / Bottleneck": f"No-shows: {tot_noshows} (rate {state['params']['no_show_rate']*100:.0f}%)"},
            {"Metric": "Educator Utilization", "Value": f"{current_metrics['educator_utilization_pct']:.1f}%", "Rule / Bottleneck": f"{current_metrics['used_hours']} / {state['params']['staff_hours']} hrs"},
            {"Metric": "Accessibility Support Line", "Value": fmt_money(state["fixed_cost_lines"]["accessibility_support"]), "Rule / Bottleneck": "Explicit nonclinical inclusion commitment"},
            {"Metric": "Participant Satisfaction (Supplied Survey)", "Value": f"{state['params']['satisfaction_mean']:.1f} / 5.0", "Rule / Bottleneck": "Nonclinical educational feedback only"},
        ]
    )

    # 6. Marketing Report
    mkt_rows = []
    for c in state["campaigns"]:
        cost_d = to_money(c["total_cost"])
        conv = int(c["conversions"])
        cac_str = fmt_money(cost_d / Decimal(conv)) if conv > 0 else "N/A (Zero Conversions)"
        attr_rev = to_money(p0 * Decimal(conv))
        contrib = to_money((p0 - v_unit) * Decimal(conv) - cost_d)
        mkt_rows.append(
            {
                "Campaign ID": c["campaign_id"],
                "Variant / Channel": f"{c['variant_label']} | {c['channel']}",
                "Status": c["status"],
                "Total Cost ($)": f"{cost_d:.2f}",
                "Assumed Reach": c["reach_assumed"],
                "Inquiries": c["inquiries"],
                "Conversions": conv,
                "CAC ($/learner)": cac_str,
                "Attributed Contribution ($)": f"{contrib:.2f}",
                "Uncertainty Note": "Assumed attribution; not causal proof without randomized experiment",
            }
        )
    df_mkt = pd.DataFrame(mkt_rows)

    # 7. Forecast Scenarios (Baseline 1.00x, Optimistic 1.10x, Pessimistic 0.90x)
    calc_info = analyze_ap_calculus(
        D0,
        float(p0),
        float(v_unit),
        float(f_month),
        current_metrics["derived_capacity"],
        float(state["params"]["no_show_rate"]),
    )
    scen_rows = []
    for label, mult in [("Pessimistic (0.90x Demand)", 0.90), ("Baseline (1.00x Demand)", 1.00), ("Optimistic (1.10x Demand)", 1.10)]:
        req_s = int(math.floor(D0 * mult))
        m_s = compute_nur_period_metrics(
            price=p0,
            requested=req_s,
            staff_hours=float(state["params"]["staff_hours"]),
            hours_per_unit=float(state["params"]["hours_per_unit"]),
            venue_seats=int(state["params"]["venue_seats"]),
            no_show_rate=float(state["params"]["no_show_rate"]),
            unit_var_cost=v_unit,
            fixed_cost=f_month,
        )
        be_seats = math.ceil(float(f_month) / float(p0 - v_unit)) if p0 > v_unit else -1
        scen_rows.append(
            {
                "Scenario (Projected)": label,
                "Demand Multiplier": f"{mult:.2f}x",
                "Requested Seats": req_s,
                "Delivered Seats": m_s["sold"],
                "Unserved Demand": m_s["unserved_demand"],
                "Projected Revenue ($)": f"{m_s['revenue']:.2f}",
                "Projected Profit ($)": f"{m_s['net_profit']:.2f}",
                "Break-Even Seats @ $25": f"{be_seats} seats (Cap: {m_s['capacity']})",
                "Constrained Corner Price": fmt_money(calc_info["p_opt_constrained"]),
            }
        )
    df_forecast = pd.DataFrame(scen_rows)

    # 8. Simplified Balance Sheet (Reconciles Assets = Liabilities + Equity)
    b_cash = to_money(state["balances"]["cash"])
    b_equip = to_money(state["balances"]["equipment_net"])
    b_inv = to_money(state["balances"]["inventory"])
    tot_assets = to_money(b_cash + b_equip + b_inv)

    b_debt = to_money(state["balances"]["debt"])
    b_cap = to_money(state["balances"]["paid_in_capital"])
    b_re = to_money(state["balances"]["retained_earnings"])
    tot_liab_eq = to_money(b_debt + b_cap + b_re)
    recon_diff = to_money(tot_assets - tot_liab_eq)

    df_balance = pd.DataFrame(
        [
            {"Account": "Assets: Spendable Cash", "Balance ($)": f"{b_cash:.2f}", "Classification": "Current Asset"},
            {"Account": "Assets: Demonstration & Accessibility Equipment (Net)", "Balance ($)": f"{b_equip:.2f}", "Classification": "Fixed Asset"},
            {"Account": "Assets: Inventory", "Balance ($)": f"{b_inv:.2f}", "Classification": "Current Asset"},
            {"Account": "TOTAL ASSETS", "Balance ($)": f"{tot_assets:.2f}", "Classification": "Asset Total"},
            {"Account": "Liabilities: Simulated Loan Debt", "Balance ($)": f"{b_debt:.2f}", "Classification": "Liability"},
            {"Account": "Equity: Owner Paid-In Capital & Grants", "Balance ($)": f"{b_cap:.2f}", "Classification": "Equity"},
            {"Account": "Equity: Retained Earnings (incl. -$500 pre-opening)", "Balance ($)": f"{b_re:.2f}", "Classification": "Equity"},
            {"Account": "TOTAL LIABILITIES + EQUITY", "Balance ($)": f"{tot_liab_eq:.2f}", "Classification": f"Reconciled Diff: {fmt_money(recon_diff)}"},
        ]
    )

    return {
        "1. Income Statement": df_income,
        "2. Cash-Flow Report": df_cash,
        "3. Budget Variance": df_variance,
        "4. Sales Report": df_sales,
        "5. Operations Report": df_ops,
        "6. Marketing Report": df_mkt,
        "7. Forecast Scenarios": df_forecast,
        "8. Simplified Balance Sheet": df_balance,
    }


def build_pdf_report(state: Dict[str, Any], reports: Dict[str, pd.DataFrame]) -> bytes:
    """Generate a clean, Latin-1 safe printable PDF package of all 8 reports & math metadata."""
    pdf = FPDF(orientation="P", unit="mm", format="Letter")
    pdf.set_auto_page_break(auto=True, margin=14)
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(0, 8, sanitize_pdf_text("CAREER VENTURE LAB -- OFFICIAL SIMULATION REPORT PACKAGE"), ln=True)
    pdf.set_font("Helvetica", "", 10)
    meta = (
        f"Team: {state['team_id']} | Pathway: {state['pathway_id']} ({state['business_name']}) | "
        f"Course: {state['course']} ({state['selected_level']})\n"
        f"Snapshot ID: {state['snapshot_id']} | State Version: v{state['state_version']} | "
        f"Simulation Date: {state['simulation_date']} | Currency: {state['currency']}\n"
        f"Status: {'ACTUAL COMMITTED LEDGER' if state['ledger'] else 'BASELINE SCENARIO PREVIEW'} | "
        f"Generated: {dt.datetime.now().strftime('%Y-%m-%d %H:%M')}"
    )
    pdf.multi_cell(0, 5, sanitize_pdf_text(meta))
    pdf.ln(3)

    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 6, sanitize_pdf_text("Active AP Calculus Skill & Assumptions:"), ln=True)
    pdf.set_font("Helvetica", "", 9)
    skill_info = AP_CALC_SKILLS[state["selected_level"]]
    pdf.multi_cell(
        0,
        4.5,
        sanitize_pdf_text(
            f"Skill ({state['selected_level']}): {skill_info['skill_name']}\n"
            f"Domain: {skill_info['domain_text']}\n"
            f"Units: {skill_info['units_text']}\n"
            f"Assumptions: {skill_info['assumptions']}"
        ),
    )
    pdf.ln(3)

    for title, df in reports.items():
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 7, sanitize_pdf_text(title), ln=True)
        pdf.set_font("Helvetica", "B", 8)
        cols = list(df.columns)
        col_w = 188.0 / max(1, len(cols))
        for c in cols:
            pdf.cell(col_w, 6, sanitize_pdf_text(str(c)[:26]), border=1)
        pdf.ln()

        pdf.set_font("Helvetica", "", 8)
        for _, row in df.iterrows():
            for c in cols:
                pdf.cell(col_w, 5.5, sanitize_pdf_text(str(row[c])[:28]), border=1)
            pdf.ln()
        pdf.ln(3)

    pdf.set_font("Helvetica", "I", 8)
    pdf.multi_cell(
        0,
        4,
        sanitize_pdf_text(
            "Provenance & Ethical Guardrail: Fictional educational simulation only. "
            "No real customer, patient, or student-grade data. Nurse-led health education simulation "
            "models general wellness workshops only -- never clinical diagnosis or medical treatment."
        ),
    )
    raw_out = pdf.output(dest="S")
    if isinstance(raw_out, str):
        return raw_out.encode("latin-1")
    return bytes(raw_out)


# ==============================================================================
# 5. BUILT-IN HANDBOOK ACCEPTANCE TEST RUNNER (NON-MUTATING SANDBOX)
# ==============================================================================


def run_handbook_acceptance_tests() -> pd.DataFrame:
    """Execute all NUR-3 + AP Calculus handbook acceptance tests on isolated copies."""
    results = []

    def add_res(name: str, expected: str, actual: str, passed: bool):
        results.append(
            {
                "Test Case": name,
                "Expected Result": expected,
                "Actual Result": actual,
                "Status": "PASS" if passed else "FAIL",
            }
        )

    # Test 1: Zero sales
    m_zero = compute_nur_period_metrics(
        price=Decimal("25.00"),
        requested=0,
        staff_hours=30.0,
        hours_per_unit=0.5,
        venue_seats=60,
        no_show_rate=0.0,
        unit_var_cost=Decimal("5.00"),
        fixed_cost=Decimal("700.00"),
    )
    add_res(
        "1. Zero Sales Check",
        "Revenue=$0.00, Variable=$0.00, Profit=-$700.00, No crash",
        f"Revenue={fmt_money(m_zero['revenue'])}, Variable={fmt_money(m_zero['variable_expense'])}, Profit={fmt_money(m_zero['operating_profit'])}",
        m_zero["revenue"] == Decimal("0.00") and m_zero["operating_profit"] == Decimal("-700.00"),
    )

    # Test 2: Demand exceeds capacity (baseline 80 req vs 60 cap)
    m_cap = compute_nur_period_metrics(
        price=Decimal("25.00"),
        requested=80,
        staff_hours=30.0,
        hours_per_unit=0.5,
        venue_seats=60,
        no_show_rate=0.0,
        unit_var_cost=Decimal("5.00"),
        fixed_cost=Decimal("700.00"),
    )
    add_res(
        "2. Demand Exceeds Capacity",
        "Sold=60, Unserved=20, Revenue=$1,500.00, Profit=$500.00",
        f"Sold={m_cap['sold']}, Unserved={m_cap['unserved_demand']}, Revenue={fmt_money(m_cap['revenue'])}, Profit={fmt_money(m_cap['operating_profit'])}",
        m_cap["sold"] == 60 and m_cap["unserved_demand"] == 20 and m_cap["operating_profit"] == Decimal("500.00"),
    )

    # Test 3: NUR Business-Specific Test (no_show_rate = 0.20)
    m_ns = compute_nur_period_metrics(
        price=Decimal("25.00"),
        requested=80,
        staff_hours=30.0,
        hours_per_unit=0.5,
        venue_seats=60,
        no_show_rate=0.20,
        unit_var_cost=Decimal("5.00"),
        fixed_cost=Decimal("700.00"),
    )
    add_res(
        "3. NUR No-Show Rate = 20%",
        "Booked=60, Attended=48, Revenue=$1,200.00, Var=$240.00, Profit=$260.00",
        f"Booked={m_ns['booked']}, Attended={m_ns['attended']}, Revenue={fmt_money(m_ns['revenue'])}, Var={fmt_money(m_ns['variable_expense'])}, Profit={fmt_money(m_ns['operating_profit'])}",
        m_ns["booked"] == 60
        and m_ns["attended"] == 48
        and m_ns["revenue"] == Decimal("1200.00")
        and m_ns["variable_expense"] == Decimal("240.00")
        and m_ns["operating_profit"] == Decimal("260.00"),
    )

    # Test 4: 3-Month Fixture Replay (monthly_fixture.csv)
    st_fix = build_default_state()
    expected_fix = [
        ("2027-01", 60, Decimal("1500.00"), Decimal("1000.00"), Decimal("500.00"), Decimal("4500.00")),
        ("2027-02", 60, Decimal("1500.00"), Decimal("1100.00"), Decimal("400.00"), Decimal("4900.00")),
        ("2027-03", 45, Decimal("1125.00"), Decimal("925.00"), Decimal("200.00"), Decimal("5100.00")),
    ]
    fix_ok = True
    fix_summary = []
    for row, exp in zip(NUR_FIXTURE_ROWS, expected_fix):
        mf = compute_nur_period_metrics(
            price=Decimal("25.00"),
            requested=row["requested"],
            staff_hours=30.0,
            hours_per_unit=0.5,
            venue_seats=60,
            no_show_rate=0.0,
            unit_var_cost=Decimal("5.00"),
            fixed_cost=Decimal("700.00"),
            ad_cost=to_money(row["ad_cost"]),
            capacity_override=row["capacity_override"],
        )
        record_period_in_ledger(st_fix, f"FIX-{row['month']}", row["month"], f"{row['month']}-28", mf)
        last_tx = st_fix["ledger"][-1]
        got = (
            row["month"],
            last_tx["sold"],
            to_money(last_tx["revenue"]),
            to_money(last_tx["total_expense"]),
            to_money(last_tx["operating_profit"]),
            to_money(last_tx["closing_cash"]),
        )
        if got != exp:
            fix_ok = False
        fix_summary.append(f"{row['month']}: cash={fmt_money(got[5])}")
    add_res(
        "4. NUR 3-Month Fixture Replay",
        "2027-01 cash=$4,500; 2027-02 cash=$4,900; 2027-03 cash=$5,100",
        "; ".join(fix_summary),
        fix_ok,
    )

    # Test 5: Duplicate Run ID Protection
    dup_ok, dup_msg = record_period_in_ledger(st_fix, "FIX-2027-01", "2027-01", "2027-01-28", m_cap)
    add_res(
        "5. Duplicate Run ID Rejection",
        "Second execution of same run_id rejected; ledger count stays 3",
        f"Rejected={not dup_ok}, Ledger rows={len(st_fix['ledger'])}",
        (not dup_ok) and len(st_fix["ledger"]) == 3,
    )

    # Test 6: Loan Received (Cash & Debt rise, Revenue unchanged)
    st_loan = build_default_state()
    m_loan = compute_nur_period_metrics(
        price=Decimal("25.00"),
        requested=0,
        staff_hours=30.0,
        hours_per_unit=0.5,
        venue_seats=60,
        no_show_rate=0.0,
        unit_var_cost=Decimal("5.00"),
        fixed_cost=Decimal("700.00"),
        loan_proceeds=Decimal("1000.00"),
    )
    record_period_in_ledger(st_loan, "RUN-LOAN-1", "2027-01", "2027-01-31", m_loan)
    add_res(
        "6. Loan Received Accounting Check",
        "Revenue=$0.00, Debt=$1,000.00, Cash=$4,300.00 ($4000+$1000-$700)",
        f"Revenue={fmt_money(m_loan['revenue'])}, Debt={fmt_money(st_loan['balances']['debt'])}, Cash={fmt_money(st_loan['balances']['cash'])}",
        m_loan["revenue"] == Decimal("0.00")
        and to_money(st_loan["balances"]["debt"]) == Decimal("1000.00")
        and to_money(st_loan["balances"]["cash"]) == Decimal("4300.00"),
    )

    # Test 7: AP Calculus Constrained Optimum & Corner Check (NUR-3 Baseline)
    calc_nur = analyze_ap_calculus(D0=80, p0=25.0, v=5.0, F=700.0, raw_capacity=60, no_show_rate=0.0)
    add_res(
        "7. AP Calculus Corner vs. Unconstrained Optimum",
        "Unconstrained p=$27.50 (q=72 infeasible); Constrained corner p=$31.25, Profit=$875.00",
        f"p_unc=${calc_nur['p_unc']:.2f} (q={calc_nur['q_unc']:.0f}); p_corner=${calc_nur['p_corner']:.2f}, Profit={fmt_money(calc_nur['prof_opt_constrained'])}",
        abs(calc_nur["p_unc"] - 27.50) < 1e-4
        and abs(calc_nur["p_corner"] - 31.25) < 1e-4
        and abs(calc_nur["prof_opt_constrained"] - 875.00) < 1e-4,
    )

    # Test 8: AP Calculus Course Adapter Fixture (D0=60, p0=30, v=10, F=500, cap=40)
    calc_adp = analyze_ap_calculus(D0=60, p0=30.0, v=10.0, F=500.0, raw_capacity=40, no_show_rate=0.0)
    add_res(
        "8. AP Calculus Adapter Fixture (p. 70)",
        "p_unc=$35.00 infeasible; Constrained p=$40.00, Profit=$700.00",
        f"p_unc=${calc_adp['p_unc']:.2f}; Constrained p=${calc_adp['p_opt_constrained']:.2f}, Profit=${calc_adp['prof_opt_constrained']:.2f}",
        abs(calc_adp["p_unc"] - 35.00) < 1e-4
        and abs(calc_adp["p_opt_constrained"] - 40.00) < 1e-4
        and abs(calc_adp["prof_opt_constrained"] - 700.00) < 1e-4,
    )

    # Test 9: AP Calculus Integral Rate Fixture (r=100+20t, c=60+5t, 0<=t<=10)
    integ = compute_integral_exercise(10.0)
    add_res(
        "9. AP Calculus Definite Integral Fixture",
        "Accumulated Rev=$2,000.00, Cost=$850.00, Profit=$1,150.00",
        f"Rev=${integ['acc_rev']:.2f}, Cost=${integ['acc_cost']:.2f}, Profit=${integ['acc_profit']:.2f}",
        integ["acc_rev"] == 2000.0 and integ["acc_cost"] == 850.0 and integ["acc_profit"] == 1150.0,
    )

    # Test 10: JSON Save & Restore Reconciliation
    serialized = json.dumps(st_fix)
    restored = json.loads(serialized)
    add_res(
        "10. JSON Export & Restore Reconciliation",
        f"Version v{st_fix['state_version']} & Cash {fmt_money(st_fix['balances']['cash'])} match identically",
        f"Restored v{restored['state_version']} & Cash {fmt_money(restored['balances']['cash'])}",
        restored["state_version"] == st_fix["state_version"] and restored["balances"]["cash"] == st_fix["balances"]["cash"],
    )

    return pd.DataFrame(results)


# ==============================================================================
# 6. STREAMLIT UI & APPLICATION LAYOUT
# ==============================================================================


def init_session_state():
    if "sim_state" not in st.session_state:
        st.session_state.sim_state = build_default_state()
    if "state_epoch" not in st.session_state:
        st.session_state.state_epoch = 0
    if "frozen_assessment" not in st.session_state:
        st.session_state.frozen_assessment = None
    if "practice_submitted" not in st.session_state:
        st.session_state.practice_submitted = False


def bump_epoch():
    st.session_state.state_epoch += 1


def main():
    st.set_page_config(
        page_title="Career Venture Lab | NUR-3 AP Calculus (Advanced)",
        page_icon="🩺",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    init_session_state()
    state = st.session_state.sim_state
    epoch = st.session_state.state_epoch

    # Accessible high-contrast styling
    st.markdown(
        """
        <style>
        .main-banner {
            background: linear-gradient(90deg, #0f3d5e 0%, #165b7d 100%);
            color: #ffffff;
            padding: 14px 20px;
            border-radius: 10px;
            margin-bottom: 12px;
        }
        .main-banner h2 { margin: 0; font-size: 1.45rem; color: #ffffff; }
        .main-banner p { margin: 4px 0 0 0; font-size: 0.92rem; color: #e2f1f8; }
        .guardrail-box {
            background-color: #f0f7fb;
            border-left: 5px solid #0f3d5e;
            padding: 10px 14px;
            border-radius: 6px;
            margin-bottom: 10px;
            font-size: 0.9rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------------------------
    # SIDEBAR: GOVERNANCE, CHALLENGE LEVEL, SAVE/LOAD JSON, QUICK ACTIONS
    # --------------------------------------------------------------------------
    with st.sidebar:
        st.header("Simulation Control Center")
        new_team = st.text_input("Team ID", value=state["team_id"], key=f"sb_team_{epoch}")
        if new_team.strip() and new_team.strip() != state["team_id"]:
            state["team_id"] = new_team.strip()
            state["snapshot_id"] = compute_snapshot_id(state)

        st.info(
            f"**Pathway:** `{state['pathway_id']}` (Healthcare / Nursing)\n\n"
            f"**Course:** `{state['course']}`\n\n"
            f"**Snapshot:** `{state['snapshot_id']}`\n\n"
            f"**State Version:** `v{state['state_version']}`"
        )

        st.subheader("Challenge Level & Approval")
        levels = ["Foundation", "Applied", "Advanced"]
        curr_idx = levels.index(state["selected_level"])
        chosen_level = st.selectbox(
            "Active AP Calculus Challenge Level",
            levels,
            index=curr_idx,
            key=f"sb_level_{epoch}",
            help="Changing level updates equations, graphs, tasks, and assessment items per Handbook p. 6 & p. 70.",
        )
        if chosen_level != state["selected_level"]:
            state["selected_level"] = chosen_level
            state["selected_skill"] = AP_CALC_SKILLS[chosen_level]["skill_name"]
            state["level_history"].append(
                {
                    "version": len(state["level_history"]) + 1,
                    "level": chosen_level,
                    "status": "Pending Teacher Approval",
                    "timestamp": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }
            )
            state["snapshot_id"] = compute_snapshot_id(state)
            st.rerun()

        if st.button("Record Teacher Level Approval", use_container_width=True):
            state["teacher_approved_level"] = state["selected_level"]
            state["level_history"].append(
                {
                    "version": len(state["level_history"]) + 1,
                    "level": state["selected_level"],
                    "status": "Teacher Approved",
                    "timestamp": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }
            )
            st.success(f"Approved level: {state['selected_level']}")

        st.divider()
        st.subheader("Required NUR-3 Disruption")
        st.caption("Handbook p. 37: Venue availability falls by 20%; add accessible delivery without exceeding budget.")
        if not state["disruption_applied"]:
            if st.button("Apply -20% Venue Disruption (60 -> 48 seats)", use_container_width=True):
                state["params"]["venue_seats"] = int(math.floor(state["params"]["venue_seats"] * 0.80))
                state["disruption_applied"] = True
                state["snapshot_id"] = compute_snapshot_id(state)
                bump_epoch()
                st.rerun()
        else:
            st.warning("Disruption Active: Venue seats reduced by 20%.")

        if st.button("Reset to NUR-3 Baseline Defaults", use_container_width=True):
            st.session_state.sim_state = build_default_state(state["team_id"])
            st.session_state.frozen_assessment = None
            st.session_state.practice_submitted = False
            bump_epoch()
            st.rerun()

        st.divider()
        st.subheader("Durable Save / Restore (JSON)")
        st.caption("Browser memory is temporary. Download JSON to save and restore state cleanly.")
        json_bytes = json.dumps(state, indent=2).encode("utf-8")
        st.download_button(
            label="Save Snapshot JSON",
            data=json_bytes,
            file_name=f"{state['team_id']}_{state['pathway_id']}_v{state['state_version']}.json",
            mime="application/json",
            use_container_width=True,
        )

        uploaded_json = st.file_uploader("Restore Snapshot JSON", type=["json"], key=f"sb_upload_{epoch}")
        if uploaded_json is not None:
            if st.button("Load Uploaded Snapshot", use_container_width=True):
                try:
                    loaded = json.loads(uploaded_json.read().decode("utf-8"))
                    required_keys = {"team_id", "pathway_id", "state_version", "params", "balances", "ledger"}
                    if not required_keys.issubset(set(loaded.keys())):
                        st.error("Rejected: JSON missing required data-contract fields.")
                    elif loaded.get("pathway_id") != "NUR-3":
                        st.error(f"Rejected: Pathway '{loaded.get('pathway_id')}' does not match NUR-3.")
                    else:
                        st.session_state.sim_state = loaded
                        bump_epoch()
                        st.success(f"Restored {loaded['snapshot_id']} (v{loaded['state_version']})!")
                        st.rerun()
                except Exception as exc:
                    st.error(f"Invalid JSON file: {exc}")

    # --------------------------------------------------------------------------
    # TOP HEADER BANNER & ETHICAL GUARDRAIL
    # --------------------------------------------------------------------------
    st.markdown(
        f"""
        <div class="main-banner">
            <h2>Career Venture Lab &mdash; {state['business_name']} ({state['pathway_id']})</h2>
            <p>Course: <strong>{state['course']}</strong> | Challenge Level: <strong>{state['selected_level']}</strong>
            (Approved: {state['teacher_approved_level']}) | Team: <strong>{state['team_id']}</strong> |
            Snapshot: <strong>{state['snapshot_id']}</strong></p>
        </div>
        <div class="guardrail-box">
            <strong>Workflow &amp; Nonclinical Guardrail:</strong> Manual AI Handoff Mode (no paid API required).
            This simulation models <em>fictional general wellness education workshops</em> only &mdash;
            no clinical diagnosis, medical treatment, or real patient/student data.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Compute live scenario metrics from current parameters
    unit_v = get_unit_variable_cost(state)
    fixed_f = get_monthly_fixed_cost(state)
    live_metrics = compute_nur_period_metrics(
        price=to_money(state["params"]["base_price"]),
        requested=int(state["params"]["requested_demand"]),
        staff_hours=float(state["params"]["staff_hours"]),
        hours_per_unit=float(state["params"]["hours_per_unit"]),
        venue_seats=int(state["params"]["venue_seats"]),
        no_show_rate=float(state["params"]["no_show_rate"]),
        unit_var_cost=unit_v,
        fixed_cost=fixed_f,
    )
    calc_analysis = analyze_ap_calculus(
        D0=int(state["params"]["D0"]),
        p0=float(state["params"]["base_price"]),
        v=float(unit_v),
        F=float(fixed_f),
        raw_capacity=live_metrics["derived_capacity"],
        no_show_rate=float(state["params"]["no_show_rate"]),
    )

    tabs = st.tabs(
        [
            "1. Overview & System Map",
            "2. Inputs, Funding & Ledger",
            "3. AP Calculus Math Lab",
            "4. Operations & Customers",
            "5. AI Handoffs & Reviewer",
            "6. Marketing Studio",
            "7. Report Center & Memo",
            "8. Cross-Course, Quiz & Tests",
        ]
    )

    # ==========================================================================
    # TAB 1: OVERVIEW, PERSISTENT MATH PANEL, SYSTEM MAP & LINKED VISUAL
    # ==========================================================================
    with tabs[0]:
        st.subheader("Executive Business & Operating Overview")
        k1, k2, k3, k4, k5 = st.columns(5)
        cash_now = to_money(state["balances"]["cash"])
        reserve_min = to_money(state["startup_financing"]["reserve_threshold"])
        k1.metric(
            "Spendable Cash (Ledger)",
            fmt_money(cash_now),
            delta="Safe (>= $800)" if cash_now >= reserve_min else "LOW CASH (< $800)",
            delta_color="normal" if cash_now >= reserve_min else "inverse",
        )
        k2.metric(
            "Scenario Monthly Profit",
            fmt_money(live_metrics["operating_profit"]),
            delta=f"Rev {fmt_money(live_metrics['revenue'])} - Exp {fmt_money(live_metrics['total_expense'])}",
        )
        k3.metric(
            "Attended (Sold) Seats",
            f"{live_metrics['sold']} seats",
            delta=f"{live_metrics['unserved_demand']} unserved | {live_metrics['no_shows']} no-show",
        )
        k4.metric(
            "Effective Seat Capacity",
            f"{live_metrics['capacity']} seats",
            delta=live_metrics["bottleneck"],
        )
        k5.metric(
            "AP Calculus Corner Optimum",
            f"{fmt_money(calc_analysis['p_opt_constrained'])} / seat",
            delta=f"Max Profit {fmt_money(calc_analysis['prof_opt_constrained'])}",
        )

        # Persistent Math Skills & Challenge Level Panel (Handbook p. 6)
        st.markdown("---")
        skill_meta = AP_CALC_SKILLS[state["selected_level"]]
        with st.container(border=True):
            st.markdown(f"### Persistent Math Skills Panel | `{state['course']}` &mdash; **{state['selected_level']} Level**")
            st.markdown(f"**Named Skill:** {skill_meta['skill_name']}")
            st.markdown(f"**Business Question:** {skill_meta['business_question']}")
            st.latex(skill_meta["equation_latex"])
            c_m1, c_m2, c_m3 = st.columns(3)
            c_m1.markdown(f"**Domain:** {skill_meta['domain_text']}")
            c_m2.markdown(f"**Units:** {skill_meta['units_text']}")
            c_m3.markdown(f"**Key Assumptions:** {skill_meta['assumptions']}")

        # Linked Interactive Price-Profit Graph + Synchronized Data Table (Handbook p. 7)
        st.markdown("---")
        st.subheader("Linked Profit/Loss Visual & Synchronized Data Table")
        st.caption(
            "Select a price point below to highlight its exact row in the synchronized table and inspect "
            "revenue, cost, profit, zero break-even line, and capacity feasibility."
        )
        grid_df = build_calc_grid(
            D0=int(state["params"]["D0"]),
            p0=float(state["params"]["base_price"]),
            v=float(unit_v),
            F=float(fixed_f),
            raw_capacity=live_metrics["derived_capacity"],
            no_show_rate=float(state["params"]["no_show_rate"]),
            num_points=51,
        )
        selected_probe_p = st.slider(
            "Inspect Price Point p ($/seat) on Graph & Table",
            min_value=0.0,
            max_value=float(calc_analysis["p_max"]),
            value=float(calc_analysis["p_opt_constrained"]),
            step=0.25,
            key=f"ov_probe_p_{epoch}",
        )

        # Find nearest row in grid_df
        nearest_idx = int((grid_df["Price ($/seat)"] - selected_probe_p).abs().idxmin())
        probe_row = grid_df.iloc[nearest_idx]

        fig_ov = go.Figure()
        fig_ov.add_trace(
            go.Scatter(
                x=grid_df["Price ($/seat)"],
                y=grid_df["Revenue R(p) ($)"],
                mode="lines",
                name="Revenue R(p) ($)",
                line=dict(color="#1f77b4", width=2, dash="dash"),
            )
        )
        fig_ov.add_trace(
            go.Scatter(
                x=grid_df["Price ($/seat)"],
                y=grid_df["Total Cost C(p) ($)"],
                mode="lines",
                name="Total Cost C(p) ($)",
                line=dict(color="#ff7f0e", width=2, dash="dot"),
            )
        )
        fig_ov.add_trace(
            go.Scatter(
                x=grid_df["Price ($/seat)"],
                y=grid_df["Constrained Profit P(p) ($)"],
                mode="lines+markers",
                marker=dict(size=4),
                name="Constrained Profit P(p) ($)",
                line=dict(color="#0f3d5e", width=3),
            )
        )
        fig_ov.add_hline(y=0, line_dash="solid", line_color="#444444", annotation_text="Break-Even Zero Line ($0)")
        if calc_analysis["has_corner"]:
            fig_ov.add_vline(
                x=calc_analysis["p_corner"],
                line_dash="dashdot",
                line_color="#8c2d04",
                annotation_text=f"Capacity Corner p=${calc_analysis['p_corner']:.2f}",
            )
        fig_ov.add_trace(
            go.Scatter(
                x=[probe_row["Price ($/seat)"]],
                y=[probe_row["Constrained Profit P(p) ($)"]],
                mode="markers+text",
                text=[f"Selected (${probe_row['Price ($/seat)']:.2f}, ${probe_row['Constrained Profit P(p) ($)']:.2f})"],
                textposition="top center",
                marker=dict(size=13, symbol="diamond", color="#d62728"),
                name="Selected Probe Point",
            )
        )
        fig_ov.update_layout(
            title=(
                f"NUR-3 Profit/Loss & Capacity Boundary | Snapshot {state['snapshot_id']} | "
                f"Units: Price ($/seat), Monthly ($/mo) | Actual/Forecast: Scenario Projection"
            ),
            xaxis_title="Workshop Seat Price p ($/seat) [Domain: 0 <= p <= 2*p0]",
            yaxis_title="Monthly USD ($/month)",
            hovermode="x unified",
            height=430,
        )

        col_g, col_t = st.columns([3, 2])
        with col_g:
            st.plotly_chart(fig_ov, use_container_width=True)
            st.download_button(
                "Export Graph + Assumptions + Data (HTML Print View)",
                data=fig_ov.to_html(include_plotlyjs="cdn").encode("utf-8"),
                file_name=f"NUR3_Profit_Chart_{state['snapshot_id']}.html",
                mime="text/html",
            )
        with col_t:
            st.markdown("**Synchronized Price-Volume Table (Selected Row Highlighted)**")

            def highlight_selected(row):
                if row.name == nearest_idx:
                    return ["background-color: #ffe8a1; font-weight: bold"] * len(row)
                return [""] * len(row)

            disp_cols = [
                "Price ($/seat)",
                "Delivered Seats s(p)",
                "Revenue R(p) ($)",
                "Constrained Profit P(p) ($)",
                "Feasible (q <= cap)",
            ]
            st.dataframe(
                grid_df[disp_cols].style.apply(highlight_selected, axis=1),
                use_container_width=True,
                height=390,
            )

        st.info(
            "**Interpretation Prompts (Handbook p. 7):** "
            "1) What changed as p moved across the corner price? "
            "2) Where are the break-even points where P(p) = 0? "
            "3) Which price interval satisfies q(p) <= capacity? "
            "4) Why does larger revenue at low prices NOT produce maximum profit?"
        )

        # System Map with Feedback Loop (Handbook p. 5 & p. 7)
        st.markdown("---")
        st.subheader("System Map & Feedback Loop (Automated vs. Student-Managed)")
        st.markdown(
            """
            | Step | Department / Stage | Flow & Feedback Loop | Control Type |
            | :--- | :--- | :--- | :--- |
            | **1** | **Marketing Studio** | Drafts campaign & approved health-literacy message &rarr; *Requested Demand* | **Student-Managed** (Fact-check & Approval) |
            | **2** | **Customers & Demand** | Price $p$ & demand $q(p)$ generate *Requested Workshop Seats* | **Deterministic Equation** ($q = 160 - 3.2p$) |
            | **3** | **Operations** | Caps bookings by $\\min(\\text{staff\\_hours}/0.5, \\text{venue\\_seats})$ &rarr; *Booked Seats* | **Deterministic Algorithm** |
            | **4** | **Attendance & Delivery** | Applies `no_show_rate` &rarr; *Attended (Sold) Seats* & unserved demand | **Deterministic Algorithm** |
            | **5** | **Satisfaction Feedback** | Workshop quality & accessibility support &rarr; *Participant Satisfaction & Later Demand* | **Student-Reviewed** (Nonclinical survey) |
            | **6** | **Finance & Reviewer** | Updates dated ledger, reconciles cash & equity &rarr; *Next Round Budget* | **Human Approval Gate** (Stale versions rejected) |
            """
        )

    # ==========================================================================
    # TAB 2: INPUTS, STARTUP FINANCING, SCENARIO SANDBOX & DATED LEDGER
    # ==========================================================================
    with tabs[1]:
        st.subheader("Capital, Operating Inputs, Scenario Sandbox & Dated Event Ledger")
        st.caption(
            "Input changes update the live scenario preview immediately without mutating committed ledger history "
            "until you click 'Approve & Record Period in Ledger'."
        )

        col_in1, col_in2, col_in3 = st.columns(3)
        with col_in1:
            st.markdown("#### 1. Operating & Demand Inputs")
            p_in = st.number_input(
                "Ticket Price per Seat ($)",
                min_value=1.0,
                max_value=200.0,
                value=float(state["params"]["base_price"]),
                step=1.0,
                key=f"in_price_{epoch}",
            )
            d0_in = st.number_input(
                "Baseline Demand D0 at Base Price (seats/mo)",
                min_value=0,
                max_value=500,
                value=int(state["params"]["D0"]),
                step=5,
                key=f"in_d0_{epoch}",
            )
            req_in = st.number_input(
                "Requested Workshop Seats This Period",
                min_value=0,
                max_value=500,
                value=int(state["params"]["requested_demand"]),
                step=5,
                key=f"in_req_{epoch}",
            )
            staff_in = st.number_input(
                "Educator Productive Staff Hours / Month",
                min_value=0.0,
                max_value=500.0,
                value=float(state["params"]["staff_hours"]),
                step=2.0,
                key=f"in_staff_{epoch}",
            )
            hpu_in = st.number_input(
                "Educator Hours per Seat (> 0)",
                min_value=0.1,
                max_value=10.0,
                value=float(state["params"]["hours_per_unit"]),
                step=0.1,
                key=f"in_hpu_{epoch}",
            )
            seats_in = st.number_input(
                "Venue Seats Capacity / Month",
                min_value=0,
                max_value=500,
                value=int(state["params"]["venue_seats"]),
                step=5,
                key=f"in_seats_{epoch}",
            )
            ns_in = st.slider(
                "No-Show Rate [0, 1]",
                min_value=0.0,
                max_value=1.0,
                value=float(state["params"]["no_show_rate"]),
                step=0.05,
                key=f"in_ns_{epoch}",
            )

        with col_in2:
            st.markdown("#### 2. Editable Cost Line Items")
            st.markdown("**Per Attended Seat Variable Costs ($/unit):**")
            v_ed = st.number_input(
                "educator_delivery ($/seat)",
                min_value=0.0,
                value=float(state["variable_cost_lines"]["educator_delivery"]),
                step=0.5,
                key=f"in_ved_{epoch}",
            )
            v_mat = st.number_input(
                "materials ($/seat)",
                min_value=0.0,
                value=float(state["variable_cost_lines"]["materials"]),
                step=0.5,
                key=f"in_vmat_{epoch}",
            )
            v_cln = st.number_input(
                "cleaning_consumables ($/seat)",
                min_value=0.0,
                value=float(state["variable_cost_lines"]["cleaning_consumables"]),
                step=0.5,
                key=f"in_vcln_{epoch}",
            )
            st.info(f"**Total Variable Cost / Seat:** `{fmt_money(v_ed + v_mat + v_cln)}` (Baseline: $5.00)")

            st.markdown("**Monthly Fixed Costs ($/month):**")
            f_ven = st.number_input(
                "venue ($/mo)",
                min_value=0.0,
                value=float(state["fixed_cost_lines"]["venue"]),
                step=25.0,
                key=f"in_fven_{epoch}",
            )
            f_sch = st.number_input(
                "scheduling_software ($/mo)",
                min_value=0.0,
                value=float(state["fixed_cost_lines"]["scheduling_software"]),
                step=25.0,
                key=f"in_fsch_{epoch}",
            )
            f_acc = st.number_input(
                "accessibility_support ($/mo) [Required Explicit Line]",
                min_value=0.0,
                value=float(state["fixed_cost_lines"]["accessibility_support"]),
                step=25.0,
                key=f"in_facc_{epoch}",
            )
            f_adm = st.number_input(
                "admin_insurance_assumption ($/mo)",
                min_value=0.0,
                value=float(state["fixed_cost_lines"]["admin_insurance_assumption"]),
                step=25.0,
                key=f"in_fadm_{epoch}",
            )
            st.info(f"**Total Monthly Fixed Cost:** `{fmt_money(f_ven + f_sch + f_acc + f_adm)}` (Baseline: $700.00)")

        with col_in3:
            st.markdown("#### 3. Startup Financing & Period Cash Events")
            st.markdown(
                "**Opening Balance Sheet (NUR-3):**\n"
                "- Owner Contribution: `$6,000.00`\n"
                "- Equipment Assets: `$1,500.00` (not expensed again)\n"
                "- Pre-Opening Expense: `$500.00` (Retained Earnings `-$500.00`)\n"
                "- Post-Startup Spendable Cash: `$4,000.00`"
            )
            new_owner = st.number_input("New Owner Contribution ($)", min_value=0.0, value=0.0, step=100.0, key=f"in_own_{epoch}")
            new_grant = st.number_input("Educational Grant Proceeds ($)", min_value=0.0, value=0.0, step=100.0, key=f"in_grt_{epoch}")
            loan_in = st.number_input("New Simulated Loan Proceeds ($)", min_value=0.0, value=0.0, step=100.0, key=f"in_loan_{epoch}")
            prin_out = st.number_input("Loan Principal Repayment ($)", min_value=0.0, value=0.0, step=50.0, key=f"in_prin_{epoch}")
            int_exp = st.number_input("Interest Expense ($)", min_value=0.0, value=0.0, step=10.0, key=f"in_int_{epoch}")
            capex_out = st.number_input("Additional Equipment CapEx ($)", min_value=0.0, value=0.0, step=100.0, key=f"in_capex_{epoch}")

        # Sync updated parameters into state
        state["params"]["base_price"] = f"{to_money(p_in):.2f}"
        state["params"]["D0"] = int(d0_in)
        state["params"]["requested_demand"] = int(req_in)
        state["params"]["staff_hours"] = float(staff_in)
        state["params"]["hours_per_unit"] = float(hpu_in)
        state["params"]["venue_seats"] = int(seats_in)
        state["params"]["no_show_rate"] = float(ns_in)
        state["variable_cost_lines"] = {
            "educator_delivery": f"{to_money(v_ed):.2f}",
            "materials": f"{to_money(v_mat):.2f}",
            "cleaning_consumables": f"{to_money(v_cln):.2f}",
        }
        state["fixed_cost_lines"] = {
            "venue": f"{to_money(f_ven):.2f}",
            "scheduling_software": f"{to_money(f_sch):.2f}",
            "accessibility_support": f"{to_money(f_acc):.2f}",
            "admin_insurance_assumption": f"{to_money(f_adm):.2f}",
        }
        state["snapshot_id"] = compute_snapshot_id(state)

        preview_m = compute_nur_period_metrics(
            price=to_money(p_in),
            requested=int(req_in),
            staff_hours=float(staff_in),
            hours_per_unit=float(hpu_in),
            venue_seats=int(seats_in),
            no_show_rate=float(ns_in),
            unit_var_cost=get_unit_variable_cost(state),
            fixed_cost=get_monthly_fixed_cost(state),
            interest_cost=to_money(int_exp),
            principal_repayment=to_money(prin_out),
            new_owner_cash=to_money(new_owner),
            grant_cash=to_money(new_grant),
            loan_proceeds=to_money(loan_in),
            additional_capex=to_money(capex_out),
        )

        st.markdown("---")
        c_commit1, c_commit2 = st.columns([2, 1])
        with c_commit1:
            st.markdown("#### Commit Normal Period to Dated Ledger")
            p_lbl = st.text_input("Period Label (YYYY-MM)", value=f"2027-{len(state['ledger']) + 1:02d}", key=f"in_plbl_{epoch}")
            s_dt = st.text_input("Transaction Date (YYYY-MM-DD)", value=f"2027-{len(state['ledger']) + 1:02d}-28", key=f"in_sdt_{epoch}")
            r_id = st.text_input("Unique Run ID (Duplicate Protection)", value=f"RUN-{state['pathway_id']}-{len(state['ledger']) + 1:03d}", key=f"in_rid_{epoch}")
            if st.button("Approve & Record Period in Ledger", type="primary"):
                ok, msg = record_period_in_ledger(state, r_id.strip(), p_lbl.strip(), s_dt.strip(), preview_m)
                if ok:
                    bump_epoch()
                    st.success(msg)
                    st.rerun()
                else:
                    st.error(msg)

        with c_commit2:
            st.markdown("#### Fixture-Test Mode (Handbook p. 85)")
            st.warning("Fixture mode overrides capacity using monthly_fixture.csv. Never mix with normal runs.")
            if st.button("Reset & Replay 3-Month NUR Fixture"):
                fresh = build_default_state(state["team_id"])
                for row in NUR_FIXTURE_ROWS:
                    mf = compute_nur_period_metrics(
                        price=Decimal("25.00"),
                        requested=row["requested"],
                        staff_hours=30.0,
                        hours_per_unit=0.5,
                        venue_seats=60,
                        no_show_rate=0.0,
                        unit_var_cost=Decimal("5.00"),
                        fixed_cost=Decimal("700.00"),
                        ad_cost=to_money(row["ad_cost"]),
                        capacity_override=row["capacity_override"],
                    )
                    record_period_in_ledger(
                        fresh,
                        f"FIX-{row['month']}",
                        row["month"],
                        f"{row['month']}-28",
                        mf,
                        note="NUR Fixture Replay",
                    )
                st.session_state.sim_state = fresh
                bump_epoch()
                st.rerun()

        # Dated Ledger with Daily / Monthly / Yearly Aggregation (Handbook p. 9)
        st.markdown("---")
        st.subheader("Dated Event Ledger & Time-Horizon Aggregation")
        view_horizon = st.radio(
            "Select Time Aggregation View",
            ["Monthly (Committed Ledger)", "Daily (Modeled Working-Day Allocation)", "Yearly (Aggregated)"],
            horizontal=True,
        )
        if not state["ledger"]:
            st.info("No transactions committed yet. Use 'Approve & Record Period in Ledger' or 'Reset & Replay 3-Month NUR Fixture'.")
        else:
            df_led = pd.DataFrame(state["ledger"])
            if view_horizon.startswith("Monthly"):
                st.dataframe(df_led, use_container_width=True, hide_index=True)
            elif view_horizon.startswith("Daily"):
                wd = state["working_days_per_month"]
                cd = state["calendar_days_per_month"]
                st.caption(
                    f"Modeled daily allocation using {wd} working days/month (calendar days = {cd}/month; "
                    "never divides/multiplies by 30 arbitrarily). Explicitly labeled as modeled allocation, not observed daily sales."
                )
                daily_rows = []
                for r in state["ledger"]:
                    daily_rows.append(
                        {
                            "Month": r["period"],
                            "Allocation Basis": f"Modeled ({wd} working days)",
                            "Modeled Daily Seats": round(r["sold"] / wd, 2),
                            "Modeled Daily Revenue ($)": f"{float(r['revenue']) / wd:.2f}",
                            "Modeled Daily Expense ($)": f"{float(r['total_expense']) / wd:.2f}",
                            "Modeled Daily Profit ($)": f"{float(r['net_profit']) / wd:.2f}",
                        }
                    )
                st.dataframe(pd.DataFrame(daily_rows), use_container_width=True, hide_index=True)
            else:
                df_led_num = df_led.copy()
                df_led_num["Year"] = df_led_num["period"].str.slice(0, 4)
                for col in ["sold", "requested", "unserved_demand"]:
                    df_led_num[col] = df_led_num[col].astype(int)
                for col in ["revenue", "total_expense", "operating_profit", "net_profit"]:
                    df_led_num[col] = df_led_num[col].astype(float)
                yr_df = (
                    df_led_num.groupby("Year", as_index=False)[
                        ["requested", "sold", "unserved_demand", "revenue", "total_expense", "operating_profit", "net_profit"]
                    ]
                    .sum()
                )
                st.dataframe(yr_df, use_container_width=True, hide_index=True)

    # ==========================================================================
    # TAB 3: AP CALCULUS MATH LAB (FOUNDATION / APPLIED / ADVANCED)
    # ==========================================================================
    with tabs[2]:
        lvl = state["selected_level"]
        s_meta = AP_CALC_SKILLS[lvl]
        st.subheader(f"AP Calculus Math Lab — {lvl} Level ({state['pathway_id']})")
        st.markdown(f"**Active Skill:** {s_meta['skill_name']}")
        st.latex(s_meta["equation_latex"])
        st.info(f"**Required {lvl} Explanation:** {s_meta['required_explanation']}")

        # Key analytic metrics row
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Demand Model q(p) = a - bp", f"160 - {calc_analysis['b']:.1f}p" if state["params"]["D0"] == 80 else f"{calc_analysis['a']:.0f} - {calc_analysis['b']:.2f}p")
        m2.metric(
            "Unconstrained Critical Point p_unc",
            f"${calc_analysis['p_unc']:.2f}",
            delta=f"q={calc_analysis['q_unc']:.1f} ({'Feasible' if calc_analysis['unconstrained_feasible'] else 'INFEASIBLE > Cap'})",
            delta_color="normal" if calc_analysis["unconstrained_feasible"] else "inverse",
        )
        m3.metric(
            "Capacity Corner p_corner",
            f"${calc_analysis['p_corner']:.2f}",
            delta=f"Effective Cap = {calc_analysis['eff_cap']:.0f} seats",
        )
        m4.metric(
            "Constrained Analytic Optimum",
            f"${calc_analysis['p_opt_constrained']:.2f}",
            delta=f"Profit {fmt_money(calc_analysis['prof_opt_constrained'])} (Int: {fmt_money(calc_analysis['best_int_profit'])})",
        )

        # Aligned Profit P(p) and Derivative P'(p) Subplots (Handbook p. 37 & p. 70)
        full_grid = build_calc_grid(
            D0=int(state["params"]["D0"]),
            p0=float(state["params"]["base_price"]),
            v=float(unit_v),
            F=float(fixed_f),
            raw_capacity=live_metrics["derived_capacity"],
            no_show_rate=float(state["params"]["no_show_rate"]),
            num_points=121,
        )

        fig_calc = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.10,
            subplot_titles=(
                f"Piecewise Constrained Profit P(p) vs. Unconstrained Profit ($/month) [{lvl} Mode]",
                "First Derivative P'(p) ($/month per $/seat = seats) with Genuine Corner Marker",
            ),
        )
        fig_calc.add_trace(
            go.Scatter(
                x=full_grid["Price ($/seat)"],
                y=full_grid["Unconstrained Profit ($)"],
                mode="lines",
                name="Unconstrained Profit P_unc(p)",
                line=dict(color="#9ecae1", width=2, dash="dash"),
            ),
            row=1,
            col=1,
        )
        fig_calc.add_trace(
            go.Scatter(
                x=full_grid["Price ($/seat)"],
                y=full_grid["Constrained Profit P(p) ($)"],
                mode="lines",
                name="Constrained Piecewise Profit P(p)",
                line=dict(color="#08519c", width=3),
            ),
            row=1,
            col=1,
        )
        fig_calc.add_hline(y=0, line_color="#555555", row=1, col=1)

        if calc_analysis["has_corner"] and lvl in ("Applied", "Advanced"):
            fig_calc.add_vline(
                x=calc_analysis["p_corner"],
                line_dash="dashdot",
                line_color="#a63603",
                annotation_text=f"Corner p=${calc_analysis['p_corner']:.2f}",
                row=1,
                col=1,
            )
            fig_calc.add_vline(x=calc_analysis["p_corner"], line_dash="dashdot", line_color="#a63603", row=2, col=1)

        fig_calc.add_trace(
            go.Scatter(
                x=full_grid["Price ($/seat)"],
                y=full_grid["Derivative P'(p) (seats)"],
                mode="lines",
                name="Derivative P'(p) (seats)",
                line=dict(color="#238b45", width=3),
                connectgaps=False,
            ),
            row=2,
            col=1,
        )
        fig_calc.add_hline(y=0, line_color="#555555", row=2, col=1)

        # Open-circle markers for genuine corner jump discontinuity in P'(p)
        if calc_analysis["has_corner"] and lvl in ("Applied", "Advanced"):
            fig_calc.add_trace(
                go.Scatter(
                    x=[calc_analysis["p_corner"], calc_analysis["p_corner"]],
                    y=[calc_analysis["left_deriv_at_corner"], calc_analysis["right_deriv_at_corner"]],
                    mode="markers+text",
                    text=[
                        f"Left Limit P'- = {calc_analysis['left_deriv_at_corner']:.0f}",
                        f"Right Limit P'+ = {calc_analysis['right_deriv_at_corner']:.1f} (P' DNE)",
                    ],
                    textposition=["top right", "bottom right"],
                    marker=dict(size=11, color="#ffffff", line=dict(color="#d62728", width=2.5)),
                    name="Undefined Corner (P' DNE)",
                ),
                row=2,
                col=1,
            )

        fig_calc.update_xaxes(title_text="Ticket Price p ($/seat) [Domain: 0 <= p <= 2*p0]", row=2, col=1)
        fig_calc.update_yaxes(title_text="Profit ($/mo)", row=1, col=1)
        fig_calc.update_yaxes(title_text="P'(p) (seats)", row=2, col=1)
        fig_calc.update_layout(height=580, hovermode="x unified")
        st.plotly_chart(fig_calc, use_container_width=True)

        # Level-Specific Interactive Tasks
        if lvl == "Foundation":
            st.markdown("### Foundation Level Task: Endpoint & Critical Point Verification")
            ep_df = pd.DataFrame(
                [
                    {"Point": "Left Endpoint (p = $0.00)", "q(p)": f"{calc_analysis['a']:.1f}", "Unconstrained P(p)": fmt_money(-float(unit_v) * calc_analysis["a"] - float(fixed_f)), "P'(p)": f"{calc_analysis['a'] + calc_analysis['b']*float(unit_v):.1f}"},
                    {"Point": f"Critical Point (p = ${calc_analysis['p_unc']:.2f})", "q(p)": f"{calc_analysis['q_unc']:.1f}", "Unconstrained P(p)": fmt_money(calc_analysis["prof_unc"]), "P'(p)": "0.0 (Maximum unconstrained)"},
                    {"Point": f"Right Endpoint (p = ${calc_analysis['p_max']:.2f})", "q(p)": "0.0", "Unconstrained P(p)": fmt_money(-float(fixed_f)), "P'(p)": f"{calc_analysis['a'] + calc_analysis['b']*float(unit_v) - 2*calc_analysis['b']*calc_analysis['p_max']:.1f}"},
                ]
            )
            st.dataframe(ep_df, use_container_width=True, hide_index=True)

        elif lvl == "Applied":
            st.markdown("### Applied Level Task: Continuous Analytic Optimum vs. Integer Whole-Seat Sales")
            c_ap1, c_ap2 = st.columns(2)
            with c_ap1:
                st.write(
                    f"- **Capacity Branch ($0 \\le p < {calc_analysis['p_corner']:.2f}$):** $P'(p) = {calc_analysis['left_deriv_at_corner']:.0f} > 0$ (strictly increasing up to corner).\n"
                    f"- **Demand Branch (${calc_analysis['p_corner']:.2f} < p \\le {calc_analysis['p_max']:.2f}$):** $P'(p) = {calc_analysis['right_deriv_at_corner']:.2f} < 0$ at corner (strictly decreasing).\n"
                    f"- **Conclusion:** Because $P'(p)$ switches from $+{calc_analysis['left_deriv_at_corner']:.0f}$ to ${calc_analysis['right_deriv_at_corner']:.2f}$ at $p = \\${calc_analysis['p_corner']:.2f}$, the corner is the global constrained maximum!"
                )
            with c_ap2:
                st.dataframe(calc_analysis["int_table"].tail(12), use_container_width=True, hide_index=True, height=240)

        else:  # Advanced Level
            st.markdown("### Advanced Level Module A: Definite Integral Rate Accumulation (Handbook p. 70)")
            st.caption(
                "Isolated campaign rate exercise: r(t) = 100 + 20t ($/day) and c(t) = 60 + 5t ($/day) over [0, T]. "
                "Explicitly separated from monthly ledger rows."
            )
            t_horizon = st.slider("Campaign Accumulation Horizon T (days)", min_value=1.0, max_value=20.0, value=10.0, step=1.0, key=f"adv_T_{epoch}")
            integ = compute_integral_exercise(t_horizon)
            ic1, ic2, ic3 = st.columns(3)
            ic1.metric(f"Accumulated Revenue Integral [0, {t_horizon:.0f}d]", fmt_money(integ["acc_rev"]), delta="Units: ($/day)*days = $")
            ic2.metric(f"Accumulated Cost Integral [0, {t_horizon:.0f}d]", fmt_money(integ["acc_cost"]), delta="Units: ($/day)*days = $")
            ic3.metric(f"Accumulated Net Profit Integral [0, {t_horizon:.0f}d]", fmt_money(integ["acc_profit"]), delta="Exact fixture at T=10: $1,150.00")

            fig_int = go.Figure()
            fig_int.add_trace(
                go.Scatter(
                    x=integ["df"]["Day t"],
                    y=integ["df"]["Revenue Rate r(t) ($/day)"],
                    mode="lines",
                    name="Revenue Rate r(t) = 100 + 20t ($/day)",
                    line=dict(color="#1f77b4", width=3),
                )
            )
            fig_int.add_trace(
                go.Scatter(
                    x=integ["df"]["Day t"],
                    y=integ["df"]["Cost Rate c(t) ($/day)"],
                    mode="lines",
                    fill="tonexty",
                    fillcolor="rgba(31, 119, 180, 0.18)",
                    name="Cost Rate c(t) = 60 + 5t (Shaded = Net Profit Integral)",
                    line=dict(color="#d62728", width=3, dash="dash"),
                )
            )
            fig_int.update_layout(
                title=f"Shaded Accumulation: Integral_0^{t_horizon:.0f} (r(t) - c(t)) dt = {fmt_money(integ['acc_profit'])}",
                xaxis_title="Campaign Time t (days)",
                yaxis_title="Daily Dollar Rate ($/day)",
                height=340,
            )
            st.plotly_chart(fig_int, use_container_width=True)

            st.markdown("### Advanced Level Module B: Interacting Constraints, Disruption Sensitivity & Marginal vs. Average Cost")
            sens_rows = []
            for label, d_mult, cap_mod, ns_mod in [
                ("Baseline (D0=80, Cap=60, No-Show=0%)", 1.00, 60, 0.00),
                ("Downside Demand (-10% D0=72, Cap=60)", 0.90, 60, 0.00),
                ("Upside Demand (+10% D0=88, Cap=60)", 1.10, 60, 0.00),
                ("NUR-3 Disruption (-20% Venue Cap=48)", 1.00, 48, 0.00),
                ("No-Show Shock (20% No-Show -> Eff Cap=48)", 1.00, 60, 0.20),
            ]:
                an = analyze_ap_calculus(
                    D0=int(round(int(state["params"]["D0"]) * d_mult)),
                    p0=float(state["params"]["base_price"]),
                    v=float(unit_v),
                    F=float(fixed_f),
                    raw_capacity=cap_mod,
                    no_show_rate=ns_mod,
                )
                sens_rows.append(
                    {
                        "Constraint / Uncertainty Scenario": label,
                        "Effective Cap": f"{an['eff_cap']:.0f} seats",
                        "Unconstrained p_unc": f"${an['p_unc']:.2f}",
                        "Constrained Corner p*": f"${an['p_opt_constrained']:.2f}",
                        "Delivered Seats": f"{an['s_opt_constrained']:.0f}",
                        "Max Constrained Profit": fmt_money(an["prof_opt_constrained"]),
                    }
                )
            st.dataframe(pd.DataFrame(sens_rows), use_container_width=True, hide_index=True)

        # Student Evidence & Transfer Check Recorder (Handbook p. 6)
        st.markdown("---")
        st.subheader("Student Evidence & Transfer Check (Saved in Snapshot)")
        ev1, ev2 = st.columns(2)
        with ev1:
            state["student_evidence"]["prediction_before_slider"] = st.text_area(
                "1. Transfer Prediction (Before Moving Sliders)",
                value=state["student_evidence"]["prediction_before_slider"],
                height=85,
                key=f"ev_pred_{epoch}",
            )
            state["student_evidence"]["independent_calculation"] = st.text_area(
                "2. Independent Hand Calculation (No AI)",
                value=state["student_evidence"]["independent_calculation"],
                height=85,
                key=f"ev_calc_{epoch}",
            )
        with ev2:
            state["student_evidence"]["business_interpretation"] = st.text_area(
                "3. Business Interpretation (Units & Capacity Meaning)",
                value=state["student_evidence"]["business_interpretation"],
                height=85,
                key=f"ev_interp_{epoch}",
            )
            state["student_evidence"]["model_limitation"] = st.text_area(
                "4. Model Limitation & Unexpected Result Reflection",
                value=state["student_evidence"]["model_limitation"],
                height=85,
                key=f"ev_lim_{epoch}",
            )

    # ==========================================================================
    # TAB 4: OPERATIONS & CUSTOMERS (NUR-3 SPECIFIC)
    # ==========================================================================
    with tabs[3]:
        st.subheader("Healthcare / Nursing Operations & Customer Funnel (NUR-3)")
        st.caption(
            "Exact NUR capacity rule: booked = min(requested, floor(staff_hours / hours_per_unit), venue_seats); "
            "attended = floor(booked * (1 - no_show_rate)); sold = attended."
        )

        o1, o2, o3, o4 = st.columns(4)
        o1.metric("Educator Utilization", f"{live_metrics['educator_utilization_pct']:.1f}%", delta=f"{live_metrics['used_hours']} / {state['params']['staff_hours']} hrs")
        o2.metric("Venue Seat Utilization", f"{live_metrics['seat_utilization_pct']:.1f}%", delta=f"{live_metrics['booked']} / {state['params']['venue_seats']} seats")
        acc_cost_val = to_money(state["fixed_cost_lines"]["accessibility_support"])
        acc_per_seat = fmt_money(acc_cost_val / Decimal(live_metrics["attended"])) if live_metrics["attended"] > 0 else "N/A (0 attended)"
        o3.metric("Accessibility Cost / Attended Seat", acc_per_seat, delta=f"Monthly line: {fmt_money(acc_cost_val)}")
        o4.metric("Survey Satisfaction (Nonclinical)", f"{state['params']['satisfaction_mean']:.1f} / 5.0", delta="Supplied survey summary")

        # Operations Funnel Chart (Handbook p. 7 & p. 84)
        fig_ops = go.Figure(
            data=[
                go.Bar(
                    x=["Requested Demand", "Effective Capacity", "Booked Seats", "Attended (Sold)", "Unserved Demand", "No-Shows"],
                    y=[
                        live_metrics["requested"],
                        live_metrics["capacity"],
                        live_metrics["booked"],
                        live_metrics["attended"],
                        live_metrics["unserved_demand"],
                        live_metrics["no_shows"],
                    ],
                    text=[
                        live_metrics["requested"],
                        live_metrics["capacity"],
                        live_metrics["booked"],
                        live_metrics["attended"],
                        live_metrics["unserved_demand"],
                        live_metrics["no_shows"],
                    ],
                    textposition="outside",
                    marker_color=["#6baed6", "#3182bd", "#08519c", "#238b45", "#d95f0e", "#756bb1"],
                )
            ]
        )
        fig_ops.update_layout(
            title=f"NUR-3 Monthly Seat Funnel & Bottleneck Analysis ({live_metrics['bottleneck']})",
            yaxis_title="Workshop Seats (count)",
            yaxis_range=[0, max(100, live_metrics["requested"] + 15)],
            height=380,
        )
        st.plotly_chart(fig_ops, use_container_width=True)

        st.markdown("#### NUR-3 Approved Offerings Directory (No Double-Counting Revenue)")
        st.dataframe(pd.DataFrame(NUR_OFFERINGS), use_container_width=True, hide_index=True)

    # ==========================================================================
    # TAB 5: AI DEPARTMENTAL HANDOFFS & INDEPENDENT REVIEWER
    # ==========================================================================
    with tabs[4]:
        st.subheader("Departmental AI Assistants (Manual Handoff) & Reviewer Gate")
        st.info(
            "Manual Workflow Notice: Copy a role prompt below along with your exported Snapshot JSON into your "
            "teacher-approved Gem, Skill, or ChatGPT Project. Import the structured JSON recommendation below for student review."
        )

        role_choice = st.selectbox(
            "Select Departmental Assistant Role Prompt",
            ["Finance Analyst", "Operations Planner", "Market & Marketing Specialist", "Independent Reviewer"],
        )
        common_hdr = (
            f"ROLE: {role_choice} | TEAM: {state['team_id']} | PATHWAY: NUR-3 (Nurse-Led Health Education Simulation)\n"
            f"COURSE: AP Calculus | LEVEL: {state['selected_level']} | SNAPSHOT: {state['snapshot_id']} (v{state['state_version']})\n"
            f"CURRENT PARAMS: price=${state['params']['base_price']}, D0={state['params']['D0']}, "
            f"staff_hours={state['params']['staff_hours']}, hours_per_unit={state['params']['hours_per_unit']}, "
            f"venue_seats={state['params']['venue_seats']}, no_show_rate={state['params']['no_show_rate']}, "
            f"v=${unit_v}, F=${fixed_f}, cash=${state['balances']['cash']}.\n"
            "RULES: Use only this snapshot version; reject stale records; distinguish demand from attended/sold seats, "
            "profit from cash, and financing from revenue. Do not provide clinical diagnosis or medical advice. "
            "Return structured JSON with keys: role, team_id, pathway_id, state_version, question, method, recommendation, "
            "uncertainty, proposed_changes, checks_needed, approval_status='pending'."
        )
        role_specific = {
            "Finance Analyst": "TASK: Reconcile revenue, variable expense ($5/unit), fixed expense ($700/mo incl. $200 accessibility), cash, and equity. Flag cash below $800. Compare baseline price $25 with AP Calculus constrained corner price $31.25.",
            "Operations Planner": "TASK: Apply booked=min(requested, floor(staff_hours/0.5), venue_seats); attended=floor(booked*(1-no_show_rate)). Identify bottleneck, unserved demand, and evaluate the -20% venue capacity disruption (60 -> 48 seats).",
            "Market & Marketing Specialist": "TASK: Propose a truthful, nonclinical wellness workshop flyer/campaign with itemized cost (design + assets + print + distribution + placement). Do not assume automatic demand lift; propose an A/B comparison.",
            "Independent Reviewer": "TASK: Independently recompute profit, capacity, and the NUR no-show test (no_show_rate=0.20 -> booked=60, attended=48, profit=$260). Verify units, domain, accessibility line item, and profit vs. cash distinction.",
        }
        st.code(common_hdr + "\n" + role_specific[role_choice], language="text")

        st.markdown("---")
        st.subheader("Import Structured Recommendation JSON (Stale-Version & Team Guarded)")
        sample_rec = json.dumps(
            {
                "role": "Finance Analyst",
                "team_id": state["team_id"],
                "pathway_id": "NUR-3",
                "state_version": state["state_version"],
                "question": "Should we adjust price from $25.00 toward the AP Calculus constrained corner $31.25?",
                "method": "Piecewise profit optimization P(p) = (p-5)*min(60, 160-3.2p) - 700",
                "recommendation": "Test $31.25 seat price while preserving $200/mo accessibility support for low-income participants.",
                "uncertainty": "Linear demand slope b=3.2 is an instructional estimate; monitor actual booking conversion.",
                "proposed_changes": {"base_price": "31.25"},
                "checks_needed": "Verify corner derivative limits P'-(31.25)=60 vs P'+(31.25)=-24 independently.",
                "approval_status": "pending",
            },
            indent=2,
        )
        rec_raw = st.text_area("Paste Recommendation JSON", value=sample_rec, height=190, key=f"ai_rec_box_{epoch}")
        if st.button("Validate & Queue Recommendation for Student Review"):
            try:
                rec_obj = json.loads(rec_raw)
                if rec_obj.get("team_id") != state["team_id"]:
                    st.error(f"REJECTED (Wrong Team): Expected '{state['team_id']}', got '{rec_obj.get('team_id')}'.")
                elif rec_obj.get("pathway_id") != "NUR-3":
                    st.error(f"REJECTED (Wrong Pathway): Expected 'NUR-3', got '{rec_obj.get('pathway_id')}'.")
                elif int(rec_obj.get("state_version", -1)) != int(state["state_version"]):
                    st.error(
                        f"REJECTED (Stale State Version): Recommendation targets v{rec_obj.get('state_version')}, "
                        f"but business is at v{state['state_version']}. Refresh snapshot and re-run."
                    )
                else:
                    rec_obj["approval_status"] = "pending"
                    state["recommendations"].append(rec_obj)
                    st.success("Valid recommendation queued as PENDING. Student reviewer must approve below before applying.")
            except Exception as exc:
                st.error(f"Invalid Recommendation JSON: {exc}")

        if state["recommendations"]:
            st.markdown("#### Pending & Reviewed Recommendations")
            for idx, rec in enumerate(state["recommendations"]):
                with st.container(border=True):
                    st.markdown(
                        f"**#{idx + 1} | {rec['role']}** (Target `v{rec['state_version']}`) &mdash; "
                        f"Status: **{rec['approval_status'].upper()}**"
                    )
                    st.write(f"**Recommendation:** {rec['recommendation']}")
                    st.write(f"**Proposed Changes:** `{rec.get('proposed_changes', {})}` | **Check Needed:** {rec.get('checks_needed', '')}")
                    if rec["approval_status"] == "pending":
                        ca1, ca2 = st.columns(2)
                        if ca1.button(f"Approve & Apply Changes (#{idx + 1})", key=f"app_rec_{idx}_{epoch}"):
                            changes = rec.get("proposed_changes", {})
                            if "base_price" in changes:
                                state["params"]["base_price"] = f"{to_money(changes['base_price']):.2f}"
                            if "venue_seats" in changes:
                                state["params"]["venue_seats"] = int(changes["venue_seats"])
                            if "staff_hours" in changes:
                                state["params"]["staff_hours"] = float(changes["staff_hours"])
                            rec["approval_status"] = "approved"
                            state["state_version"] += 1
                            state["snapshot_id"] = compute_snapshot_id(state)
                            bump_epoch()
                            st.rerun()
                        if ca2.button(f"Reject Recommendation (#{idx + 1})", key=f"rej_rec_{idx}_{epoch}"):
                            rec["approval_status"] = "rejected"
                            st.rerun()

    # ==========================================================================
    # TAB 6: COSTED MARKETING STUDIO (DUPLICATE CAMPAIGN ID PROTECTION)
    # ==========================================================================
    with tabs[5]:
        st.subheader("Costed Marketing Studio & A/B Variant Planner")
        st.caption(
            "Advertising has a real economic cost. Draft -> Student Fact-Check -> Reviewer Approval -> Launch in Simulation. "
            "Launching the same campaign_id twice is strictly blocked."
        )

        mc1, mc2 = st.columns(2)
        with mc1:
            c_id = st.text_input("Unique Campaign ID", value=f"CAMP-NUR-{len(state['campaigns']) + 1:03d}", key=f"mk_id_{epoch}")
            c_off = st.selectbox("Target Offering", [o["name"] for o in NUR_OFFERINGS], key=f"mk_off_{epoch}")
            c_var = st.selectbox(
                "A/B Variant Label (Equal-Budget Comparison)",
                ["Variant A (Plain-Language Flyer)", "Variant B (Caregiver Checklist Postcard)"],
                key=f"mk_var_{epoch}",
            )
            c_aud = st.text_input("Target Audience (Fictional)", value="Local adults & family caregivers", key=f"mk_aud_{epoch}")
            c_claim = st.text_area(
                "Approved Nonclinical Health-Literacy Claim",
                value="Join our nurse-led general wellness & first-aid awareness workshop. Educational content only; no medical diagnosis or treatment.",
                height=75,
                key=f"mk_clm_{epoch}",
            )
            c_cta = st.text_input("Call to Action & Fictional Contact", value="Register at Community Health Desk | (555) 019-2834 (Fictional)", key=f"mk_cta_{epoch}")

        with mc2:
            d_hrs = st.number_input("Design Labor Hours", min_value=0.0, value=2.0, step=0.5, key=f"mk_dhrs_{epoch}")
            d_rate = st.number_input("Design Hourly Rate ($/hr)", min_value=0.0, value=25.0, step=5.0, key=f"mk_drate_{epoch}")
            in_kind = st.checkbox("Mark Design Labor as In-Kind Student Labor (disclosed separately)", value=False, key=f"mk_ik_{epoch}")
            assets_c = st.number_input("Licensed Assets Cost ($)", min_value=0.0, value=10.0, step=5.0, key=f"mk_ast_{epoch}")
            pr_qty = st.number_input("Print Quantity (flyers)", min_value=0, value=100, step=25, key=f"mk_pqty_{epoch}")
            pr_unit = st.number_input("Print Unit Cost ($/flyer)", min_value=0.0, value=0.20, step=0.05, key=f"mk_punit_{epoch}")
            dist_c = st.number_input("Distribution Cost ($)", min_value=0.0, value=10.0, step=5.0, key=f"mk_dist_{epoch}")
            plac_c = st.number_input("Placement Cost ($)", min_value=0.0, value=10.0, step=5.0, key=f"mk_plac_{epoch}")
            reach_est = st.number_input("Assumed Reach (impressions)", min_value=0, value=600, step=50, key=f"mk_reach_{epoch}")
            inq_est = st.number_input("Tracked Inquiries", min_value=0, value=45, step=5, key=f"mk_inq_{epoch}")
            conv_est = st.number_input("Attributed Conversions (0 -> CAC = N/A)", min_value=0, value=0, step=1, key=f"mk_conv_{epoch}")

        cash_design = 0.0 if in_kind else (d_hrs * d_rate)
        tot_camp_cost = to_money(cash_design + assets_c + (pr_qty * pr_unit) + dist_c + plac_c)
        cac_display = fmt_money(tot_camp_cost / Decimal(conv_est)) if conv_est > 0 else "N/A (Zero Conversions)"
        st.info(
            f"**Itemized Campaign Cash Cost:** `{fmt_money(tot_camp_cost)}` | "
            f"**In-Kind Student Labor Disclosed:** `{fmt_money(d_hrs * d_rate if in_kind else 0)}` | "
            f"**Customer Acquisition Cost (CAC):** `{cac_display}`"
        )

        fact_ok = st.checkbox("Student Fact-Check Verified (Nonclinical claim, fictional contact, accessible plain language)", value=True, key=f"mk_fc_{epoch}")
        rev_ok = st.checkbox("Independent Reviewer Approval Granted", value=True, key=f"mk_rv_{epoch}")

        if st.button("Save / Launch Campaign in Ledger (Once per Campaign ID)", type="primary"):
            existing_ids = {c["campaign_id"] for c in state["campaigns"] if c.get("charged_in_ledger")}
            if c_id.strip() in existing_ids:
                st.error(f"REJECTED: Campaign ID '{c_id.strip()}' has already been launched and expensed once!")
            elif not (fact_ok and rev_ok):
                st.error("Both Student Fact-Check and Reviewer Approval are required before launching.")
            else:
                new_camp = {
                    "campaign_id": c_id.strip(),
                    "offering": c_off,
                    "audience": c_aud,
                    "approved_claim": c_claim,
                    "channel": "Printable Flyer & Community Posting",
                    "dates": state["simulation_date"],
                    "design_hours": d_hrs,
                    "design_hourly_rate": f"{to_money(d_rate):.2f}",
                    "in_kind_student_labor": in_kind,
                    "licensed_assets": f"{to_money(assets_c):.2f}",
                    "print_quantity": int(pr_qty),
                    "print_unit_cost": f"{to_money(pr_unit):.2f}",
                    "distribution": f"{to_money(dist_c):.2f}",
                    "placement": f"{to_money(plac_c):.2f}",
                    "total_cost": f"{tot_camp_cost:.2f}",
                    "reach_assumed": int(reach_est),
                    "inquiries": int(inq_est),
                    "conversions": int(conv_est),
                    "status": "Launched (Expensed Once)",
                    "fact_checked": True,
                    "reviewer_approved": True,
                    "charged_in_ledger": True,
                    "variant_label": c_var,
                }
                state["campaigns"].append(new_camp)
                # Record zero-seat marketing expense transaction in ledger so cash & operating profit update once
                m_ad = compute_nur_period_metrics(
                    price=to_money(state["params"]["base_price"]),
                    requested=0,
                    staff_hours=float(state["params"]["staff_hours"]),
                    hours_per_unit=float(state["params"]["hours_per_unit"]),
                    venue_seats=int(state["params"]["venue_seats"]),
                    no_show_rate=float(state["params"]["no_show_rate"]),
                    unit_var_cost=Decimal("0.00"),
                    fixed_cost=Decimal("0.00"),
                    ad_cost=tot_camp_cost,
                )
                record_period_in_ledger(
                    state,
                    run_id=f"AD-{c_id.strip()}",
                    period_label=state["base_period"],
                    sim_date=state["simulation_date"],
                    metrics=m_ad,
                    note=f"Campaign Launch {c_id.strip()} ({c_var})",
                )
                bump_epoch()
                st.success(f"Launched {c_id.strip()} and recorded {fmt_money(tot_camp_cost)} expense once!")
                st.rerun()

        st.markdown("#### Printable Flyer Draft Preview")
        with st.container(border=True):
            st.markdown(f"### {c_off}")
            st.markdown(f"**Audience:** {c_aud} | **Campaign ID:** `{c_id}` ({c_var})")
            st.write(c_claim)
            st.markdown(f"**Workshop Fee:** {fmt_money(state['params']['base_price'])} per seat | **Action:** {c_cta}")
            st.caption("Accessibility: 14pt high-contrast print layout, plain-language summary, and bilingual handout available.")

    # ==========================================================================
    # TAB 7: REPORT CENTER (ALL 8 REPORTS + CSV/PDF) & DECISION MEMO
    # ==========================================================================
    with tabs[6]:
        st.subheader("Report Center (All 8 Reconciled Handbook Reports)")
        st.caption(
            f"Team: {state['team_id']} | Pathway: {state['pathway_id']} | Snapshot: {state['snapshot_id']} | "
            "Viewing or exporting reports never mutates state or creates duplicate transactions."
        )

        reports = generate_all_reports(state, live_metrics)
        pdf_bytes = build_pdf_report(state, reports)
        st.download_button(
            label="Download Complete Printable PDF Report Package (All 8 Reports)",
            data=pdf_bytes,
            file_name=f"{state['team_id']}_NUR3_Reports_{state['snapshot_id']}.pdf",
            mime="application/pdf",
            type="primary",
        )

        rep_names = list(reports.keys())
        chosen_rep = st.selectbox("Select Report to Inspect & Export CSV", rep_names)
        rep_df = reports[chosen_rep]
        st.dataframe(rep_df, use_container_width=True, hide_index=True)
        st.download_button(
            label=f"Export '{chosen_rep}' as CSV",
            data=rep_df.to_csv(index=False).encode("utf-8"),
            file_name=f"{state['team_id']}_{chosen_rep.replace(' ', '_')}_{state['snapshot_id']}.csv",
            mime="text/csv",
        )

        st.markdown("---")
        st.subheader("Round Decision Memo Builder (Handbook p. 10)")
        st.caption("Every business round requires citing at least TWO reports, comparing baseline with TWO alternatives, and addressing non-profit criteria.")
        dm1, dm2 = st.columns(2)
        with dm1:
            dq = st.text_input("Decision Question", value="How should we price workshop seats and protect accessibility under capacity limits?")
            cited_reps = st.multiselect("Cite At Least TWO Reports", rep_names, default=[rep_names[0], rep_names[4]])
            alt_text = st.text_area(
                "Compare Baseline vs. Two Alternatives",
                value=(
                    "Baseline ($25/seat, 60 seats): Profit = $500/mo, 20 unserved seats.\n"
                    "Alternative 1 ($31.25/seat corner optimum, 60 seats): Profit = $875/mo, 0 unserved seats.\n"
                    "Alternative 2 ($35.00/seat under -20% venue disruption = 48 seats): Profit = $740/mo, preserves $200/mo accessibility support."
                ),
                height=95,
            )
        with dm2:
            pred_out = st.text_input("Student Prediction & Measurable Outcome", value="Operating profit rises to $875/mo while closing cash stays > $4,500.")
            non_prof = st.text_area(
                "Service Quality, Accessibility, Ethical Nonclinical Claims & Cash Reserve Check",
                value="Maintains $200/mo accessibility support line, keeps claims strictly educational (no diagnosis), and maintains cash well above $800 reserve.",
                height=75,
            )
            owner_rev = st.text_input("Decision Owner, Checker & Review Date", value="Owner: Ops Lead | Checker: Finance Lead | Review: 2027-02-28")

        if st.button("Save Decision Memo to Snapshot"):
            if len(cited_reps) < 2:
                st.error("Please cite at least TWO reports per Handbook p. 10.")
            else:
                state["decisions"].append(
                    {
                        "decision_id": f"DEC-{len(state['decisions']) + 1:03d}",
                        "snapshot_id": state["snapshot_id"],
                        "question": dq,
                        "cited_reports": cited_reps,
                        "alternatives": alt_text,
                        "prediction": pred_out,
                        "non_profit_ethics": non_prof,
                        "owner_and_review": owner_rev,
                    }
                )
                st.success(f"Saved Decision Memo DEC-{len(state['decisions']):03d} to snapshot!")

        if state["decisions"]:
            st.dataframe(pd.DataFrame(state["decisions"]), use_container_width=True, hide_index=True)

    # ==========================================================================
    # TAB 8: CROSS-COURSE CONNECTIONS, FROZEN ASSESSMENT & ACCEPTANCE TESTS
    # ==========================================================================
    with tabs[7]:
        st.subheader("Cross-Course Connections Panel (At Least Two Substantive Connections)")
        cc1, cc2 = st.columns(2)
        with cc1:
            with st.container(border=True):
                st.markdown(f"#### Connection 1: {state['cross_course']['conn_1_subject']}")
                st.write(f"**Named Concept:** {state['cross_course']['conn_1_concept']}")
                st.write(f"**Complexity Level:** {state['cross_course']['conn_1_complexity']}")
                st.write(f"**Supporting Artifact:** {state['cross_course']['conn_1_artifact']}")
                st.write(state["cross_course"]["conn_1_explanation"])
        with cc2:
            with st.container(border=True):
                st.markdown(f"#### Connection 2: {state['cross_course']['conn_2_subject']}")
                st.write(f"**Named Concept:** {state['cross_course']['conn_2_concept']}")
                st.write(f"**Complexity Level:** {state['cross_course']['conn_2_complexity']}")
                st.write(f"**Supporting Artifact:** {state['cross_course']['conn_2_artifact']}")
                st.write(state["cross_course"]["conn_2_explanation"])

        st.markdown("---")
        st.subheader("Snapshot-Frozen Practice Assessment Panel (Handbook p. 12 & p. 62)")
        st.caption(
            "Freezes snapshot_id, state_version, parameters, and seed before generating questions. "
            "Hints and worked solutions reveal only after submission. Graded keys remain private/LMS-controlled."
        )
        if st.button("Freeze Current Snapshot & Generate Practice Assessment"):
            st.session_state.frozen_assessment = {
                "snapshot_id": state["snapshot_id"],
                "state_version": state["state_version"],
                "level": state["selected_level"],
                "D0": int(state["params"]["D0"]),
                "p0": float(state["params"]["base_price"]),
                "v": float(unit_v),
                "F": float(fixed_f),
                "cap": live_metrics["derived_capacity"],
                "no_show": float(state["params"]["no_show_rate"]),
                "calc": copy.deepcopy(calc_analysis),
            }
            st.session_state.practice_submitted = False

        fa = st.session_state.frozen_assessment
        if fa is not None:
            st.info(f"Frozen Assessment Snapshot: `{fa['snapshot_id']}` (v{fa['state_version']}) | Level: **{fa['level']}**")
            ca = fa["calc"]
            q1_ans = st.number_input(
                f"Q1 (Calculation): Using frozen D0={fa['D0']}, p0=${fa['p0']:.2f}, and effective capacity={ca['eff_cap']:.0f} seats, "
                "what is the exact capacity-corner ticket price p_corner ($/seat)?",
                min_value=0.0,
                max_value=200.0,
                value=0.0,
                step=0.25,
                key="qa_q1",
            )
            q2_ans = st.selectbox(
                "Q2 (Graph & Derivative Reasoning): At p = p_corner, why is P'(p) undefined?",
                [
                    "Select an explanation...",
                    "Left-hand derivative (+cap) does not equal right-hand derivative (a + bv - 2bp), creating a corner",
                    "Total revenue is zero at the corner price",
                    "Fixed cost F becomes negative at capacity",
                ],
                key="qa_q2",
            )
            q3_ans = st.selectbox(
                "Q3 (Cross-Course & Error Diagnosis): If an AI specialist returns a recommendation tagged v1 after the simulation advanced to v2, what must happen?",
                [
                    "Select an action...",
                    "Reject the stale recommendation with an explanation and request a new run on v2",
                    "Automatically merge it into the ledger",
                    "Double the marketing expense to compensate",
                ],
                key="qa_q3",
            )
            if st.button("Submit Practice Assessment"):
                st.session_state.practice_submitted = True

            if st.session_state.practice_submitted:
                q1_ok = abs(q1_ans - ca["p_corner"]) <= 0.05
                q2_ok = q2_ans.startswith("Left-hand derivative")
                q3_ok = q3_ans.startswith("Reject the stale")
                st.markdown(
                    f"**Practice Feedback (Frozen `{fa['snapshot_id']}`):**\n"
                    f"- **Q1:** {'CORRECT' if q1_ok else 'REVIEW'} &mdash; Worked Solution: $p_{{\\text{{corner}}}} = (a - \\text{{cap}})/b = ({ca['a']:.0f} - {ca['eff_cap']:.0f}) / {ca['b']:.2f} = \\${ca['p_corner']:.2f}$.\n"
                    f"- **Q2:** {'CORRECT' if q2_ok else 'REVIEW'} &mdash; Left limit $P'_- = +{ca['left_deriv_at_corner']:.0f}$ seats vs. right limit $P'_+ = {ca['right_deriv_at_corner']:.2f}$ seats.\n"
                    f"- **Q3:** {'CORRECT' if q3_ok else 'REVIEW'} &mdash; Handbook data contract strictly rejects stale versions."
                )

        st.markdown("---")
        st.subheader("Automated Handbook Acceptance Test Suite (NUR-3 + AP Calculus)")
        st.caption("Executes all required numerical, ledger, duplicate-run, loan, calculus, and JSON restore checks on isolated copies.")
        if st.button("Run All 10 Handbook Acceptance Tests", type="primary"):
            test_df = run_handbook_acceptance_tests()
            st.dataframe(test_df, use_container_width=True, hide_index=True)
            passed_cnt = int((test_df["Status"] == "PASS").sum())
            st.success(f"Acceptance Test Result: {passed_cnt} / {len(test_df)} PASSED.")


if __name__ == "__main__":
    main()
