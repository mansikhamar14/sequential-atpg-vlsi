"""
5-Valued Logic Engine for Extended D-Algorithm
================================================
Values: 0, 1, X, D, D_BAR

  D     = good circuit has 1, faulty circuit has 0
  D_BAR = good circuit has 0, faulty circuit has 1

Priority order for implication: 0 > X > D/D_BAR > 1

Truth tables implement the standard 5-valued calculus for
AND, OR, NAND, NOR, NOT, XOR, XNOR, BUF gates.
"""

# ── Value Constants ──────────────────────────────────────────────────────────

ZERO  = 0
ONE   = 1
X     = "X"
D     = "D"
D_BAR = "D_BAR"

FIVE_VALUES = [ZERO, ONE, X, D, D_BAR]


# ── NOT table ────────────────────────────────────────────────────────────────

def not_5v(a):
    """NOT gate in 5-valued logic."""
    if a == ZERO:  return ONE
    if a == ONE:   return ZERO
    if a == D:     return D_BAR
    if a == D_BAR: return D
    return X


# ── AND table ────────────────────────────────────────────────────────────────
#
#         0    1    X    D    D'
#    0  | 0    0    0    0    0
#    1  | 0    1    X    D    D'
#    X  | 0    X    X    X    X
#    D  | 0    D    X    D    0
#    D' | 0    D'   X    0    D'

_AND_TABLE = {
    (ZERO, ZERO): ZERO, (ZERO, ONE): ZERO, (ZERO, X): ZERO, (ZERO, D): ZERO, (ZERO, D_BAR): ZERO,
    (ONE, ZERO): ZERO,  (ONE, ONE): ONE,   (ONE, X): X,     (ONE, D): D,     (ONE, D_BAR): D_BAR,
    (X, ZERO): ZERO,    (X, ONE): X,       (X, X): X,       (X, D): X,       (X, D_BAR): X,
    (D, ZERO): ZERO,    (D, ONE): D,       (D, X): X,       (D, D): D,       (D, D_BAR): ZERO,
    (D_BAR, ZERO): ZERO,(D_BAR, ONE): D_BAR,(D_BAR, X): X,  (D_BAR, D): ZERO,(D_BAR, D_BAR): D_BAR,
}

def and_5v(a, b):
    return _AND_TABLE.get((a, b), X)


# ── OR table ─────────────────────────────────────────────────────────────────
#
#         0    1    X    D    D'
#    0  | 0    1    X    D    D'
#    1  | 1    1    1    1    1
#    X  | X    1    X    X    X
#    D  | D    1    X    D    1
#    D' | D'   1    X    1    D'

_OR_TABLE = {
    (ZERO, ZERO): ZERO, (ZERO, ONE): ONE, (ZERO, X): X,     (ZERO, D): D,     (ZERO, D_BAR): D_BAR,
    (ONE, ZERO): ONE,   (ONE, ONE): ONE,  (ONE, X): ONE,    (ONE, D): ONE,    (ONE, D_BAR): ONE,
    (X, ZERO): X,       (X, ONE): ONE,    (X, X): X,        (X, D): X,        (X, D_BAR): X,
    (D, ZERO): D,       (D, ONE): ONE,    (D, X): X,        (D, D): D,        (D, D_BAR): ONE,
    (D_BAR, ZERO): D_BAR,(D_BAR, ONE): ONE,(D_BAR, X): X,   (D_BAR, D): ONE,  (D_BAR, D_BAR): D_BAR,
}

def or_5v(a, b):
    return _OR_TABLE.get((a, b), X)


# ── N-input gate evaluation ─────────────────────────────────────────────────

def and_5v_n(*args):
    """N-input AND gate."""
    result = args[0]
    for v in args[1:]:
        result = and_5v(result, v)
    return result

def or_5v_n(*args):
    """N-input OR gate."""
    result = args[0]
    for v in args[1:]:
        result = or_5v(result, v)
    return result

def nand_5v_n(*args):
    """N-input NAND = NOT(AND)."""
    return not_5v(and_5v_n(*args))

def nor_5v_n(*args):
    """N-input NOR = NOT(OR)."""
    return not_5v(or_5v_n(*args))

def xor_5v(a, b):
    """2-input XOR in 5-valued logic."""
    if a == X or b == X:
        return X
    # XOR using AND/OR/NOT: a⊕b = (a·b') + (a'·b)
    return or_5v(and_5v(a, not_5v(b)), and_5v(not_5v(a), b))

def xor_5v_n(*args):
    """N-input XOR."""
    result = args[0]
    for v in args[1:]:
        result = xor_5v(result, v)
    return result

def xnor_5v_n(*args):
    """N-input XNOR = NOT(XOR)."""
    return not_5v(xor_5v_n(*args))

def buf_5v(a):
    return a


# ── Gate dispatch ────────────────────────────────────────────────────────────

GATE_FN_5V = {
    "and":  and_5v_n,
    "or":   or_5v_n,
    "not":  lambda a: not_5v(a),
    "nand": nand_5v_n,
    "nor":  nor_5v_n,
    "xor":  xor_5v_n,
    "xnor": xnor_5v_n,
    "buf":  lambda a: buf_5v(a),
}


def evaluate_gate_5v(gate_type, input_values):
    """Evaluate a gate given its type and list of 5-valued inputs."""
    fn = GATE_FN_5V.get(gate_type.lower())
    if fn is None:
        raise ValueError(f"Unknown gate type: {gate_type}")
    return fn(*input_values)


# ── Helpers ──────────────────────────────────────────────────────────────────

def is_d_value(v):
    """True if value is D or D_BAR (fault effect present)."""
    return v == D or v == D_BAR

def complement_5v(v):
    """Complement: 0↔1, D↔D_BAR, X→X."""
    return not_5v(v)

def fault_activation_value_5v(fault_type):
    """
    Value to place on the fault wire to activate the fault.
      SA0: good=1, faulty=0 → D
      SA1: good=0, faulty=1 → D_BAR
    """
    if fault_type.upper() == "SA0":
        return D
    elif fault_type.upper() == "SA1":
        return D_BAR
    raise ValueError(f"Unknown fault type: {fault_type}")

def ctrl_val_5v(gate_type):
    """Controlling value of a gate type."""
    if gate_type in ("and", "nand"):
        return ZERO
    if gate_type in ("or", "nor"):
        return ONE
    return None

def non_ctrl_val_5v(gate_type):
    """Non-controlling value of a gate type."""
    cv = ctrl_val_5v(gate_type)
    if cv is None:
        return None
    return ONE if cv == ZERO else ZERO

def is_inverting_5v(gate_type):
    """True if gate inverts output."""
    return gate_type in ("nand", "nor", "not", "xnor")


# ── Self-test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== 5-Valued Logic Self-Test ===\n")
    tests = [
        ("AND(D,1)",     and_5v(D, ONE),       D),
        ("AND(D,0)",     and_5v(D, ZERO),      ZERO),
        ("AND(D,D')",    and_5v(D, D_BAR),     ZERO),
        ("AND(D,D)",     and_5v(D, D),         D),
        ("OR(D,0)",      or_5v(D, ZERO),       D),
        ("OR(D,1)",      or_5v(D, ONE),        ONE),
        ("OR(D,D')",     or_5v(D, D_BAR),      ONE),
        ("NOT(D)",       not_5v(D),            D_BAR),
        ("NOT(D')",      not_5v(D_BAR),        D),
        ("NAND(1,D)",    nand_5v_n(ONE, D),    D_BAR),
        ("NOR(0,D)",     nor_5v_n(ZERO, D),    D_BAR),
        ("BUF(D)",       buf_5v(D),            D),
    ]

    ok = True
    for label, result, expected in tests:
        status = "PASS" if result == expected else "FAIL"
        if status == "FAIL": ok = False
        print(f"  [{status}] {label}: got={result}, expected={expected}")
    print(f"\n{'All passed ✓' if ok else 'FAILURES ✗'}")
