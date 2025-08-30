"""NeuroSymbolic demo: symbolic rules + learned (toy) scorer.
We classify loan approvals. Symbolic rule: reject if debt/income > 0.6.
Neural-ish score: logistic regression with fixed weights (pretend pre-trained).
Final decision: rules can veto, otherwise use score threshold.
"""
from math import exp


# Toy logistic regression (pretrained weights)
W = {
"bias": -1.2,
"income": 0.002, # higher income -> more likely approve
"credit": 0.015, # higher credit score helps
"debt": -0.004, # more debt hurts
}


def sigmoid(z):
return 1 / (1 + exp(-z))


def neural_score(x):
z = (W["bias"] + W["income"]*x["income"] + W["credit"]*x["credit"] + W["debt"]*x["debt"])
return sigmoid(z)


# Symbolic rule system
RULES = [
(lambda x: x["debt"]/max(1,x["income"]) > 0.6, "REJECT: debt-to-income too high"),
(lambda x: x["credit"] < 500, "REJECT: credit below policy floor"),
]


def decide(app):
for cond, reason in RULES:
if cond(app):
return False, f"Rule veto → {reason}"
p = neural_score(app)
return (p >= 0.5), f"Model p={p:.2f}"


if __name__ == "__main__":
examples = [
{"income": 8000, "debt": 3000, "credit": 720},
{"income": 4000, "debt": 2800, "credit": 640},
{"income": 2500, "debt": 1800, "credit": 480},
]
for i, x in enumerate(examples, 1):
ok, why = decide(x)
print(f"Case {i}: {'APPROVE' if ok else 'REJECT'} — {why}")