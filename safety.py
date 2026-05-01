from models import PlannedStep, NodeData

RISKY_KEYWORDS = [
    "pay", "send money", "transfer", "delete", "remove",
    "confirm payment", "upi", "buy now", "place order",
    "submit", "share", "post", "publish"
]

RISKY_PACKAGES = [
    "com.google.android.apps.nbu.paisa.user",   # GPay
    "net.one97.paytm",                           # Paytm
    "com.phonepe.app",                           # PhonePe
]

def classify_step_risk(step: PlannedStep, app_package: str) -> bool:
    text = step.description.lower()
    keyword_match = any(kw in text for kw in RISKY_KEYWORDS)
    package_match = app_package in RISKY_PACKAGES
    return keyword_match or package_match

def build_hitl_description(step: PlannedStep, node: NodeData) -> str:
    return (
        f"Aether is about to: {step.description}\n"
        f"Action: {step.actionType} on '{node.text or node.contentDescription}'\n"
        f"App: {step.appPackage or 'unknown'}\n\n"
        f"Do you want to allow this?"
    )
