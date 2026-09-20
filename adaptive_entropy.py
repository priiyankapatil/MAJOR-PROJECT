import json
import os
from datetime import datetime

FEEDBACK_LOG_PATH = "entropy_feedback_log.json"
INITIAL_THRESHOLD = 1.5
LEARNING_RATE = 0.05
MIN_THRESHOLD = 0.8
MAX_THRESHOLD = 2.5
MIN_FEEDBACK_COUNT = 3


def load_feedback_log(log_path: str = None):
    path = log_path or FEEDBACK_LOG_PATH
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass

    return {
        "current_threshold": INITIAL_THRESHOLD,
        "total_feedbacks": 0,
        "fast_path_feedback": [],
        "slow_path_feedback": [],
        "threshold_history": []
    }


def save_feedback_log(log: dict, log_path: str = None):
    path = log_path or FEEDBACK_LOG_PATH
    with open(path, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2)


def record_feedback(query, entropy, path_used, was_accurate: bool, log_path: str = None):
    target_path = log_path or FEEDBACK_LOG_PATH
    log = load_feedback_log(target_path)
    entry = {
        "query": query,
        "entropy": float(entropy),
        "was_accurate": bool(was_accurate),
        "timestamp": datetime.now().isoformat()
    }

    key = "fast_path_feedback" if path_used.lower().startswith("f") else "slow_path_feedback"
    log.setdefault(key, []).append(entry)
    log["total_feedbacks"] = int(log.get("total_feedbacks", 0)) + 1

    if log["total_feedbacks"] >= MIN_FEEDBACK_COUNT:
        adapt_threshold(log)

    save_feedback_log(log, target_path)
    print(f"🧠 Feedback recorded: path={path_used}, accurate={was_accurate}, entropy={entropy}")


def compute_path_accuracy(entries: list):
    if not entries:
        return None
    correct = sum(1 for e in entries if e.get("was_accurate") is True)
    return correct / len(entries)


def adapt_threshold(log: dict, min_path_observations: int = 3):
    fast = log.get("fast_path_feedback", [])
    slow = log.get("slow_path_feedback", [])

    fast_acc = compute_path_accuracy(fast)
    slow_acc = compute_path_accuracy(slow)

    old = float(log.get("current_threshold", INITIAL_THRESHOLD))
    new = old
    reason = "no_change"

    # Only adapt if we have sufficient observations for the path under evaluation
    if fast_acc is not None and len(fast) >= min_path_observations and fast_acc < 0.6:
        new = old - LEARNING_RATE
        reason = "fast_low_accuracy"
    elif (fast_acc is not None and len(fast) >= min_path_observations and fast_acc > 0.85 and
          slow_acc is not None and len(slow) >= min_path_observations and slow_acc > 0.85):
        new = old + (LEARNING_RATE / 2)
        reason = "both_high_accuracy"
    else:
        reason = "no_change"

    # Clamp
    new = max(MIN_THRESHOLD, min(MAX_THRESHOLD, new))

    log["current_threshold"] = round(new, 3)
    hist = {
        "timestamp": datetime.now().isoformat(),
        "old_threshold": round(old, 3),
        "new_threshold": round(new, 3),
        "fast_accuracy": round(fast_acc, 3) if fast_acc is not None else None,
        "slow_accuracy": round(slow_acc, 3) if slow_acc is not None else None,
        "reason": reason,
    }
    log.setdefault("threshold_history", []).append(hist)

    print(f"🔄 Threshold adapted: {old} → {new} (reason={reason}, fast_acc={fast_acc}, slow_acc={slow_acc})")
    return log


def get_current_threshold():
    log = load_feedback_log()
    th = log.get("current_threshold", INITIAL_THRESHOLD)
    total = int(log.get("total_feedbacks", 0))
    print(f"📊 Current entropy threshold: {th} (adapted from {total} feedbacks)")
    return float(th)


def get_threshold_report():
    log = load_feedback_log()
    current = log.get("current_threshold", INITIAL_THRESHOLD)
    total = log.get("total_feedbacks", 0)
    fast_count = len(log.get("fast_path_feedback", []))
    slow_count = len(log.get("slow_path_feedback", []))

    def acc_percent(entries):
        if not entries:
            return 0.0
        return 100.0 * sum(1 for e in entries if e.get("was_accurate")) / len(entries)

    fast_acc = acc_percent(log.get("fast_path_feedback", []))
    slow_acc = acc_percent(log.get("slow_path_feedback", []))
    history_len = len(log.get("threshold_history", []))

    print("╔══════════════════════════════════════╗")
    print("║   ADAPTIVE ENTROPY THRESHOLD REPORT  ║")
    print("╠══════════════════════════════════════╣")
    print(f"║ Current Threshold : {current}")
    print(f"║ Total Feedbacks   : {total}")
    print(f"║ Fast Path Entries : {fast_count}")
    print(f"║ Slow Path Entries : {slow_count}")
    print(f"║ Fast Accuracy     : {fast_acc:.2f}%")
    print(f"║ Slow Accuracy     : {slow_acc:.2f}%")
    print(f"║ Threshold History : {history_len} changes")
    print("╚══════════════════════════════════════╝")


if __name__ == "__main__":
    # Simulate feedback
    # 3 fast path wrong
    record_feedback("q1", 1.2, "FAST", False)
    record_feedback("q2", 1.1, "FAST", False)
    record_feedback("q3", 1.3, "FAST", False)
    # 2 slow path correct
    record_feedback("q4", 1.8, "SLOW", True)
    record_feedback("q5", 1.9, "SLOW", True)

    get_threshold_report()
