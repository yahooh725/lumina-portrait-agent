"""Single-image portrait agent prototype: Chinese request -> plan -> state -> pixels."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
RETOUCH = ROOT / "vendor/photo_retouch/retouch.py"
INITIAL = json.loads((ROOT / "configs/tool_registry.json").read_text(encoding="utf-8"))["initial_state"]
ALIASES = {
    "slim_face": ("瘦脸", "脸小", "脸型", "脸稍微收", "脸收"),
    "enlarge_eyes": ("大眼", "眼睛", "眼部"),
    "remove_acne": ("祛痘", "去痘", "痘痘", "斑点"),
    "smooth_skin": ("磨皮", "皮肤平滑", "皮肤磨", "磨一下皮"),
    "blur_bg": ("背景虚化", "虚化背景", "背景模糊", "模糊背景"),
    "whiten_teeth": ("牙齿美白", "美白牙齿", "牙齿白", "牙白"),
}
FLAGS = {"slim_face": "--slim-face", "enlarge_eyes": "--enlarge-eyes",
         "remove_acne": "--remove-acne", "smooth_skin": "--smooth-skin",
         "blur_bg": "--blur-bg", "whiten_teeth": "--whiten-teeth"}
DEFAULT_HALF_REQUESTS = {
    "脸小点": ("slim_face",),
    "帮我磨皮": ("smooth_skin",),
    "来点大眼": ("enlarge_eyes",),
    "皮肤好点": ("smooth_skin", "remove_acne"),
    "帮我祛痘": ("remove_acne",),
    "帮我虚化背景": ("blur_bg",),
    "帮我美白牙齿": ("whiten_teeth",),
}
UNSUPPORTED = ("头包脸", "皮肤美白", "全身美白", "小头", "瘦身", "换背景", "去皱", "美颜")
COLLECTIVE_DISABLE = re.compile(
    r"(?:其余|其他|其它|剩下|别的)(?:已经开的|目前开启的)?"
    r"(?:美颜|修图|效果|项目|工具)?(?:都|全部)?(?:先)?"
    r"(?:停用|关闭|关掉|取消|去掉|停掉|关了)"
)
ALL_ACNE_REQUEST = re.compile(
    r"(?:痘痘|祛痘|去痘).{0,10}(?:全部|全都|所有|一个不留)"
    r"|(?:全部|所有|全都).{0,10}(?:痘痘|祛痘|去痘)"
    r"|痘痘.{0,4}都去掉"
)


def requests_collective_disable(instruction: str) -> bool:
    for clause in re.split(r"[，,。；;！？!?]", instruction):
        match = COLLECTIVE_DISABLE.search(clause)
        if match and not re.search(r"(?:不想|不希望|不要|别|不必|不是要).{0,8}$",
                                   clause[:match.start()]):
            return True
    return False


def mentions_tool(text: str, tool: str) -> bool:
    if tool in DEFAULT_HALF_REQUESTS.get(text.strip().rstrip("。！？!?"), ()):
        return True
    if any(alias in text for alias in ALIASES[tool]):
        return True
    if tool in ("smooth_skin", "remove_acne") and re.search(r"皮肤.{0,6}(?:好|弄好)", text):
        return True
    if tool == "remove_acne" and re.search(r"祛.{0,4}痘", text):
        return True
    if tool == "blur_bg":
        return bool(re.search(r"背景.{0,8}(?:虚化|模糊)|(?:虚化|模糊).{0,8}背景", text))
    if tool == "whiten_teeth":
        return bool(re.search(r"牙齿?.{0,8}(?:美白|变白|白一点)|美白.{0,5}牙齿?", text))
    if tool == "slim_face":
        return bool(re.search(r"(?:脸|面部).{0,8}瘦|瘦.{0,8}(?:脸|面部)|脸.{0,2}收|脸修小|脸部收窄", text))
    return False


def save(path: Path, value: object) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run_tool(args: list[str], timeout: int = 180) -> tuple[dict, dict]:
    started = time.monotonic()
    env = dict(os.environ)
    env.setdefault("PHOTO_RETOUCH_MODEL_DIR", str(ROOT / "tool_acceptance/models"))
    env.setdefault("MPLCONFIGDIR", str(ROOT / "tool_acceptance/.cache/matplotlib"))
    proc = subprocess.run([sys.executable, str(RETOUCH), *args], capture_output=True,
                          text=True, timeout=timeout, env=env, check=False)
    meta = {"argv": [sys.executable, str(RETOUCH), *args],
            "exit_code": proc.returncode, "seconds": round(time.monotonic() - started, 2),
            "stderr": proc.stderr[-4000:]}
    if proc.returncode:
        raise RuntimeError(f"retouch exit {proc.returncode}: {proc.stderr[-1000:]}")
    return json.loads(proc.stdout), meta


def clarify(question: str) -> dict:
    return {"action": "clarify", "updates": [], "question": question}


def is_relative_feedback(text: str) -> bool:
    return bool(re.search(r"更明显|加重|重一点|加一点|调高|提高|加强|轻一点|淡一点|减轻|调淡|调轻|减弱|降低|减一点|减小|低一点|往下|退一|松一点|收回|缩回|小回", text))


def confirm_zero_strength(tool: str) -> dict:
    label = {"slim_face": "瘦脸", "enlarge_eyes": "大眼", "smooth_skin": "磨皮",
             "blur_bg": "背景虚化", "whiten_teeth": "牙齿美白"}[tool]
    return clarify(f"再降低一点会让{label}强度变成 0，效果将消失。要关闭{label}吗？")


def rule_plan(instruction: str, state: dict, dialogue: list[dict] | None = None) -> dict:
    """Conservative preview planner; unsupported/unclear requests ask a question."""
    text = instruction.strip()
    if not text:
        return clarify("请说出要调整的项目和程度。")
    if dialogue and dialogue[-1].get("role") == "assistant":
        previous = dialogue[-1].get("content", "")
        for tool in ("slim_face", "enlarge_eyes", "smooth_skin", "blur_bg", "whiten_teeth"):
            if previous == confirm_zero_strength(tool)["question"] and state[tool] == {"enabled": True, "strength": 0.1}:
                if re.fullmatch(r"(?:是|是的|确认|确定|好|好的|可以|执行|关闭|关掉|关了|关吧|关闭吧|关掉吧)[。！! ]*", text):
                    return {"action": "update", "updates": [{"tool": tool, "enabled": False, "strength": 0.0}], "question": None}
                label = {"slim_face": "瘦脸", "enlarge_eyes": "大眼", "smooth_skin": "磨皮",
                         "blur_bg": "背景虚化", "whiten_teeth": "牙齿美白"}[tool]
                if re.fullmatch(rf"(?:(?:不|不用|不要|先不|不关|别关|取消|保持|算了)|{label}先保留)[。！! ]*", text):
                    return clarify("好的，保持当前设置。还想调整哪一项？")
    if text.rstrip("。！？!?") in DEFAULT_HALF_REQUESTS:
        updates = []
        for tool in DEFAULT_HALF_REQUESTS[text.rstrip("。！？!?")]:
            parameter = "sensitivity" if tool == "remove_acne" else "strength"
            updates.append({"tool": tool, "enabled": True, parameter: 0.5})
        return {"action": "update", "updates": updates, "question": None}
    if requests_collective_disable(text):
        keep = set()
        for clause in re.split(r"[，,。；;！？!?]", text):
            if COLLECTIVE_DISABLE.search(clause):
                continue
            if any(word in clause for word in ("留着", "保留", "只留", "保持", "维持", "不变", "别动")):
                keep.update(tool for tool in FLAGS if mentions_tool(clause, tool))
        if not keep:
            return clarify("你想保留哪一项效果，并关闭哪些其他效果？")
        updates = []
        for tool in FLAGS:
            if tool not in keep and state[tool]["enabled"]:
                key = "sensitivity" if tool == "remove_acne" else "strength"
                updates.append({"tool": tool, "enabled": False,
                                key: state[tool][key] if tool == "remove_acne" else 0.0})
        if not updates:
            return clarify("其他效果当前已经关闭；你还想调整哪一项？")
        return {"action": "update", "updates": updates, "question": None}
    if any(word in text for word in UNSUPPORTED):
        return clarify("目前支持瘦脸、大眼、祛痘、磨皮、背景虚化和牙齿美白。你想调整哪一项？")
    if any(word in text for word in ("撤销上一步", "撤回上一步", "退回上一步", "撤销刚才", "撤回刚才")):
        return {"action": "undo", "updates": [], "question": None}
    if any(word in text for word in ("恢复原图", "全部重置", "全部恢复", "重置全部", "回到最初")):
        return {"action": "reset", "updates": [], "question": None}
    if any(word in text for word in ("这个去掉", "那个去掉", "它去掉")):
        return clarify("你想关闭哪一项修图效果？")

    clauses = re.split(r"[，,。；;、]|但是|但|并且|以及", text)
    updates = {}
    mentioned = set()
    for clause in clauses:
        found = [tool for tool in ALIASES if mentions_tool(clause, tool)]
        if len(found) > 1:
            if any(word in clause for word in ("都开启", "同时开启", "一起开启", "和", "与")):
                for grouped_tool in found:
                    grouped_key = "sensitivity" if grouped_tool == "remove_acne" else "strength"
                    updates[grouped_tool] = {"tool": grouped_tool, "enabled": True, grouped_key: 0.5}
                mentioned.update(found)
                continue
            return clarify("请把每项调整分开描述，并说明各自程度。")
        if not found:
            continue
        tool = found[0]
        mentioned.add(tool)
        if any(w in clause for w in ("别动", "不动", "保持", "不变", "保留", "维持", "照旧", "别关", "不关", "不用关")) or ("继续" in clause and any(w in clause for w in ("现在", "原来", "原本"))):
            continue
        key = "sensitivity" if tool == "remove_acne" else "strength"
        if (tool == "remove_acne" and ALL_ACNE_REQUEST.search(clause)
                and not any(w in clause for w in
                            ("不要", "别", "关闭", "关掉", "停用", "取消", "暂停"))):
            updates[tool] = {"tool": tool, "enabled": True, "sensitivity": 1.0}
            continue
        if any(w in clause for w in ("不要", "别", "关闭", "关掉", "关了", "去掉", "撤销", "取消", "不磨", "不瘦")):
            value = state[tool][key] if tool == "remove_acne" else 0.0
            updates[tool] = {"tool": tool, "enabled": False, key: value}
            continue
        number = re.search(r"(?:强度|敏感度|到|为)?\s*(0(?:\.\d+)?|1(?:\.0+)?)", clause)
        if number:
            level = float(number.group(1))
        elif "只" in clause and state[tool]["enabled"]:
            level = state[tool][key]
        elif "更明显" in clause:
            level = 0.7
        elif (state[tool]["enabled"] and
              not any(w in clause for w in ("轻微", "常规", "正常", "明显档", "明显强度", "最高")) and
              re.search(r"加重|加一点|调高|提高|更明显|加强|轻一点|淡一点|调淡|调轻|减轻|减一点|低一点|弱一点|收敛|减弱|降低|往下|退一|松一点|收回|缩回|小回|放宽一点|纹理多留", clause)):
            delta = 0.1 if re.search(r"加重|加一点|调高|提高|更明显|加强", clause) else -0.1
            level = round(min(1.0, max(0.0, state[tool][key] + delta)), 3)
            if tool != "remove_acne" and delta < 0 and level == 0:
                return confirm_zero_strength(tool)
        elif tool == "remove_acne" and ALL_ACNE_REQUEST.search(clause):
            level = 1.0
        elif tool == "remove_acne" and "最高" in clause:
            level = 1.0
        elif tool == "smooth_skin" and "明显" in clause:
            level = 0.7
        elif any(w in clause for w in ("明显", "强一点", "多一点")):
            level = 0.5
        elif tool == "slim_face" and "突出一点" in clause:
            level = 0.7
        elif any(w in clause for w in ("常规", "正常")):
            level = 0.5 if tool in ("remove_acne", "smooth_skin") else 0.3
        elif re.search(r"轻微|稍微|轻一点|做轻|调轻", clause):
            level = 0.2 if tool != "remove_acne" else 0.3
        else:
            level = 0.5
        updates[tool] = {"tool": tool, "enabled": True, key: level}
    if not updates and not mentioned and is_relative_feedback(text):
        active = [tool for tool, item in state.items() if item["enabled"]]
        if len(active) != 1:
            return clarify("你想调整哪一项效果？")
        tool = active[0]
        key = "sensitivity" if tool == "remove_acne" else "strength"
        current = state[tool][key]
        if "更明显" in text:
            level = 0.7
        elif re.search(r"加重|重一点|加一点|调高|提高|加强", text):
            level = round(min(1.0, current + 0.1), 3)
        else:
            level = round(max(0.0, current - 0.1), 3)
            if tool != "remove_acne" and level == 0:
                return confirm_zero_strength(tool)
        updates[tool] = {"tool": tool, "enabled": True, key: level}
    if "只" in text and updates:
        enabled = {tool for tool, u in updates.items() if u["enabled"]}
        if len(enabled) != 1:
            return clarify("“只”指保留哪一项效果？")
        for tool in FLAGS:
            if tool not in enabled and state[tool]["enabled"]:
                key = "sensitivity" if tool == "remove_acne" else "strength"
                updates[tool] = {"tool": tool, "enabled": False,
                                 key: state[tool][key] if tool == "remove_acne" else 0.0}
    if not updates:
        return clarify("请明确要调整哪一项：瘦脸、大眼、祛痘、磨皮、背景虚化或牙齿美白？")
    return {"action": "update", "updates": list(updates.values()), "question": None}


def validate(plan: object, instruction: str, source: str, state: dict | None = None) -> None:
    """Check the plan protocol and executable parameter bounds only.

    The instruction/source/state arguments are retained for existing callers. Whether
    the plan matches the user's Chinese request is a model-quality question evaluated
    against reviewed labels, not a keyword-based runtime rejection rule.
    """
    if not isinstance(plan, dict) or set(plan) != {"action", "updates", "question"}:
        raise ValueError("plan must contain only action, updates, question")
    action, updates, question = plan["action"], plan["updates"], plan["question"]
    if action not in ("update", "undo", "reset", "clarify") or not isinstance(updates, list):
        raise ValueError("invalid action or updates")
    if action == "clarify":
        if updates or not isinstance(question, str) or not question.strip():
            raise ValueError("clarify needs a question and no updates")
        return
    if question is not None or (action == "update" and not updates) or (action != "update" and updates):
        raise ValueError("invalid action/updates/question combination")
    if len(updates) > len(FLAGS):
        raise ValueError("too many updates")
    seen = set()
    for item in updates:
        if not isinstance(item, dict):
            raise ValueError("update must be an object")
        tool = item.get("tool")
        if tool not in FLAGS or tool in seen:
            raise ValueError("unknown or repeated tool")
        seen.add(tool)
        key = "sensitivity" if tool == "remove_acne" else "strength"
        if set(item) != {"tool", "enabled", key} or type(item["enabled"]) is not bool:
            raise ValueError(f"invalid fields for {tool}")
        value = item[key]
        if type(value) not in (int, float) or not 0 <= value <= 1:
            raise ValueError(f"invalid {key} for {tool}")
        if not item["enabled"] and tool != "remove_acne" and value != 0:
            raise ValueError("disabled strength must be zero")


def session_dir(name: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", name):
        raise ValueError("session name: letters, digits, underscores, dashes only")
    return RUNS / name


def init(name: str, image: Path) -> dict:
    source = image.resolve()
    if not source.is_file() or source.suffix.lower() not in (".png", ".jpg", ".jpeg"):
        raise ValueError("image must be an existing PNG or JPEG")
    folder = session_dir(name)
    if folder.exists():
        raise ValueError("session already exists")
    folder.mkdir(parents=True)
    original = folder / ("original" + source.suffix.lower())
    shutil.copy2(source, original)
    try:
        analysis, command = run_tool(["analyze", str(original), "--overlay", str(folder / "analysis.png")], 900)
        people = analysis.get("people", [])
        if len(people) != 1 or not people[0].get("has_face"):
            raise ValueError("expected exactly one person with a face")
        record = {"version": 1, "source": str(source), "original": str(original),
                 "original_sha256": digest(original), "state": copy.deepcopy(INITIAL),
                 "history": [], "dialogue": [], "turn": 0, "current_image": str(original),
                  "analysis": analysis, "analysis_command": command}
        save(folder / "session.json", record)
        return {"session": name, "image": str(original), "detected_people": 1,
                "detected_faces": 1, "state": record["state"]}
    except Exception:
        shutil.rmtree(folder)
        raise


def chat(name: str, instruction: str, plan_file: Path | None, planner: str = "rules",
         model_path: Path | None = None, request_id: str | None = None) -> dict:
    folder = session_dir(name)
    path = folder / "session.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    for old_state in [record["state"], *record["history"]]:
        for tool in ("blur_bg", "whiten_teeth"):
            old_state.setdefault(tool, copy.deepcopy(INITIAL[tool]))
    original = Path(record["original"])
    if digest(original) != record["original_sha256"]:
        raise RuntimeError("original image changed")
    source = "external" if plan_file else planner
    model_meta = None
    if plan_file:
        plan = json.loads(plan_file.read_text(encoding="utf-8"))
    elif planner == "qwen":
        from tool_expand_v1.qwen_planner_v2 import generate_plan
        plan, model_meta = generate_plan(original, instruction, record["state"], model_path,
                                         record.get("dialogue", [])[-8:])
    else:
        plan = rule_plan(instruction, record["state"], record.get("dialogue", []))
    turn = record["turn"] + 1
    audit = {"turn": turn, "instruction": instruction, "planner": source, "raw_plan": plan,
             "previous_state": record["state"]}
    if request_id is not None:
        audit['request_id'] = request_id
    if model_meta is not None:
        audit["model"] = model_meta
    try:
        validate(plan, instruction, "rules" if source == "rules" else "external", record["state"])
        action = plan["action"]
        if action == "update" and any(u["tool"] == "whiten_teeth" and u["enabled"] for u in plan["updates"]):
            teeth = record.get("analysis", {}).get("people", [{}])[0].get("teeth_target")
            # Old sessions cached target checks before the obstruction screen existed.
            if teeth is None or "suspected_occlusion" not in teeth:
                analysis, _ = run_tool(["analyze", str(original), "--overlay", str(folder / "analysis.png")], 900)
                record["analysis"] = analysis
                teeth = analysis["people"][0].get("teeth_target", {})
            if not teeth.get("eligible", False):
                audit["skipped"] = [{"tool": "whiten_teeth", "reason": "visible teeth not confirmed", "check": teeth}]
                plan = copy.deepcopy(plan)
                plan["updates"] = [u for u in plan["updates"] if u["tool"] != "whiten_teeth"]
                audit["effective_plan"] = plan
                if not plan["updates"]:
                    plan = clarify("未确认到可处理的露齿区域，牙齿美白未执行。请换一张露齿照片，或继续调整其他项目。")
                    action = "clarify"
        if action == "clarify":
            audit["status"] = "clarify"
            audit["question"] = plan["question"]
            result = {"status": "clarify", "question": plan["question"], "state": record["state"]}
        else:
            state = copy.deepcopy(record["state"])
            history = copy.deepcopy(record["history"])
            if action == "undo":
                if not history:
                    raise ValueError("nothing to undo")
                state = history.pop()
            else:
                if action == "reset":
                    state = copy.deepcopy(INITIAL)
                else:
                    for update in plan["updates"]:
                        tool = update["tool"]
                        state[tool] = {k: v for k, v in update.items() if k != "tool"}
                history.append(copy.deepcopy(record["state"]))
            active = [(tool, item) for tool, item in state.items() if item["enabled"]]
            # A no-effect turn copies the original bytes, so preserve its media type.
            suffix = ".png" if active else original.suffix.lower()
            target = folder / f"turn_{turn:03d}{suffix}"
            if active:
                args = ["edit", str(original), "-o", str(target), "--person", "0"]
                for tool, item in active:
                    args += [FLAGS[tool], str(item["sensitivity" if tool == "remove_acne" else "strength"])]
                tool_result, command = run_tool(args)
                if tool_result.get("people_detected") != 1 or tool_result.get("targets") != [0]:
                    raise RuntimeError("render target mismatch")
                audit["render"] = tool_result
                audit["command"] = command
                if tool_result.get("skipped"):
                    audit.setdefault("skipped", []).extend(tool_result["skipped"])
                    for item in tool_result["skipped"]:
                        if item["tool"] == "whiten_teeth":
                            state["whiten_teeth"] = copy.deepcopy(INITIAL["whiten_teeth"])
            else:
                shutil.copyfile(original, target)
                audit["render"] = {"applied": [], "warnings": [], "copied_original": True}
            record.update(state=state, history=history, current_image=str(target), turn=turn)
            audit.update(status="rendered", state=state, output=str(target),
                         output_sha256=digest(target))
            result = {"status": "rendered", "image": str(target), "state": state,
                      "applied": audit["render"].get("applied", []),
                      "warnings": audit["render"].get("warnings", []),
                      "skipped": audit.get("skipped", [])}
    except Exception as exc:
        audit.update(status="rejected", reason=str(exc))
        result = {"status": "rejected", "reason": str(exc), "state": record["state"]}
    if result["status"] == "rendered":
        if isinstance(plan, dict) and plan.get("action") == "undo":
            assistant_message = "已撤销上一步。"
        elif isinstance(plan, dict) and plan.get("action") == "reset":
            assistant_message = "已回到最初设置。"
        else:
            assistant_message = "已完成本轮调整。" + ("牙齿美白未执行：未确认到露齿区域。" if audit.get("skipped") else "")
    elif result["status"] == "clarify":
        assistant_message = result["question"]
    else:
        assistant_message = "本轮未执行：" + result["reason"]
    record["dialogue"] = (record.get("dialogue", []) + [
        {"role": "user", "content": instruction},
        {"role": "assistant", "content": assistant_message},
    ])[-8:]
    record["turn"] = turn
    save(folder / f"turn_{turn:03d}.json", audit)
    save(path, record)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    a = sub.add_parser("init")
    a.add_argument("session")
    a.add_argument("image", type=Path)
    a = sub.add_parser("chat")
    a.add_argument("session")
    a.add_argument("instruction")
    a.add_argument("--plan-json", type=Path, help="external planner JSON; validated before execution")
    a.add_argument("--planner", choices=("rules", "qwen"), default="rules")
    a.add_argument("--model-path", type=Path, help="local Qwen3-VL directory")
    a = sub.add_parser("show")
    a.add_argument("session")
    args = parser.parse_args()
    if args.command == "init":
        result = init(args.session, args.image)
    elif args.command == "chat":
        result = chat(args.session, args.instruction, args.plan_json, args.planner, args.model_path)
    else:
        result = json.loads((session_dir(args.session) / "session.json").read_text(encoding="utf-8"))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result.get("status") == "rejected":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
