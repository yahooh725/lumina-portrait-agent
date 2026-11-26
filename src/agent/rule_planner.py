"""Lumina 内置中文规则规划器（无 GPU 环境下可直接驱动原生六工具引擎）。"""
from __future__ import annotations

import re

TOOL_NAMES = (
    'slim_face',
    'enlarge_eyes',
    'remove_acne',
    'smooth_skin',
    'blur_bg',
    'whiten_teeth',
)

ALIASES = {
    'slim_face': ('瘦脸', '脸小', '脸型', '脸稍微收', '脸收'),
    'enlarge_eyes': ('大眼', '眼睛', '眼部'),
    'remove_acne': ('祛痘', '去痘', '痘痘', '斑点'),
    'smooth_skin': ('磨皮', '皮肤平滑', '皮肤磨', '磨一下皮'),
    'blur_bg': ('背景虚化', '虚化背景', '背景模糊', '模糊背景'),
    'whiten_teeth': ('牙齿美白', '美白牙齿', '牙齿白', '牙白'),
}

DEFAULT_HALF_REQUESTS = {
    '脸小点': ('slim_face',),
    '帮我磨皮': ('smooth_skin',),
    '来点大眼': ('enlarge_eyes',),
    '皮肤好点': ('smooth_skin', 'remove_acne'),
    '帮我祛痘': ('remove_acne',),
    '帮我虚化背景': ('blur_bg',),
    '帮我美白牙齿': ('whiten_teeth',),
}

UNSUPPORTED = ('头包脸', '皮肤美白', '全身美白', '小头', '瘦身', '换背景', '去皱', '美颜')

COLLECTIVE_DISABLE = re.compile(
    r'(?:其余|其他|其它|剩下|别的)(?:已经开的|目前开启的)?'
    r'(?:美颜|修图|效果|项目|工具)?(?:都|全部)?(?:先)?'
    r'(?:停用|关闭|关掉|取消|去掉|停掉|关了)'
)

ALL_ACNE_REQUEST = re.compile(
    r'(?:痘痘|祛痘|去痘).{0,10}(?:全部|全都|所有|一个不留)'
    r'|(?:全部|所有|全都).{0,10}(?:痘痘|祛痘|去痘)'
    r'|痘痘.{0,4}都去掉'
)

LABELS = {
    'slim_face': '瘦脸',
    'enlarge_eyes': '大眼',
    'smooth_skin': '磨皮',
    'blur_bg': '背景虚化',
    'whiten_teeth': '牙齿美白',
}


def _clarify(question: str) -> dict:
    return {'action': 'clarify', 'updates': [], 'question': question}


def _confirm_zero_strength(tool: str) -> dict:
    label = LABELS[tool]
    return _clarify(f'再降低一点会让{label}强度变成 0，效果将消失。要关闭{label}吗？')


def _requests_collective_disable(instruction: str) -> bool:
    for clause in re.split(r'[，,。；;！？!?]', instruction):
        match = COLLECTIVE_DISABLE.search(clause)
        if match and not re.search(r'(?:不想|不希望|不要|别|不必|不是要).{0,8}$', clause[:match.start()]):
            return True
    return False


def _mentions_tool(text: str, tool: str) -> bool:
    if tool in DEFAULT_HALF_REQUESTS.get(text.strip().rstrip('。！？!?'), ()):
        return True
    if any(alias in text for alias in ALIASES[tool]):
        return True
    if tool in ('smooth_skin', 'remove_acne') and re.search(r'皮肤.{0,6}(?:好|弄好)', text):
        return True
    if tool == 'remove_acne' and re.search(r'祛.{0,4}痘', text):
        return True
    if tool == 'blur_bg':
        return bool(re.search(r'背景.{0,8}(?:虚化|模糊)|(?:虚化|模糊).{0,8}背景', text))
    if tool == 'whiten_teeth':
        return bool(re.search(r'牙齿?.{0,8}(?:美白|变白|白一点)|美白.{0,5}牙齿?', text))
    if tool == 'slim_face':
        return bool(re.search(r'(?:脸|面部).{0,8}瘦|瘦.{0,8}(?:脸|面部)|脸.{0,2}收|脸修小|脸部收窄', text))
    return False


def _is_relative_feedback(text: str) -> bool:
    return bool(re.search(
        r'更明显|加重|重一点|加一点|调高|提高|加强|轻一点|淡一点|减轻|调淡|调轻|减弱|降低|减一点|减小|低一点|往下|退一|松一点|收回|缩回|小回',
        text,
    ))


def rule_plan(instruction: str, state: dict, dialogue: list[dict] | None = None) -> dict:
    """将中文自然语言修图指令解析为结构化操作计划（保守策略：不确定时发起澄清）。"""
    text = instruction.strip()
    if not text:
        return _clarify('请说出要调整的项目和程度。')

    if dialogue and dialogue[-1].get('role') == 'assistant':
        previous = dialogue[-1].get('content', '')
        for tool in LABELS:
            if previous == _confirm_zero_strength(tool)['question'] and state[tool] == {'enabled': True, 'strength': 0.1}:
                if re.fullmatch(r'(?:是|是的|确认|确定|好|好的|可以|执行|关闭|关掉|关了|关吧|关闭吧|关掉吧)[。！! ]*', text):
                    return {'action': 'update', 'updates': [{'tool': tool, 'enabled': False, 'strength': 0.0}], 'question': None}
                label = LABELS[tool]
                if re.fullmatch(rf'(?:(?:不|不用|不要|先不|不关|别关|取消|保持|算了)|{label}先保留)[。！! ]*', text):
                    return _clarify('好的，保持当前设置。还想调整哪一项？')

    stripped = text.rstrip('。！？!?')
    if stripped in DEFAULT_HALF_REQUESTS:
        updates = []
        for tool in DEFAULT_HALF_REQUESTS[stripped]:
            param = 'sensitivity' if tool == 'remove_acne' else 'strength'
            updates.append({'tool': tool, 'enabled': True, param: 0.5})
        return {'action': 'update', 'updates': updates, 'question': None}

    if _requests_collective_disable(text):
        keep = set()
        for clause in re.split(r'[，,。；;！？!?]', text):
            if COLLECTIVE_DISABLE.search(clause):
                continue
            if any(word in clause for word in ('留着', '保留', '只留', '保持', '维持', '不变', '别动')):
                keep.update(tool for tool in TOOL_NAMES if _mentions_tool(clause, tool))
        if not keep:
            return _clarify('你想保留哪一项效果，并关闭哪些其他效果？')
        updates = []
        for tool in TOOL_NAMES:
            if tool not in keep and state[tool]['enabled']:
                key = 'sensitivity' if tool == 'remove_acne' else 'strength'
                updates.append({
                    'tool': tool,
                    'enabled': False,
                    key: state[tool][key] if tool == 'remove_acne' else 0.0,
                })
        if not updates:
            return _clarify('其他效果当前已经关闭；你还想调整哪一项？')
        return {'action': 'update', 'updates': updates, 'question': None}

    if any(word in text for word in UNSUPPORTED):
        return _clarify('目前支持瘦脸、大眼、祛痘、磨皮、背景虚化和牙齿美白。你想调整哪一项？')
    if any(word in text for word in ('撤销上一步', '撤回上一步', '退回上一步', '撤销刚才', '撤回刚才', '撤销上一轮', '撤回上一轮')):
        return {'action': 'undo', 'updates': [], 'question': None}
    if any(word in text for word in ('恢复原图', '全部重置', '全部恢复', '重置全部', '回到最初')):
        return {'action': 'reset', 'updates': [], 'question': None}
    if any(word in text for word in ('这个去掉', '那个去掉', '它去掉')):
        return _clarify('你想关闭哪一项修图效果？')

    clauses = re.split(r'[，,。；;、]|但是|但|并且|以及', text)
    updates: dict[str, dict] = {}
    mentioned: set[str] = set()
    for clause in clauses:
        found = [tool for tool in ALIASES if _mentions_tool(clause, tool)]
        if len(found) > 1:
            if any(word in clause for word in ('都开启', '同时开启', '一起开启', '和', '与')):
                for grouped_tool in found:
                    grouped_key = 'sensitivity' if grouped_tool == 'remove_acne' else 'strength'
                    updates[grouped_tool] = {'tool': grouped_tool, 'enabled': True, grouped_key: 0.5}
                mentioned.update(found)
                continue
            return _clarify('请把每项调整分开描述，并说明各自程度。')
        if not found:
            continue
        tool = found[0]
        mentioned.add(tool)
        if any(w in clause for w in ('别动', '不动', '保持', '不变', '保留', '维持', '照旧', '别关', '不关', '不用关')) or (
            '继续' in clause and any(w in clause for w in ('现在', '原来', '原本'))
        ):
            continue
        key = 'sensitivity' if tool == 'remove_acne' else 'strength'
        if (
            tool == 'remove_acne'
            and ALL_ACNE_REQUEST.search(clause)
            and not any(w in clause for w in ('不要', '别', '关闭', '关掉', '停用', '取消', '暂停'))
        ):
            updates[tool] = {'tool': tool, 'enabled': True, 'sensitivity': 1.0}
            continue
        if any(w in clause for w in ('不要', '别', '关闭', '关掉', '关了', '去掉', '撤销', '取消', '不磨', '不瘦')):
            value = state[tool][key] if tool == 'remove_acne' else 0.0
            updates[tool] = {'tool': tool, 'enabled': False, key: value}
            continue

        number = re.search(r'(?:强度|敏感度|到|为)?\s*(0(?:\.\d+)?|1(?:\.0+)?)', clause)
        if number:
            level = float(number.group(1))
        elif '只' in clause and state[tool]['enabled']:
            level = state[tool][key]
        elif '更明显' in clause:
            level = 0.7
        elif (
            state[tool]['enabled']
            and not any(w in clause for w in ('轻微', '常规', '正常', '明显档', '明显强度', '最高'))
            and re.search(r'加重|加一点|调高|提高|更明显|加强|轻一点|淡一点|调淡|调轻|减轻|减一点|低一点|弱一点|收敛|减弱|降低|往下|退一|松一点|收回|缩回|小回|放宽一点|纹理多留', clause)
        ):
            delta = 0.1 if re.search(r'加重|加一点|调高|提高|更明显|加强', clause) else -0.1
            level = round(min(1.0, max(0.0, state[tool][key] + delta)), 3)
            if tool != 'remove_acne' and delta < 0 and level == 0:
                return _confirm_zero_strength(tool)
        elif tool == 'remove_acne' and (ALL_ACNE_REQUEST.search(clause) or '最高' in clause):
            level = 1.0
        elif tool == 'smooth_skin' and '明显' in clause:
            level = 0.7
        elif any(w in clause for w in ('明显', '强一点', '多一点')):
            level = 0.5
        elif tool == 'slim_face' and '突出一点' in clause:
            level = 0.7
        elif any(w in clause for w in ('常规', '正常')):
            level = 0.5 if tool in ('remove_acne', 'smooth_skin') else 0.3
        elif re.search(r'轻微|稍微|轻一点|做轻|调轻', clause):
            level = 0.2 if tool != 'remove_acne' else 0.3
        else:
            level = 0.5
        updates[tool] = {'tool': tool, 'enabled': True, key: level}

    if not updates and not mentioned and _is_relative_feedback(text):
        active = [tool for tool, item in state.items() if item['enabled']]
        if len(active) != 1:
            return _clarify('你想调整哪一项效果？')
        tool = active[0]
        key = 'sensitivity' if tool == 'remove_acne' else 'strength'
        current = state[tool][key]
        if '更明显' in text:
            level = 0.7
        elif re.search(r'加重|重一点|加一点|调高|提高|加强', text):
            level = round(min(1.0, current + 0.1), 3)
        else:
            level = round(max(0.0, current - 0.1), 3)
            if tool != 'remove_acne' and level == 0:
                return _confirm_zero_strength(tool)
        updates[tool] = {'tool': tool, 'enabled': True, key: level}

    if '只' in text and updates:
        enabled = {tool for tool, u in updates.items() if u['enabled']}
        if len(enabled) != 1:
            return _clarify('“只”指保留哪一项效果？')
        for tool in TOOL_NAMES:
            if tool not in enabled and state[tool]['enabled']:
                key = 'sensitivity' if tool == 'remove_acne' else 'strength'
                updates[tool] = {
                    'tool': tool,
                    'enabled': False,
                    key: state[tool][key] if tool == 'remove_acne' else 0.0,
                }

    if not updates:
        return _clarify('请明确要调整哪一项：瘦脸、大眼、祛痘、磨皮、背景虚化或牙齿美白？')
    return {'action': 'update', 'updates': list(updates.values()), 'question': None}
