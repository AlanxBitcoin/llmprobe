from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

from transformers import AutoTokenizer

ATTR_FILE = Path(r"C:\AI_Model\probe\data\attributes.txt")
OUT_FILE = Path(r"C:\AI_Model\probe\data\cache\attribute_groups.en_tmp.json")
MODEL_PATH = "C:/AI_Model/Llama3_8B_Instruct"


def _clean_attr(text: str) -> str:
    t = str(text).strip()
    t = t.replace("\u3000", " ").strip()
    t = re.sub(r"（.*?）", "", t)
    t = t.replace(" ", "")
    return t.strip("：:、,，")


def _split_items(text: str) -> list[str]:
    parts = re.split(r"[、,，/]", text)
    out: list[str] = []
    for p in parts:
        c = _clean_attr(p)
        if c:
            out.append(c)
    return out


def parse_attributes_in_order(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    attrs: list[str] = []
    seen: set[str] = set()
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        if line.startswith("目标：") or line.startswith("属性来源：") or line.startswith("建议属性加类型："):
            continue
        if "不是属性，是导引" in line:
            continue

        left = ""
        right = ""
        if "：" in line:
            left, right = line.split("：", 1)
        elif ":" in line:
            left, right = line.split(":", 1)
        else:
            left = line

        left_c = _clean_attr(left)
        if left_c:
            if left_c not in seen:
                attrs.append(left_c)
                seen.add(left_c)

        for item in _split_items(right):
            if item not in seen:
                attrs.append(item)
                seen.add(item)
    return attrs


BASE_BANKS: dict[str, list[str]] = {
    "generic": [
        "time","space","value","state","level","class","group","type","form","mode",
        "point","line","field","area","case","name","word","text","image","sound",
        "light","dark","warm","cold","hard","soft","high","low","fast","slow",
        "open","close","start","end","first","last","early","late","near","far",
        "left","right","up","down","front","back","inner","outer","local","global",
        "daily","total","basic","major","minor","simple","complex","public","private","direct",
        "human","animal","plant","metal","wood","stone","water","fire","earth","air",
        "color","shape","size","mass","speed","force","power","energy","money","trade",
    ],
    "noun": [
        "person","child","adult","man","woman","student","teacher","doctor","worker","friend",
        "family","father","mother","son","daughter","brother","sister","city","town","village",
        "road","street","bridge","house","room","door","window","table","chair","book",
        "paper","phone","screen","music","movie","story","poem","river","lake","ocean",
        "mountain","forest","field","garden","flower","tree","grass","bird","fish","horse",
        "cat","dog","apple","bread","water","coffee","sugar","salt","money","bank",
    ],
    "verb": [
        "run","walk","sit","stand","jump","swim","drive","ride","fly","move",
        "eat","drink","cook","wash","clean","cut","draw","paint","write","read",
        "speak","listen","watch","think","learn","teach","work","build","make","create",
        "open","close","push","pull","hold","drop","throw","catch","carry","lift",
        "buy","sell","pay","save","send","call","meet","help","share","change",
    ],
    "adj": [
        "big","small","long","short","high","low","wide","narrow","thick","thin",
        "hot","cold","warm","cool","bright","dark","clean","dirty","new","old",
        "young","adult","strong","weak","hard","soft","sweet","bitter","sour","salty",
        "happy","sad","angry","calm","fast","slow","rich","poor","safe","risky",
    ],
    "number": [
        "zero","one","two","three","four","five","six","seven","eight","nine",
        "ten","eleven","twelve","twenty","thirty","forty","fifty","sixty","seventy","eighty",
        "ninety","hundred","thousand","million","billion","first","second","third","fourth","fifth",
        "single","double","triple","half","whole","total","count","sum","index","rank",
    ],
    "unit": [
        "meter","inch","foot","yard","mile","gram","pound","ton","liter","gallon",
        "second","minute","hour","day","week","month","year","volt","watt","byte",
        "pixel","ratio","score","grade","point","percent","degree","newton","pascal","bar",
    ],
    "phonetics": [
        "sound","voice","tone","pitch","stress","accent","rhyme","syllable","phoneme","vowel",
        "consonant","nasal","plosive","fricative","diphthong","intonation","speech","pronounce","oral","audio",
    ],
    "spelling": [
        "spell","letter","word","text","script","alphabet","prefix","suffix","root","morpheme",
        "capital","lower","hyphen","apostrophe","punctuation","comma","period","quote","typo","orthography",
    ],
    "language": [
        "english","latin","french","german","spanish","chinese","japanese","korean","arabic","hindi",
        "russian","italian","greek","portuguese","vietnamese","thai","turkish","dutch","swedish","polish",
    ],
    "emotion_pos": [
        "happy","joyful","cheerful","glad","delight","smile","laugh","pleased","content","merry",
        "calm","peace","hope","love","trust","proud","brave","kind","warm","sunny",
    ],
    "emotion_neg": [
        "sad","angry","fear","panic","hate","grief","shock","jealous","regret","worry",
        "pain","hurt","stress","tense","upset","gloom","rage","cry","doubt","shame",
    ],
    "sense_touch": [
        "warm","cold","hot","cool","soft","hard","rough","smooth","wet","dry",
        "pain","itch","press","touch","hold","grip","sharp","dull","numb","tingle",
    ],
    "vision": [
        "color","light","dark","bright","dim","flash","glow","shadow","move","shift",
        "large","small","focus","blur","image","frame","pixel","screen","shape","line",
    ],
    "audio": [
        "music","song","tone","voice","noise","sound","speech","talk","echo","beat",
        "rhythm","melody","radio","audio","volume","quiet","loud","listen","hear","call",
    ],
    "taste": [
        "sweet","sour","bitter","salty","spicy","fresh","rich","mild","sauce","sugar",
        "salt","honey","lemon","pepper","chili","flavor","taste","snack","meal","dish",
    ],
    "smell": [
        "fragrant","smelly","odor","aroma","stink","perfume","scent","fresh","rotten","musty",
        "sharp","sweet","smoke","flower","fruit","oil","gas","soap","spice","air",
    ],
    "shape": [
        "round","square","line","curve","cube","sphere","cylinder","cone","triangle","circle",
        "flat","solid","angle","edge","face","point","form","model","frame","grid",
    ],
    "material": [
        "metal","wood","plastic","glass","paper","stone","iron","steel","copper","silver",
        "gold","fiber","cotton","wool","leather","rubber","ceramic","clay","foam","fabric",
    ],
    "state": [
        "solid","liquid","gas","steam","ice","vapor","melt","freeze","boil","flow",
        "drip","dry","wet","mix","phase","state","form","change","stable","fluid",
    ],
    "animal": [
        "animal","bird","fish","insect","mammal","reptile","frog","snake","tiger","lion",
        "horse","cow","sheep","goat","pig","dog","cat","duck","eagle","shark",
    ],
    "human": [
        "human","man","woman","child","adult","youth","elder","baby","person","people",
        "body","head","hand","foot","heart","brain","skin","bone","muscle","hair",
    ],
    "body": [
        "head","face","eye","ear","nose","mouth","tooth","tongue","neck","chest",
        "back","arm","hand","finger","leg","foot","heart","lung","brain","skin",
    ],
    "action_body": [
        "eat","drink","speak","listen","smell","breathe","bite","kiss","lick","blow",
        "grab","push","pull","touch","hit","throw","wave","shake","kick","step",
        "walk","run","sit","stand","sleep","jump","dance","swim","drive","cut",
    ],
    "plant": [
        "plant","tree","grass","flower","fruit","seed","root","leaf","branch","forest",
        "wood","grain","rice","wheat","corn","bean","rose","lily","apple","orange",
    ],
    "weather": [
        "wind","rain","snow","storm","cloud","sun","fog","hail","thunder","lightning",
        "hot","cold","warm","cool","spring","summer","autumn","winter","day","night",
    ],
    "geography": [
        "mountain","river","lake","ocean","sea","island","coast","plain","desert","plateau",
        "valley","hill","forest","field","land","earth","region","area","zone","border",
    ],
    "building": [
        "building","house","hotel","office","factory","airport","station","store","restaurant","bar",
        "school","hospital","gym","park","plaza","road","bridge","room","indoor","outdoor",
    ],
    "technology": [
        "software","hardware","screen","video","photo","audio","camera","phone","computer","network",
        "signal","power","device","system","tool","message","call","record","display","ui",
    ],
    "time": [
        "past","present","future","today","tomorrow","yesterday","early","late","quick","slow",
        "hour","day","week","month","year","age","classic","modern","fresh","stale",
    ],
    "space": [
        "place","position","location","distance","near","far","left","right","up","down",
        "front","back","inside","outside","center","edge","top","bottom","move","shift",
    ],
    "law": [
        "legal","crime","court","judge","law","rule","policy","tax","permit","license",
        "election","party","state","office","official","case","proof","claim","right","duty",
    ],
    "art": [
        "music","dance","paint","film","drama","novel","poem","artist","famous","classic",
        "work","image","scene","style","voice","stage","gallery","museum","craft","design",
    ],
    "economy": [
        "money","wealth","income","cost","price","trade","market","bank","ticket","account",
        "office","manage","budget","tax","profit","loss","pay","salary","cash","credit",
    ],
    "religion": [
        "christian","buddhist","islamic","church","temple","mosque","faith","prayer","ritual","saint",
        "monk","god","spirit","soul","sacred","holy","festival","tradition","belief","moral",
    ],
}


RULES: list[tuple[list[str], str]] = [
    (["名词", "名字"], "noun"),
    (["动词", "动作", "驾驶", "剪", "洗", "挖", "削"], "verb"),
    (["形容词"], "adj"),
    (["单位", "物理单位"], "unit"),
    (["数字", "数值"], "number"),
    (["发音"], "phonetics"),
    (["拼写"], "spelling"),
    (["语言", "英语", "拉丁语", "法语", "德语", "西班牙语", "汉语", "日语", "方言"], "language"),
    (["高兴", "爱", "舒适", "美", "经典", "吉祥"], "emotion_pos"),
    (["生气", "沮丧", "嫉妒", "遗憾", "恨", "恐惧", "悲伤", "不适", "丑", "严重", "违法", "腐败"], "emotion_neg"),
    (["触觉", "温暖", "冰冷", "疼痛"], "sense_touch"),
    (["视觉", "颜色", "亮暗", "变大", "变小", "移动"], "vision"),
    (["听觉", "音乐"], "audio"),
    (["味觉", "酸", "甜", "苦", "咸", "鲜", "辣"], "taste"),
    (["嗅觉", "香", "臭", "刺鼻", "腥", "馊", "臊"], "smell"),
    (["形状", "方", "圆", "线", "柱", "球"], "shape"),
    (["材料", "金属", "木材", "动物性", "塑料"], "material"),
    (["物质态", "固态", "气态", "液态", "相变", "冻结", "蒸发"], "state"),
    (["动物", "鱼", "虫子", "脊椎动物", "鸟", "宠物", "畜牧"], "animal"),
    (["人", "男", "女", "人种", "白人", "黑人", "亚洲人", "混血"], "human"),
    (["身体部位", "头", "五官", "手", "脚", "胸", "腹", "乳房", "胃肠", "肺", "骨骼", "肌肉", "头发", "皮肤"], "body"),
    (["头部", "上肢", "下肢", "全身", "吃", "说话", "听", "闻", "喝", "吹", "吸", "拿", "按键", "演奏", "击打", "抱", "拧", "摸", "敲", "推", "拉", "压", "抹", "指", "扔", "甩", "挥", "摇", "踢", "踩", "走", "跑", "站", "坐", "躺", "跳", "舞蹈", "游泳"], "action_body"),
    (["植物", "树", "草", "花", "果实", "种子", "生长"], "plant"),
    (["天气", "风", "雨", "气温", "四季", "白天", "黑夜", "晴天"], "weather"),
    (["地理", "山", "水体", "陆地", "沙漠", "高原"], "geography"),
    (["建筑", "商场", "宾馆", "办公楼", "住宅楼", "工厂", "历史建筑", "机场", "车站", "室内", "室外", "商店", "餐厅", "酒吧", "健身房", "办公室", "住宅", "停车场", "球场", "游乐场", "道路", "操场", "广场", "公园", "交通工具内"], "building"),
    (["IT", "UI", "视频", "照片", "工具", "录音", "拍照", "录像", "通话", "msg", "用电", "显示器", "音响"], "technology"),
    (["时间", "时间点", "历法", "快", "慢", "年龄", "新鲜", "陈旧", "过时"], "time"),
    (["空间", "位置", "相对位置", "位移"], "space"),
    (["法律", "诉讼", "政治", "选举", "政党", "官员", "政策", "制度", "税收", "证件"], "law"),
    (["艺术", "绘画", "影视", "文学", "名作", "典故", "体育", "比赛", "历史", "大事件", "王朝"], "art"),
    (["经济", "钱", "财富", "收入", "支出", "经营", "管理", "办公", "票", "账号"], "economy"),
    (["宗教", "基督教", "佛教", "伊斯兰教", "风俗", "婚嫁", "丧事", "节日", "生日", "纪念日", "图腾"], "religion"),
]


def _bank_for_attr(attr: str) -> list[str]:
    for keys, bank_key in RULES:
        for key in keys:
            if key and key in attr:
                return BASE_BANKS.get(bank_key, [])
    return BASE_BANKS["generic"]


def _token_filter(tokenizer, words: Iterable[str]) -> list[str]:
    kept: list[str] = []
    seen: set[str] = set()
    for w in words:
        x = w.strip().lower()
        if not x or x in seen:
            continue
        seen.add(x)
        ids = tokenizer(x, add_special_tokens=False).get("input_ids") or []
        if len(ids) == 1:
            kept.append(x)
    return kept


def _collect_40(tokenizer, attr: str) -> list[str]:
    bank = _bank_for_attr(attr)
    if not bank:
        return []
    cursor = 0
    kept: list[str] = []
    used: set[str] = set()
    rounds = 0
    while len(kept) < 40 and rounds < 12:
        rounds += 1
        chunk = bank[cursor : cursor + 40]
        cursor += 40
        if not chunk:
            chunk = BASE_BANKS["generic"]
        filtered = _token_filter(tokenizer, chunk)
        for w in filtered:
            if w in used:
                continue
            used.add(w)
            kept.append(w)
            if len(kept) == 40:
                break
    return kept[:40]


def build() -> None:
    attrs = parse_attributes_in_order(ATTR_FILE)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)

    groups: list[dict[str, str]] = []
    for attr in attrs:
        words = _collect_40(tokenizer, attr)
        if len(words) < 40:
            continue
        groups.append({"group_name": attr, "tokens": ", ".join(words)})

    OUT_FILE.write_text(json.dumps({"groups": groups}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"attrs_total={len(attrs)}")
    print(f"groups_written={len(groups)}")
    print(f"output={OUT_FILE}")


if __name__ == "__main__":
    build()
