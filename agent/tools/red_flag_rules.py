"""危险信号规则数据 v0.1（危险信号与红线清单的代码化，单一来源）。

⚠️ 工程初稿：正式上线前必须经医学专业人员评审修订。
规则同时服务两处（单一来源）：
- check_red_flags 工具（模型调用核对）；
- scan_input 前置安全扫描（代码直接拦截，不经过模型）。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RedFlagRule:
    """一条危险信号规则。"""

    rule_id: str
    category: str
    level: str  # emergency=立即急诊/拨打120；high=尽快就医
    patterns: tuple[str, ...]
    summary: str
    advice: str


EMERGENCY_RULES: tuple[RedFlagRule, ...] = (
    RedFlagRule(
        rule_id="chest_pain",
        category="心血管",
        level="emergency",
        patterns=(
            r"(胸口|胸前|心口|胸骨后|胸).{0,10}(压榨|压得|刀割|针扎|紧缩).{0,12}(疼|痛|闷)?",
            r"(胸痛|胸口疼|胸口痛|胸闷|心口疼|心口痛).{0,12}(大汗|冷汗|冒汗|出汗|呼吸困难|喘不上气|放射|恶心|呕吐|左肩|左臂|后背|濒死|不缓解)",
            r"(胸口|胸前|心口|胸骨后).{0,8}(疼|痛|闷).{0,12}(大汗|冷汗|冒汗|出汗|喘|呼吸困难|恶心|呕吐|放射|左肩|左臂|后背)",
            r"(胸痛|胸口疼|胸口痛|胸闷).{0,6}(持续|一直|不停)",
            r"(胸痛|胸口疼|胸口痛|胸闷).{0,10}(十几分钟|20分钟|半小时|一小时)",
        ),
        summary="胸痛/胸闷持续不缓解，或伴大汗、恶心、放射痛等",
        advice="立即拨打 120 或前往最近的急诊科",
    ),
    RedFlagRule(
        rule_id="breathing",
        category="呼吸",
        level="emergency",
        patterns=(
            r"(呼吸困难|喘不上气|喘不过气|窒息)",
            r"(口唇发紫|嘴唇发紫|嘴唇青紫)",
            r"(憋气|憋得慌).{0,12}(厉害|严重|加重|越来越|受不了)",
        ),
        summary="严重呼吸困难、口唇发紫等缺氧表现",
        advice="立即拨打 120 或前往最近的急诊科",
    ),
    RedFlagRule(
        rule_id="stroke",
        category="神经（卒中）",
        level="emergency",
        patterns=(
            r"(嘴角歪|嘴角歪斜|口角歪斜|嘴歪了|面部歪)",
            r"(说话不清|言语不清|说不清话|口齿不清|听不懂话)",
            r"(半身|偏瘫|一侧肢体无力|一侧手脚发麻|一侧手脚无力|胳膊腿没力气|手脚不听使唤)",
        ),
        summary="突发面歪、言语不清、单侧肢体无力等卒中征象",
        advice="立即拨打 120 或前往最近的急诊科",
    ),
    RedFlagRule(
        rule_id="consciousness",
        category="神经（意识）",
        level="emergency",
        patterns=(
            r"(昏迷|意识不清|意识模糊|叫不醒|抽搐|抽风|晕厥|晕倒|休克)",
            r"(突然|突发|剧烈|炸裂|雷击样).{0,6}头痛",
            r"(头痛|头疼).{0,3}欲裂",
            r"头痛.{0,8}(呕吐|看不清|视物模糊|脖子硬|颈部僵硬)",
        ),
        summary="意识障碍、抽搐、晕厥，或突发剧烈头痛",
        advice="立即拨打 120 或前往最近的急诊科",
    ),
    RedFlagRule(
        rule_id="bleeding",
        category="出血",
        level="emergency",
        patterns=(
            r"(呕血|吐血|咯血|咳出血|大出血|血流不止|止不住血)",
            r"(黑便|柏油样便|血便|便血|大便带血)",
            r"(阴道大出血|经期大出血|出血不止)",
        ),
        summary="呕血、咯血、黑便/血便、大出血等",
        advice="立即拨打 120 或前往最近的急诊科",
    ),
    RedFlagRule(
        rule_id="anaphylaxis",
        category="过敏",
        level="emergency",
        patterns=(
            r"(喉咙|喉头|嗓子).{0,4}(发紧|紧缩|堵|肿胀)",
            r"(皮疹|风团|荨麻疹|全身痒).{0,10}(呼吸困难|喘不上气|喉|头晕|胸闷)",
        ),
        summary="严重过敏反应（喉部发紧、皮疹伴呼吸困难等）",
        advice="立即拨打 120 或前往最近的急诊科",
    ),
    RedFlagRule(
        rule_id="abdominal",
        category="腹部",
        level="emergency",
        patterns=(
            r"(肚子|腹部|腹痛).{0,10}(剧痛|剧烈|疼得直不起|疼得打滚|拒按|硬邦邦|板状)",
        ),
        summary="剧烈持续腹痛（腹部拒按/板硬）",
        advice="立即拨打 120 或前往最近的急诊科",
    ),
    RedFlagRule(
        rule_id="sepsis_meningitis",
        category="感染/全身",
        level="emergency",
        patterns=(
            r"(高烧|高热|发烧|发热).{0,10}(脖子硬|颈部僵硬|皮疹|瘀点|瘀斑|意识|抽搐|叫不醒|寒战)",
        ),
        summary="高热伴意识改变、颈硬、皮疹等（脑膜炎/脓毒症征象）",
        advice="立即拨打 120 或前往最近的急诊科",
    ),
    RedFlagRule(
        rule_id="pregnancy_emergency",
        category="孕产",
        level="emergency",
        patterns=(
            r"(怀孕|孕妇|孕期|妊娠|产后|哺乳期).{0,16}(剧烈腹痛|肚子疼|阴道出血|大出血|看不见|看不清|剧烈头痛|胎动)",
        ),
        summary="孕期/产后：剧烈腹痛、出血、剧烈头痛或胎动异常",
        advice="立即拨打 120 或前往最近的急诊科",
    ),
    RedFlagRule(
        rule_id="trauma_poison",
        category="外伤/中毒",
        level="emergency",
        patterns=(
            r"(误服|吃错药|喝农药|中毒|一氧化碳)",
            r"(头.{0,3}(撞|摔|磕|碰)).{0,12}(呕吐|意识|晕|叫不醒|头痛)",
            r"(骨折|骨头.{0,4}(露|戳))",
        ),
        summary="头部外伤后意识改变、骨折畸形、误服中毒等",
        advice="立即拨打 120 或前往最近的急诊科",
    ),
)

HIGH_RULES: tuple[RedFlagRule, ...] = (
    RedFlagRule(
        rule_id="fever_duration",
        category="发热",
        level="high",
        patterns=(r"(发烧|发热|高烧).{0,12}(三天|3天|四天|4天|五天|5天|一周|七天|7天)",),
        summary="发热持续多日不退（>3 天）",
        advice="建议尽快就医",
    ),
    RedFlagRule(
        rule_id="weight_loss",
        category="全身",
        level="high",
        patterns=(r"体重.{0,6}(下降|减轻|变轻|往下掉|掉得)",),
        summary="不明原因体重明显下降",
        advice="建议近期就医检查",
    ),
    RedFlagRule(
        rule_id="cough_long",
        category="呼吸",
        level="high",
        patterns=(
            r"咳嗽.{0,10}(三周|3周|一个月|四周|4周)",
            r"(痰|咳).{0,4}带血",
        ),
        summary="持续咳嗽超过 3 周，或痰中带血",
        advice="建议尽快就医",
    ),
    RedFlagRule(
        rule_id="leg_swelling",
        category="循环",
        level="high",
        patterns=(
            r"(小腿|腿).{0,6}(肿|胀).{0,6}(疼|痛)",
            r"(小腿|腿).{0,6}(疼|痛).{0,6}(肿|胀)",
        ),
        summary="单侧小腿肿胀疼痛（血栓可能）",
        advice="建议尽快就医",
    ),
    RedFlagRule(
        rule_id="mass",
        category="全身",
        level="high",
        patterns=(r"(肿块|疙瘩|淋巴结).{0,8}(变大|增大|新出现|长出来)",),
        summary="新出现或增大的肿块/淋巴结",
        advice="建议近期就医检查",
    ),
    RedFlagRule(
        rule_id="dehydration",
        category="消化",
        level="high",
        patterns=(r"(呕吐|腹泻|拉肚子|拉稀).{0,12}(口渴|尿少|没尿|脱水|乏力|站不起来)",),
        summary="呕吐/腹泻伴脱水表现",
        advice="建议尽快就医",
    ),
)

# 前置扫描：自伤/伤人表达（pattern, 说明）
SELF_HARM_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"(想死|不想活|活不下去|活着没意思|活着没意义|活够了|一了百了)", "自伤念头表达"),
    (r"(自杀|轻生|割腕|跳楼|结束(自己的)?(生命|一生)|伤害自己|自残|自伤)", "自伤相关表达"),
    (r"(杀了?自己|弄死自己)", "自伤相关表达"),
    (r"(杀(了|掉)?(他|她|别人)|弄死(他|她)|报复社会|同归于尽)", "伤人相关表达"),
)

# 超范围场景（仅首次对话时识别）
PSYCH_PATTERNS: tuple[str, ...] = (
    r"(精神病|精神分裂|分裂症|双相|躁郁|抑郁症|抑郁发作|焦虑症|强迫症|妄想|幻觉)",
)

CHILD_PATTERNS: tuple[str, ...] = (
    r"(孩子|小孩|儿童|宝宝|婴儿|幼儿|闺女|儿子|女儿).{0,12}(发烧|发热|咳嗽|喘|呕吐|吐了|腹泻|拉肚子|出疹|起疹|抽搐|哭闹|不吃|烫伤|磕|摔)",
)

PREGNANCY_PATTERNS: tuple[str, ...] = (
    r"(怀孕|孕妇|孕期|妊娠期|哺乳期|产后).{0,14}(腹痛|肚子疼|出血|头痛|发烧|发热|呕吐|恶心|不适|头晕|心慌)",
)
