# 提示词与话术变更记录

| 日期 | 版本 | 变更 | 原因 | 回归结果 |
| --- | --- | --- | --- | --- |
| 2026-09-12 | v1 | 首版：system_prompt_v1 / review_fix_v1 / emergency_templates_v1 | 一期开发启动 | 待跑（§6.2 场景测试） |

## 约定

- 修改任何提示词或话术文件 → 更新本表 → 跑 `tests/test_scenarios_fake_model.py` 回归 → 升版本号。
- 固定话术（emergency_templates）修改需医学评审。
