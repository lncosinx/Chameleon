# 基线仓库内部脚本备份（换服务器重放用）
这些脚本运行时必须放回对应克隆仓库内部（依赖其内部模块），此处仅为版本保存。
- MAPF-GPT__run_nothing_bench.py  -> 放回 MAPF-GPT/run_nothing_bench.py（nothing 协议评测 MAPF-GPT-2M/DDG-2M）
- EPH__eval_instances.py          -> 放回 EPH/eval_instances.py（EPH 纯策略在外部实例集评测）
- EPH__configs__eph-quick.py      -> 放回 EPH/configs/eph-quick.py
- marl-gpt__run_light_bench.py    -> 放回 marl-gpt/run_light_bench.py（通才 marl-gpt，已弃用作对比）
另: 我还改了 lacam/_lacam.so 加载告警(lacam_runner.py, 已在Chameleon内), MAPF-GPT/gpt/inference.py 无改动。
