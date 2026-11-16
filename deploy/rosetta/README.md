# Conda + PyRosetta API

此版直接使用 Conda 环境中的 PyRosetta，在独立 Slurm 子作业内打分。不需要 SIF、Singularity/Apptainer、Rosetta CLI 二进制或单独配置 Rosetta database。

## 部署

将 `main.py` 和 `rosetta-api.sbatch` 放在：

```text
/lustre/project/apis/rosetta/
```

替换前备份已有文件。部署脚本不包含个人集群路径。启动 Harness 前导出实际路径；API 日志默认写入提交目录，作业输出默认在 API 目录的 jobs/：

```bash
export ROSETTA_API_DIR=/lustre/project/apis/rosetta
export ROSETTA_ALLOWED_INPUT_ROOTS=/lustre/project
export ROSETTA_API_WORK=/lustre/project/rosetta_api_jobs
mkdir -p "$ROSETTA_API_WORK"
```

使用你**已经安装 PyRosetta 的真实 Conda 环境**，其中还需要 FastAPI、Pydantic 2 和 uvicorn。没有提供环境名，因此脚本不会虚构一个固定名称，也不会安装或更换 PyRosetta。

两种方式任选一种：

```bash
# 方式一：在提交任务的终端先激活已有环境。
conda activate <你的PyRosetta环境名>
# 然后从同一环境启动 Harness，让它按已有注册提交 API sbatch。
```

```bash
# 方式二：让 sbatch 显式激活环境。填写实际名称或绝对 prefix 路径。
export ROSETTA_CONDA_ENV=<你的PyRosetta环境名或绝对环境路径>
# 仅在集群无法从 CONDA_EXE / PATH 找到 conda 时设置真实 conda.sh 路径：
export ROSETTA_CONDA_SH=/你的conda安装目录/etc/profile.d/conda.sh
# 从保留这些导出变量的环境启动 Harness。
```

脚本使用激活后的 `$CONDA_PREFIX/bin/python` 启动 uvicorn；科学子作业也使用同一 Python。`sbatch --export=ALL` 保留环境和动态库设置。Conda 环境与 API 文件必须在子作业节点可读；目录默认在共享 `/lustre` 上。

也可在集群手工验证环境依赖和启动检查：

```bash
python -c 'import pyrosetta, fastapi, pydantic, uvicorn; print(pydantic.__version__)'
python main.py --check-pyrosetta
```

启动检查实际初始化 ref2015，在临时两链结构上执行残基对打分和 InterfaceAnalyzer，检查 PyRosetta bindings 和输出。检查失败时服务不会伪装成就绪。`/health` 在独立子进程执行该检查，并缓存当前服务进程的结果；更换环境后需重启服务。

## 当前适配范围

- `POST /v1/tools/interface_analyze`：本次 autonomous round 所需接口。
- `POST /v1/tools/structure_score`：PyRosetta 整体结构评分。
- 原来的 `structure_relax`、`constrained_relax`、`interface_local_relax`、`mutation_binding_ddg` 属于额外协议，此版本未移植，明确返回 HTTP 501；不会用别的计算冒充其结果。`/health.supported_tools` 公布实际支持的接口。

PyRosetta 不在 FastAPI 请求线程内初始化；每个科学请求提交独立 Slurm 子进程，避免同一进程中的并发状态干扰。

## Interface 分数与固定点

接口请求字段保持为：

```json
{
  "input_pdb": "/lustre/实际路径/generated.pdb",
  "chain_group_a": "B",
  "chain_group_b": "A",
  "pack_input": false,
  "pack_separated": true
}
```

GitHub 将 group_a 设置为 binder，group_b 设置为 target。两组需完整覆盖复合物蛋白链，链 ID 为大写 A–Z、标准蛋白残基、正整数 PDB 编号，无 insertion code。

子作业会：

1. 从提交时保留的 `input.pdb` 快照加载原始复合物。
2. 用 `ref2015` 评分，开启 backbone H-bond 的 pair 分解。
3. 读取 EnergyGraph 中的短程 pair 能量，并补上已启用的 long-range two-body energy methods；将每个 score term 乘以权重，得到实际 pair total。
4. 在 **InterfaceAnalyzer 打包或分离结构之前** 写出 binder–target pair 表 `outputs/residue_energies.out`。
5. 按请求的 `pack_input` / `pack_separated` 参数运行 PyRosetta InterfaceAnalyzer，保留 `dG_separated` 到 `interface.sc`。

输出保持上传分析脚本和当前 GitHub 分支需要的格式：

```text
ResResE Res1 Res2 total
ResResE ---- ---- -----
ResResE ALA_661 GLY_651 -6.25
```

`661` 表示 B 链（ASCII 66）的 PDB 残基 1。真实零能量也保留；非有限能量、错误的残基映射或 PyRosetta 异常会让 Slurm 作业非零退出。

CB/CA 距离 ≤12 Å、负能量按 binder 残基累加、每残基能量 < -5 选作固定点，仍由 GitHub 客户端执行。整体 `dG_separated` 不代替该计算。

## GitHub 协议配置

保持原有注册，更新 Rosetta `service` 字段为对应部署：

```json
{
  "url": "http://slurm-api.invalid:8050",
  "health_path": "/health",
  "sbatch_script": "/lustre/project/apis/rosetta/rosetta-api.sbatch",
  "sbatch_resolve_host": true,
  "status_field": "status",
  "exit_code_field": "exit_code",
  "require_exit_code": true,
  "files_field": "files",
  "metrics_field": "metrics"
}
```

URL 主机是占位符；Harness 根据自己创建的 allocation 解析节点，端口仍为 8050。保持共享路径映射，使客户端能读取 `ROSETTA_API_WORK` 下的输出。旧注册中的 `files_field` 若指向 `score_files` / `pdb_files`，需改为 `files`。

任务 YAML 保持：

```yaml
evaluation:
  scoring_backend: torchscore
  rosetta_residue_glob: '*.out'
```

状态接口同时提供 `status` / `state` 和 `exit_code`。只有 Slurm 父作业 COMPLETED 且退出码为 `0:0` 才允许获取结果；accounting 尚未就绪时为 UNKNOWN。结果的 `files` 含唯一的 `residue_energies.out`，`metrics` 保留整体分数；Slurm stdout 不混入结果文件列表。

已有旧作业若缺少标准化 pair 输出，需要新任务，不能当作新流程的完整结果复用。

## 验证及限制

离线测试不要求安装 PyRosetta 或 Slurm，不能验证真实集群计算结果。测试使用**测试目录中的 PyRosetta 替身**，覆盖实际生成的 shell/worker 子进程、加权短程及长程 pair 累加、失败传播、Conda 环境选择、状态协议，以及真实 GitHub 客户端对输出的读取。生产代码没有替身或模拟分数。

```bash
uv run --locked pytest tests/test_rosetta_api_deployment.py -q
bash -n deploy/rosetta/rosetta-api.sbatch
```

以上测试命令从仓库根目录执行。客户端对接测试还需要安装本仓库的 `opendde_harness`；其它测试需要 API 依赖、pytest 和 pytest-asyncio。真实环境请执行上面的 `--check-pyrosetta`，再使用实际复合物验证。
