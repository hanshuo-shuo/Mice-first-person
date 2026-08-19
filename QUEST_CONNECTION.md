# Quest / Slurm setup（项目 B）

这份文档记录从本地 Mac 同步代码并向 Northwestern Quest 提交 Slurm 任务的完整流程。

## 已验证的配置

| 项目 | 值 |
| --- | --- |
| GitHub 仓库 | `https://github.com/hanshuo-shuo/Mice-first-person.git` |
| 本地目录 | `/Users/hanshuo/Desktop/Mice` |
| Quest 目录 | `~/projects/Mice-first-person` |
| Git 分支 | `velocity-action-env` |
| Quest 用户 | `shv7753` |
| Slurm account | `p31777`（当前默认 account） |
| 默认分区 | `normal` |
| Conda 环境 | `Mice-BotEvade` |
| SSH socket | `/tmp/quest.sock` |

`/tmp/quest.sock` 只代表 Mac 到 Quest 的 SSH 连接，不属于某个项目。项目通过 Quest 目录、Git 远端和分支区分。

## 验证状态（2026-08-19）

- 项目已 clone 到 Quest，远端和 `velocity-action-env` 分支正确；
- `Mice-BotEvade` 已在 Quest 创建；
- `setup/sac_train.sbatch` 已通过 Quest `sbatch --test-only`；
- compute-node smoke job `9882250` 已 `COMPLETED`，exit code 为 `0:0`；
- smoke test 首次加载峰值约 8 GiB，因此模板预留 12 GiB。

## 首次配置

### 1. 建立 SSH 连接

在本地 Mac 终端运行：

```bash
ssh -M -S /tmp/quest.sock -o ControlPersist=8h -fN \
  quest.northwestern.edu
```

登录 Quest：

```bash
ssh -S /tmp/quest.sock quest.northwestern.edu
```

### 2. Clone 项目

项目已经 clone 到 Quest。重新安装时可以在 Quest 运行：

```bash
mkdir -p ~/projects
cd ~/projects
git clone --branch velocity-action-env \
  https://github.com/hanshuo-shuo/Mice-first-person.git
cd Mice-first-person
```

如果目录已经存在，不要再次 clone。

### 3. 创建 Quest Conda 环境

在 Quest 运行：

```bash
cd ~/projects/Mice-first-person
bash setup/quest_setup.sh
```

脚本会创建或更新 `Mice-BotEvade` 并建立训练输出目录。只有 `environment.yaml` 改变后才需要再次执行。

不要在登录节点加载完整的 PyTorch 训练栈。环境安装完成后，提交一个十分钟上限的小型 smoke job，在计算节点验证依赖和 Cellworld 环境：

```bash
cd ~/projects/Mice-first-person
sbatch setup/quest_smoke.sbatch
```

查看返回的 job ID 和日志：

```bash
squeue -j <JOB_ID>
cat slurm_logs/mice-smoke-<JOB_ID>.out
cat slurm_logs/mice-smoke-<JOB_ID>.err
```

成功日志会包含 `Quest compute-node smoke test passed`。

## 以后每次提交任务（推荐）

### 1. 在本地提交并推送代码

```bash
cd /Users/hanshuo/Desktop/Mice
git status
git add <本次修改的文件>
git commit -m "描述本次实验"
git push origin velocity-action-env
```

不要把尚未 commit 或 push 的代码提交到 Quest；Quest 只能拉取 GitHub 上已有的 commit。

### 2. 建立 Quest 连接

```bash
ssh -M -S /tmp/quest.sock -o ControlPersist=8h -fN \
  quest.northwestern.edu
```

如果连接已经存在，可以跳过这一步。

检查连接：

```bash
ssh -O check -S /tmp/quest.sock quest.northwestern.edu
```

### 3. 从本地一条命令提交

提交默认 SAC 训练任务：

```bash
cd /Users/hanshuo/Desktop/Mice
bash setup/submit_quest.sh
```

这个脚本会自动：

1. 检查本地工作区是否干净；
2. 检查当前分支是否为 `velocity-action-env`；
3. 检查本地 commit 是否已经 push 到 GitHub；
4. 在 Quest 上 fast-forward 更新同一分支；
5. 提交 `setup/sac_train.sbatch` 并返回 job ID。

以后增加其他任务脚本后，可以指定仓库内的 `.sbatch` 文件：

```bash
bash setup/submit_quest.sh setup/another_job.sbatch
```

提交不训练模型、且强制无 OpenRouter key 的 EXP-00 Gaze Oracle Headroom：

```bash
bash setup/submit_quest.sh setup/peekbench_exp00.sbatch
```

该作业使用 `configs/peekbench/exp00.yaml`，并按 Slurm job ID 写入独立的
`results/peekbench/exp00_gaze_oracle_headroom_<JOB_ID>/` 目录。

提交第一人称双目 CNN SAC（1×A100），训练结束后自动做配对评估和 GIF
渲染：

```bash
bash setup/submit_quest.sh setup/sac_cnn_train.sbatch
```

产物写入
`results/sac/sac_cnn_active_gaze_<JOB_ID>/`。该作业使用公开 Dict observation
和 active-gaze 三维动作；旧的 `setup/sac_train.sbatch` 仍是状态向量 MLP
基线，两者不要混淆。

## 手动提交方式

如果不使用本地辅助脚本：

```bash
ssh -S /tmp/quest.sock quest.northwestern.edu
cd ~/projects/Mice-first-person
git pull --ff-only origin velocity-action-env
mkdir -p slurm_logs Saved_Models logs
sbatch setup/sac_train.sbatch
```

不要直接在 Quest 登录节点运行长时间训练；训练应通过 `sbatch` 进入计算节点。

## 查看和管理任务

登录 Quest 后：

```bash
# 查看自己的任务
squeue -u shv7753

# 查看指定任务
squeue -j <JOB_ID>

# 查看完成任务的资源和状态
sacct -j <JOB_ID> \
  --format=JobID,JobName,Partition,State,Elapsed,ExitCode,MaxRSS

# 查看标准输出；把 JOB_ID 换成真实编号
tail -f ~/projects/Mice-first-person/slurm_logs/mice-sac-<JOB_ID>.out

# 取消任务
scancel <JOB_ID>
```

`setup/sac_train.sbatch` 当前申请：

- account：`p31777`
- partition：`normal`
- 时间：8 小时
- CPU：4 cores
- 内存：16 GB
- GPU：不申请（当前 `MlpPolicy` SAC 训练主要是 CPU 工作负载）

资源不足或浪费时，直接修改 `.sbatch` 文件顶部的 `#SBATCH` 参数，commit 并 push 后再提交。Quest 要求 job script 明确指定 account、partition 和 time。

## 防止进入错误项目

在 Quest 运行任务前可以检查：

```bash
cd ~/projects/Mice-first-person
pwd
git remote get-url origin
git branch --show-current
git status --short --branch
```

预期分别包含：

```text
.../projects/Mice-first-person
https://github.com/hanshuo-shuo/Mice-first-person.git
velocity-action-env
```

## 关闭连接

在本地 Mac 运行：

```bash
ssh -O exit -S /tmp/quest.sock quest.northwestern.edu
```

## 官方参考

- [Northwestern Quest Slurm job scheduler](https://rcdsdocs.it.northwestern.edu/systems/quest/user-guide/slurm/slurm.html)
- [Northwestern Quest resources and partitions](https://rcdsdocs.it.northwestern.edu/systems/quest/resources/quest-resource.html)
- [Northwestern Quest GPU guide](https://rcdsdocs.it.northwestern.edu/systems/quest/user-guide/gpu/gpu.html)
