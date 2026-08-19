# Quest 连接与项目目录

本文档用于在 Northwestern Quest 上使用 **Mice-first-person（项目 B）**。

## 项目信息

| 项目 | 值 |
| --- | --- |
| GitHub 仓库 | `https://github.com/hanshuo-shuo/Mice-first-person.git` |
| Quest 目录 | `~/projects/Mice-first-person` |
| 工作分支 | `velocity-action-env` |
| SSH 主机 | `quest.northwestern.edu` |

> `/tmp/quest.sock` 只代表本机到 Quest 的 SSH 连接，不属于任何一个项目。项目之间通过 Quest 上的目录和 Git 仓库区分。

## 1. 建立并登录 Quest

在本地 Mac 终端建立一个可复用 8 小时的 SSH 连接：

```bash
ssh -M -S /tmp/quest.sock -o ControlPersist=8h -fN quest.northwestern.edu
```

然后登录 Quest：

```bash
ssh -S /tmp/quest.sock quest.northwestern.edu
```

如果连接已经建立，直接运行第二条命令即可。

## 2. 第一次把项目 clone 到 Quest

以下命令在 **Quest 终端**中运行：

```bash
mkdir -p ~/projects
cd ~/projects
git clone --branch velocity-action-env \
  https://github.com/hanshuo-shuo/Mice-first-person.git
cd Mice-first-person
```

如果 `~/projects/Mice-first-person` 已经存在，不要再次 clone，按照下面的日常更新流程操作。

## 3. 日常更新流程

先在本地 Mac 提交并推送代码：

```bash
cd /Users/hanshuo/Desktop/Mice
git status
git push origin velocity-action-env
```

然后登录 Quest 并更新项目 B：

```bash
cd ~/projects/Mice-first-person
git pull --ff-only origin velocity-action-env
```

## 4. 运行前防止进入错误项目

每次运行训练或提交任务前执行：

```bash
pwd
git remote get-url origin
git branch --show-current
```

项目 B 应该分别显示：

```text
.../projects/Mice-first-person
https://github.com/hanshuo-shuo/Mice-first-person.git
velocity-action-env
```

只要其中一项不一致，就先不要运行任务。

## 5. 使用独立的 tmux 会话（推荐）

在 Quest 上为项目 B 创建或重新进入名为 `mice` 的会话：

```bash
tmux new-session -A -s mice -c ~/projects/Mice-first-person
```

项目 A 使用不同的会话名，例如 `crashbench`。SSH 连接可以共用，但项目目录和 tmux 会话名不要共用。

## 6. 关闭 SSH 复用连接

在本地 Mac 终端运行：

```bash
ssh -O exit -S /tmp/quest.sock quest.northwestern.edu
```
