# MINCO 远端部署包

适用目标：Linux x86_64、CPython 3.10。

将本目录上传到远端后执行：

```bash
cd /path/to/minco_processor_remote
chmod +x deploy_remote.sh
./deploy_remote.sh /root/NavDP
```

默认使用：

```text
/root/autodl-tmp/navdp/conda/envs/isaaclab/bin/python
```

如果环境位于其他位置：

```bash
ISAACLAB_PYTHON=/实际路径/bin/python ./deploy_remote.sh /root/NavDP
```

脚本会把扩展安装到：

```text
/root/NavDP/minco_processor/build/_minco_processor.cpython-310-x86_64-linux-gnu.so
```

并在 `isaaclab` Conda 环境中验证新版接口同时包含：

- `optimization_safe_dist`
- `validation_safe_dist`

如果远端不是 CPython 3.10 或不是 x86_64，不要使用该二进制，应在远端从源码编译。
