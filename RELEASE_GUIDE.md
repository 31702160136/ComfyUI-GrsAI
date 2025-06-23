# 🚀 发布指南

## 版本发布流程

自 v1.0.5 起，本项目采用基于 Git 标签的发布流程，以提供稳定的版本管理。

### 📋 发布新版本的步骤

#### 1. 准备发布

```bash
# 确保在 main 分支上
git checkout main
git pull origin main
```

#### 2. 更新版本号

**重要：必须同时修改两个文件的版本号，确保完全一致**

- 修改 `pyproject.toml` 中的 `version = "1.0.6"`
- 修改 `__init__.py` 中的 `__version__ = "1.0.6"`

#### 3. 提交版本更新

```bash
git add pyproject.toml __init__.py
git commit -m "Bump version to 1.0.6"
git push origin main
```

#### 4. 创建并推送标签

```bash
# 创建带注释的标签
git tag -a v1.0.6 -m "Release version 1.0.6 - 修复宽高比参数传递问题"

# 推送标签到远程仓库
git push origin v1.0.6
```

#### 5. 自动发布

- GitHub Actions 会自动检测到新标签
- 自动将新版本发布到 ComfyUI Registry
- 用户可在 ComfyUI Manager 中看到新的稳定版本

### 📊 版本类型说明

| 版本类型 | 描述 | 用户群体 |
|---------|------|---------|
| **latest** | 最新的稳定版本（基于 Git 标签） | 普通用户推荐 |
| **nightly** | 开发版本（main 分支最新提交） | 体验最新功能的用户 |

### 🔍 验证发布是否成功

1. 访问 [ComfyUI Registry](https://registry.comfy.org/) 搜索你的节点
2. 确认新版本出现在版本历史中
3. 在 ComfyUI Manager 中测试安装和更新功能

### ⚠️ 注意事项

- **版本号格式**：使用语义化版本号（如 v1.0.6, v1.1.0）
- **版本一致性**：`pyproject.toml` 和 `__init__.py` 的版本号必须完全一致
- **标签命名**：标签名必须以 `v` 开头（如 v1.0.6）
- **发布频率**：建议每次发布前充分测试，避免频繁发布

### 🐛 故障排除

如果发布失败：

1. 检查 GitHub Actions 日志
2. 确认 `REGISTRY_ACCESS_TOKEN` 密钥设置正确
3. 验证版本号格式是否正确
4. 确认标签是否成功推送到远程仓库
