# ComfyUI-GrsAI Agents 说明

本文档基于当前代码库梳理 ComfyUI-GrsAI 自定义节点的“代理”体系（API 客户端、节点包装器与工具函数），帮助快速理解主要组件的职责与交互方式。

## 1. 项目定位与整体流程
- **目标**：把 GrsAI 平台的多模型生成功能封装成 ComfyUI 节点，覆盖文生图、图生图、多图融合与 Nano Banana 专用模型。
- **核心流程**：节点收集输入 → 上传参考图（可选）→ 调用 `GrsaiAPI` → 并发等待结果 → 下载图像 → 转换为 ComfyUI 张量并回传状态文本。
- **三大模型族**：Flux.1 Kontext 系列、GPT Image（sora-image）、Nano Banana（含 fast 变体）。

## 2. 模块结构速览

| 模块 | 角色 | 关键点 |
| --- | --- | --- |
| `config.py` | 配置代理 | 负责加载 `.env`，提供默认参数校验、统一的缺失密钥提示。 |
| `api_client.py` | API 代理 | 封装 HTTP 调用、重试、并行下载；暴露 Flux/GPT/Nano Banana 三套接口。 |
| `flux_nodes.py` | Flux 节点代理 | 三个节点类共享 `_GrsaiFluxKontextNodeBase`；支持并发批量生成。 |
| `gpt_image_nodes.py` | GPT 节点代理 | 文生图与图生图共用 `_GPTImageNodeBase`；自动上传 1–6 张参考图。 |
| `nano_banana_nodes.py` | Nano Banana 节点 | 文生图 / 图生图合并为单节点；支持宽高比控制与多图上传。 |
| `upload.py` | 上传代理 | 提供表单/直传两种上传流程，自动获取 token。 |
| `utils.py` | 工具集 | 下载图片、PIL↔Tensor 转换、透明度处理、宽高比计算等。 |
| `tests/*.py` | 脚本代理 | 独立脚本用于 API 连通性、图像处理与上传测试，便于离线验证。 |

## 3. 配置层（`config.py`）
- `GrsaiConfig` 在初始化时尝试加载根目录 `.env`，并提供全局实例 `default_config`。
- 统一的错误信息 `api_key_error_message` 强调两种密钥注入方式与重启提醒。
- 内置多组枚举：Flux 与 Nano Banana 宽高比列表、输出格式和安全容忍度校验。
- 配置读取均通过 `get_config`/`set_config`，节点层不直接持久化配置。

## 4. API 客户端层（`api_client.py`）
- `GrsaiAPI` 在构造时要求合法的 `sk-` 前缀密钥，复用 `requests.Session` 共享头信息。
- `_make_request` 实现指数退避重试，针对 401/429/5xx/超时给出分支处理。
- **Flux**：`flux_generate_image` 返回单图 `(PIL, url)`；支持种子、宽高比、安全容忍度、指导强度、步数等可选参数，最终统一下载校验 URL。
- **GPT Image**：`gpt_image_generate_image` 返回多图列表；节点端可通过 `variants` 控制数量。
- **Nano Banana**：`banana_generate_image` 自动兼容 `results` 与 `url` 两种返回格式，并并发下载。
- 下载阶段普遍使用 `ThreadPoolExecutor`，失败时抽象为易读错误字符串，减少技术细节泄露。

## 5. 节点层代理

### 5.1 Flux.1 Kontext 系列（`flux_nodes.py`）
- 公共基类 `_GrsaiFluxKontextNodeBase`：
  - 统一的错误封装 `_create_error_result` 与并发调度 `_execute_generation`。
  - 批量生成时针对 `seed=0` 自动生成随机种子（限制在 32 位整形范围）。
- **节点划分**：
  - `GrsaiFluxKontext_TextToImage`：纯文本输入，支持 1–4 张批量生成。
  - `GrsaiFluxKontext_ImageToImage`：单图编辑，先落地到临时文件 → 上传 → 调用 `flux_generate_image`。
  - `GrsaiFluxKontext_MultiImageToImage`：最多三张参考图，逐张保存上传并传给 API。
- 返回值统一为 `(IMAGE, STRING)`，UI 提示成功/失败统计；失败时回退空张量或原图。

### 5.2 GPT Image 系列（`gpt_image_nodes.py`）
- `_GPTImageNodeBase` 负责文生图与图生图共享逻辑。
- `variants` 控制批量数量；图生图可接入 1–6 张参考图，上传流程与 Flux 多图类似。
- 输出为将多张 PIL 图拼接成批量张量，状态文案区分参考图数量与成功计数。

### 5.3 Nano Banana 节点（`nano_banana_nodes.py`）
- 单节点兼顾文生图/图生图：无输入图则文生，有图则编辑。
- 可选 `use_aspect_ratio` 控制是否下发 `aspectRatio`，默认 `auto`。
- 上传路径必须显式传入 API Key（避免依赖环境污染），并对每张输入图做清理。
- `SuppressFalLogs` 在上传/请求阶段降低 HTTP 噪声，提升终端可读性。

## 6. 上传与工具层
- `upload.py`：
  - `upload_file`（PUT）与 `upload_file_zh`（表单 POST）覆盖国际/国内加速域。
  - 当调用方未显式传入密钥时，会回退到 `default_config.get_api_key()` 并复用统一报错。
  - 通过 `get_upload_token[_zh]` 获取一次性凭证，上传成功后拼合 `domain/key` 为可访问 URL。
- `utils.py`：
  - `download_image` 对所有外链下载统一使用友好 UA 与异常捕获。
  - `tensor_to_pil` 与 `pil_to_tensor` 支持 RGBA 保真；`handle_transparent_background` 提供透明背景修复策略。
  - 额外提供宽高比校验、尺寸估算、错误格式化等辅助函数。

## 7. 测试与脚本
- `run_test.py` / `test_generate_single_image.py`：快速验证 Flux API，可批量生成并缓存图片到 `test_outputs/`。
- `test_banana.py`：覆盖 Nano Banana 各模型与宽高比枚举，内置 32 像素对齐逻辑便于对比官方输出。
- `test_transparency_fix.py`、`test_transparent_background.py`：验证 RGBA → Tensor → PIL 的往返与背景处理。
- `test_upload_file_zh.py`：构造临时图像测试上传接口，并覆盖异常路径（空路径、丢失文件、无效密钥）。
- 所有脚本均独立运行，默认读取 `.env`，对命令行无侵入。

## 8. 典型调用顺序（以图生图为例）
1. 节点入口接收 `IMAGE` → 使用 `tensor_to_pil` 转换为 PIL。
2. 落地临时 PNG → 调用 `upload_file_zh` 获得 R2 URL。
3. 组装 API 请求参数（提示词、模型、参考图 URL、额外控制项）。
4. `GrsaiAPI` 发送请求并根据结果列表启动并行下载。
5. 成功的 PIL 图像通过 `pil_to_tensor` 拼装为批量张量；状态文本带成功/失败统计。
6. 失败时返回占位张量与可读的失败提示，便于 ComfyUI 节点面板展示。

## 9. 扩展与注意事项
- 统一使用 `default_config.get_api_key()`，避免在节点内部硬编码密钥或缓存状态。
- 并发下载数量与生成批次数同步，如需扩容建议在节点层控制 `max_workers`。
- 上传临时文件记得在 `finally` 中清理，当前实现已覆盖常见异常路径。
- 若扩展新模型，可参考 `_GrsaiFluxKontextNodeBase` 的基类模式，将公共逻辑抽离后复用。
- 测试脚本默认访问真实 API，使用前确认余额与 `GRSAI_API_KEY` 已配置。

以上内容可作为继续开发或排障时的结构化参考。

## 10. 开发计划：支持 `nano-banana-pro` 与 `imageSize`
- 文档确认：依据 `ref/nano-banana-pro.html`，新增模型 `nano-banana-pro` 支持可选参数 `imageSize`（字符串），取值 `1K` / `2K` / `4K`，默认 `1K`，仅该模型支持。
- 配置与校验：在 `config.py` 增加 Nano Banana 输出尺寸枚举与校验方法，作为节点下拉和 API 入参的单一来源；明确非 `nano-banana-pro` 模型传 `imageSize` 时需拒绝或自动忽略，并给出友好错误提示。
- API 客户端：为 `GrsaiAPI.banana_generate_image` 添加 `image_size` 可选参数，序列化为 `imageSize` 下发；在调用前绑定模型与尺寸的合法性校验，保持旧模型调用不变。
- 节点层：在 `GrsaiNanoBanana_Node.INPUT_TYPES` 中加入模型选项 `nano-banana-pro` 与 `image_size` 选择（默认 `1K`），执行时仅在选用 `nano-banana-pro` 时传递该字段；状态文本可包含所用模型/尺寸以便排障。
- 测试：扩展 `test_banana.py` 覆盖 `nano-banana-pro` 基本生成流程，并补充 `imageSize` 取值校验（含非法组合的预期错误/忽略行为）；必要时更新占位尺寸估算逻辑。
- 文档同步：在 `README.md`/节点说明处标注新模型与 `imageSize` 用法、限制与默认值，保证使用者能从 UI 与文档快速获取差异信息。
