"""
nanoPoet Web 应用 - 古典诗词生成系统
简洁优雅的苹果风格界面
"""

from flask import Flask, render_template_string, request, jsonify
from pathlib import Path
import torch

from nanopoet.common import (
    AUTHOR_START, AUTHOR_END,
    TITLE_START, TITLE_END,
    STYLE_START, STYLE_END,
    CONTENT_START, CONTENT_END,
    filter_poem, update_poem_author
)
from nanopoet.dataset import load_raw_data


def create_app(model_configs, tokenizer, device, authors_list=None, styles_list=None):
    """
    创建 Flask 应用

    Args:
        model_configs: 模型配置列表，每个元素是字典 {"name": "显示名称", "path": "模型路径", "model": 模型实例}
                      第一个模型会作为默认模型
        tokenizer: 分词器
        device: 设备 (cpu/cuda/mps)
        authors_list: 作者列表（可选）
        styles_list: 风格列表（可选）

    Returns:
        Flask app 实例
    """
    app = Flask(__name__)

    # 存储当前使用的模型（默认第一个）
    app.config['current_model_idx'] = 0
    app.config['model_configs'] = model_configs
    app.config['tokenizer'] = tokenizer
    app.config['device'] = device

    # 如果没有提供作者和风格列表，从数据中提取
    if authors_list is None or styles_list is None:
        data = load_raw_data()
        filtered_data = [d for d in data if filter_poem(d)]
        updated_data = [update_poem_author(d) for d in filtered_data]

        if authors_list is None:
            authors_list = sorted(list(set(d.get('author', '') for d in updated_data if d.get('author'))))
        if styles_list is None:
            styles_list = sorted(list(set(d.get('style', '') for d in updated_data if d.get('style'))))

    app.config['authors_list'] = authors_list
    app.config['styles_list'] = styles_list

    # ============ 诗词生成 ============

    def generate_poem(title=None, author=None, style=None, max_tokens=150, temperature=0.8, top_k=10):
        """生成诗词"""
        current_idx = app.config['current_model_idx']
        model = app.config['model_configs'][current_idx]['model']
        tok = app.config['tokenizer']
        dev = app.config['device']

        # 构建prompt
        parts = []
        if author:
            parts.append(f"{AUTHOR_START}{author}{AUTHOR_END}")
        if style:
            parts.append(f"{STYLE_START}{style}{STYLE_END}")
        if title:
            parts.append(f"{TITLE_START}{title}{TITLE_END}")
        parts.append(CONTENT_START)

        prompt = "".join(parts)

        # 编码prompt
        context = torch.tensor([tok.encode(prompt)], dtype=torch.long, device=dev)

        # 获取END_TOKEN的token id
        end_token_id = tok.encode(CONTENT_END)[0]

        # 生成
        model.eval()
        with torch.no_grad():
            generated = model.generate(
                context,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                stop_token_ids=[end_token_id]
            )

        # 解码
        full_text = tok.decode(generated[0].tolist())

        # 调试：打印完整文本
        print(f"\n生成的完整文本: {full_text}")
        print(f"CONTENT_START: {CONTENT_START}")
        print(f"CONTENT_END: {CONTENT_END}")

        # 提取诗词内容 - 只提取START和END之间的内容
        is_complete = CONTENT_END in full_text

        # 使用更精确的方式提取 - 找到最后一个START标记
        start_idx = full_text.rfind(CONTENT_START)
        end_idx = full_text.find(CONTENT_END, start_idx) if start_idx != -1 else -1

        if start_idx != -1:
            # 找到了START_TOKEN
            poem_start = start_idx + len(CONTENT_START)

            if end_idx != -1 and end_idx > start_idx:
                # 找到了END_TOKEN，提取中间内容
                poem_content = full_text[poem_start:end_idx]
            else:
                # 没找到END_TOKEN，提取START_TOKEN之后的所有内容
                poem_content = full_text[poem_start:]
        else:
            # 没找到START_TOKEN，返回整个文本
            poem_content = full_text

        print(f"提取的诗词内容: {poem_content}")

        return poem_content.strip(), is_complete

    # ============ HTML模板 ============

    HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>nanoPoet - 古典诗词生成</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, "SF Pro SC", "SF Pro Text", "PingFang SC", "Helvetica Neue", "Helvetica", "Arial", sans-serif;
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 40px 20px;
            color: #1d1d1f;
        }

        .container {
            max-width: 1400px;
            width: 100%;
        }

        .header {
            text-align: center;
            margin-bottom: 60px;
            animation: fadeInDown 0.8s ease;
        }

        .content-wrapper {
            display: grid;
            grid-template-columns: 450px 1fr;
            gap: 32px;
            align-items: start;
        }

        .header h1 {
            font-size: 56px;
            font-weight: 600;
            letter-spacing: -0.02em;
            margin-bottom: 12px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        .header p {
            font-size: 21px;
            color: #6e6e73;
            font-weight: 400;
        }

        .card {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(20px);
            border-radius: 24px;
            padding: 40px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.08);
            animation: fadeInUp 0.8s ease;
            height: fit-content;
            position: sticky;
            top: 40px;
        }

        .form-group {
            margin-bottom: 24px;
        }

        .form-group:last-of-type {
            margin-bottom: 32px;
        }

        .form-group label {
            display: block;
            font-size: 17px;
            font-weight: 500;
            margin-bottom: 12px;
            color: #1d1d1f;
        }

        .form-group select,
        .form-group input {
            width: 100%;
            padding: 16px 20px;
            font-size: 17px;
            border: 2px solid #e5e5e7;
            border-radius: 12px;
            background: #fbfbfb;
            color: #1d1d1f;
            transition: all 0.3s ease;
            font-family: inherit;
        }

        .form-group select:focus,
        .form-group input:focus {
            outline: none;
            border-color: #667eea;
            background: white;
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.15);
        }

        .form-group select:hover,
        .form-group input:hover {
            border-color: #d1d1d6;
        }

        .btn-generate {
            width: 100%;
            padding: 18px;
            font-size: 19px;
            font-weight: 600;
            color: white;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border: none;
            border-radius: 12px;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 8px 24px rgba(102, 126, 234, 0.3);
        }

        .btn-generate:hover {
            transform: translateY(-2px);
            box-shadow: 0 12px 32px rgba(102, 126, 234, 0.4);
        }

        .btn-generate:active {
            transform: translateY(0);
        }

        .btn-generate:disabled {
            background: #d1d1d6;
            cursor: not-allowed;
            transform: none;
            box-shadow: none;
        }

        .result-card {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(20px);
            border-radius: 24px;
            padding: 48px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.08);
            animation: fadeInUp 0.8s ease;
            min-height: 500px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: flex-start;
        }

        .result-content {
            width: 100%;
            display: none;
        }

        .result-content.show {
            display: block;
        }

        .result-placeholder {
            text-align: center;
            color: #86868b;
            font-size: 17px;
            flex: 1;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
        }

        .result-placeholder.hide {
            display: none;
        }

        .poem-title {
            font-size: 28px;
            font-weight: 600;
            text-align: center;
            margin-bottom: 32px;
            color: #1d1d1f;
        }

        .poem-content {
            font-size: 21px;
            line-height: 2;
            color: #1d1d1f;
            white-space: pre-wrap;
            text-align: center;
            font-weight: 400;
            letter-spacing: 0.02em;
        }

        .poem-meta {
            margin-top: 32px;
            padding-top: 24px;
            border-top: 1px solid #e5e5e7;
            text-align: center;
            font-size: 15px;
            color: #86868b;
        }

        .loading {
            text-align: center;
            color: #667eea;
            font-size: 17px;
            margin-top: 24px;
            display: none;
        }

        .loading.show {
            display: block;
        }

        .loading::after {
            content: '...';
            animation: dots 1.5s steps(4, end) infinite;
        }

        @keyframes dots {
            0%, 20% { content: '.'; }
            40% { content: '..'; }
            60% { content: '...'; }
            80%, 100% { content: ''; }
        }

        @keyframes fadeInDown {
            from {
                opacity: 0;
                transform: translateY(-20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        @keyframes fadeInUp {
            from {
                opacity: 0;
                transform: translateY(20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        .footer {
            text-align: center;
            margin-top: 60px;
            color: #86868b;
            font-size: 15px;
            width: 100%;
        }

        @media (max-width: 1024px) {
            .content-wrapper {
                grid-template-columns: 1fr;
                gap: 24px;
            }

            .card {
                position: static;
            }
        }

        @media (max-width: 768px) {
            .header h1 {
                font-size: 40px;
            }

            .header p {
                font-size: 17px;
            }

            .card, .result-card {
                padding: 32px 24px;
            }

            .poem-title {
                font-size: 24px;
            }

            .poem-content {
                font-size: 19px;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>nanoPoet</h1>
            <p>古典诗词生成系统</p>
        </div>

        <div class="content-wrapper">
            <div class="card">
                <form id="poemForm">
                    {% if models|length > 1 %}
                    <div class="form-group">
                        <label for="modelSelect">模型选择</label>
                        <select id="modelSelect" name="model">
                            {% for model in models %}
                            <option value="{{ loop.index0 }}" {% if loop.index0 == current_model %}selected{% endif %}>
                                {{ model.name }}
                            </option>
                            {% endfor %}
                        </select>
                    </div>
                    {% endif %}

                    <div class="form-group">
                        <label for="author">作者</label>
                        <select id="author" name="author">
                            <option value="">随机</option>
                            {% for author in authors %}
                            <option value="{{ author }}">{{ author }}</option>
                            {% endfor %}
                        </select>
                    </div>

                    <div class="form-group">
                        <label for="style">形式</label>
                        <select id="style" name="style">
                            <option value="">随机</option>
                            {% for style in styles %}
                            <option value="{{ style }}">{{ style }}</option>
                            {% endfor %}
                        </select>
                    </div>

                    <div class="form-group">
                        <label for="title">标题（可选）</label>
                        <input type="text" id="title" name="title" placeholder="输入诗词标题">
                    </div>

                    <button type="submit" class="btn-generate" id="generateBtn">
                        生成诗词
                    </button>
                </form>

                <div class="loading" id="loading">正在生成</div>
            </div>

            <div class="result-card" id="resultCard">
                <div class="result-placeholder" id="resultPlaceholder">
                    <div style="font-size: 48px; margin-bottom: 24px;">📝</div>
                    <p style="font-size: 19px; font-weight: 500; margin-bottom: 8px;">等待生成</p>
                    <p style="font-size: 15px; opacity: 0.7;">选择条件并点击生成按钮开始创作</p>
                </div>
                <div class="result-content" id="resultContent">
                    <div class="poem-title" id="poemTitle"></div>
                    <div class="poem-content" id="poemContent"></div>
                    <div class="poem-meta" id="poemMeta"></div>
                </div>
            </div>
        </div>

        <div class="footer">
            <p>Powered by nanoPoet | GPT-style Language Model</p>
        </div>
    </div>

    <script>
        const form = document.getElementById('poemForm');
        const generateBtn = document.getElementById('generateBtn');
        const loading = document.getElementById('loading');
        const resultCard = document.getElementById('resultCard');
        const resultContent = document.getElementById('resultContent');
        const resultPlaceholder = document.getElementById('resultPlaceholder');
        const poemTitle = document.getElementById('poemTitle');
        const poemContent = document.getElementById('poemContent');
        const poemMeta = document.getElementById('poemMeta');
        const modelSelect = document.getElementById('modelSelect');

        // 切换模型
        if (modelSelect) {
            modelSelect.addEventListener('change', async (e) => {
                const selectedModelIdx = parseInt(e.target.value);

                // 禁用所有控件
                generateBtn.disabled = true;
                modelSelect.disabled = true;
                generateBtn.textContent = '正在切换模型...';

                try {
                    const response = await fetch('/api/switch_model', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            model_idx: selectedModelIdx
                        })
                    });

                    const result = await response.json();

                    if (result.success) {
                        console.log('模型切换成功:', result.model_name);
                    } else {
                        alert('切换模型失败: ' + result.error);
                    }
                } catch (error) {
                    alert('切换模型失败: ' + error.message);
                } finally {
                    // 恢复控件状态
                    generateBtn.disabled = false;
                    modelSelect.disabled = false;
                    generateBtn.textContent = '生成诗词';
                }
            });
        }

        form.addEventListener('submit', async (e) => {
            e.preventDefault();

            // 禁用按钮，显示加载状态
            generateBtn.disabled = true;
            generateBtn.textContent = '生成中...';
            loading.classList.add('show');
            resultContent.classList.remove('show');
            resultPlaceholder.classList.remove('hide');

            // 获取表单数据
            const formData = new FormData(form);
            const data = {
                author: formData.get('author'),
                style: formData.get('style'),
                title: formData.get('title')
            };

            try {
                // 发送请求
                const response = await fetch('/generate', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify(data)
                });

                const result = await response.json();

                if (result.success) {
                    // 显示结果
                    if (data.title) {
                        poemTitle.textContent = `《${data.title}》`;
                        poemTitle.style.display = 'block';
                    } else {
                        poemTitle.style.display = 'none';
                    }

                    poemContent.textContent = result.poem;

                    // 显示元信息
                    let metaText = '';
                    if (data.author) metaText += `作者：${data.author}`;
                    if (data.style) {
                        if (metaText) metaText += ' | ';
                        metaText += `形式：${data.style}`;
                    }
                    if (!result.is_complete) {
                        if (metaText) metaText += ' | ';
                        metaText += '⚠️ 可能未完成';
                    }
                    poemMeta.textContent = metaText || '随机生成';

                    // 隐藏占位符，显示结果内容
                    resultPlaceholder.classList.add('hide');
                    resultContent.classList.add('show');
                } else {
                    alert('生成失败：' + result.error);
                }
            } catch (error) {
                alert('生成失败：' + error.message);
            } finally {
                // 恢复按钮状态
                generateBtn.disabled = false;
                generateBtn.textContent = '生成诗词';
                loading.classList.remove('show');
            }
        });
    </script>
</body>
</html>
"""

    # ============ 路由 ============

    @app.route('/')
    def index():
        """首页"""
        return render_template_string(
            HTML_TEMPLATE,
            authors=app.config['authors_list'],
            styles=app.config['styles_list'],
            models=app.config['model_configs'],
            current_model=app.config['current_model_idx']
        )

    @app.route('/api/switch_model', methods=['POST'])
    def switch_model():
        """切换模型"""
        try:
            data = request.get_json()
            model_idx = data.get('model_idx')

            if model_idx is None or model_idx < 0 or model_idx >= len(app.config['model_configs']):
                return jsonify({
                    'success': False,
                    'error': '无效的模型索引'
                })

            # 切换模型
            app.config['current_model_idx'] = model_idx
            model_name = app.config['model_configs'][model_idx]['name']

            print(f"\n切换到模型: {model_name}")

            return jsonify({
                'success': True,
                'model_name': model_name,
                'model_idx': model_idx
            })
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            })

    @app.route('/generate', methods=['POST'])
    def generate():
        """生成诗词API"""
        try:
            data = request.get_json()
            author = data.get('author') or None
            style = data.get('style') or None
            title = data.get('title') or None

            poem, is_complete = generate_poem(
                title=title,
                author=author,
                style=style,
                max_tokens=150,
                temperature=0.8,
                top_k=10
            )

            return jsonify({
                'success': True,
                'poem': poem,
                'is_complete': is_complete
            })
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            })

    return app


def run_app(models, tokenizer, device, authors_list=None, styles_list=None, host='0.0.0.0', port=54321, debug=False):
    """
    启动 Web 应用

    Args:
        models: 模型列表，每个元素是字典 {"name": "显示名称", "model": 模型实例}
                或者是单个模型实例（会自动命名为 "模型"）
                第一个模型会作为默认模型
        tokenizer: 分词器
        device: 设备 (cpu/cuda/mps)
        authors_list: 作者列表（可选）
        styles_list: 风格列表（可选）
        host: 监听地址，默认 0.0.0.0
        port: 端口号，默认 54321
        debug: 是否开启调试模式

    Example:
        # 方式1：传入多个模型
        run_app(
            models=[
                {"name": "SFT模型", "model": sft_model},
                {"name": "Mid模型", "model": mid_model},
            ],
            tokenizer=tokenizer,
            device=device
        )

        # 方式2：传入单个模型
        run_app(
            models=sft_model,
            tokenizer=tokenizer,
            device=device
        )
    """
    print("\n" + "=" * 70)
    print("  nanoPoet Web 应用启动中...")
    print("=" * 70)

    # 标准化 models 格式为列表
    if not isinstance(models, list):
        # 单个模型，包装成列表
        model_configs = [{"name": "模型", "model": models}]
    else:
        model_configs = []
        for i, item in enumerate(models):
            if isinstance(item, dict):
                model_configs.append({
                    "name": item.get("name", f"模型 {i + 1}"),
                    "model": item["model"]
                })
            else:
                # 直接是模型实例
                model_configs.append({
                    "name": f"模型 {i + 1}",
                    "model": item
                })

    print(f"\n模型信息:")
    print(f"  - 模型数量: {len(model_configs)}")
    for i, cfg in enumerate(model_configs):
        param_count = sum(p.numel() for p in cfg['model'].parameters())
        print(f"  - {cfg['name']}: {param_count:,} 参数")
    print(f"  - 默认模型: {model_configs[0]['name']}")
    print(f"  - 词表大小: {tokenizer.vocab_size}")
    print(f"  - 设备: {device}")

    if authors_list:
        print(f"  - 作者数量: {len(authors_list)}")
    if styles_list:
        print(f"  - 风格数量: {len(styles_list)}")

    app = create_app(model_configs, tokenizer, device, authors_list, styles_list)

    print("\n" + "=" * 70)
    print("  服务器启动成功！")
    print("=" * 70)
    print(f"\n  访问地址：http://127.0.0.1:{port}")
    print("  按 Ctrl+C 停止服务器\n")
    print("=" * 70 + "\n")

    app.run(host=host, port=port, debug=debug)
