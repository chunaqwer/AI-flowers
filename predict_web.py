import os
import torch
import numpy as np
from PIL import Image
from train import get_model
import torchvision.transforms as transforms
from flask import Flask, request, render_template, redirect, url_for, flash, jsonify
from werkzeug.utils import secure_filename

MODEL_LIST = [
    'resnet18',
    'alexnet',
    'mobilenet_v3_small',
    'vit_b_16',
    'vgg16',
    'deit_base_patch16_ls',
    'efficientvim_m3',
    'convnext_small',
    'custom_fused',
]

FLOWERS102_CLASSES = [
    '粉色报春花', '硬叶袋兰', '坎特伯雷钟', '甜豌豆', '英国金盏菊',
    '百合', '月亮兰', '天堂鸟', '乌头', '球蓟', '金鱼草', '马蹄莲',
    '帝王花', '矛蓟', '黄鸢尾', '球花', '紫锥菊', '秘鲁百合',
    '气球花', '巨型白马蹄莲', '火百合', '针垫花', '棋盘花', '红姜',
    '葡萄风信子', '虞美人', '威尔士王子羽毛', '无梗龙胆', '洋蓟', '须苞石竹',
    '康乃馨', '花园福禄考', '迷雾之恋', '墨西哥翠菊', '高山海冬青', '红唇卡特兰',
    '披针叶花', '大阿司特', '暹罗郁金香', '四旬玫瑰', '巴贝顿雏菊', '水仙', '剑兰',
    '一品红', '深蓝波莱罗', '墙花', '万寿菊', '毛茛', '牛眼菊', '普通蒲公英',
    '矮牵牛', '野三色堇', '报春花', '向日葵', '天竺葵', '兰达夫主教', '高雪轮', '天竺葵',
    '橙色大丽花', '粉黄大丽花', '穗花姜', '日本银莲花', '黑心菊', '银叶灌木',
    '加州罂粟', '非洲菊', '春番红花', '德国鸢尾', '银莲花', '树罂粟', '勋章菊',
    '杜鹃花', '睡莲', '玫瑰', '曼陀罗', '牵牛花', '西番莲', '莲花', '油点百合',
    '火鹤花', '鸡蛋花', '铁线莲', '朱槿', '耧斗菜', '沙漠玫瑰', '锦葵', '木兰',
    '仙客来', '水田芥', '美人蕉', '朱顶红', '蜂香薄荷', '气生苔', '毛地黄', '九重葛',
    '山茶花', '锦葵', '墨西哥牵牛', '凤梨', '毛毯花', '凌霄花', '黑果鸢尾',
    '普通郁金香', '野蔷薇'
]




transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app = Flask(__name__)
app.secret_key = 'flowers102_secret'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_image(model, img_path, device, confidence_threshold):
    import time
    img = Image.open(img_path).convert('RGB')
    x = transform(img).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        start_time = time.time()
        out = model(x)
        infer_time = time.time() - start_time
        prob_all = torch.softmax(out, dim=1)[0]
        pred = prob_all.argmax(dim=0).item()
        prob = prob_all[pred].item()

    if prob < confidence_threshold:
        return None, None, infer_time # Indicate prediction failure due to low confidence

    return pred, prob, infer_time

model_cache = {}

def get_loaded_model(model_name, device, weight_type='pretrain'):
    cache_key = f"{model_name}_{weight_type}"
    if cache_key in model_cache:
        return model_cache[cache_key]

    if weight_type == 'pretrain':
        weight_dir = 'pretrain-weights'
    elif weight_type == 'custom':
        weight_dir = 'weights' # Assuming custom weights are in 'weights' folder
    else:
        return None # Invalid weight type

    weight_path = os.path.join(weight_dir, f'{model_name}_best.pth')
    if not os.path.exists(weight_path):
        return None
    checkpoint = torch.load(weight_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    if model_name.startswith('convnext'):
        # For ConvNeXt, we need to pass the variant (e.g., 'small')
        # Assuming the variant is part of the model_name string, e.g., 'convnext_small'
        variant = model_name.split('_')[-1] if '_' in model_name else 'small'
        model = get_model(model_name, num_classes=102, variant=variant).to(device)
    else:
        model = get_model(model_name, num_classes=102).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    model_cache[cache_key] = model
    return model

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        model_name = request.form.get('model_name')
        file = request.files.get('file')
        confidence_threshold = float(request.form.get('confidence_threshold', 0.5)) # Default to 0.5 if not provided
        if not model_name or model_name not in MODEL_LIST:
            return jsonify({'error': '请选择模型'})
        if not file or file.filename == '':
            return jsonify({'error': '请选择图片文件'})
        if not allowed_file(file.filename):
            return jsonify({'error': '仅支持png/jpg/jpeg格式图片'})
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        weight_type = request.form.get('weight_type', 'pretrain') # Get weight_type from form
        model = get_loaded_model(model_name, device, weight_type)
        if model is None:
            return jsonify({'error': f'未找到权重文件: weights/{model_name}_best.pth'})
        pred, prob, infer_time = predict_image(model, file_path, device, confidence_threshold)

        if pred is None or prob is None:
            return jsonify({'error': f'预测置信度过低 ({prob:.2f})，低于阈值 ({confidence_threshold:.2f})，预测失败。', 'infer_time': infer_time})
        cls_name = FLOWERS102_CLASSES[pred] if pred < len(FLOWERS102_CLASSES) else str(pred)
        return jsonify({
            'image_url': url_for('uploaded_file', filename=filename),
            'filename': filename,
            'cls_name': cls_name,
            'prob': prob,
            'model_name': model_name,
            'weight_type_display': '预训练' if request.form.get('weight_type') == 'pretrain' else '自定义',
            'infer_time': infer_time
        })
    return render_template('index.html', model_list=MODEL_LIST)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    from flask import send_from_directory
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

import pandas as pd

@app.route('/chart')
def show_chart():
    print(f"Request args: {request.args}")
    model_name = request.args.get('model_name', MODEL_LIST[0])
    weight_type = request.args.get('weight_type', 'pretrain') # 'pretrain' or 'custom'

    # Determine CSV file path based on weight_type
    if weight_type == 'pretrain':
        csv_file_path = os.path.join('pretrain-weights', f'{model_name}_train_log.csv')
    elif weight_type == 'custom':
        csv_file_path = os.path.join('weights', f'{model_name}_train_log.csv')
    else:
        flash('无效的权重类型')
        return redirect(url_for('index'))

    print(f"Attempting to load CSV from: {csv_file_path}")
    print(f"File exists: {os.path.exists(csv_file_path)}")

    data = {
        'epoch': [],
        'val_acc': [],
        'val_precision': [],
        'val_recall': [],
        'val_f1': []
    }
    weight_type_display = '预训练' if weight_type == 'pretrain' else '自定义'

    if os.path.exists(csv_file_path):
        try:
            df = pd.read_csv(csv_file_path)
            data['epoch'] = df['epoch'].tolist() if 'epoch' in df else []
            data['val_acc'] = df['val_acc'].tolist() if 'val_acc' in df else [] # Common column name for accuracy
            data['val_precision'] = df['val_precision'].tolist() if 'val_precision' in df else []
            data['val_recall'] = df['val_recall'].tolist() if 'val_recall' in df else []
            data['val_f1'] = df['val_f1_score'].tolist() if 'val_f1_score' in df else (df['val_f1'].tolist() if 'val_f1' in df else []) # Check for f1_score or f1
        except Exception as e:
            flash(f'读取日志文件失败: {e}')
    else:
        flash(f'未找到日志文件: {csv_file_path}')

    return render_template('chart.html', 
                           model_list=MODEL_LIST, 
                           data=data, 
                           model_name=model_name, 
                           weight_type=weight_type,
                           weight_type_display=weight_type_display)

if __name__ == '__main__':
    app.run(debug=True)
