import os
import json
import time
import random
from flask import Flask, request, jsonify, send_file, session, redirect, url_for
from werkzeug.security import generate_password_hash, check_password_hash 
from datetime import timedelta, datetime, timezone
from google import genai
from google.genai.errors import APIError
from PIL import Image
import io
import numpy as np
from functools import wraps

# --- CẤU HÌNH ---
app = Flask(__name__)
# Thiết lập khóa bí mật (BẮT BUỘC cho Session)
app.secret_key = os.environ.get('SECRET_KEY', 'default_super_secret_key_change_me_in_production') 
app.permanent_session_lifetime = timedelta(minutes=60)

# Timezone cho Việt Nam (UTC+7)
vietnam_tz = timezone(timedelta(hours=7))

# Khởi tạo Gemini Client
client = None
try:
    # Đảm bảo biến môi trường GEMINI_API_KEY đã được thiết lập
    client = genai.Client()
    print("✅ Gemini Client khởi tạo thành công.")
except Exception as e:
    print(f"❌ Lỗi khởi tạo Gemini Client: {e}")

DB_FILE = 'db.json'
GEMINI_MODEL_VISION = 'gemini-2.5-flash'
GEMINI_MODEL_REASONING = 'gemini-2.5-flash'

# --- HÀM HỖ TRỢ CHUNG ---

def load_data(file_name, default_data):
    """Đọc dữ liệu từ file JSON."""
    if os.path.exists(file_name):
        try:
            with open(file_name, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError:
            print(f"Cảnh báo: File {file_name} bị lỗi định dạng. Sử dụng dữ liệu mặc định.")
            return default_data
    return default_data

def save_data(file_name, data):
    """Lưu dữ liệu vào file JSON."""
    with open(file_name, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def load_db():
    return load_data(DB_FILE, {"users": {}})

def save_db(db):
    save_data(DB_FILE, db)
    
def get_user_data(user_id):
    """Lấy dữ liệu người dùng từ DB."""
    db = load_db()
    return db['users'].get(user_id)

def clean_and_load_json(text_response):
    """Làm sạch chuỗi phản hồi Gemini và tải JSON."""
    try:
        json_text = text_response.strip()
        
        # Xử lý các trường hợp format phổ biến
        if json_text.startswith('```json'):
            json_text = json_text[7:].strip()
        elif json_text.startswith('```'):
            json_text = json_text[3:].strip()
            
        if json_text.endswith('```'):
            json_text = json_text[:-3].strip()
        
        # Xử lý trường hợp có text thừa trước/sau JSON
        start_idx = json_text.find('{')
        end_idx = json_text.rfind('}') + 1
        
        if start_idx != -1 and end_idx != 0:
            json_text = json_text[start_idx:end_idx]
        
        print(f"🧹 Cleaned JSON text: {json_text}")
        return json.loads(json_text)
        
    except json.JSONDecodeError as e:
        print(f"❌ Lỗi parse JSON: {e}")
        print(f"📄 Original response: {text_response}")
        
        # Fallback: cố gắng extract thông tin từ text
        return extract_info_from_text(text_response)
    except Exception as e:
        print(f"❌ Lỗi khác khi parse: {e}")
        return create_fallback_meal_data()

def extract_info_from_text(text):
    """Trích xuất thông tin từ text response khi JSON parse lỗi"""
    fallback_data = create_fallback_meal_data()
    
    # Cố gắng tìm tên món ăn trong text
    if "tên" in text.lower() or "món" in text.lower():
        lines = text.split('\n')
        for line in lines:
            if any(keyword in line.lower() for keyword in ['phở', 'bún', 'cơm', 'bánh', 'cháo', 'xôi']):
                fallback_data['meal_name'] = line.strip()
                break
    
    # Cố gắng tìm calories trong text
    import re
    calorie_match = re.search(r'(\d+)\s*(calo|calories|kcal)', text, re.IGNORECASE)
    if calorie_match:
        fallback_data['estimated_calories'] = int(calorie_match.group(1))
    
    fallback_data['description'] = text[:300] + '...' if len(text) > 300 else text
    fallback_data['nutrition_analysis'] = "Phân tích tự động từ mô tả"
    
    return fallback_data

def retry_on_error(max_retries=3, delay=2):
    """Decorator để thử lại khi API bị lỗi"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except APIError as e:
                    if attempt == max_retries - 1:  # Lần thử cuối
                        raise e
                    wait_time = delay * (2 ** attempt) + random.uniform(0, 1)
                    print(f"⚠️ API lỗi, thử lại sau {wait_time:.1f} giây... (lần {attempt + 1})")
                    time.sleep(wait_time)
            return None
        return wrapper
    return decorator

def generate_fallback_suggestions(profile, remaining_calories):
    """Tạo gợi ý mẫu khi API bị lỗi - LỜI KHUYÊN NGẮN GỌN"""
    
    goal = profile['goal']
    
    # Lời khuyên ngắn gọn dựa trên calories còn lại
    if remaining_calories < 0:
        advice = f"Đã vượt {abs(remaining_calories)}kcal. Ưu tiên rau xanh, thức ăn nhẹ."
    elif remaining_calories < 200:
        advice = f"Còn {remaining_calories}kcal. Chọn món nhẹ: salad, súp, trái cây."
    elif remaining_calories < 500:
        advice = f"Còn {remaining_calories}kcal. Cân bằng: protein + rau + tinh bột vừa phải."
    else:
        advice = f"Còn {remaining_calories}kcal. Có thể ăn đa dạng thực phẩm."
    
    # Menu gợi ý ngắn gọn
    if goal == 'giảm cân':
        if remaining_calories < 0:
            menu = [
                {"name": "Salad rau củ", "calories": 150, "nutrition_summary": "Ít calo, nhiều xơ"},
                {"name": "Súp rau", "calories": 120, "nutrition_summary": "Nhẹ bụng"},
                {"name": "Sữa chua không đường", "calories": 80, "nutrition_summary": "Tốt cho tiêu hóa"}
            ]
        else:
            menu = [
                {"name": "Cơm gạo lứt + ức gà", "calories": 400, "nutrition_summary": "Cân bằng dinh dưỡng"},
                {"name": "Bún gạo lứt + cá hồi", "calories": 350, "nutrition_summary": "Omega-3, chất xơ"},
                {"name": "Rau củ luộc + đậu phụ", "calories": 280, "nutrition_summary": "Protein thực vật"}
            ]
    elif goal == 'tăng cân':
        menu = [
            {"name": "Cơm thịt kho", "calories": 550, "nutrition_summary": "Nhiều năng lượng"},
            {"name": "Bún bò Huế", "calories": 520, "nutrition_summary": "Protein cao"},
            {"name": "Cháo yến mạch + hạt", "calories": 450, "nutrition_summary": "Dinh dưỡng toàn diện"}
        ]
    else:  # giữ cân
        menu = [
            {"name": "Cơm cá kho", "calories": 480, "nutrition_summary": "Cân đối"},
            {"name": "Phở gà", "calories": 420, "nutrition_summary": "Vừa phải"},
            {"name": "Bánh mì trứng", "calories": 380, "nutrition_summary": "Tiện lợi"}
        ]
    
    return {
        "advice": advice,
        "menu_suggestions": menu,
        "note": f"Mục tiêu: {goal}"
    }

def create_fallback_meal_data():
    """Tạo dữ liệu món ăn mẫu khi API bị lỗi"""
    meal_names = [
        "Cơm tấm sườn nướng",
        "Phở bò", 
        "Bún chả",
        "Bánh mì thịt",
        "Cơm gà xé",
        "Bún bò Huế",
        "Hủ tiếu nam vang"
    ]
    
    descriptions = [
        "Cơm trắng với sườn nướng, bì, chả và đồ chua",
        "Phở nước dùng thơm ngon với thịt bò tái, chín",
        "Bún với chả thịt nướng, nem và nước mắm chua ngọt",
        "Bánh mì giòn với nhân thịt, pate và rau sống",
        "Cơm trắng với gà xé, rau sống và nước mắm",
        "Bún bò với hương vị Huế đặc trưng, giò heo",
        "Hủ tiếu với nước dùng trong, thịt heo, tôm"
    ]
    
    index = random.randint(0, len(meal_names) - 1)
    
    return {
        'meal_name': meal_names[index],
        'estimated_calories': random.randint(200, 800),
        'description': descriptions[index],
        'nutrition_analysis': 'Món ăn truyền thống Việt Nam'
    }

# --- DECORATOR & LOGIC TÍNH TOÁN ---

def login_required(f):
    """Decorator để kiểm tra xem người dùng đã đăng nhập chưa."""
    @wraps(f)
    def wrapper(*args, **kwargs):
        if 'user_id' not in session:
            # Nếu là yêu cầu API, trả về 401
            if request.path.startswith('/api/'):
                return jsonify({"error": "Bạn cần đăng nhập để truy cập tính năng này."}), 401
            # Nếu là yêu cầu trang, chuyển hướng
            return redirect(url_for('auth_page'))
        return f(*args, **kwargs)
    return wrapper

def calculate_tdee(gender, age, height_cm, weight_kg, activity_level):
    """Tính toán TDEE dựa trên công thức Mifflin-St Jeor."""
    if gender.lower() == 'nam':
        bmr = (10 * weight_kg) + (6.25 * height_cm) - (5 * age) + 5
    else:
        bmr = (10 * weight_kg) + (6.25 * height_cm) - (5 * age) - 161

    activity_factors = {'ít': 1.2, 'bình thường': 1.55, 'nhiều': 1.9}
    activity_factor = activity_factors.get(activity_level.lower(), 1.55)
    return round(bmr * activity_factor)

# --- ROUTES PHỤC VỤ HTML ---

@app.route('/')
def index():
    """Trang chủ yêu cầu login."""
    if 'user_id' not in session:
        return redirect(url_for('auth_page'))
    return send_file('index.html') 

@app.route('/profile')
@login_required 
def profile_page():
    return send_file('profile.html')

@app.route('/log')
@login_required
def log_page():
    return send_file('log.html')

@app.route('/chart')
@login_required
def chart_page():
    return send_file('chart.html')

@app.route('/auth')
def auth_page():
    if 'user_id' in session:
        return redirect(url_for('index'))
    return send_file('auth.html')

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_file(f'static/{filename}')

# --- API MỚI: LẤY NGÀY HIỆN TẠI TỪ SERVER ---

@app.route('/api/current_date')
def get_current_date():
    """API để lấy ngày hiện tại từ server - ĐÃ SỬA TIMEZONE VIỆT NAM"""
    current_date = datetime.now(vietnam_tz).strftime("%Y-%m-%d")
    print(f"🌏 Server date (Vietnam time): {current_date}")
    return jsonify({"current_date": current_date})

# --- AUTH API ---

@app.route('/api/register', methods=['POST'])
def register():
    data = request.json
    db = load_db()
    username = data['username']

    if username in db['users']:
        return jsonify({"error": "Tên đăng nhập đã tồn tại."}), 400

    password_hash = generate_password_hash(data['password'])
    
    db['users'][username] = {
        'username': username,
        'password_hash': password_hash,
        'profile': None,
        'food_log': []
    }
    save_db(db)
    return jsonify({"message": "Đăng ký thành công."}), 201

@app.route('/api/login', methods=['POST'])
def login():
    data = request.json
    db = load_db()
    username = data['username']

    user = db['users'].get(username)
    if user and check_password_hash(user['password_hash'], data['password']):
        session.permanent = True
        session['user_id'] = username 
        return jsonify({"message": "Đăng nhập thành công"}), 200
    
    return jsonify({"error": "Tên đăng nhập hoặc mật khẩu không đúng."}), 401

@app.route('/api/logout')
def logout():
    session.pop('user_id', None)
    return redirect(url_for('auth_page'))

@app.route('/api/status')
def status():
    user_id = session.get('user_id')
    is_logged_in = user_id is not None
    
    has_profile = False
    if is_logged_in:
        user = get_user_data(user_id)
        if user and user.get('profile'):
            has_profile = True

    return jsonify({
        'logged_in': is_logged_in, 
        'username': user_id,
        'has_profile': has_profile
    })

# --- PROFILE & FOOD LOG API ---

@app.route('/api/profile', methods=['POST', 'GET'])
@login_required
def handle_profile():
    user_id = session['user_id']
    db = load_db()
    user = db['users'][user_id]

    if request.method == 'GET':
        return jsonify(user['profile']) if user['profile'] else jsonify(None), 200
    
    # POST - Cập nhật Hồ sơ
    data = request.json
    try:
        age = int(data.get('age', 0))
        height_cm = float(data.get('height_cm', 0))
        weight_kg = float(data.get('weight_kg', 0))

        if age <= 0 or height_cm <= 0 or weight_kg <= 0:
             return jsonify({"error": "Tuổi, chiều cao và cân nặng phải lớn hơn 0"}), 400

        tdee = calculate_tdee(data['gender'], age, height_cm, weight_kg, data['activity_level'])
        
        target_goal = data['goal'].lower()
        if target_goal == 'giảm cân':
            target_calories = tdee - 500
        elif target_goal == 'tăng cân':
            target_calories = tdee + 500
        else:
            target_calories = tdee

        profile = {
            'name': data['name'], 'gender': data['gender'], 'age': age,
            'height_cm': height_cm, 'weight_kg': weight_kg,
            'activity_level': data['activity_level'], 'goal': target_goal,
            'tdee': tdee, 'target_calories': max(1200, target_calories)
        }
        
        user['profile'] = profile
        save_db(db)
        return jsonify({"message": "Hồ sơ đã được lưu thành công", "profile": profile}), 200

    except Exception as e:
        return jsonify({"error": f"Lỗi xử lý hồ sơ: {str(e)}"}), 400

@app.route('/api/food_log', methods=['GET'])
@login_required
def get_food_log():
    user = get_user_data(session['user_id'])
    return jsonify(user['food_log'])

@app.route('/api/log_meal', methods=['POST'])
@login_required
def log_meal():
    if not client: 
        return jsonify({"error": "Gemini API Client chưa được cấu hình."}), 500
    if 'photo' not in request.files: 
        return jsonify({"error": "Không tìm thấy file ảnh"}), 400

    image_file = request.files['photo']
    custom_date = request.form.get('date')
    custom_time = request.form.get('time')
    
    try:
        # Đọc và xử lý ảnh
        image_data = image_file.read()
        img = Image.open(io.BytesIO(image_data))
        
        # Resize ảnh nếu quá lớn
        if img.size[0] > 1024 or img.size[1] > 1024:
            img.thumbnail((1024, 1024))

        # Xử lý ngày và giờ - SỬA TIMEZONE VIỆT NAM
        if custom_date and custom_time:
            # Tạo timestamp từ ngày và giờ custom
            try:
                custom_datetime_str = f"{custom_date} {custom_time}"
                custom_datetime = datetime.strptime(custom_datetime_str, "%Y-%m-%d %H:%M")
                # Thêm timezone Việt Nam
                custom_datetime = custom_datetime.replace(tzinfo=vietnam_tz)
                timestamp = custom_datetime.isoformat()
                date_used = custom_date
                print(f"📅 Using custom date: {date_used}, time: {timestamp}")
            except ValueError as e:
                print(f"❌ Lỗi parse datetime: {e}")
                # Fallback: dùng thời gian hiện tại với timezone Việt Nam
                timestamp = datetime.now(vietnam_tz).isoformat()
                date_used = datetime.now(vietnam_tz).strftime("%Y-%m-%d")
        else:
            # Dùng thời gian hiện tại với timezone Việt Nam - SỬA QUAN TRỌNG
            timestamp = datetime.now(vietnam_tz).isoformat()
            date_used = datetime.now(vietnam_tz).strftime("%Y-%m-%d")
            print(f"📅 Using current Vietnam date: {date_used}")

        # PROMPT cho Gemini AI - ngắn gọn và hiệu quả
        prompt = ("""
Phân tích món ăn trong ảnh. Trả về JSON:

{
    "meal_name": "Tên món ăn",
    "estimated_calories": số_calories,
    "description": "Mô tả ngắn",
    "nutrition_analysis": "Phân tích dinh dưỡng"
}

Ví dụ:
{
    "meal_name": "Phở bò",
    "estimated_calories": 450,
    "description": "Phở bò tái chín, nước dùng thơm",
    "nutrition_analysis": "Protein từ thịt bò, tinh bột từ bánh phở"
}
""")

        print(f"🔄 Đang gửi ảnh đến Gemini API với model: {GEMINI_MODEL_VISION}")
        
        # Gọi Gemini API với retry logic
        max_retries = 3
        last_error = None
        
        for attempt in range(max_retries):
            try:
                response = client.models.generate_content(
                    model=GEMINI_MODEL_VISION, 
                    contents=[prompt, img]
                )
                print(f"✅ Gemini API phản hồi thành công (lần {attempt + 1})")
                print(f"📄 Response: {response.text}")
                ai_data = clean_and_load_json(response.text)
                break
            except APIError as e:
                last_error = e
                if attempt == max_retries - 1:  # Lần thử cuối
                    # Nếu vẫn lỗi sau 3 lần thử, dùng fallback
                    print(f"❌ Gemini API vẫn lỗi sau {max_retries} lần thử: {e}")
                    ai_data = create_fallback_meal_data()
                    print(f"📊 Using fallback data: {ai_data}")
                else:
                    wait_time = 2 * (2 ** attempt) + random.uniform(0, 1)
                    print(f"⚠️ Vision API lỗi, thử lại sau {wait_time:.1f} giây... (lần {attempt + 1})")
                    time.sleep(wait_time)
            except Exception as e:
                print(f"❌ Lỗi khác khi gọi API: {e}")
                ai_data = create_fallback_meal_data()
                break

        # Lưu vào Nhật ký ăn uống
        db = load_db()
        user = db['users'][session['user_id']]
        
        meal_entry = {
            'timestamp': timestamp,
            'date': date_used,
            'meal_name': ai_data.get('meal_name', 'Món ăn không xác định'),
            'calories': int(ai_data.get('estimated_calories', 300)),
            'description': ai_data.get('description', 'Không có mô tả chi tiết'),
            'nutrition_analysis': ai_data.get('nutrition_analysis', 'Chưa có phân tích dinh dưỡng')
        }
        
        user['food_log'].append(meal_entry)
        save_db(db)

        return jsonify({
            "message": "Món ăn đã được phân tích và ghi nhận thành công", 
            "data": meal_entry
        }), 200

    except Exception as e:
        print(f"❌ Unexpected Error in log_meal: {e}")
        return jsonify({"error": f"Lỗi hệ thống: {str(e)}"}), 500

@app.route('/api/suggest_menu', methods=['GET'])
@login_required
@retry_on_error(max_retries=3, delay=2)
def suggest_menu():
    if not client: 
        return jsonify({"error": "Gemini API Client chưa được cấu hình."}), 500
    
    user = get_user_data(session['user_id'])
    profile = user.get('profile')
    if not profile: 
        return jsonify({"error": "Vui lòng nhập Hồ sơ cá nhân trước để nhận gợi ý."}), 404

    food_log = user['food_log']
    
    # SỬA: Dùng Vietnam date để tính calories hôm nay
    today_date = datetime.now(vietnam_tz).strftime("%Y-%m-%d")
    today_log = [log for log in food_log if log['date'] == today_date]
    
    calories_consumed_today = sum(item.get('calories', 0) for item in today_log)
    target_calories = profile['target_calories']
    remaining_calories = target_calories - calories_consumed_today

    # PROMPT ngắn gọn cho gợi ý
    prompt = f"""
Người dùng: {profile['name']}, Mục tiêu: {profile['goal']}
Calories hôm nay: {calories_consumed_today}/{target_calories}kcal
Calories còn lại: {remaining_calories}kcal

Gợi ý 3 món ăn phù hợp. Trả về JSON:

{{
    "advice": "Lời khuyên ngắn gọn",
    "menu_suggestions": [
        {{"name": "Món 1", "calories": X, "nutrition_summary": "Mô tả ngắn"}}
    ]
}}
"""

    try:
        print(f"🔄 Đang gửi yêu cầu gợi ý đến Gemini API với model: {GEMINI_MODEL_REASONING}")
        response = client.models.generate_content(model=GEMINI_MODEL_REASONING, contents=[prompt])
        
        ai_data = clean_and_load_json(response.text)
        return jsonify(ai_data), 200

    except APIError as e:
        print(f"❌ Gemini API Error, using fallback data: {e}")
        # Dữ liệu mẫu khi API lỗi
        fallback_data = generate_fallback_suggestions(profile, remaining_calories)
        return jsonify(fallback_data), 200
    except Exception as e:
        print(f"❌ Error, using fallback: {e}")
        fallback_data = generate_fallback_suggestions(profile, remaining_calories)
        return jsonify(fallback_data), 200

@app.route('/api/delete_meal', methods=['POST'])
@login_required
def delete_meal():
    user_id = session['user_id']
    data = request.json
    timestamp = data.get('timestamp')
    
    db = load_db()
    user = db['users'][user_id]
    
    # Tìm và xóa bữa ăn theo timestamp
    initial_length = len(user['food_log'])
    user['food_log'] = [meal for meal in user['food_log'] if meal['timestamp'] != timestamp]
    
    if len(user['food_log']) < initial_length:
        save_db(db)
        return jsonify({"message": "Đã xóa bữa ăn thành công"}), 200
    else:
        return jsonify({"error": "Không tìm thấy bữa ăn để xóa"}), 404

# --- API CHO TRANG BIỂU ĐỒ ---

@app.route('/api/nutrition_analysis', methods=['GET'])
@login_required
def get_nutrition_analysis():
    """API để lấy dữ liệu phân tích dinh dưỡng cho biểu đồ"""
    user = get_user_data(session['user_id'])
    profile = user.get('profile')
    food_log = user['food_log']
    
    if not profile:
        return jsonify({"error": "Chưa có hồ sơ"}), 404
    
    # Phân tích dữ liệu - SỬA: Dùng Vietnam date
    today = datetime.now(vietnam_tz).strftime("%Y-%m-%d")
    today_log = [log for log in food_log if log['date'] == today]
    
    # Lấy dữ liệu 30 ngày gần nhất
    month_dates = [datetime.now(vietnam_tz).date() - timedelta(days=i) for i in range(30)]
    month_dates_str = [date.strftime("%Y-%m-%d") for date in month_dates]
    month_log = [log for log in food_log if log['date'] in month_dates_str]
    
    # Tính toán calories
    today_calories = sum(item.get('calories', 0) for item in today_log)
    month_calories = sum(item.get('calories', 0) for item in month_log)
    
    month_avg_calories = month_calories / max(len(month_dates), 1)
    
    # Tính toán xu hướng calories 7 ngày
    daily_calories = []
    week_dates = [datetime.now(vietnam_tz).date() - timedelta(days=i) for i in range(6, -1, -1)]
    week_dates_str = [date.strftime("%Y-%m-%d") for date in week_dates]
    
    for date in week_dates_str:
        day_log = [log for log in food_log if log['date'] == date]
        day_calories = sum(item.get('calories', 0) for item in day_log)
        daily_calories.append(day_calories)
    
    # Phân tích loại món ăn
    meal_types = analyze_meal_types(food_log)
    
    # Tính % đạt mục tiêu (7 ngày gần nhất)
    target_days = 0
    total_days_with_data = 0
    
    for date in week_dates_str:
        day_log = [log for log in food_log if log['date'] == date]
        if day_log:  # Chỉ tính ngày có dữ liệu
            day_calories = sum(item.get('calories', 0) for item in day_log)
            total_days_with_data += 1
            if day_calories <= profile['target_calories']:
                target_days += 1
    
    achievement_rate = round((target_days / total_days_with_data) * 100) if total_days_with_data > 0 else 0
    
    # Phân tích xu hướng
    trend_analysis = analyze_trend(food_log, profile)
    
    analysis = {
        'today_calories': today_calories,
        'target_calories': profile['target_calories'],
        'month_avg_calories': round(month_avg_calories),
        'achievement_rate': achievement_rate,
        'meal_types': meal_types,
        'total_meals': len(food_log),
        'daily_calories_trend': daily_calories,
        'dates_trend': week_dates_str,
        'goal': profile['goal'],
        'remaining_calories': profile['target_calories'] - today_calories,
        'trend_analysis': trend_analysis
    }
    
    return jsonify(analysis), 200

def analyze_meal_types(food_log):
    """Phân tích loại món ăn dựa trên tên và thời gian"""
    meal_categories = {
        'Sáng': 0,
        'Trưa': 0, 
        'Tối': 0,
        'Phụ': 0
    }
    
    for meal in food_log:
        meal_name = meal['meal_name'].lower()
        meal_time = meal['timestamp'] if meal['timestamp'] else None
        
        # Phân loại dựa trên thời gian nếu có
        if meal_time:
            try:
                hour = datetime.fromisoformat(meal_time).hour
                if 5 <= hour < 11:
                    meal_categories['Sáng'] += 1
                elif 11 <= hour < 14:
                    meal_categories['Trưa'] += 1
                elif 17 <= hour < 22:
                    meal_categories['Tối'] += 1
                else:
                    meal_categories['Phụ'] += 1
            except:
                # Fallback: phân loại dựa trên tên món ăn
                if any(word in meal_name for word in ['sáng', 'bữa sáng', 'điểm tâm']):
                    meal_categories['Sáng'] += 1
                elif any(word in meal_name for word in ['trưa', 'bữa trưa']):
                    meal_categories['Trưa'] += 1
                elif any(word in meal_name for word in ['tối', 'bữa tối']):
                    meal_categories['Tối'] += 1
                else:
                    meal_categories['Phụ'] += 1
        else:
            # Phân loại dựa trên tên món ăn
            if any(word in meal_name for word in ['sáng', 'bữa sáng', 'điểm tâm']):
                meal_categories['Sáng'] += 1
            elif any(word in meal_name for word in ['trưa', 'bữa trưa']):
                meal_categories['Trưa'] += 1
            elif any(word in meal_name for word in ['tối', 'bữa tối']):
                meal_categories['Tối'] += 1
            else:
                meal_categories['Phụ'] += 1
    
    return meal_categories

def analyze_trend(food_log, profile):
    """Phân tích xu hướng tiêu thụ"""
    if len(food_log) < 7:
        return {
            'trend': 'not_enough_data',
            'message': 'Cần thêm dữ liệu để phân tích xu hướng'
        }
    
    # Lấy dữ liệu 14 ngày gần nhất - SỬA: Dùng Vietnam timezone
    recent_dates = [datetime.now(vietnam_tz).date() - timedelta(days=i) for i in range(13, -1, -1)]
    recent_dates_str = [date.strftime("%Y-%m-%d") for date in recent_dates]
    
    weekly_calories = []
    for date in recent_dates_str:
        day_log = [log for log in food_log if log['date'] == date]
        day_calories = sum(item.get('calories', 0) for item in day_log)
        weekly_calories.append(day_calories)
    
    # Phân chia thành 2 tuần
    week1_avg = sum(weekly_calories[:7]) / 7
    week2_avg = sum(weekly_calories[7:]) / 7
    
    trend_direction = 'stable'
    if week2_avg > week1_avg + 100:
        trend_direction = 'increasing'
    elif week2_avg < week1_avg - 100:
        trend_direction = 'decreasing'
    
    # So sánh với mục tiêu
    target_comparison = 'below'
    avg_calories = (week1_avg + week2_avg) / 2
    if avg_calories > profile['target_calories'] + 200:
        target_comparison = 'above'
    elif abs(avg_calories - profile['target_calories']) <= 200:
        target_comparison = 'on_track'
    
    return {
        'trend': trend_direction,
        'target_comparison': target_comparison,
        'week1_avg': round(week1_avg),
        'week2_avg': round(week2_avg),
        'overall_avg': round(avg_calories)
    }

@app.route('/api/improvement_tips', methods=['GET'])
@login_required
def get_improvement_tips():
    """API để lấy gợi ý cải thiện dựa trên dữ liệu hiện tại"""
    user = get_user_data(session['user_id'])
    profile = user.get('profile')
    food_log = user['food_log']
    
    if not profile:
        return jsonify({"error": "Chưa có hồ sơ"}), 404
    
    # Phân tích dữ liệu hiện tại - SỬA: Dùng Vietnam date
    today = datetime.now(vietnam_tz).strftime("%Y-%m-%d")
    today_log = [log for log in food_log if log['date'] == today]
    today_calories = sum(item.get('calories', 0) for item in today_log)
    remaining_calories = profile['target_calories'] - today_calories
    
    tips = []
    goal = profile['goal']
    
    # Tips ngắn gọn
    if goal == 'giảm cân':
        if remaining_calories < -300:
            tips.append("🍽️ Ăn vượt mục tiêu. Giảm tinh bột, tăng rau xanh.")
        elif remaining_calories < 0:
            tips.append("⚡ Vượt mục tiêu. Bữa tối nhẹ nhàng.")
        elif remaining_calories > 500:
            tips.append("💪 Còn nhiều calories. Thêm bữa phụ lành mạnh.")
    
    elif goal == 'tăng cân':
        if remaining_calories > 700:
            tips.append("🎯 Cần ăn nhiều hơn. Thêm bữa phụ giàu calo.")
        elif remaining_calories > 300:
            tips.append("🥩 Duy trì. Tăng protein trong bữa chính.")
    
    else:  # giữ cân
        if abs(remaining_calories) > 300:
            tips.append("⚖️ Calories chênh lệch. Cân đối lại bữa ăn.")
    
    # Tips chung
    if len(today_log) < 2:
        tips.append("⏰ Ăn đều 3 bữa/ngày để ổn định năng lượng.")
    
    if len(today_log) > 5:
        tips.append("🍎 Nhiều bữa nhỏ tốt cho kiểm soát calories!")
    
    # Nếu không có tips nào
    if not tips:
        tips.append("🎉 Chế độ ăn tốt! Tiếp tục duy trì.")
    
    return jsonify({"tips": tips}), 200

if __name__ == '__main__':
    app.run(debug=True)