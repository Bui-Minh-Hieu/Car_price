from django import forms

MODEL_NAME_LIST = [
    'Vitz', 'Corolla', 'Alto', 'Civic', 'Wagon', 'Mirage', 'Prado', 'Vezel', 
    'City', 'N', 'Cultus', 'Fortuner', 'Mehran', 'Khyber', 'Dayz', 'Mira', 
    'Prius', 'Juke', 'Passo', 'Bolan', 'Swift', 'Liana', 'Margalla', 'Belta', 
    'Camry', 'A4', 'Noah', 'Hijet', 'Land', 'Aqua', 'Fit', 'Accord', 'Joy', 
    'Cast', 'Probox', 'Move', 'MR', 'Lancer', 'C-HR', 'Benz', 'Voxy', 'Mark', 
    'Cuore', 'Platz', 'A6', 'Baleno', 'Premio', 'Terios', 'Surf', 'Crown', 
    'Santro', 'Boon', 'Ek', 'Racer', 'Tanto', 'Esse', 'Aygo', '7', 'Grace', 
    'Kizashi', '3', 'Clipper', 'Ciaz', 'X-PV', 'Moco', 'Allion', 'Vamos', 
    'Pixis', 'Bego', 'Sunny', 'A5', 'Porte', 'ISIS', 'Rover', 'Hustler', 
    'Glory', 'Life', 'Rush', 'Jimny', 'Otti', 'Sportage', 'Excel', 'RX', 
    'Bravo', 'Kei', 'Sienta', 'Cross', 'EK', 'Corona', 'Note', 'Hilux', 
    'Spectra', 'Pajero', 'Tiida', 'Freed', 'C37', 'March', '5', 'A3', 
    'CT200h', 'Insight', 'X', 'Picanto', 'Celerio', 'Bluebird', 'Potohar', 
    'Stavic', 'Vitara', 'X6', 'Estima', 'Galant', 'Q7', 'V2', 'Flair', 
    'Roox', 'APV', 'FX', 'Stella', 'Carol', 'HR-V', 'Rav4', 'Hiace', 'Stream', 
    'Cayenne', 'Caldina', 'Pleo', 'Azwagon', 'Q2', 'Q3', 'Starlet', 'LX', 
    'Axela', 'Sonica', 'Patrol', 'CR-V', 'Wake', 'X1', 'Panamera', 'IST', 
    'Macan', 'Duet', 'Outlander', 'Acty', 'Pino', 'QQ', 'Zest', 'Wish', 
    'Aveo', 'Scrum', 'Dias', 'Terrano', 'Exclusive', 'B', 'Pride', 'Sirius', 
    'Uno', 'Splash', 'Solio', 'Cervo', 'Blue', 'I', 'Cooper', 'Succeed', 
    'Auris', 'Charade', 'iQ', 'Optra', 'H2', 'X5', 'Wrangler', 'Vanguard', 
    'Minica', 'Thats', 'Justy', 'Impreza', 'XF', 'Familia', 'Sx4', 'Latio', 
    'Cami', 'Avanza', 'Alphard', 'Rumion', 'Coupe', 'Palette', 'Skyline', 
    'Convoy', 'Altezza', 'L300', 'Revo', 'Cefiro', 'A7', 'X3', 'S40', 
    'Ractis', 'CJ'
]
MAU_XE_CHOICES = [(model, model) for model in MODEL_NAME_LIST]
HANG_XE_LIST = [
    'Toyota', 'Suzuki', 'Honda', 'Mitsubishi', 'Nissan', 'Daihatsu',
    'Audi', 'Chevrolet', 'Mercedes', 'Hyundai', 'Daewoo', 'BMW', 'FAW',
    'Range', 'DFSK', 'KIA', 'Lexus', 'United', 'SsangYong', 'Mazda',
    'Subaru', 'Porsche', 'Chery', 'Fiat', 'Land', 'MINI', 'Hummer',
    'Jeep', 'Jaguar', 'Adam', 'Volvo'
]
HANG_XE_CHOICES = [(company, company) for company in HANG_XE_LIST]
LOAI_DONG_CO_CHOICES = [('Petrol', 'Xăng'), ('Diesel', 'Dầu Diesel'), ('Hybrid', 'Hybrid')]
MAU_NGOAI_THAT_LIST = [
    'Silver', 'White', 'Black', 'Beige', 'Grey', 'Brown', 'Pink',
    'Assembly', 'Maroon', 'Burgundy', 'Gold', 'Blue', 'Red', 'Indigo',
    'Unlisted', 'Green', 'Turquoise', 'Orange', 'Bronze', 'Purple',
    'Yellow', 'Navy', 'Magenta', 'Wine'
]
MAU_NGOAI_THAT_CHOICES = [(color, color) for color in MAU_NGOAI_THAT_LIST]
DAY_CHUYEN_LAP_RAP_CHOICES = [('Imported', 'Nhập khẩu'), ('Local', 'Trong nước')]

KIEU_DANG_THAN_XE_LIST = ['Hatchback', 'Sedan', 'SUV', 'Cross Over', 'Van', 'Mini Van']
KIEU_DANG_THAN_XE_CHOICES = [(body_type, body_type) for body_type in KIEU_DANG_THAN_XE_LIST]

KIEU_HOP_SO_CHOICES = [('Automatic', 'Tự động'), ('Manual', 'Sàn')]
DANG_KY_XE_CHOICES = [('Un-Registered', 'Chưa đăng ký'), ('Registered', 'Đã đăng ký')]

# Tạo form
class CarPricePredictionForm(forms.Form):
    hang_xe = forms.ChoiceField(label='Hãng xe', choices=HANG_XE_CHOICES)  
    model_xe = forms.ChoiceField(label='Mẫu xe', choices=MAU_XE_CHOICES) 
    nam_san_xuat = forms.IntegerField(label='Năm sản xuất', min_value=1950, initial=2018)   
    so_km = forms.FloatField(label='Số km đã đi (log10)', initial=4.0)
    loai_dong_co = forms.ChoiceField(label='Loại động cơ', choices=LOAI_DONG_CO_CHOICES)
    dung_tich_dong_co = forms.FloatField(label='Dung tích động cơ (lít)', initial=2.0)
    mau_xe = forms.ChoiceField(label='Màu ngoại thất', choices=MAU_NGOAI_THAT_CHOICES)
    day_chuyen_lap_rap = forms.ChoiceField(label='Dây chuyền lắp ráp', choices=DAY_CHUYEN_LAP_RAP_CHOICES)
    kieu_dang_than_xe = forms.ChoiceField(label='Kiểu dáng thân xe', choices=KIEU_DANG_THAN_XE_CHOICES)
    loai_hop_so = forms.ChoiceField(label='Kiểu hộp số', choices=KIEU_HOP_SO_CHOICES)
    dang_ky_xe = forms.ChoiceField(label='Nơi đăng ký xe', choices=DANG_KY_XE_CHOICES)
    