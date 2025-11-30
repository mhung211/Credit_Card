# Credit Card Customer

## Mô tả ngắn gọn
Dự án này tập trung vào phân tích và dự đoán khách hàng thẻ tín dụng có khả năng rời bỏ ngân hàng dựa trên dữ liệu hành vi sử dụng thẻ. 
---

## Mục lục
- [Giới thiệu](#giới-thiệu)  
- [Dataset](#dataset)  
- [Method](#method)  
- [Installation & Setup](#installation--setup)  
- [Usage](#usage)  
- [Results](#results)  
- [Project Structure](#project-structure)  
- [Challenges & Solutions](#challenges--solutions)  
- [Future Improvements](#future-improvements)  
- [Contributors](#contributors)  
- [Thông tin tác giả](#thông-tin-tác-giả)  
- [Contact](#contact)  
- [License](#license)  

---

## Giới thiệu
### Mô tả bài toán
Quản lý của một ngân hàng nhận thấy số lượng khách hàng ngừng sử dụng dịch vụ thẻ tín dụng tăng lên đáng kể. Vì vậy, ngân hàng muốn xây dựng một hệ thống có khả năng dự đoán sớm khách hàng có khả năng rời bỏ dịch vụ để có thể cung cấp dịch vụ tốt hơn để thay đổi ý định của họ. 

### Động lực và ứng dụng thực tế
Khách hàng rời bỏ ngân hàng gây tổn thất doanh thu và ảnh hưởng đến chiến lược kinh doanh. Dự án nhằm giúp ngân hàng dự đoán hành vi khách hàng để thực hiện các chương trình giữ chân khách hàng hiệu quả.  

### Mục tiêu cụ thể
- Làm sạch và xử lý dữ liệu thẻ tín dụng.  
- Mã hóa các biến phân loại và chuẩn hóa số liệu.  
- Xây dựng mô hình dự đoán khách hàng rời bỏ ngân hàng.  

---

## Dataset
### Nguồn dữ liệu
Dữ liệu từ Kaggle: [Credit Card Customers](https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers)  

### Mô tả các features
- **CLIENTNUM**: Mã khách hàng  
- **Attrition_Flag**: Attrited/Existing  
- **Customer_Age**: Tuổi khách hàng  
- **Gender**: Giới tính  
- **Dependent_count**: Số người phụ thuộc  
- **Education_Level**: Trình độ học vấn  
- **Marital_Status**: Tình trạng hôn nhân  
- **Income_Category**: Thu nhập  
- **Card_Category**: Loại thẻ  
- **Months_on_book**: Số tháng là khách hàng  
- **Total_Relationship_Count**: Số lượng sản phẩm khách hàng nắm 
- **Months_Inactive_12_mon**: Số tháng không hoạt động trong 12 tháng vừa  
- **Contacts_Count_12_mon**: Số lần liên hệ 12 tháng  
- **Credit_Limit**: Hạn mức tín dụng  
- **Total_Revolving_Bal**: Tổng số dư luân chuyển trên thẻ tín dụng  
- **Avg_Open_To_Buy**: Hạn mức tín dụng mở để mua (Trung bình 12 tháng qua) 
- **Total_Amt_Chng_Q4_Q1**: Thay đổi số tiền giao dịch Q4 so với Q1  
- **Total_Trans_Amt**: Tổng số tiền giao dịch trong 12 tháng qua 
- **Total_Trans_Ct**: Tổng số lượng giao dịch trong 12 tháng   
- **Total_Ct_Chng_Q4_Q1**: Thay đổi số lượng giao dịch Q4 so với Q1  
- **Avg_Utilization_Ratio**: Tỷ lệ sử dụng thẻ trung bình  

### Kích thước và đặc điểm dữ liệu
- Số lượng mẫu: ~10,000 khách hàng  
- Tổng số features: 23
- Số biến mục tiêu: 2
- Dữ liệu không có giá trị null

---

## Method
### Quy trình xử lý dữ liệu
1. **Đọc dữ liệu**: Sử dụng `read_BankChurners` từ `data_processing.py`.  
2. **Xử lý outlier**: Dùng hàm `remove_outlier` để loại bỏ giá trị ngoại lai.  
3. **Chuẩn hóa dữ liệu**: Áp dụng MinMax scaling với `minmax_scaler`.  
4. **Mã hóa biến phân loại**: `feature_encode` và `encode_label` cho biến phân loại và biến mục tiêu.  

### Thuật toán sử dụng
- Logistic Regression:
$$
\sigmoid(z) = \frac{1}{1 + e^{-z}}, \quad z = X\theta
$$
Hàm loss:
$$
L(\theta) = -\sum_{i} y_i \log(\hat{y_i}) + (1-y_i) \log(1-\hat{y_i})
$$

### Giải thích cách implement bằng NumPy
- Tính dot product giữa ma trận `X` và vector `theta`  
- Áp dụng sigmoid để chuyển sang xác suất
- Tính loss để theo dõi
- Cập nhật gradient bằng:  
$$
\theta := \theta - \eta \cdot X^T(\hat{y}-y)
$$  
- Tính loss của tập validation sau mỗi epochs để theo dõi overfitting
---

## Installation & Setup
```bash
# Clone repository
git clone https://github.com/mhung211/Credit_Card.git
cd Credict_Card

# Tạo virtual environment
python -m venv venv
venv\Scripts\activate     # Windows

# Cài đặt dependencies
pip install -r requirements.txt
