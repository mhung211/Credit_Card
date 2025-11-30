# Credit Card Customer Churn Prediction

## Mô tả ngắn gọn
Dự án này tập trung vào phân tích và dự đoán khách hàng thẻ tín dụng có khả năng rời bỏ ngân hàng (churn) dựa trên dữ liệu hành vi sử dụng thẻ. Chúng tôi xử lý dữ liệu, mã hóa biến, chuẩn hóa số liệu, và áp dụng các thuật toán học máy để xây dựng mô hình dự đoán.

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
Dự án giải quyết vấn đề dự đoán khách hàng sẽ rời bỏ ngân hàng (Attrition) dựa trên dữ liệu hành vi sử dụng thẻ tín dụng.  

### Động lực và ứng dụng thực tế
Khách hàng rời bỏ ngân hàng gây tổn thất doanh thu và ảnh hưởng đến chiến lược marketing. Dự đoán sớm giúp ngân hàng thực hiện các chương trình retention hiệu quả.  

### Mục tiêu cụ thể
- Làm sạch và xử lý dữ liệu thẻ tín dụng.  
- Mã hóa các biến phân loại và chuẩn hóa số liệu.  
- Xây dựng mô hình dự đoán khách hàng rời bỏ ngân hàng.  

---

## Dataset
### Nguồn dữ liệu
Dữ liệu từ Kaggle: [Credit Card Customers Dataset](https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers)  

### Mô tả các features
- **CLIENTNUM**: Mã khách hàng  
- **Attrition_Flag**: Churn/Existing  
- **Customer_Age**: Tuổi khách hàng  
- **Gender**: Giới tính  
- **Dependent_count**: Số người phụ thuộc  
- **Education_Level**: Trình độ học vấn  
- **Marital_Status**: Tình trạng hôn nhân  
- **Income_Category**: Thu nhập  
- **Card_Category**: Loại thẻ  
- **Months_on_book**: Số tháng là khách hàng  
- **Total_Relationship_Count**: Số lượng mối quan hệ với ngân hàng  
- **Months_Inactive_12_mon**: Tháng không hoạt động trong 12 tháng  
- **Contacts_Count_12_mon**: Số lần liên hệ 12 tháng  
- **Credit_Limit**: Hạn mức tín dụng  
- **Total_Revolving_Bal**: Tổng dư nợ quay vòng  
- **Avg_Open_To_Buy**: Dư tín dụng khả dụng trung bình  
- **Total_Amt_Chng_Q4_Q1**: Thay đổi tổng số tiền từ Q4 sang Q1  
- **Total_Trans_Amt**: Tổng giao dịch  
- **Total_Trans_Ct**: Tổng số giao dịch  
- **Total_Ct_Chng_Q4_Q1**: Thay đổi số giao dịch Q4 sang Q1  
- **Avg_Utilization_Ratio**: Tỷ lệ sử dụng trung bình  

### Kích thước và đặc điểm dữ liệu
- Số lượng mẫu: ~10,000 khách hàng  
- Các biến phân loại: 5  
- Các biến số liên tục: 15  
- Dữ liệu gốc cần làm sạch và loại bỏ outlier.  

---

## Method
### Quy trình xử lý dữ liệu
1. **Đọc dữ liệu**: Sử dụng `read_BankChurners` từ `data_processing.py`.  
2. **Xử lý outlier**: Dùng hàm `remove_outlier` để loại bỏ giá trị ngoại lai.  
3. **Chuẩn hóa dữ liệu**: Áp dụng MinMax scaling với `minmax_scaler`.  
4. **Mã hóa biến phân loại**: `feature_encode` và `encode_label` cho biến target.  

### Thuật toán sử dụng
- Logistic Regression (ví dụ):
\[
\sigma(z) = \frac{1}{1 + e^{-z}}, \quad z = X\theta
\]
Hàm loss: 
\[
L(\theta) = -\sum_{i} y_i \log(\hat{y_i}) + (1-y_i) \log(1-\hat{y_i})
\]

### Giải thích cách implement bằng NumPy
- Tính dot product giữa ma trận `X` và vector `theta`  
- Áp dụng sigmoid để chuyển sang xác suất  
- Cập nhật gradient bằng:  
\[
\theta := \theta - \eta \cdot X^T(\hat{y}-y)
\]  

---

## Installation & Setup
```bash
# Clone repository
git clone <repo_url>
cd project-name

# Tạo virtual environment
python -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows

# Cài đặt dependencies
pip install -r requirements.txt
