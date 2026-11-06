# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:**

**Runtime Colab:** CPU/GPU

**Python / PyTorch / Ultralytics:**

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ            | Nghĩa ngắn                                                    |
| ------------- | ------------------------------------------------------------- |
| Prediction    | Kết quả model vừa đoán ra; chưa phải nhãn đúng                |
| Ground truth  | Nhãn do người gán theo quy tắc                                |
| Taxonomy      | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint    | File model `.pt` đã huấn luyện sẵn                            |
| Guideline     | Quy tắc gán nhãn do người viết                                |
| Annotator     | Người gán nhãn                                                |
| Reviewer / QC | Người kiểm tra nhãn                                           |
| Escalation    | Ca khó → hỏi Lab Coach, không tự bịa                          |
| Model score   | Số “tin” của model; không phải điểm chất lượng nhãn người     |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: `class_id`=… ; `class_name`=… ; `rank`=1 ; `score`=… ; `taxonomy_name`=…
   [
   {
   "sample_id": "traffic",
   "taxonomy_name": "ImageNet-1K",
   "rank": 1,
   "class_id": 468,
   "class_name": "cab",
   "score": 0.510915
   },
   ]

2. Record này nói về **cả bức ảnh** hay về **một vật thể riêng** trong ảnh? Vì sao bạn nghĩ vậy?
   - Record này nói về cả bức ảnh , vì record này không chứa bất kì thông tin tọa độ hay segmentation mask để khoanh từng vị trí cụ thể
3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?
   - Danh sách tên lớp đi kèm với taxonomy và model không thể tự nghĩ ra
4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?

- Việc ghi ra cả ba thay vì chỉ ghi mỗi tên lớp nhằm đảm bảo tính toàn vẹn dữ liệu, chống nhầm lẫn và giúp được người đọc lẫn máy tính hiểu được.

5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp?
   - Nếu trong ảnh có nhiều vật hoặc nhiều chủ đề, guideline cần xác định rõ các tiêu chí sau:
   * Xác định rõ các vật chủ thể chính thông qua các đặc điểm vật lý trong ảnh
   * Kiểm tra độ ưu tiên của các vật thể
   * Xác định ngưỡng che khuất và cắt xén vật thể
6. Vì sao không được lấy `score` của model làm ground truth?
   - Không được lấy 'model score' làm ground truth vì score chỉ là xác suất dự đoán của model

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name`=… ; `score`=… ; `bbox_xyxy`=[…, …, …, …] ; `bbox_width`=… ; `bbox_height`=…
   [
   {
   "class_id": 0,
   "class_name": "person",
   "score": 0.912625,
   "bbox_format": "xyxy",
   "bbox_xyxy": [
   385.33,
   69.24,
   498.92,
   348.92
   ],
   "bbox_width": 113.58,
   "bbox_height": 279.68
   },
   ]
2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?
   - Góc bên trên- trái nằm ở tọa độ [385.33,69.24]
   - Góc dưới phải nằm ở tọa độ [498.22,348.92]
   - Chiều rộng là 113.58 pixel
   - Chiều cao là 279.68 pixel
3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?
   - Khi hạ xuống ngưỡng 0.20 sẽ xuất hiện 17 vật thể
   - Khi hạ xuống ngưỡng 0.60 xuất hiện 6 vật thể
4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì?
   - Sửa các nhãn bị nhận diện nhầm
   - Xóa các nhãn bị diện diện sai
   - Căn chỉnh lại các tọa độ box
5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).
   - Không cắt xén: Cạnh của hộp không được cắt lẹm vào bất kỳ bộ phận chính nào của vật thể
   - Không thừa thãi: Không để dư thừa quá nhiều khoảng trống (ảnh nền/background) bên trong hộp.
   - Xử lý che khuất (Occlusion): Nếu vật thể bị một vật khác che khuất một phần, chỉ vẽ hộp ôm sát phần diện tích thực tế còn nhìn thấy của vật thể đó
6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?
   - Nhãn dán person với score là 0.61 nằm sát viền bên trái của khung hình, đã bị cắt mất toàn bộ phần thân, chỉ còn hở ra một phần cánh tay và bàn tay. Trường hợp này đưa lên hỏi Lab Coach trước khi hỏi guideline, vì tránh tranh cãi về định nghĩa đối tượng, phụ thuộc vào mục tiêu dự án, đồng nhất quy trình chuẩn hóa

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1.  Chép một record: `instance_id`=… ; `class_name`=… ; `score`=… ; số điểm polygon=`polygon_point_count`=… ; 3–5 điểm đầu của `polygon_xy`=…
    [{
    "instance_id": "kitchen-011",
    "class_name": "spoon",
    "score": 0.359147,
    "polygon_point_count": 37,
    "polygon_xy": [
    [
    529.0,
    66.0
    ],
    [
    529.0,
    72.0
    ],
    [
    528.0,
    73.0
    ],
    ]
    }
    ]
2.  So với box, polygon cho thêm thông tin gì về hình dạng vật thể?
    - So với hộp giới hạn (bounding box) vốn chỉ là một hình chữ nhật thô cứng, polygon (đa giác) cung cấp thêm những thông tin chi tiết sau về hình dạng vật thể:
    * Đường viền chính xác
    * Diện tích thực
    * Hình dáng bất quy tắc
    * Chi tiết che khuất
3.  `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?
    - instance_id (mã định danh thực thể) được sử dụng để phân biệt các cá thể vật thể độc lập nằm trong cùng một bức ảnh, đặc biệt là để tách biệt những vật thể thuộc cùng một loại với nhau.
    - class_id (Loại vật thể): Xác định vật đó là gì
    - instance_id (Cá thể trong 1 ảnh): Đánh số thứ tự để tách bạch các vật thể cùng loại đứng cạnh nhau trong một bức ảnh tĩnh
    - tracking_id (Theo dõi qua video): Ghi nhớ danh tính của vật thể xuyên suốt nhiều khung hình theo thời gian. instance_id không có chức năng này vì nó chỉ có giá trị tại một ảnh duy nhất và sẽ đánh số lại từ đầu ở ảnh tiếp theo

4.  Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.
    Quy tắc bám sát viền ở mức điểm ảnh (Pixel-perfect / Tight Boundary Rule):
    "Đường biên của mask phải đi sát chính xác vào ranh giới thực tế của vật thể, không được dư và không được thiếu. Cụ thể:
    Không lấn ra ngoài (No Overshooting): Tuyệt đối không bao gồm các điểm ảnh (pixel) của nền (background) hoặc của vật thể khác vào bên trong mask của vật đang vẽ.
    Không lẹm vào trong (No Undershooting): Không được vẽ gọt bớt hay cắt phạm vào bất kỳ phần rìa nào của vật thể.
    Chỉ vẽ phần nhìn thấy: Đối với các vị trí vật thể bị che khuất (occlusion), viền mask phải đi men theo sát mép của vật cản, không tự ý tưởng tượng hay vẽ xuyên qua phần bị lấp."
5.  Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?
    - Chiếc bát (bowl 0.50) đang bị chiếc cốc màu xanh (cup 0.45) che khuất một phần góc phải, hai vật này nằm sát và chạm đè lên nhau trên mặt bàn, Sử dụng Guideline.
    - Hiện tượng đồ vật thông thường che lấp nhau (occlusion) trong không gian 3D là tình huống cực kỳ phổ biến và tiêu chuẩn trong mọi bộ dữ liệu. Nó không gây ra sự mơ hồ về mặt định nghĩa đối tượng (ai cũng biết đó vẫn là cái bát).
    - Do đó, người gán nhãn chỉ cần tuân thủ quy tắc xử lý che khuất đã được chuẩn hóa sẵn trong Guideline (ví dụ: "Chỉ vẽ box/mask ôm sát phần diện tích của chiếc bát còn lộ ra, tuyệt đối không vẽ xuyên qua phần diện tích của chiếc cốc"). Việc đưa một tình huống phổ thông như thế này lên Lab Coach (Escalation) là lãng phí nguồn lực và không cần thiết, vì Escalation chỉ dành cho các ca thực sự dị biệt, chưa từng có tiền lệ.

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ                | Ground truth trông như thế nào? (đơn vị nhãn)                                                                         | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON                                                                                                  | Annotator làm gì?                                                                                                                 | Reviewer xem gì?                                                                                                            |
| --------------------- | --------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| Phân loại ảnh         | Một (hoặc một vài) nhãn chữ đại diện cho toàn bộ ảnh, kèm class_id và taxonomy_name                                   | Ảnh có bối cảnh hỗn hợp, nhiều vật thể đan xen nhau                                                                                                 | Nhìn tổng thể ảnh, áp dụng quy tắc ưu tiên trong guideline                                                                        | Kiểm tra xem nhãn được chọn có đại diện đúng cho chủ thể chính chưa,                                                        |
| Phát hiện vật thể     | Tọa độ của hộp giới hạn                                                                                               | Vật thể bị che khuất                                                                                                                                | Vẽ hộp (box) ôm sát (tight-fit) vào mép ngoài cùng của vật thể phần còn nhìn thấy được                                            | Soi các cạnh box xem có bị lẹm vào vật hay dư ra nền quá nhiều không;                                                       |
| Instance segmentation | Đa giác (Polygon/Mask) chi tiết đến từng điểm ảnh, kèm class_name và bắt buộc có instance_id để phân biệt các cá thể. | Ranh giới giữa hai vật thể cùng màu bị mờ (hòa vào nhau), đường viền vật thể có hình thù phức tạp/đứt gãy, các vật thể cùng lớp chồng lấp lên nhau. | Chấm/vẽ polygon bám sát đường viền pixel thực tế, không lấn nền hay lẹm vật; đánh số instance_id tách bạch cho các vật dính nhau. | Phóng to (zoom) vùng biên để soi đường polygon; kiểm tra xem các vật thể cùng loại đứng cạnh nhau có bị gộp chung ID không. |

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)
   - Quy tắc bảo vệ dữ liệu đối với bài lab này là tuyệt đối bảo mật, không sao chép, tải xuống hay chia sẻ bất kỳ hình ảnh và dữ liệu nào ra bên ngoài hệ thống gán nhãn quy định.
2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?

- Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi (out-of-scope), tôi sẽ dừng thao tác và báo cáo (escalate) ngay cho Lab Coach để xin chỉ thị xử lý tiếp theo.

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [ ] `REPORT.md` (file này đã điền)
- [ ] `day1_lab_outputs/classification_predictions.json`
- [ ] `day1_lab_outputs/detection_predictions.json`
- [ ] `day1_lab_outputs/segmentation_predictions.json`
- [ ] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [ ] `day1_lab_outputs/visuals/classification_top5.png`
- [ ] `day1_lab_outputs/visuals/detection_predictions.png`
- [ ] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [ ] Ô validation cuối notebook báo `PASS`
- [ ] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [ ] Đã nộp **link repository** trên LMS
